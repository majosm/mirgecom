"""Demonstrate multiple coupled volumes."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
from mirgecom.mpi import mpi_entry_point
import numpy as np
from functools import partial
from pytools.obj_array import make_obj_array
import pyopencl as cl
import pyopencl.tools as cl_tools

from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from meshmode.dof_array import DOFArray
from arraycontext import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from grudge.trace_pair import (
    TracePair,
    interior_trace_pairs
)
from grudge.dof_desc import DTAG_BOUNDARY, DISCR_TAG_BASE, DOFDesc, as_dofdesc
import grudge.op as op
from mirgecom.navierstokes import ns_operator
from mirgecom.diffusion import (
    diffusion_operator,
    NeumannDiffusionBoundary,
    InterfaceDiffusionBoundary
)
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    AdiabaticSlipBoundary,
    TemperatureCoupledSlipBoundary
)
from mirgecom.initializers import (
    Lump,
    AcousticPulse
)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.flux import (
    gradient_flux_central
)
from mirgecom.operators import (
    grad_operator
)
from mirgecom.gas_model import (
    project_fluid_state,
    make_fluid_state_trace_pairs
)

from mirgecom.gas_model import (
    GasModel,
    make_fluid_state
)
from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_cl_device_info,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def fluid_temp_gradient(
        discr, gas_model, state, boundaries, time=0.0, quadrature_tag=None,
        mask=None):
    actx = state.array_context

    if mask is None:
        mask = discr.zeros(actx) + 1

    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quadrature_tag)
    dd_faces = DOFDesc("all_faces", quadrature_tag)

    mask_quad = discr.project(dd_base, dd_vol, mask)
    mask_allfaces = discr.project(dd_vol, dd_faces, mask_quad)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quadrature_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(discr, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(discr, local_dd, local_dd_quad, utpair.ext)
        )

    boundary_states = {
        btag: project_fluid_state(
            discr, dd_base,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            state, gas_model) for btag in boundaries
    }

    cv_interior_pairs = [
        # Get the interior trace pairs onto the surface quadrature
        # discretization (if any)
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, state.cv)
    ]

    mask_interior_pairs = [
        interp_to_surf_quad(tpair)
        for tpair in interior_trace_pairs(discr, mask)
    ]

    tseed_interior_pairs = None
    if state.is_mixture > 0:
        # If this is a mixture, we need to exchange the temperature field because
        # mixture pressure (used in the flux calculations) depends on
        # temperature and we need to seed the temperature calculation for the
        # (+) part of the partition boundary with the remote temperature data.
        tseed_interior_pairs = [
            # Get the interior trace pairs onto the surface quadrature
            # discretization (if any)
            interp_to_surf_quad(tpair)
            for tpair in interior_trace_pairs(discr, state.temperature)
        ]

    quadrature_state = \
        project_fluid_state(discr, dd_base, dd_vol, state, gas_model)
    interior_state_pairs = make_fluid_state_trace_pairs(cv_interior_pairs,
                                                        gas_model,
                                                        tseed_interior_pairs)

    def gradient_flux_interior(tpair, mask_tpair):
        dd = tpair.dd
        normal = thaw(discr.normal(dd), actx)
        both_inside = mask_tpair.int * mask_tpair.ext
        flux = gradient_flux_central(tpair, normal) * both_inside
        return op.project(discr, dd, dd.with_dtag("all_faces"), flux)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # Capture the temperature for the interior faces for grad(T) calc
    # Note this is *all interior faces*, including partition boundaries
    # due to the use of *interior_state_pairs*.
    t_interior_pairs = [TracePair(state_pair.dd,
                                  interior=state_pair.int.temperature,
                                  exterior=state_pair.ext.temperature)
                        for state_pair in interior_state_pairs]

    t_flux_bnd = (

        # Domain boundaries
        mask_allfaces
        * sum(boundaries[btag].temperature_gradient_flux(
            discr,
            # Make sure we get the state on the quadrature grid
            # restricted to the tag *btag*
            as_dofdesc(btag).with_discr_tag(quadrature_tag),
            gas_model=gas_model,
            state_minus=boundary_states[btag],
            time=time)
            for btag in boundary_states)

        # Interior boundaries
        + sum(
            gradient_flux_interior(t_tpair, mask_tpair)
            for t_tpair, mask_tpair in zip(t_interior_pairs, mask_interior_pairs))
    )

    # Fluxes in-hand, compute the gradient of temperature and mpi exchange it
    return mask_quad * grad_operator(
        discr, dd_vol, dd_faces, quadrature_state.temperature, t_flux_bnd)


def wall_temp_gradient(
        discr, temp, kappa, boundaries, quadrature_tag=None, mask=None):
    # TODO
    if quadrature_tag is not None:
        raise NotImplementedError

    actx = temp.array_context

    if mask is None:
        mask = discr.zeros(actx) + 1

    def grad_flux_interior(temp_tpair, mask_tpair):
        normal = thaw(discr.normal(temp_tpair.dd), actx)
        # Hard-coding central per [Bassi_1997]_ eqn 13
        from mirgecom.flux import gradient_flux_central
        both_inside = mask_tpair.int * mask_tpair.ext
        flux_weak = gradient_flux_central(temp_tpair, normal) * both_inside
        return discr.project(temp_tpair.dd, "all_faces", flux_weak)

    # Temperature gradient for conductive heat flux: [Ihme_2014]_ eqn (3b)
    # - now computed, *not* communicated
    def grad_flux_bnd(btag):
        return -boundaries[btag].get_gradient_flux(
            discr, quad_tag=DISCR_TAG_BASE, dd=as_dofdesc(btag), alpha=kappa, u=temp)

    from grudge.trace_pair import (
        interior_trace_pair,
        cross_rank_trace_pairs)

    temp_int_tpair = interior_trace_pair(discr, temp)
    temp_part_pairs = cross_rank_trace_pairs(discr, temp)

    mask_int_tpair = interior_trace_pair(discr, mask)
    mask_part_pairs = cross_rank_trace_pairs(discr, mask)

    flux_bnd = (
        grad_flux_interior(temp_int_tpair, mask_int_tpair)
        + sum(
            grad_flux_interior(temp_tpair, mask_tpair)
            for temp_tpair, mask_tpair in zip(temp_part_pairs, mask_part_pairs))
        + (
            discr.project("vol", "all_faces", mask)
            * sum(grad_flux_bnd(btag) for btag in boundaries))
        )

    from mirgecom.operators import grad_operator
    return mask * grad_operator(
        discr, as_dofdesc("vol"), as_dofdesc("all_faces"), temp, flux_bnd)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_overintegration=False,
         use_leap=False, use_profiling=False, casename=None,
         rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    # timestepping control
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    t_final = 1.2
    current_cfl = 1.0
    current_dt = .000625
    current_t = 0
    constant_cfl = False

    assert np.abs(t_final/current_dt - np.floor(t_final/current_dt)) < 1e-12

    # some i/o frequencies
    nstatus = 1
    nrestart = 100
    nviz = 10
    nhealth = 1

    dim = 2
    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_filename:  # read the grid from restart data
        rst_filename = f"{rst_filename}-{rank:04d}.pkl"
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_filename)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_parts"] == num_parts
    else:  # generate the grid from scratch
        def generate_mesh():
            from meshmode.mesh.io import read_gmsh
            return read_gmsh("multivolume.msh", force_ambient_dim=2)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        default_simplex_group_factory, QuadratureSimplexGroupFactory

    order = 3
    discr = EagerDGDiscretization(
        actx, local_mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=local_mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
        },
        mpi_communicator=comm
    )
    nodes = thaw(discr.nodes(), actx)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    y_np = actx.to_numpy(nodes[1][0])
    y_element_avg = np.sum(y_np, axis=1)/y_np.shape[1]
    fluid_mask_np = (
        np.broadcast_to(y_element_avg > 0, y_np.shape[::-1]).astype(np.float64).T)
    fluid_mask = DOFArray(actx, (actx.from_numpy(fluid_mask_np),))
    wall_mask = 1 - fluid_mask

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    mu = 0.
    wall_kappa = 0.5
    fluid_kappa = 2
    eos = IdealSingleGas()
    transport = SimpleTransport(
        viscosity=mu,
        thermal_conductivity=fluid_kappa)
    gas_model = GasModel(eos=eos, transport=transport)
    wall_density = 1
    wall_heat_capacity = eos.heat_capacity_cv()
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    orig[1] += 0.25
    base_initializer = Lump(dim=dim, center=orig, velocity=vel, rhoamp=0.0)
    base_fluid_state = make_fluid_state(base_initializer(nodes), gas_model)
    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        current_wall_temperature = restart_data["wall_temperature"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        acoustic_pulse = AcousticPulse(dim=dim, amplitude=1.0, width=.05,
                                       center=orig)
        pulse_fluid_state = make_fluid_state(
            acoustic_pulse(x_vec=nodes, cv=base_fluid_state.cv, eos=eos),
            gas_model)
        current_cv = (
            fluid_mask * pulse_fluid_state.cv
            + (1 - fluid_mask) * base_fluid_state.cv)
        current_wall_temperature = (
            wall_mask * pulse_fluid_state.temperature
            + (1 - wall_mask) * base_fluid_state.temperature)

    current_state = make_obj_array([current_cv, current_wall_temperature])

    visualizer = make_visualizer(discr, vis_order=order+1)

    initname = "multivolume"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def get_energies(state):
        from grudge.op import integral
        fluid_energy = integral(discr, "vol", fluid_mask * state[0].energy)
        wall_energy = integral(
            discr, "vol", wall_mask * wall_density * wall_heat_capacity * state[1])
        return fluid_energy, wall_energy, fluid_energy + wall_energy

    fluid_energy_init, wall_energy_init, total_energy_init = get_energies(
        current_state)

    def my_write_status(step, t, state):
        fluid_energy, wall_energy, total_energy = get_energies(state)
        fluid_diff = (fluid_energy - fluid_energy_init)/fluid_energy_init
        wall_diff = (wall_energy - wall_energy_init)/wall_energy_init
        total_diff = (total_energy - total_energy_init)/total_energy_init
        print(
            f"{fluid_energy=} ({fluid_diff/100}%), "
            f"{wall_energy=}, ({wall_diff/100}%), "
            f"{total_energy=}, ({total_diff/100}%)")

    def my_write_viz(step, t, state, dv=None):
        if dv is None:
            fluid_state = make_fluid_state(state[0], gas_model)
            dv = fluid_state.dv
        rhs = my_rhs(t, state)
        viz_fields = [("fluid_cv", state[0]),
                      ("fluid_dv", dv),
                      ("wall_temperature", state[1]),
                      ("wall_energy", wall_density * wall_heat_capacity * state[1]),
                      ("temperature", dv.temperature*fluid_mask+state[1]*wall_mask),
                      ("fluid_rhs", rhs[0]),
                      ("wall_rhs", rhs[1]),
                      ("fluid_mask", fluid_mask),
                      ("wall_mask", wall_mask)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state[0],
                "wall_temperature": state[1],
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": num_parts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(pressure):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", pressure) \
           or check_range_local(discr, "vol", pressure, .8, 1.5):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")
        return health_error

    def my_pre_step(step, t, dt, state):
        fluid_state = make_fluid_state(state[0], gas_model)
        dv = fluid_state.dv

        try:
            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_status = check_step(step=step, interval=nstatus)
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_status:
                my_write_status(step=step, t=t, state=state)

            if do_health:
                health_errors = global_reduce(my_health_check(dv.pressure), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(step=step, t=t, state=state, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)
        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state[0], eos)
            logmgr.tick_after()
        return state, dt

    from meshmode.discretization.connection import make_opposite_face_connection
    interface_swap_connection = make_opposite_face_connection(
        actx, discr.connection_from_dds("vol", DTAG_BOUNDARY("Interface")))

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state[0], gas_model=gas_model)
        wall_temperature = state[1]
        # Compute thermal conductivity
        kappa = (
            fluid_mask * fluid_state.thermal_conductivity
            + wall_mask * wall_kappa)
        # Project and face-swap
        fluid_interface_temperature = discr.project(
            "vol", DTAG_BOUNDARY("Interface"), fluid_state.temperature)
        fluid_interface_temperature_swapped = interface_swap_connection(
            fluid_interface_temperature)
        wall_interface_temperature = discr.project(
            "vol", DTAG_BOUNDARY("Interface"), wall_temperature)
        wall_interface_temperature_swapped = interface_swap_connection(
            wall_interface_temperature)
        interface_kappa = discr.project("vol", DTAG_BOUNDARY("Interface"), kappa)
        interface_kappa_swapped = interface_swap_connection(interface_kappa)
        # Create wall BCs without the temperature gradient
        wall_boundaries_no_grad = {
            DTAG_BOUNDARY("Lower Sides"): NeumannDiffusionBoundary(0),
            DTAG_BOUNDARY("Interface"): InterfaceDiffusionBoundary(
                fluid_interface_temperature_swapped,
                (0*wall_interface_temperature,)*dim,
                interface_kappa_swapped)
        }
        # Compute the temperature gradient in the wall, project and face-swap
        wall_grad_temperature = wall_temp_gradient(
            discr, wall_temperature, kappa, wall_boundaries_no_grad,
            quadrature_tag=None, mask=wall_mask)
        wall_interface_grad_temperature = discr.project(
            "vol", DTAG_BOUNDARY("Interface"), wall_grad_temperature)
        wall_interface_grad_temperature_swapped = interface_swap_connection(
            wall_interface_grad_temperature)
        # Construct the fluid boundaries
        fluid_boundaries = {
            DTAG_BOUNDARY("Upper Sides"): AdiabaticSlipBoundary(),
            DTAG_BOUNDARY("Interface"): TemperatureCoupledSlipBoundary(
                wall_interface_temperature_swapped,
                wall_interface_grad_temperature_swapped,
                interface_kappa_swapped)
        }
        # Compute the temperature gradient in the fluid, project and face-swap
        fluid_grad_temperature = fluid_temp_gradient(
            discr, gas_model, fluid_state, fluid_boundaries, time=t, mask=fluid_mask)
        fluid_interface_grad_temperature = discr.project(
            "vol", DTAG_BOUNDARY("Interface"), fluid_grad_temperature)
        fluid_interface_grad_temperature_swapped = interface_swap_connection(
            fluid_interface_grad_temperature)
        # Construct the wall BCs (now with the temperature gradient)
        wall_boundaries = {
            DTAG_BOUNDARY("Lower Sides"): NeumannDiffusionBoundary(0),
            DTAG_BOUNDARY("Interface"): InterfaceDiffusionBoundary(
                fluid_interface_temperature_swapped,
                fluid_interface_grad_temperature_swapped,
                interface_kappa_swapped)
        }
        # Compute the RHS
        return make_obj_array([
            ns_operator(
                discr, state=fluid_state, boundaries=fluid_boundaries,
                gas_model=gas_model, time=t, quadrature_tag=quadrature_tag,
                mask=fluid_mask),
            1/(wall_density * wall_heat_capacity) * diffusion_operator(
                discr, quad_tag=quadrature_tag, alpha=kappa,
                boundaries=wall_boundaries, u=wall_temperature, mask=wall_mask)])

    current_fluid_state = make_fluid_state(current_state[0], gas_model)
    current_dt = get_sim_timestep(discr, current_fluid_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    my_write_viz(step=current_step, t=current_t, state=current_state)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "multivolume"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--lazy", action="store_true",
        help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--restart_file", help="root name of restart file")
    parser.add_argument("--casename", help="casename to use for i/o")
    args = parser.parse_args()
    if args.profiling:
        if args.lazy:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        actx_class = PytatoPyOpenCLArrayContext if args.lazy \
            else PyOpenCLArrayContext

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(use_logmgr=args.log, use_overintegration=args.overintegration,
         use_leap=args.leap, use_profiling=args.profiling,
         casename=casename, rst_filename=rst_filename, actx_class=actx_class)

# vim: foldmethod=marker
