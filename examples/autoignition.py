"""Demonstrate combustive mixture with Pyrometheus."""

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
import sys
import logging
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from arraycontext import thaw
from meshmode.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    generate_and_distribute_mesh,
    get_next_timestep,
    write_visfile
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)
from mirgecom.fluid import logmgr_add_fluid_quantities

import cantera
import pyrometheus as pyro

logger = logging.getLogger(__name__)


class SimError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


def main(ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, constant_cfl=False,
         casename=None, rst_filename=None, actx_class=PyOpenCLArrayContext):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    if "mpi4py.MPI" in sys.modules:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nproc = comm.Get_size()
    else:
        comm = None
        rank = 0
        nproc = 1

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

    # Some discretization parameters
    dim = 2
    nel_1d = 8
    order = 1

    # {{{ Time stepping control

    # This example runs only 3 steps by default (to keep CI ~short)
    # With the mixture defined below, equilibrium is achieved at ~40ms
    # To run to equlibrium, set t_final >= 40ms.

    # Time stepper selection
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step

    # Time loop control parameters
    t_final = 1e-8
    if constant_cfl:
        sim_dt = None
        sim_cfl = 0.001
    else:
        sim_dt = 1e-9
        sim_cfl = None

    # i.o frequencies
    nhealth = 1
    nstatus = 1
    nrestart = 5
    nviz = 5

    # }}}  Time stepping control

    debug = False

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
        assert restart_data["num_parts"] == nproc
    else:  # generate the grid from scratch
        from meshmode.mesh.generation import generate_regular_rect_mesh
        box_ll = -0.005
        box_ur = 0.005
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(discr.nodes(), actx)

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.
    from mirgecom.mechanisms import get_mechanism_cti
    mech_cti = get_mechanism_cti("uiuc")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    init_temperature = 1500.0  # Initial temperature hot enough to burn
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    stoich_ratio = 3.0
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index("C2H4")
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({init_temperature}, {one_atm}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = init_temperature, one_atm, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    can_t, can_rho, can_y = cantera_soln.TDY
    can_p = cantera_soln.P
    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    pyrometheus_mechanism = pyro.get_thermochem_class(cantera_soln)(actx.np)
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=init_temperature)

    # }}}

    # {{{ MIRGE-Com state initialization

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
    velocity = np.zeros(shape=(dim,))
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    boundaries = {BTAG_ALL: AdiabaticSlipBoundary()}

    if rst_filename:
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
        rst_order = restart_data["order"]
        rst_state = restart_data["state"]
        if order == rst_order:
            current_state = rst_state
        else:
            old_discr = EagerDGDiscretization(actx, local_mesh, order=rst_order,
                                              mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(actx, discr.discr_from_dd("vol"),
                                                   old_discr.discr_from_dd("vol"))
            current_state = connection(rst_state)
    else:
        # Set the current state from time 0
        current_step = 0
        current_t = 0
        current_state = initializer(eos=eos, x_vec=nodes)

    # Inspection at physics debugging time
    if debug:
        print("Initial MIRGE-Com state:")
        print(f"{current_state=}")
        print(f"Initial DV pressure: {eos.pressure(current_state)}")
        print(f"Initial DV temperature: {eos.temperature(current_state)}")

    # }}}

    vis_timer = None

    if logmgr:
        logmgr_add_fluid_quantities(logmgr, discr, eos, nspecies=nspecies)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("min_pressure", "------- P (min, max) (Pa) = ({value:1.9e}, "),
            ("max_pressure",    "{value:1.9e})\n"),
            ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
            ("max_temperature",    "{value:7g})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    visualizer = make_visualizer(discr)

    init_message = make_init_message(
        casename=casename,
        dim=dim, order=order,
        nelements=local_nelements, global_nelements=global_nelements,
        extra_params_dict={
            "Initialization": initializer.__class__.__name__,
            "EOS": eos.__class__.__name__,
            "Final time": t_final,
            "Timestep": sim_dt,
            "CFL": sim_cfl,
        })

    # Cantera equilibrate calculates the expected end state @ chemical equilibrium
    # i.e. the expected state after all reactions
    cantera_soln.equilibrate("UV")
    eq_temperature, eq_density, eq_mass_fractions = cantera_soln.TDY
    eq_pressure = cantera_soln.P

    # Report the expected final state to the user
    if rank == 0:
        logger.info(init_message)
        logger.info(f"Expected equilibrium state:"
                    f" {eq_pressure=}, {eq_temperature=},"
                    f" {eq_density=}, {eq_mass_fractions=}")

    from mirgecom.inviscid import get_inviscid_timestep
    get_nodal_timestep = partial(get_inviscid_timestep, discr, eos)

    def get_timestep_and_cfl(t, state, *, nodal_dt=None):
        if nodal_dt is None:
            nodal_dt = get_nodal_timestep(state)
        from grudge.op import nodal_min
        min_nodal_dt = actx.to_numpy(nodal_min(discr, "vol", nodal_dt))[()]
        if constant_cfl:
            dt = sim_cfl*min_nodal_dt
            cfl = sim_cfl
        else:
            dt = sim_dt
            cfl = sim_dt/min_nodal_dt
        return get_next_timestep(t, t_final, dt), cfl

    # FIXME: Can this be done with logging?
    def write_status(dt, cfl):
        status_msg = f"------ {dt=}" if constant_cfl else f"----- {cfl=}"
        if rank == 0:
            logger.info(status_msg)

    def write_viz(step, t, state, *, dv=None, production_rates=None, nodal_dt=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if production_rates is None:
            production_rates = eos.get_production_rates(state)
        if nodal_dt is None:
            nodal_dt = get_nodal_timestep(state)
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("production_rates", production_rates)]
        if constant_cfl:
            viz_fields.append(("dt", sim_cfl*nodal_dt))
        else:
            viz_fields.append(("cfl", sim_dt/nodal_dt))
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname == rst_filename:
            if rank == 0:
                logger.info("Skipping overwrite of restart file.")
        else:
            rst_data = {
                "local_mesh": local_mesh,
                "state": state,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nproc
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def health_check(dv):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure) \
           or check_range_local(discr, "vol", dv.pressure, 1e5, 2.4e5):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        if check_range_local(discr, "vol", dv.temperature, 1.498e3, 1.52e3):
            health_error = True
            logger.info(f"{rank=}: Invalid temperature data found.")

        return health_error

    def pre_step(step, t, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)
            do_restart = check_step(step=step, interval=nrestart)
            do_viz = check_step(step=step, interval=nviz)

            if do_health:
                dv = eos.dependent_vars(state)
                health_errors = global_reduce(health_check(dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise SimError("Failed simulation health check.")

            nodal_dt = get_nodal_timestep(state)
            dt, cfl = get_timestep_and_cfl(t, state, nodal_dt=nodal_dt)

            if do_status:
                write_status(dt, cfl)

            if do_restart:
                write_restart(step=step, t=t, state=state)

            if do_viz:
                write_viz(step=step, t=t, state=state, dv=dv, nodal_dt=nodal_dt)

        except SimError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            write_restart(step=step, t=t, state=state)
            write_viz(step=step, t=t, state=state)
            raise

        return state, dt

    def post_step(step, t, dt, state):
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, state)
            logmgr.tick_after()
        return state

    def rhs(t, state):
        return (euler_operator(discr, cv=state, time=t,
                               boundaries=boundaries, eos=eos)
                + eos.get_species_source_terms(state))

    current_step, current_t, current_state = \
        advance_state(rhs=rhs, timestepper=timestepper,
                      pre_step_callback=pre_step,
                      post_step_callback=post_step,
                      step=current_step, t=current_t, state=current_state,
                      t_final=t_final)

    current_dt, current_cfl = get_timestep_and_cfl(current_t, current_state)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    write_status(dt=current_dt, cfl=current_cfl)
    write_restart(step=current_step, t=current_t, state=current_state)
    write_viz(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Autoignition")
    parser.add_argument("--mpi", action="store_true", help="run with MPI")
    # Not working yet
    # parser.add_argument("--lazy", action="store_true",
    #     help="switch to a lazy computation mode")
    parser.add_argument("--profiling", action="store_true",
        help="turn on detailed performance profiling")
    parser.add_argument("--log", action="store_true", default=True,
        help="turn on logging")
    parser.add_argument("--leap", action="store_true",
        help="use leap timestepper")
    parser.add_argument("--constant-cfl", action="store_true",
        help="maintain a constant CFL")
    parser.add_argument("--casename", help="casename to use for i/o")
    parser.add_argument("--restart_file", help="root name of restart file")
    args = parser.parse_args()

    if args.mpi:
        main_func = mpi_entry_point(main)
    else:
        main_func = main

    if args.profiling:
        # if args.lazy:
        if False:
            raise ValueError("Can't use lazy and profiling together.")
        actx_class = PyOpenCLProfilingArrayContext
    else:
        # actx_class = PytatoPyOpenCLArrayContext if args.lazy \
        actx_class = PytatoPyOpenCLArrayContext if False \
            else PyOpenCLArrayContext

    if args.casename:
        casename = args.casename
    else:
        casename = "autoignition"

    if args.restart_file:
        rst_filename = args.restart_file
    else:
        rst_filename = None

    main_func(use_logmgr=args.log, use_leap=args.leap, use_profiling=args.profiling,
        constant_cfl=args.constant_cfl, casename=casename, rst_filename=rst_filename,
        actx_class=actx_class)

# vim: foldmethod=marker