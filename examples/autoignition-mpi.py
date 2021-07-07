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
import logging
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
from functools import partial

from meshmode.array_context import PyOpenCLArrayContext

from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext

from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import AdiabaticSlipBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture

from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    logmgr_set_time,
    set_sim_state
)

import cantera
import pyrometheus as pyro

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=False,
         use_leap=False, use_profiling=False, casename=None,
         start_step=0):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            logmgr=logmgr)
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

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
    current_cfl = 1.0
    current_dt = 1e-9
    constant_cfl = False

    # i.o frequencies
    nstatus = 1
    nviz = 5
    nhealth = 1
    nrestart = 5

    # }}}  Time stepping control

    debug = False

    def get_rst_fname(step, rank):
        rst_path = "restart_data/"
        rst_pattern = rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
        return rst_pattern.format(cname=casename, step=step, rank=rank)

    if start_step > 0:
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, get_rst_fname(start_step, rank))
        assert restart_data["num_parts"] == nproc
    else:
        restart_data = None

    from mirgecom.restart import memoize_from_restart, RESTART_TAG

    @memoize_from_restart(restart_data,
        (RESTART_TAG("local_mesh"), RESTART_TAG("global_nelements")))
    def get_mesh():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_and_distribute_mesh(comm,
            lambda: generate_regular_rect_mesh(
                a=(-0.005,)*dim,
                b=(+0.005,)*dim,
                nelements_per_axis=(nel_1d,)*dim))

    local_mesh, global_nelements = get_mesh()
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                             extract_vars_for_logging, units_for_logging)

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

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

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

    my_boundary = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: my_boundary}

    @memoize_from_restart(restart_data,
        (start_step, RESTART_TAG("t"), RESTART_TAG("state")))
    def get_initial_state(initializer, eos, nodes):
        state = initializer(eos=eos, x_vec=nodes, t=0)
        return start_step, 0., state

    current_step, current_t, current_state = get_initial_state(
        initializer, eos, nodes)

    if logmgr:
        logmgr_set_time(logmgr, current_step, current_t)

    # Inspection at physics debugging time
    if debug:
        print("Initial MIRGE-Com state:")
        print(f"{current_state=}")
        print(f"Initial DV pressure: {eos.pressure(current_state)}")
        print(f"Initial DV temperature: {eos.temperature(current_state)}")

    # }}}

    visualizer = make_visualizer(discr)
    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)

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

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_graceful_exit(cv, step, t, do_viz=False, do_restart=False, message=None):
        if rank == 0:
            logger.info("Errors detected; attempting graceful exit.")
        if do_viz:
            my_write_viz(cv, step, t)
        if do_restart:
            my_write_restart(state=cv, step=step, t=t)
        if message is None:
            message = "Fatal simulation errors detected."
        raise RuntimeError(message)

    def my_write_viz(cv, step, t, dv=None, production_rates=None):
        viz_fields = [("cv", cv)]
        if dv is not None:
            viz_fields.append(("dv", dv))
        if production_rates is not None:
            viz_fields.append(("production_rates", production_rates))
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(state, step, t):
        rst_data = {
            "local_mesh": local_mesh,
            "state": state,
            "t": t,
            "step": step,
            "global_nelements": global_nelements,
            "num_parts": nproc
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, get_rst_fname(step, rank), comm)

    def my_health_check(dv, dt):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure) \
           or check_range_local(discr, "vol", dv.pressure, 1e5, 2.4e5):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        if check_range_local(discr, "vol", dv.temperature, 1.4e3, 3.3e3):
            health_error = True
            logger.info(f"{rank=}: Invalid temperature data found.")

        if dt < 0:
            health_error = True
            if rank == 0:
                logger.info("Global DT is negative!")

        return health_error

    def my_rhs(t, state):
        return (euler_operator(discr, cv=state, t=t,
                               boundaries=boundaries, eos=eos)
                + eos.get_species_source_terms(state))

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_pre_step(step, t, dt, state):
        dv = None
        pre_step_errors = False

        if logmgr:
            logmgr.tick_before()

        from mirgecom.simutil import check_step
        do_viz = check_step(step=step, interval=nviz)
        do_restart = check_step(step=step, interval=nrestart)
        do_health = check_step(step=step, interval=nhealth)

        if step == start_step:  # don't do viz or restart @ restart
            do_viz = False
            do_restart = False

        if do_health:
            dv = eos.dependent_vars(state)
            local_health_error = my_health_check(dv, dt)
            health_errors = False
            if comm is not None:
                health_errors = comm.allreduce(local_health_error, op=MPI.LOR)
            if health_errors and rank == 0:
                logger.info("Fluid solution failed health check.")
            pre_step_errors = pre_step_errors or health_errors

        if do_restart:
            my_write_restart(state, step, t)

        if do_viz:
            production_rates = eos.get_production_rates(state)
            if dv is None:
                dv = eos.dependent_vars(state)
            my_write_viz(cv=state, dv=dv, step=step, t=t,
                         production_rates=production_rates)

        if pre_step_errors:
            my_graceful_exit(cv=state, step=step, t=t,
                             do_viz=(not do_viz), do_restart=(not do_restart),
                             message="Error detected at prestep, exiting.")

        return state, dt

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    finish_tol = 1e-16
    if np.abs(current_t - t_final) > finish_tol:
        my_graceful_exit(cv=current_state, step=current_step, t=current_t,
                         do_viz=True, do_restart=True,
                         message="Simulation timestepping did not complete.")

    # Dump the final data
    final_dv = eos.dependent_vars(current_state)
    final_dm = eos.get_production_rates(current_state)
    my_write_viz(cv=current_state, dv=final_dv, production_rates=final_dm,
                 step=current_step, t=current_t)
    my_write_restart(current_state, current_step, current_t)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    use_profiling = True
    use_logging = True
    use_leap = False
    casename = "autoignition"

    main(use_profiling=use_profiling, use_logmgr=use_logging, use_leap=use_leap,
         casename=casename, start_step=0)

# vim: foldmethod=marker
