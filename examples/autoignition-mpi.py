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
import math
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
    check_step,
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
    set_sim_state
)

import cantera
import pyrometheus as pyro

logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_logmgr=False,
         use_leap=False, use_profiling=False, casename=None):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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

    dim = 2
    nel_1d = 8
    order = 1

    # This example runs only 3 steps by default (to keep CI ~short)
    # With the mixture defined below, equilibrium is achieved at ~40ms
    # To run to equlibrium, set t_final >= 40ms.
    t_final = 3e-9
    current_cfl = 1.0
    velocity = np.zeros(shape=(dim,))
    current_dt = 1e-9
    current_t = 0
    constant_cfl = False
    nstatus = 1
    nviz = 5
    nhealth = 1
    nlog = 1
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    box_ll = -0.005
    box_ur = 0.005
    debug = False

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, nelements_per_axis=(nel_1d,) * dim)
    local_mesh, global_nelements = generate_and_distribute_mesh(comm, generate_mesh)
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
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     pressure=can_p, temperature=can_t,
                                     massfractions=can_y, velocity=velocity)

    my_boundary = AdiabaticSlipBoundary()
    boundaries = {BTAG_ALL: my_boundary}
    current_state = initializer(eos=eos, x_vec=nodes, t=0)

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

    def write_vis(step, t, state, *, dv=None, reaction_rates=None):
        # Probably unnecessary optimizations, but *shrug*
        if dv is None:
            dv = eos.dependent_vars(state)
        if reaction_rates is None:
            reaction_rates = eos.get_production_rates(state)
        io_fields = [
            ("cv", state),
            ("dv", dv),
            ("reaction_rates", reaction_rates)
        ]
        return write_visfile(
            discr, io_fields, visualizer, vizname=casename, step=step, t=t)

    def get_timestep(step, t, state):
        dt = inviscid_sim_timestep(
            discr=discr, t=t, dt=current_dt, cfl=current_cfl, eos=eos,
            t_final=t_final, constant_cfl=constant_cfl, state=state)
        if not math.isfinite(dt) or dt <= 0:
            write_vis(step, t, state)
            raise RuntimeError(f"Invalid timestep {dt}.")
        return dt

    def my_rhs(t, state):
        return (euler_operator(discr, cv=state, t=t,
                               boundaries=boundaries, eos=eos)
                + eos.get_species_source_terms(state))

    def post_step_stuff(step, t, dt, state):
        do_logend = check_step(step=(step-1), interval=nlog)

        if do_logend and logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state

    def my_checkpoint(step, t, dt, state, force=False):
        from mirgecom.simutil import check_step
        do_viz = force or check_step(step=step, interval=nviz)
        do_health = force or check_step(step=step, interval=nhealth)
        do_logstart = force or check_step(step=step, interval=nlog)

        if do_logstart and logmgr:
            logmgr.tick_before()

        if do_viz or do_health:
            dv = eos.dependent_vars(state)

        errored = False
        if do_health:
            health_message = ""
            from mirgecom.simutil import check_naninf_local, check_range_local
            if check_naninf_local(discr, "vol", dv.pressure) \
               or check_range_local(discr, "vol", dv.pressure, 1e5, 2.4e5):
                errored = True
                health_message += "Invalid pressure data found.\n"
            if check_range_local(discr, "vol", dv.temperature, 1.4e3, 3.3e3):
                errored = True
                health_message += "Temperature data exceeded healthy range.\n"
            comm = discr.mpi_communicator
            if comm is not None:
                errored = comm.allreduce(errored, op=MPI.LOR)
            if errored:
                if rank == 0:
                    logger.info("Fluid solution failed health check.")
                if health_message:  # capture any rank's health message
                    logger.info(f"{rank=}:{health_message}")

        if do_viz or errored:
            write_vis(step, t, state, dv=dv)

        if errored:
            raise RuntimeError("Error detected by user checkpoint, exiting.")

        return state

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_checkpoint,
                      post_step_callback=post_step_stuff,
                      get_timestep=get_timestep, state=current_state,
                      t=current_t, t_final=t_final, eos=eos, dim=dim)

    if not check_step(current_step, nviz):
        if rank == 0:
            logger.info("Checkpointing final state ...")
        write_vis(current_step, current_t, current_state)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    use_profiling = True
    use_logging = True
    use_leap = False
    casename = "autoignition"

    main(use_profiling=use_profiling, use_logmgr=use_logging, use_leap=use_leap,
         casename=casename)

# vim: foldmethod=marker
