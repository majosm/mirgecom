"""Demonstrate simple gas mixture with Pyrometheus."""

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

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer

from mirgecom.discretization import create_discretization_collection
from mirgecom.euler import euler_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.initializers import MixtureInitializer
from mirgecom.eos import PyrometheusMixture

import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_many_discretization_quantities,
    logmgr_add_device_name,
    logmgr_add_device_memory_usage,
    set_sim_state
)

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_leap=False, use_profiling=False, casename=None, rst_filename=None,
         log_dependent=True, lazy=False):
    """Drive example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr,
        filename=f"{casename}.sqlite", mode="wu", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # timestepping control
    if use_leap:
        from leap.rk import RK4MethodBuilder
        timestepper = RK4MethodBuilder("state")
    else:
        timestepper = rk4_step
    t_final = 1e-8
    current_cfl = 1.0
    current_dt = 1e-9
    current_t = 0
    current_step = 0
    constant_cfl = False

    # some i/o frequencies
    nstatus = 1
    nhealth = 1
    nrestart = 5
    nviz = 1

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
        assert restart_data["num_parts"] == nparts
    else:  # generate the grid from scratch
        nel_1d = 16
        box_ll = -5.0
        box_ur = 5.0
        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh, a=(box_ll,)*dim,
                                b=(box_ur,) * dim, nelements_per_axis=(nel_1d,)*dim)
        local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                    generate_mesh)
        local_nelements = local_mesh.nelements

    order = 3
    discr = create_discretization_collection(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = actx.thaw(discr.nodes())

    vis_timer = None

    if logmgr:
        logmgr_add_device_name(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

        if log_dependent:
            logmgr_add_many_discretization_quantities(logmgr, discr, dim,
                                                      extract_vars_for_logging,
                                                      units_for_logging)
            logmgr.add_watches([
                ("min_pressure", "\n------- P (min, max) (Pa) = ({value:1.9e}, "),
                ("max_pressure",    "{value:1.9e})\n"),
                ("min_temperature", "------- T (min, max) (K)  = ({value:7g}, "),
                ("max_temperature",    "{value:7g})\n")])

    # Pyrometheus initialization
    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input("uiuc")
    sol = cantera.Solution(name="gas", yaml=mech_input)
    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyrometheus_mechanism = make_pyrometheus_mechanism_class(sol)(actx.np)

    nspecies = pyrometheus_mechanism.num_species
    eos = PyrometheusMixture(pyrometheus_mechanism)
    from mirgecom.gas_model import GasModel, make_fluid_state
    gas_model = GasModel(eos=eos)
    from pytools.obj_array import make_obj_array

    y0s = np.zeros(shape=(nspecies,))
    for i in range(nspecies-1):
        y0s[i] = 1.0 / (10.0 ** (i + 1))
    spec_sum = sum([y0s[i] for i in range(nspecies-1)])
    y0s[nspecies-1] = 1.0 - spec_sum

    # Mixture defaults to STP (p, T) = (1atm, 300K)
    velocity = np.zeros(shape=(dim,)) + 1.0
    initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                     massfractions=y0s, velocity=velocity)

    def boundary_solution(discr, btag, gas_model, state_minus, **kwargs):
        actx = state_minus.array_context
        bnd_discr = discr.discr_from_dd(btag)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(initializer(x_vec=nodes, eos=gas_model.eos,
                                            **kwargs), gas_model,
                                temperature_seed=state_minus.temperature)

    boundaries = {
        BTAG_ALL: PrescribedFluidBoundary(boundary_state_func=boundary_solution)
    }

    if rst_filename:
        current_t = restart_data["t"]
        current_step = restart_data["step"]
        current_cv = restart_data["cv"]
        tseed = restart_data["temperature_seed"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_cv = initializer(x_vec=nodes, eos=eos)
        tseed = 300.0

    current_state = make_fluid_state(current_cv, gas_model, temperature_seed=tseed)

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
    if rank == 0:
        logger.info(init_message)

    def my_write_status(component_errors, dv=None):
        from mirgecom.simutil import allsync
        status_msg = (
            "------- errors="
            + ", ".join("%.3g" % en for en in component_errors))
        if ((dv is not None) and (not log_dependent)):
            temp = dv.temperature
            press = dv.pressure

            from grudge.op import nodal_min_loc, nodal_max_loc
            tmin = allsync(actx.to_numpy(nodal_min_loc(discr, "vol", temp)),
                           comm=comm, op=MPI.MIN)
            tmax = allsync(actx.to_numpy(nodal_max_loc(discr, "vol", temp)),
                           comm=comm, op=MPI.MAX)
            pmin = allsync(actx.to_numpy(nodal_min_loc(discr, "vol", press)),
                           comm=comm, op=MPI.MIN)
            pmax = allsync(actx.to_numpy(nodal_max_loc(discr, "vol", press)),
                           comm=comm, op=MPI.MAX)
            dv_status_msg = f"\nP({pmin}, {pmax}), T({tmin}, {tmax})"
            status_msg = status_msg + dv_status_msg

        if rank == 0:
            logger.info(status_msg)
        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, state, dv, exact=None, resid=None):
        if exact is None:
            exact = initializer(x_vec=nodes, eos=eos, time=t)
        if resid is None:
            resid = state - exact
        viz_fields = [("cv", state), ("dv", dv)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, state, tseed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != rst_filename:
            rst_data = {
                "local_mesh": local_mesh,
                "cv": state,
                "temperature_seed": tseed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            from mirgecom.restart import write_restart_file
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(dv, component_errors):
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure) \
           or check_range_local(discr, "vol", dv.pressure, 1e5, 1.1e5):
            health_error = True
            logger.info(f"{rank=}: Invalid pressure data found.")

        exittol = .09
        if max(component_errors) > exittol:
            health_error = True
            if rank == 0:
                logger.info("Solution diverged from exact soln.")

        return health_error

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv, gas_model, temperature_seed=tseed)
        dv = fluid_state.dv

        try:
            exact = None
            component_errors = None

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                exact = initializer(x_vec=nodes, eos=eos, time=t)
                from mirgecom.simutil import compare_fluid_solutions
                component_errors = compare_fluid_solutions(discr, cv, exact)
                health_errors = global_reduce(
                    my_health_check(dv, component_errors), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, state=cv, tseed=tseed)

            if do_viz:
                if exact is None:
                    exact = initializer(x_vec=nodes, eos=eos, time=t)
                resid = state - exact
                my_write_viz(step=step, t=t, state=cv, dv=dv, exact=exact,
                             resid=resid)

            if do_status:
                if component_errors is None:
                    if exact is None:
                        exact = initializer(x_vec=nodes, eos=eos, time=t)
                    from mirgecom.simutil import compare_fluid_solutions
                    component_errors = compare_fluid_solutions(discr, cv, exact)
                my_write_status(component_errors, dv=dv)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=cv, dv=dv)
            my_write_restart(step=step, t=t, state=cv, tseed=tseed)
            raise

        dt = get_sim_timestep(discr, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)
        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv, gas_model, temperature_seed=tseed)
        tseed = fluid_state.temperature
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, eos)
            logmgr.tick_after()
        return make_obj_array([fluid_state.cv, tseed]), dt

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv, gas_model, temperature_seed=tseed)
        return make_obj_array(
            [euler_operator(discr, state=fluid_state, time=t,
                            boundaries=boundaries, gas_model=gas_model),
             0*tseed])

    current_dt = get_sim_timestep(discr, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, advanced_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=make_obj_array([current_state.cv,
                                            current_state.temperature]),
                      t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    current_cv, tseed = advanced_state
    current_state = make_fluid_state(current_cv, gas_model, temperature_seed=tseed)
    final_dv = current_state.dv
    final_exact = initializer(x_vec=nodes, eos=eos, time=current_t)
    final_resid = current_state.cv - final_exact
    my_write_viz(step=current_step, t=current_t, state=current_cv, dv=final_dv,
                 exact=final_exact, resid=final_resid)
    my_write_restart(step=current_step, t=current_t, state=current_state.cv,
                     tseed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    import argparse
    casename = "uiuc-mixture"
    parser = argparse.ArgumentParser(description=f"MIRGE-Com Example: {casename}")
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
    from warnings import warn
    warn("Automatically turning off DV logging. MIRGE-Com Issue(578)")
    log_dependent = False
    lazy = args.lazy
    if args.profiling:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if args.casename:
        casename = args.casename
    rst_filename = None
    if args.restart_file:
        rst_filename = args.restart_file

    main(actx_class, use_logmgr=args.log, use_leap=args.leap, lazy=lazy,
         use_profiling=args.profiling, casename=casename, rst_filename=rst_filename,
         log_dependent=log_dependent)

# vim: foldmethod=marker
