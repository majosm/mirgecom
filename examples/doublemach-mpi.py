"""Demonstrate double mach reflection."""

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
from grudge.dof_desc import DTAG_BOUNDARY
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.fluid import make_conserved
from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import (
    av_operator,
    smoothness_indicator
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    AdiabaticNoslipMovingBoundary,
    PrescribedBoundary
)
from mirgecom.initializers import DoubleMachReflection
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport

from logpyle import set_dt
from mirgecom.euler import extract_vars_for_logging, units_for_logging
from mirgecom.profiling import PyOpenCLProfilingArrayContext
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


def get_doublemach_mesh():
    """Generate or import a grid using `gmsh`.

    Input required:
        doubleMach.msh (read existing mesh)

    This routine will generate a new grid if it does
    not find the grid file (doubleMach.msh).
    """
    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptSource,
    )
    import os
    meshfile = "doubleMach.msh"
    if not os.path.exists(meshfile):
        mesh = generate_gmsh(
            ScriptSource("""
                x0=1.0/6.0;
                setsize=0.025;
                Point(1) = {0, 0, 0, setsize};
                Point(2) = {x0,0, 0, setsize};
                Point(3) = {4, 0, 0, setsize};
                Point(4) = {4, 1, 0, setsize};
                Point(5) = {0, 1, 0, setsize};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(5) = {3, 4};
                Line(6) = {4, 5};
                Line(7) = {5, 1};
                Line Loop(8) = {-5, -6, -7, -1, -2};
                Plane Surface(8) = {8};
                Physical Surface('domain') = {8};
                Physical Curve('ic1') = {6};
                Physical Curve('ic2') = {7};
                Physical Curve('ic3') = {1};
                Physical Curve('wall') = {2};
                Physical Curve('out') = {5};
        """, "geo"), force_ambient_dim=2, dimensions=2, target_unit="M")
    else:
        mesh = read_gmsh(meshfile, force_ambient_dim=2)

    return mesh


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, use_leap=False,
         use_profiling=False, rst_step=None, rst_name=None,
         casename="doubleMach", use_logmgr=True):
    """Drive the example."""
    cl_ctx = ctx_factory()

    if casename is None:
        casename = "mirgecom"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

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
    order = 3
    # Too many steps for CI
    # t_final = 1.0e-2
    t_final = 1.0e-3
    current_cfl = 0.1
    current_dt = 1.0e-4
    current_t = 0
    # {{{ Initialize simple transport model
    kappa = 1e-5
    sigma = 1e-5
    transport_model = SimpleTransport(viscosity=sigma, thermal_conductivity=kappa)
    # }}}
    eos = IdealSingleGas(transport_model=transport_model)
    initializer = DoubleMachReflection()

    boundaries = {
        DTAG_BOUNDARY("ic1"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("ic2"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("ic3"): PrescribedBoundary(initializer),
        DTAG_BOUNDARY("wall"): AdiabaticNoslipMovingBoundary(),
        DTAG_BOUNDARY("out"): AdiabaticNoslipMovingBoundary(),
    }
    constant_cfl = False
    nstatus = 10
    nviz = 100
    current_step = 0
    timestepper = rk4_step
    nrestart = 100
    nhealth = 1

    s0 = -6.0
    kappa = 1.0
    alpha = 2.0e-2
    from mpi4py import MPI

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    if rst_step:  # read the grid from restart data
        rst_fname = rst_pattern.format(cname=rst_name, step=rst_step, rank=rank)

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, rst_fname)
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        gen_grid = partial(get_doublemach_mesh)
        from mirgecom.simutil import generate_and_distribute_mesh
        local_mesh, global_nelements = generate_and_distribute_mesh(comm, gen_grid)
        local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                                  mpi_communicator=comm)
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
            ("min_temperature", "------- T (min, max) (K) = ({value:1.9e}, "),
            ("max_temperature",    "{value:1.9e})\n"),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s")
        ])

    if rst_step:
        current_t = restart_data["t"]
        current_step = rst_step
        current_state = restart_data["state"]
        if logmgr:
            from mirgecom.logging_quantities import logmgr_set_time
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        current_state = initializer(nodes)

    visualizer = make_visualizer(discr,
                                 discr.order if discr.dim == 2 else discr.order)

    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(
        dim=dim,
        order=order,
        nelements=local_nelements,
        global_nelements=global_nelements,
        dt=current_dt,
        t_final=t_final,
        nstatus=nstatus,
        nviz=nviz,
        cfl=current_cfl,
        constant_cfl=constant_cfl,
        initname=initname,
        eosname=eosname,
        casename=casename,
    )
    if rank == 0:
        logger.info(init_message)

    def my_write_viz(step, t, state, dv=None, tagged_cells=None):
        if dv is None:
            dv = eos.dependent_vars(state)
        if tagged_cells is None:
            tagged_cells = smoothness_indicator(discr, state.mass, s0=s0,
                                                kappa=kappa)
        viz_fields = [("cv", state),
                      ("dv", dv),
                      ("tagged_cells", tagged_cells)]
        from mirgecom.simutil import write_visfile
        write_visfile(discr, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, state):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=rank)
        rst_data = {
            "local_mesh": local_mesh,
            "state": state,
            "t": t,
            "step": step,
            "order": order,
            "global_nelements": global_nelements,
            "num_parts": nparts
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(state, dv):
        # Note: This health check is tuned s.t. it is a test that
        #       the case gets the expected solution.  If dt,t_final or
        #       other run parameters are changed, this check should
        #       be changed accordingly.
        health_error = False
        from mirgecom.simutil import check_naninf_local, check_range_local
        if check_naninf_local(discr, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        from mirgecom.simutil import allsync
        if allsync(check_range_local(discr, "vol", dv.pressure, .9, 18.6),
                   comm, op=MPI.LOR):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            p_min = nodal_min(discr, "vol", dv.pressure)
            p_max = nodal_max(discr, "vol", dv.pressure)
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if check_naninf_local(discr, "vol", dv.temperature):
            health_error = True
            logger.info(f"{rank=}: NANs/INFs in temperature data.")

        if allsync(
                check_range_local(discr, "vol", dv.temperature, 2.48e-3, 1.071e-2),
                comm, op=MPI.LOR):
            health_error = True
            from grudge.op import nodal_max, nodal_min
            t_min = nodal_min(discr, "vol", dv.temperature)
            t_max = nodal_max(discr, "vol", dv.temperature)
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        return health_error

    def my_pre_step(step, t, dt, state):
        try:
            dv = None

            if logmgr:
                logmgr.tick_before()

            from mirgecom.simutil import check_step
            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                dv = eos.dependent_vars(state)
                from mirgecom.simutil import allsync
                health_errors = allsync(my_health_check(state, dv), comm,
                                        op=MPI.LOR)
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if step == rst_step:  # don't do viz or restart @ restart
                do_viz = False
                do_restart = False

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                if dv is None:
                    dv = eos.dependent_vars(state)
                tagged_cells = smoothness_indicator(discr, state.mass, s0=s0,
                                                    kappa=kappa)
                my_write_viz(step=step, t=t, state=state, dv=dv,
                             tagged_cells=tagged_cells)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, state=state)
            my_write_restart(step=step, t=t, state=state)
            raise

        t_remaining = max(0, t_final - t)
        return state, min(dt, t_remaining)

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        return ns_operator(
            discr, cv=state, t=t, boundaries=boundaries, eos=eos
        ) + make_conserved(dim, q=av_operator(
            discr, q=state.join(), boundaries=boundaries,
            boundary_kwargs={"time": t, "eos": eos}, alpha=alpha,
            s0=s0, kappa=kappa)
        )

    current_step, current_t, current_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step, dt=current_dt,
                      state=current_state, t=current_t, t_final=t_final)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = eos.dependent_vars(current_state)
    my_write_viz(step=current_step, t=current_t, state=current_state, dv=final_dv)
    my_write_restart(step=current_step, t=current_t, state=current_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker