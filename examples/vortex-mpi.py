"""Demonstrate the isentropic vortex example."""

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


from mirgecom.euler import (
    inviscid_operator,
    split_conserved,
    get_inviscid_timestep,
    get_extra_init as get_extra_init_euler,
    get_extra_status as get_extra_status_euler,
)
from mirgecom.simutil import (
    create_parallel_grid,
    sim_checkpoint,
)
from mirgecom.io import (
    make_init_message,
    write_visualization_file,
)
from mirgecom.mpi import mpi_entry_point, comm_any

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import Vortex2D
from mirgecom.eos import IdealSingleGas


logger = logging.getLogger(__name__)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context):
    """Drive the example."""
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nel_1d = 16
    order = 3
    exittol = .09
    t_final = 0.1
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[:dim] = 1.0
    current_dt = .001
    current_t = 0
    eos = IdealSingleGas()
    initializer = Vortex2D(center=orig, velocity=vel)
    casename = "vortex"
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = 10
    nviz = 10
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if dim != 2:
        raise ValueError("This example must be run with dim = 2.")

    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_grid = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, n=(nel_1d,) * dim)
    local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)
    local_nelements = local_mesh.nelements

    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(nodes)

    visualizer = make_visualizer(discr, order + 3 if dim == 2 else order)

    extra_init = get_extra_init_euler(
        cfl=current_cfl,
        constant_cfl=constant_cfl,
        initname=initializer.__class__.__name__,
        eosname=eos.__class__.__name__)
    init_message = make_init_message(dim=dim, order=order, casename=casename,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, extra_init=extra_init)
    if rank == 0:
        logger.info(init_message)

    def get_timestep(state):
        return get_inviscid_timestep(discr=discr, q=state, cfl=current_cfl,
            eos=eos) if constant_cfl else current_dt

    def rhs(t, state):
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def write_vis(step, t, state):
        cv = split_conserved(dim, state)
        exact_cv = split_conserved(dim, initializer(nodes, t=t))
        io_fields = [
            ("cv", cv),
            ("dv", eos.dependent_vars(cv)),
            ("exact_cv", exact_cv),
        ]
        return write_visualization_file(visualizer, fields=io_fields,
                    basename=casename, step=step, t=t, comm=comm)

    def checkpoint(step, t, dt, state):
        exact_state = initializer(nodes, t=t)
        if comm_any(comm, discr.norm(state - exact_state, np.inf) > exittol):
            write_vis(step, t, state)
            raise RuntimeError(f"Exact solution mismatch at step {step}.")
        get_extra_status = partial(get_extra_status_euler, discr=discr, eos=eos)
        return sim_checkpoint(state=state, step=step, t=t, dt=dt, nstatus=nstatus,
            get_extra_status=get_extra_status, nviz=nviz, write_vis=write_vis,
            comm=comm)

    (current_step, current_t, current_state) = \
        advance_state(rhs=rhs, timestepper=timestepper,
                      checkpoint=checkpoint, get_timestep=get_timestep,
                      state=current_state, t=current_t, t_final=t_final)

    if rank == 0:
        logger.info("Timestepping completed.")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
