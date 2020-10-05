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
from mpi4py import MPI

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer


from mirgecom.euler import inviscid_operator
from mirgecom.simutil import (
    inviscid_sim_timestep,
    sim_checkpoint,
    create_parallel_grid,
    ExactSolutionMismatch
)
from mirgecom.io import make_init_message

from mirgecom.integrators import rk4_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedBoundary
from mirgecom.initializers import Lump
from mirgecom.eos import IdealSingleGas


logger = logging.getLogger(__name__)


def main(ctx_factory=cl.create_some_context):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from pampr import Profiler
    profiler = Profiler(comm)
    profiler.start_sync("All")

    profiler.start_sync("CreateContext")
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    profiler.stop("CreateContext")

    dim = 3
    nel_1d = 16
    order = 3
    exittol = .09
    t_final = 0.001
    current_cfl = 1.0
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    vel[:dim] = 1.0
    current_dt = .00002
    current_t = 0
    eos = IdealSingleGas()
    initializer = Lump(numdim=dim, center=orig, velocity=vel)
    casename = "lump"
    boundaries = {BTAG_ALL: PrescribedBoundary(initializer)}
    constant_cfl = False
    nstatus = -1
    nviz = -1
    rank = 0
    checkpoint_t = current_t
    current_step = 0
    timestepper = rk4_step
    box_ll = -5.0
    box_ur = 5.0

    print(f"nsteps = {(t_final-current_t)/current_dt}")

    profiler.start("GenerateMesh")
    from meshmode.mesh.generation import generate_regular_rect_mesh
    generate_grid = partial(generate_regular_rect_mesh, a=(box_ll,) * dim,
                            b=(box_ur,) * dim, n=(nel_1d,) * dim)
    local_mesh, global_nelements = create_parallel_grid(comm, generate_grid)
    local_nelements = local_mesh.nelements
    profiler.stop("GenerateMesh")

    profiler.start_sync("CreateDiscretization")
    discr = EagerDGDiscretization(
        actx, local_mesh, order=order, mpi_communicator=comm
    )
    nodes = thaw(actx, discr.nodes())
    current_state = initializer(0, nodes)
    profiler.stop("CreateDiscretization")

    profiler.start_sync("InitializeSim")
    visualizer = make_visualizer(discr, discr.order + 3
                                 if discr.dim == 2 else discr.order)
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

    get_timestep = partial(inviscid_sim_timestep, discr=discr, t=current_t,
                           dt=current_dt, cfl=current_cfl, eos=eos,
                           t_final=t_final, constant_cfl=constant_cfl)

    def my_rhs(t, state):
        return inviscid_operator(discr, q=state, t=t,
                                 boundaries=boundaries, eos=eos)

    def my_checkpoint(step, t, dt, state):
        return sim_checkpoint(discr, visualizer, eos, q=state,
                              exact_soln=initializer, vizname=casename, step=step,
                              t=t, dt=dt, nstatus=nstatus, nviz=nviz,
                              exittol=exittol, constant_cfl=constant_cfl, comm=comm)

    profiler.stop("InitializeSim")
    profiler.start_sync("Dummy")
#     n_dummy_steps, _, _ = advance_state(rhs=my_rhs, timestepper=timestepper,
#                   checkpoint=my_checkpoint,
#                   get_timestep=get_timestep, state=current_state,
#                   t=current_t, t_final=current_t + 0.1*(t_final-current_t))
#     assert n_dummy_steps >= 5
    dummy_t = current_t
    for i in range(5):
        current_state = timestepper(state=current_state, t=dummy_t, dt=current_dt, rhs=my_rhs)
        dummy_t += current_dt
    profiler.stop("Dummy")
    profiler.start_sync("RunSim")
    try:
#         (current_step, current_t, current_state) = \
#             advance_state(rhs=my_rhs, timestepper=timestepper,
#                           checkpoint=my_checkpoint,
#                           get_timestep=get_timestep, state=current_state,
#                           t=current_t, t_final=t_final)
        t = current_t
        while t < t_final:
            current_state = timestepper(state=current_state, t=t, dt=current_dt, rhs=my_rhs)
            t += current_dt
            current_step += 1
        current_t = t
    except ExactSolutionMismatch as ex:
        current_step = ex.step
        current_t = ex.t
        current_state = ex.state

    #    if current_t != checkpoint_t:
    if rank == 0:
        logger.info("Checkpointing final state ...")
        my_checkpoint(current_step, t=current_t,
                      dt=(current_t - checkpoint_t),
                      state=current_state)

    if current_t - t_final < 0:
        raise ValueError("Simulation exited abnormally")

    profiler.stop("RunSim")
    profiler.stop("All")
    profiler.write_reduced()


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

# vim: foldmethod=marker
