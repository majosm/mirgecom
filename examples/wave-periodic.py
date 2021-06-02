"""Demonstrate wave-eager serial example."""

__copyright__ = "Copyright (C) 2021 University of Illinos Board of Trustees"

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
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.array as cla  # noqa
from pytools.obj_array import flat_obj_array
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from mirgecom.wave import wave_operator
from mirgecom.integrators import rk4_step
from meshmode.dof_array import thaw
from meshmode.array_context import PyOpenCLArrayContext
import pyopencl.tools as cl_tools

from mirgecom.profiling import PyOpenCLProfilingArrayContext


def cos(x):
    if isinstance(x, np.ndarray):
        return np.cos(x)
    else:
        return math.cos(x)


def sin(x):
    if isinstance(x, np.ndarray):
        return np.sin(x)
    else:
        return math.sin(x)


def bump(actx, discr, *, x0, t=0):
    """Create a bump."""
    source_center = np.array(x0)
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, discr.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main(use_profiling=False):
    """Drive the example."""
    cl_ctx = cl.create_some_context()
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
        actx = PyOpenCLProfilingArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        queue = cl.CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    dim = 2
    nelements_per_axis = (16, 32)

    if dim != 2:
        raise ValueError("wave-periodic only works in 2D.")

    from meshmode.mesh.generation import generate_regular_rect_mesh

    unit_mesh = generate_regular_rect_mesh(
        a=(0,)*dim,
        b=(1,)*dim,
        nelements_per_axis=nelements_per_axis,
        boundary_tag_to_face={
            "-xi": ["-x"],
            "+xi": ["+x"],
            "-eta": ["-y"],
            "+eta": ["+y"],
            })

#     def transform(x):
#         r_min = 0.4
#         r_max = 1
#         theta_min = -np.pi/4
#         theta_max = np.pi/4
#         r = r_min*(1-x[0]) + r_max*x[0]
#         theta = theta_min*(1-x[1]) + theta_max*x[1]
#         return (r*cos(theta), r*sin(theta))

#     def transform(x):
#         R = np.array([
#             [math.cos(np.pi/4), -np.sin(np.pi/4)],
#             [  np.sin(np.pi/4),  np.cos(np.pi/4)]])
#         return R @ x

#     def transform(x):
#         return np.stack((x[0]+2, (1+x[0])*(x[1]-0.5)+2))

    def transform(x):
        return np.stack((x[0], x[0] + x[1]))

#     def transform(x):
#         r_min = 0.4
#         r_max = 1
#         theta_min = 0
#         theta_max = np.pi
#         r = r_min*(1-x[0]) + r_max*x[0]
#         theta = theta_min*(1-x[1]) + theta_max*x[1]
#         return (r*cos(theta), r*sin(theta))

    from meshmode.mesh.processing import map_mesh
    transformed_mesh = map_mesh(unit_mesh, lambda x: np.stack(transform(x)))

    # Figure out the periodic mapping, under the assumption that the
    # boundaries are mapped affinely by transform()
    def make_periodic_mapping(idim):
        bdry_lower = "-" + ("xi", "eta")[idim]
        bdry_upper = "+" + ("xi", "eta")[idim]
        jdim = (idim + 1) % 2
        o = np.zeros(2, dtype=np.float32)
        ei = o.copy()
        ei[idim] = 1
        ej = o.copy()
        ej[jdim] = 1
        p0 = np.asarray(transform(o))
        p1 = np.asarray(transform(o + ej))
        q0 = np.asarray(transform(o + ei))
        q1 = np.asarray(transform(o + ei + ej))
        u = p1 - p0
        v = q1 - q0
        theta_u = math.atan2(u[1], u[0])
        theta_v = math.atan2(v[1], v[0])
        theta = theta_v - theta_u
        c = math.cos(theta)
        s = math.sin(theta)
        A = np.array([[c, -s], [s, c]])
        b = q0 - A @ p0
        return bdry_lower, bdry_upper, A, b

    print(f"{make_periodic_mapping(1)=}")

    from meshmode.mesh.processing import glue_mesh_boundaries
    mesh = glue_mesh_boundaries(transformed_mesh,
        glued_boundary_mappings=[
            make_periodic_mapping(1)])

    order = 3

    # no deep meaning here, just a fudge factor
    dt = 0.1 / (max(nelements_per_axis) * order ** 2)

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(actx, mesh, order=order)

    fields = flat_obj_array(
        bump(actx, discr, x0=transform((0.5,0.85))),
        [discr.zeros(actx) for i in range(discr.dim)]
        )

    vis = make_visualizer(discr)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            if use_profiling:
                print(actx.tabulate_profiling_data())
            print(istep, t, discr.norm(fields[0], np.inf))
            vis.write_vtk_file("fld-wave-eager-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ])

        t += dt
        istep += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wave-eager (non-MPI version)")
    parser.add_argument("--profile", action="store_true",
        help="enable kernel profiling")
    args = parser.parse_args()

    main(use_profiling=args.profile)

# vim: foldmethod=marker
