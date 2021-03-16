__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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

import os
import math
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import DOFArray, thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.symbolic.primitives import QTAG_NONE
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools


def get_mesh():
    if os.path.exists("two-region-box.msh"):
        from meshmode.mesh.io import read_gmsh
        return read_gmsh("two-region-box.msh")
    else:
        from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
        return generate_gmsh(
            ScriptWithFilesSource("""
                Merge "two-region-box.brep";

                Mesh.MshFileVersion = 2.2;

                // Convert to meters
                Mesh.ScalingFactor = 0.001;

                Mesh.ElementOrder = 1;

                Mesh.Algorithm3D = 4;
                // Mesh.Smoothing = 100;
                Mesh.Smoothing = 1;

                Mesh.CharacteristicLengthMin = 5;
                Mesh.CharacteristicLengthMax = 5;
                Mesh.CharacteristicLengthExtendFromBoundary = 0;
                Mesh.CharacteristicLengthFromPoints = 0;
                Mesh.CharacteristicLengthFromCurvature = 0;

                bottom_surfaces[] = {11};
                top_surfaces[] = {6};
                side_surfaces[] = {1:4,7:10};
                interface_surfaces[] = {5};

                Physical Volume("Lower") = {2};
                Physical Volume("Upper") = {1};
                Physical Surface("Bottom") = { bottom_surfaces[] };
                Physical Surface("Sides") = { side_surfaces[] };
                Physical Surface("Top") = { top_surfaces[] };
                Physical Surface("Interface") = { interface_surfaces[] };

                Mesh 3;
                // OptimizeMesh "Netgen";
            """, ["two-region-box.brep"]), 3, target_unit='MM')


@mpi_entry_point
def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 3

    if mesh_dist.is_mananger_rank():
        mesh = get_mesh()

        print("%d elements" % mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    order = 3

    dt = 1e-8

    discr = EagerDGDiscretization(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    nodes = thaw(actx, discr.nodes())

    boundaries = {
        grudge_sym.DTAG_BOUNDARY("Bottom"): DirichletDiffusionBoundary(0.),
        grudge_sym.DTAG_BOUNDARY("Top"): DirichletDiffusionBoundary(1.),
        grudge_sym.DTAG_BOUNDARY("Sides"): NeumannDiffusionBoundary(0.)
    }

    u = discr.zeros(actx)

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    lower_region_bit = local_mesh.region_tag_bit("Lower")
    upper_region_bit = local_mesh.region_tag_bit("Upper")

    alpha = discr.empty(actx)
    alpha_np = [actx.to_numpy(alpha_i) for alpha_i in alpha]

    for igrp, grp in enumerate(local_mesh.groups):
        lower_elems, = np.where((grp.regions & lower_region_bit) != 0)
        upper_elems, = np.where((grp.regions & upper_region_bit) != 0)
        alpha_np[igrp][lower_elems, :] = 0.25
        alpha_np[igrp][upper_elems, :] = 2

    alpha = DOFArray(actx, tuple([
        actx.from_numpy(alpha_np_i) for alpha_np_i in alpha_np]))

    def rhs(t, u):
        return (diffusion_operator(
            discr, quad_tag=QTAG_NONE, alpha=alpha, boundaries=boundaries, u=u))

    rank = comm.Get_rank()

    t = 0
    t_final = 5e-5
    istep = 0

    while True:
        if istep % 1 == 0:
            print(istep, t, dt, discr.norm(u))
            vis.write_vtk_file("fld-two-region-box-%03d-%04d.vtu" % (rank, istep),
                    [
                        ("u", u),
                        ("alpha", alpha),
                        ])

        if t >= t_final:
            break

        u = rk4_step(u, t, dt, rhs)
        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
