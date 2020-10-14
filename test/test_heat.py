__copyright__ = """Copyright (C) 2020 University of Illinois Board of Trustees"""

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

import numpy as np
import pyopencl.array as cla  # noqa
import pyopencl.clmath as clmath # noqa
from pytools.obj_array import flat_obj_array, make_obj_array
import pymbolic as pmbl
import pymbolic.primitives as prim
import mirgecom.symbolic as sym
from mirgecom.heat import heat_operator
from meshmode.dof_array import thaw

from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


# Tests below take a problem description as input, which is a tuple
#   (dim, alpha, mesh_factory, sym_u)
# where:
#   dim is the problem dimension
#   alpha is the diffusivity
#   mesh_factory is a factory that creates a mesh given a characteristic size
#   sym_u is a symbolic expression for the solution


def get_decaying_cosine(dim):
    # 1D: u(x,t) = exp(-alpha*t)*cos(x)
    # 2D: u(x,y,t) = exp(-2*alpha*t)*cos(x)*cos(y)
    # 3D: u(x,y,z,t) = exp(-3*alpha*t)*cos(x)*cos(y)*cos(z)
    # on [-pi/2, pi/2]^{#dims}
    def mesh_factory(n):
        from meshmode.mesh.generation import generate_regular_rect_mesh
        return generate_regular_rect_mesh(
            a=(-0.5*np.pi,)*dim,
            b=(0.5*np.pi,)*dim,
            n=(n,)*dim)
    alpha = 2.
    sym_coords = prim.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")
    sym_cos = pmbl.var("cos")
    sym_exp = pmbl.var("exp")
    sym_u = sym_exp(-dim*alpha*sym_t)
    for i in range(dim):
        sym_u *= sym_cos(sym_coords[i])
    return (dim, alpha, mesh_factory, sym_u)


def sym_heat(dim, sym_u):
    """Return symbolic expressions for the heat equation system given a desired
    solution. (Note: In order to support manufactured solutions, we modify the heat
    equation to add a source term (f). If the solution is exact, this term should
    be 0.)
    """

    sym_alpha = pmbl.var("alpha")
    sym_coords = prim.make_sym_vector("x", dim)
    sym_t = pmbl.var("t")

    # rhs = alpha * div(grad(u))
    sym_rhs = sym_alpha * sym.div(sym.grad(dim, sym_u))

    # f = u_t - rhs
    sym_f = sym.diff(sym_t)(sym_u) - sym_rhs[0]

    return sym_f, sym_rhs


@pytest.mark.parametrize("problem",
    [
        get_decaying_cosine(2),
        get_decaying_cosine(3)
    ])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_heat_accuracy(actx_factory, problem, order, visualize=False):
    """Checks accuracy of the heat operator for a given problem setup.
    """
    actx = actx_factory()

    dim, alpha, mesh_factory, sym_u = problem

    _, sym_rhs = sym_heat(dim, sym_u)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [8, 10, 12] if dim == 3 else [4, 8, 16, 32, 64, 128]:
        mesh = mesh_factory(n)

        from grudge.eager import EagerDGDiscretization
        discr = EagerDGDiscretization(actx, mesh, order=order)

        nodes = thaw(actx, discr.nodes())

        def sym_eval(expr, t):
            return sym.EvaluationMapper({"alpha": alpha, "x": nodes, "t": t})(expr)

        t_check = 1.23456789

        fields = make_obj_array([sym_eval(sym_u, t_check)])

        rhs = heat_operator(discr, alpha=alpha, w=fields)

        expected_rhs = make_obj_array([sym_eval(sym_rhs, t_check)])

        rel_linf_err = (
            discr.norm(rhs - expected_rhs, np.inf)
            / discr.norm(expected_rhs, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(discr, discr.order+3)
            vis.write_vtk_file("heat_accuracy_{order}_{n}.vtu".format(order=order,
                        n=n), [
                            ("u", fields[0]),
                            ("rhs_actual", rhs[0]),
                            ("rhs_expected", expected_rhs[0]),
                            ])

    print("Approximation error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5 or eoc_rec.max_error() < 1e-11)


# @pytest.mark.parametrize(("problem", "timestep_scale"),
#     [
#         (get_standing_wave(2), 0.05),
#         (get_standing_wave(3), 0.05),
#         (get_manufactured_cubic(2), 0.025),
#         (get_manufactured_cubic(3), 0.025)
#     ])
# @pytest.mark.parametrize("order", [2, 3, 4])
# def test_wave_stability(actx_factory, problem, timestep_scale, order,
#             visualize=False):
#     """Checks stability of the wave operator for a given problem setup.
#     Adjust *timestep_scale* to get timestep close to stability limit.
#     """

#     actx = actx_factory()

#     dim, c, mesh_factory, sym_phi = problem

#     sym_u, sym_v, sym_f, sym_rhs = sym_wave(dim, sym_phi)

#     mesh = mesh_factory(8)

#     from grudge.eager import EagerDGDiscretization
#     discr = EagerDGDiscretization(actx, mesh, order=order)

#     nodes = thaw(actx, discr.nodes())

#     def sym_eval(expr, t):
#         return sym.EvaluationMapper({"c": c, "x": nodes, "t": t})(expr)

#     def get_rhs(t, w):
#         result = wave_operator(discr, c=c, w=w)
#         result[0] += sym_eval(sym_f, t)
#         return result

#     t = 0.

#     u = sym_eval(sym_u, t)
#     v = sym_eval(sym_v, t)

#     fields = flat_obj_array(u, v)

#     from mirgecom.integrators import rk4_step
#     dt = timestep_scale/order**2
#     for istep in range(10):
#         fields = rk4_step(fields, t, dt, get_rhs)
#         t += dt

#     expected_u = sym_eval(sym_u, 10*dt)
#     expected_v = sym_eval(sym_v, 10*dt)
#     expected_fields = flat_obj_array(expected_u, expected_v)

#     if visualize:
#         from grudge.shortcuts import make_visualizer
#         vis = make_visualizer(discr, discr.order)
#         vis.write_vtk_file("wave_stability.vtu",
#                 [
#                     ("u", fields[0]),
#                     ("v", fields[1:]),
#                     ("u_expected", expected_fields[0]),
#                     ("v_expected", expected_fields[1:]),
#                     ])

#     err = discr.norm(fields-expected_fields, np.inf)
#     max_err = discr.norm(expected_fields, np.inf)

#     assert err < max_err


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         exec(sys.argv[1])
#     else:
#         from pytest import main
#         main([__file__])
