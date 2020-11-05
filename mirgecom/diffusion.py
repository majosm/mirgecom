r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: diffusion_operator
"""

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
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.symbolic.primitives import DOFDesc, QTAG_NONE
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs


def _q_flux(discr, var_diff_quad_tag, alpha, u_tpair):
    actx = u_tpair.int.array_context

    dd = u_tpair.dd
    if var_diff_quad_tag is QTAG_NONE:
        dd_quad = dd
    else:
        dd_quad = dd.with_qtag(var_diff_quad_tag)

    normal = thaw(actx, discr.normal(dd))

    flux_weak = make_obj_array([-u_tpair.avg])*normal

    dd_allfaces_quad = dd_quad.with_dtag("all_faces")
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    flux_quad = discr.project(dd, dd_quad, flux_weak)

    return discr.project(dd_quad, dd_allfaces_quad, make_obj_array([sqrt_alpha_quad])
                * flux_quad)


def _u_flux(discr, var_diff_quad_tag, alpha, q_tpair):
    actx = q_tpair.int[0].array_context

    dd = q_tpair.dd
    if var_diff_quad_tag is QTAG_NONE:
        dd_quad = dd
    else:
        dd_quad = dd.with_qtag(var_diff_quad_tag)

    normal = thaw(actx, discr.normal(dd))

    flux_weak = np.dot(-q_tpair.avg, normal)

    dd_allfaces_quad = dd_quad.with_dtag("all_faces")
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    flux_quad = discr.project(dd, dd_quad, flux_weak)

    return discr.project(dd_quad, dd_allfaces_quad, sqrt_alpha_quad * flux_quad)


def diffusion_operator(discr, alpha, u_boundaries, q_boundaries, u,
            var_diff_quad_tag=QTAG_NONE):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\alpha\nabla u)$, where $\alpha$ is the diffusivity and
    $u$ is a scalar field.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    alpha: meshmode.dof_array.DOFArray
        the diffusivities
    u_boundaries:
        dictionary (or object array of dictionaries) of boundary functions for *u*,
        one for each valid btag
    q_boundaries:
        dictionary (or object array of dictionaries) of boundary functions for
        :math:`q = \sqrt{\alpha}\nabla u`, one for each valid btag
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array or object array of DOF arrays to which the operator should be
        applied
    var_diff_quad_tag:
        tag indicating which quadrature discretization in *discr* to use for
        overintegration (required for variable diffusivity)

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        if not isinstance(u_boundaries, np.ndarray) or len(u_boundaries) != len(u):
            raise RuntimeError("u_boundaries must be the same length as u")
        if not isinstance(q_boundaries, np.ndarray) or len(q_boundaries) != len(u):
            raise RuntimeError("q_boundaries must be the same length as u")
        return obj_array_vectorize_n_args(lambda u_boundaries_i, q_boundaries_i, u_i:
            diffusion_operator(discr, alpha, u_boundaries_i, q_boundaries_i, u_i),
            u_boundaries, q_boundaries, u)

    actx = u.array_context

    dd_quad = DOFDesc("vol", var_diff_quad_tag)
    alpha_quad = discr.project("vol", dd_quad, alpha)
    sqrt_alpha_quad = actx.np.sqrt(alpha_quad)
    u_quad = discr.project("vol", dd_quad, u)

    dd_allfaces_quad = DOFDesc("all_faces", var_diff_quad_tag)

    q = (
        discr.grad(-actx.np.sqrt(alpha))*make_obj_array([u])  # not sure how to do overintegration here
        +  # noqa: W504
        discr.inverse_mass(
            discr.weak_grad(dd_quad, -sqrt_alpha_quad*u_quad)
            -  # noqa: W504
            discr.face_mass(
                dd_allfaces_quad,
                _q_flux(discr, var_diff_quad_tag, alpha,
                    u_tpair=interior_trace_pair(discr, u))
                + sum(
                    _q_flux(discr, var_diff_quad_tag, alpha=alpha,
                        u_tpair=u_boundaries[btag](discr, u))
                    for btag in u_boundaries
                )
                + sum(
                    _q_flux(discr, var_diff_quad_tag, alpha,
                        u_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, u)
                )
            ))
        )

    q_quad = discr.project("vol", dd_quad, q)

    return (
        discr.inverse_mass(
            discr.weak_div(dd_quad, make_obj_array([-sqrt_alpha_quad])*q_quad)
            -  # noqa: W504
            discr.face_mass(
                dd_allfaces_quad,
                _u_flux(discr, var_diff_quad_tag, alpha,
                    q_tpair=interior_trace_pair(discr, q))
                + sum(
                    _u_flux(discr, var_diff_quad_tag, alpha=alpha,
                        q_tpair=q_boundaries[btag](discr, q))
                    for btag in q_boundaries
                )
                + sum(
                    _u_flux(discr, var_diff_quad_tag, alpha,
                        q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )
        )
