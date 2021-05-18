r""":mod:`mirgecom.gradient` computes the gradient operator.

Gradient Operator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: gradient_flux
.. autofunction:: gradient_operator

Gradient Boundary Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GradientBoundaryInterface
"""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
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

import abc
import numpy as np
import numpy.linalg as la  # noqa
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.dof_desc import DOFDesc, as_dofdesc
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs


class GradientBoundaryInterface(metaclass=abc.ABCMeta):
    """
    Interface for gradient boundary information retrieval.

    .. automethod:: get_gradient_flux
    """

    @abc.abstractmethod
    def get_gradient_flux(self, discr, quad_tag, dd, u, **kwargs):
        r"""Compute the numerical boundary flux for $\nabla u$."""
        raise NotImplementedError


def gradient_flux(discr, quad_tag, u_tpair):
    r"""Compute the numerical flux for $\nabla u$."""
    if isinstance(u_tpair.int, np.ndarray):
        actx = u_tpair.int[0].array_context
    else:
        actx = u_tpair.int.array_context

    dd = u_tpair.dd
    dd_quad = dd.with_discr_tag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(u, normal):
        if isinstance(u, np.ndarray):
            return -np.outer(u, normal)
        else:
            return -u * normal

    return discr.project(dd_quad, dd_allfaces_quad, flux(
        to_quad(u_tpair.avg), normal_quad))


def gradient_operator(discr, quad_tag, boundaries, u, boundary_kwargs=None):
    r"""
    Compute the gradient operator $\nabla u$.

    Uses unstabilized central numerical fluxes (for now).

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    quad_tag:
        quadrature tag indicating which discretization in *discr* to use for
        overintegration
    boundaries:
        dictionary mapping boundary tags to objects implementing
        :class:`GradientBoundaryInterface`
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    boundary_kwargs: :class:`dict`
        dictionary of extra arguments to pass through to the boundary conditions

    Returns
    -------
    numpy.ndarray
        the gradient of *u*
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    for bdry in boundaries.values():
        if not isinstance(bdry, GradientBoundaryInterface):
            raise ValueError("Incompatible boundary; boundaries must implement "
                "GradientBoundaryInterface.")

    dd_allfaces_quad = DOFDesc("all_faces", quad_tag)

    return discr.inverse_mass(
        discr.weak_grad(-u)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            gradient_flux(discr, quad_tag, interior_trace_pair(discr, u))
            + sum(
                bdry.get_gradient_flux(discr, quad_tag, as_dofdesc(btag),
                    u, **boundary_kwargs)
                for btag, bdry in boundaries.items())
            + sum(
                gradient_flux(discr, quad_tag, u_tpair)
                for u_tpair in cross_rank_trace_pairs(discr, u))
            )
        )
