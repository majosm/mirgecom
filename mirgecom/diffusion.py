r""":mod:`mirgecom.diffusion` computes the diffusion operator.

Diffusion Operator Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: diffusion_gradient_flux
.. autofunction:: diffusion_flux
.. autofunction:: diffusion_operator

Diffusion Boundary Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DiffusionBoundaryInterface
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

import abc
import numpy as np
import numpy.linalg as la  # noqa
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.dof_desc import DOFDesc, as_dofdesc
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair


class DiffusionBoundaryInterface(metaclass=abc.ABCMeta):
    """
    Interface for diffusion boundary information retrieval.

    .. automethod:: get_diffusion_gradient_flux
    .. automethod:: get_diffusion_flux
    """

    @abc.abstractmethod
    def get_diffusion_gradient_flux(self, discr, quad_tag, dd, u, **kwargs):
        """Compute the flux for grad(u) on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u, **kwargs):
        """Compute the flux for diff(u) on the boundary corresponding to *dd*."""
        raise NotImplementedError


def diffusion_gradient_flux(discr, quad_tag, u_tpair):
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


def diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair):
    r"""Compute the numerical flux for $\nabla \cdot (\alpha \nabla u)$."""
    if isinstance(grad_u_tpair.int[0], np.ndarray):
        actx = grad_u_tpair.int[0][0].array_context
    else:
        actx = grad_u_tpair.int[0].array_context

    dd = grad_u_tpair.dd
    dd_quad = dd.with_discr_tag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(alpha, grad_u, normal):
        if isinstance(grad_u[0], np.ndarray):
            return -alpha * np.stack(grad_u, axis=0) @ normal
        else:
            return -alpha * np.dot(grad_u, normal)

    flux_tpair = TracePair(dd_quad,
        interior=flux(
            to_quad(alpha_tpair.int), to_quad(grad_u_tpair.int), normal_quad),
        exterior=flux(
            to_quad(alpha_tpair.ext), to_quad(grad_u_tpair.ext), normal_quad)
        )

    return discr.project(dd_quad, dd_allfaces_quad, flux_tpair.avg)


def diffusion_operator(discr, quad_tag, alpha, boundaries, u, boundary_kwargs=None,
        return_grad_u=False):
    r"""
    Compute the diffusion operator.

    The diffusion operator is defined as
    $\nabla\cdot(\alpha\nabla u)$, where $\alpha$ is the diffusivity and
    $u$ is a scalar field.

    Uses unstabilized central numerical fluxes.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    quad_tag:
        quadrature tag indicating which discretization in *discr* to use for
        overintegration
    alpha: numbers.Number or meshmode.dof_array.DOFArray
        the diffusivity value(s)
    boundaries:
        dictionary mapping boundary tags to objects implementing
        :class:`DiffusionBoundaryInterface`
    u: meshmode.dof_array.DOFArray or numpy.ndarray
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    boundary_kwargs: :class:`dict`
        dictionary of extra arguments to pass through to the boundary conditions
    return_grad_u: bool
        an optional flag indicating whether $\nabla u$ should also be returned

    Returns
    -------
    diff_u: meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    grad_u: numpy.ndarray
        the gradient of *u*; only returned if *return_grad_u* is True
    """
    if boundary_kwargs is None:
        boundary_kwargs = dict()

    for bdry in boundaries.values():
        if not isinstance(bdry, DiffusionBoundaryInterface):
            raise ValueError("Incompatible boundary; boundaries must implement "
                "DiffusionBoundaryInterface.")

    dd_quad = DOFDesc("vol", quad_tag)
    dd_allfaces_quad = DOFDesc("all_faces", quad_tag)

    grad_u = discr.inverse_mass(
        discr.weak_grad(-u)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            diffusion_gradient_flux(discr, quad_tag, interior_trace_pair(discr, u))
            + sum(
                bdry.get_diffusion_gradient_flux(discr, quad_tag, as_dofdesc(btag),
                    u, **boundary_kwargs)
                for btag, bdry in boundaries.items())
            + sum(
                diffusion_gradient_flux(discr, quad_tag, u_tpair)
                for u_tpair in cross_rank_trace_pairs(discr, u))
            )
        )

    alpha_quad = discr.project("vol", dd_quad, alpha)
    grad_u_quad = discr.project("vol", dd_quad, grad_u)

    diff_u = discr.inverse_mass(
        discr.weak_div(dd_quad, -alpha_quad*grad_u_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            diffusion_flux(discr, quad_tag, interior_trace_pair(discr, alpha),
                interior_trace_pair(discr, grad_u))
            + sum(
                bdry.get_diffusion_flux(discr, quad_tag, as_dofdesc(btag), alpha,
                    grad_u, **boundary_kwargs)
                for btag, bdry in boundaries.items())
            + sum(
                diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair)
                for alpha_tpair, grad_u_tpair in zip(
                    cross_rank_trace_pairs(discr, alpha),
                    cross_rank_trace_pairs(discr, grad_u)))
            )
        )

    if return_grad_u:
        return diff_u, grad_u
    else:
        return diff_u
