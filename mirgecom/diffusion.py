r""":mod:`mirgecom.diffusion` computes the diffusion operator.

.. autofunction:: diffusion_operator
.. autoclass:: DiffusionBoundary
.. autoclass:: DirichletDiffusionBoundary
.. autoclass:: NeumannDiffusionBoundary
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
import math
import numpy as np
import numpy.linalg as la  # noqa
from pytools.obj_array import make_obj_array, obj_array_vectorize_n_args
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw, DOFArray
from grudge.symbolic.primitives import DOFDesc
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs
from grudge.symbolic.primitives import TracePair, as_dofdesc


def _grad(discr, x):
    if isinstance(x, DOFArray):
        return discr.grad(x)
    else:
        return 0.


def _v_flux(discr, quad_tag, alpha_tpair, u_tpair):
    actx = u_tpair.int.array_context

    dd = u_tpair.dd
    dd_quad = dd.with_qtag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(alpha, u, normal):
        return -alpha * u * normal

    return discr.project(dd_quad, dd_allfaces_quad, flux(
        to_quad(alpha_tpair.int), to_quad(u_tpair.avg), normal_quad))


def _u_flux(discr, quad_tag, v_tpair):
    actx = v_tpair.int[0].array_context

    dd = v_tpair.dd
    dd_quad = dd.with_qtag(quad_tag)
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")

    normal_quad = thaw(actx, discr.normal(dd_quad))

    def to_quad(a):
        return discr.project(dd, dd_quad, a)

    def flux(v, normal):
        return -np.dot(v, normal)

    return discr.project(dd_quad, dd_allfaces_quad, flux(
        to_quad(v_tpair.avg), normal_quad))


class DiffusionBoundary(metaclass=abc.ABCMeta):
    """
    Diffusion boundary base class.

    .. automethod:: get_v_flux
    .. automethod:: get_u_flux
    """

    @abc.abstractmethod
    def get_v_flux(self, discr, quad_tag, alpha, dd, u):
        """Compute the flux for *v* on the boundary corresponding to *dd*."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_u_flux(self, discr, quad_tag, dd, v):
        """Compute the flux for *u* on the boundary corresponding to *dd*."""
        raise NotImplementedError


class DirichletDiffusionBoundary(DiffusionBoundary):
    r"""
    Dirichlet boundary condition for the diffusion operator.

    For boundary condition $u|_\Gamma = f$, uses external data

    .. math::

                 u^+ &= 2 f - u^-

        \mathbf{v}^+ &= \mathbf{v}^-

    to compute boundary fluxes as shown in [Hesthaven_2008]_, Section 7.1.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) along the boundary
        """
        self.value = value

    # Observe: Dirichlet BC enforced on v, not u
    def get_v_flux(self, discr, quad_tag, dd, alpha, u):  # noqa: D102
        alpha_int = discr.project("vol", dd, alpha)
        alpha_tpair = TracePair(dd, interior=alpha_int, exterior=alpha_int)
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2*self.value-u_int)
        return _v_flux(discr, quad_tag, alpha_tpair, u_tpair)

    def get_u_flux(self, discr, quad_tag, dd, alpha, v):  # noqa: D102
        v_int = discr.project("vol", dd, v)
        v_tpair = TracePair(dd, interior=v_int, exterior=v_int)
        return _u_flux(discr, quad_tag, v_tpair)


class NeumannDiffusionBoundary(DiffusionBoundary):
    r"""
    Neumann boundary condition for the diffusion operator.

    For boundary condition $\frac{\partial u}{\partial \mathbf{n}}|_\Gamma = g$, uses
    external data

    .. math::

        u^+ = u^-

    to compute boundary fluxes for $\mathbf{v} = \alpha \nabla u$, and computes
    boundary fluxes for $u_t = \nabla \cdot \mathbf{v}$ using

    .. math::

        \mathbf{F}\cdot\mathbf{\hat{n}} &= -\alpha\nabla u\cdot\mathbf{\hat{n}}

                                        &= -\alpha\frac{\partial u}{\partial
                                                \mathbf{n}}

                                        &= -\alpha g

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) along the boundary
        """
        self.value = value

    def get_v_flux(self, discr, quad_tag, dd, alpha, u):  # noqa: D102
        alpha_int = discr.project("vol", dd, alpha)
        alpha_tpair = TracePair(dd, interior=alpha_int, exterior=alpha_int)
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        return _v_flux(discr, quad_tag, alpha_tpair, u_tpair)

    def get_u_flux(self, discr, quad_tag, dd, alpha, v):  # noqa: D102
        dd_quad = dd.with_qtag(quad_tag)
        dd_allfaces_quad = dd_quad.with_dtag("all_faces")
        # Compute the flux directly instead of constructing an external v value
        # (and the associated TracePair); this approach is simpler in the
        # spatially-varying alpha case (the other approach would result in a
        # v_tpair that lives in the quadrature discretization; _u_flux would need
        # to be modified to accept such values).
        alpha_int_quad = discr.project("vol", dd_quad, alpha)
        value_quad = discr.project(dd, dd_quad, self.value)
        flux_quad = -alpha_int_quad*value_quad
        return discr.project(dd_quad, dd_allfaces_quad, flux_quad)


def diffusion_operator(discr, quad_tag, alpha, boundaries, u, return_v=False):
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
    alpha: Union[numbers.Number, meshmode.dof_array.DOFArray]
        the diffusivity value(s)
    boundaries:
        dictionary (or list of dictionaries) mapping boundary tags to
        :class:`DiffusionBoundary` instances
    u: Union[meshmode.dof_array.DOFArray, numpy.ndarray]
        the DOF array (or object array of DOF arrays) to which the operator should be
        applied
    return_v: bool
        an optional flag indicating whether the auxiliary variable
        $\mathbf{v} = \alpha \nabla u$ should also be returned

    Returns
    -------
    meshmode.dof_array.DOFArray or numpy.ndarray
        the diffusion operator applied to *u*
    """
    if isinstance(u, np.ndarray):
        if not isinstance(boundaries, list):
            raise TypeError("boundaries must be a list if u is an object array")
        if len(boundaries) != len(u):
            raise TypeError("boundaries must be the same length as u")
        return obj_array_vectorize_n_args(lambda boundaries_i, u_i:
            diffusion_operator(discr, quad_tag, alpha, boundaries_i, u_i,
            return_v=return_v), make_obj_array(boundaries), u)

    for btag, bdry in boundaries.items():
        if not isinstance(bdry, DiffusionBoundary):
            raise TypeError(f"Unrecognized boundary type for tag {btag}. "
                "Must be an instance of DiffusionBoundary.")

    actx = u.array_context

    dd_quad = DOFDesc("vol", quad_tag)
    dd_allfaces_quad = DOFDesc("all_faces", quad_tag)

    alpha_quad = discr.project("vol", dd_quad, alpha)
    grad_alpha_quad = discr.project("vol", dd_quad, _grad(discr, alpha))

    u_quad = discr.project("vol", dd_quad, u)

    v = discr.inverse_mass(
        # Decompose phi_i*grad(alpha*phi_j) term via the product rule in
        # order to avoid having to define a new operator
        -discr.mass(dd_quad, grad_alpha_quad * u_quad)
        -  # noqa: W504
        discr.weak_grad(dd_quad, alpha_quad * u_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            _v_flux(discr, quad_tag, interior_trace_pair(discr, alpha),
                interior_trace_pair(discr, u))
            + sum(
                bdry.get_v_flux(discr, quad_tag, as_dofdesc(btag), alpha, u)
                for btag, bdry in boundaries.items())
            + sum(
                _v_flux(discr, quad_tag, alpha_tpair, u_tpair)
                for alpha_tpair, u_tpair in zip(
                    cross_rank_trace_pairs(discr, alpha),
                    cross_rank_trace_pairs(discr, u)))
            )
        )

    v_quad = discr.project("vol", dd_quad, v)

    result = discr.inverse_mass(
        -discr.weak_div(dd_quad, v_quad)
        -  # noqa: W504
        discr.face_mass(
            dd_allfaces_quad,
            _u_flux(discr, quad_tag, interior_trace_pair(discr, v))
            + sum(
                bdry.get_u_flux(discr, quad_tag, as_dofdesc(btag), alpha, v)
                for btag, bdry in boundaries.items())
            + sum(
                _u_flux(discr, quad_tag, v_tpair)
                for v_tpair in cross_rank_trace_pairs(discr, v))
            )
        )

    if return_v:
        return result, v
    else:
        return result
