""":mod:`mirgecom.boundary` provides methods and constructs for boundary treatments.

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: PrescribedBoundary
.. autoclass:: DummyBoundary
.. autoclass:: AdiabaticSlipBoundary
.. autoclass:: DirichletBoundary
.. autoclass:: NeumannBoundary
.. autoclass:: AggregateBoundary
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

import numpy as np
from pytools.obj_array import make_obj_array
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import as_dofdesc
from grudge.symbolic.primitives import TracePair
from mirgecom.fluid import (
    split_conserved,
    join_conserved
)
from mirgecom.artificial_viscosity import AVBoundary
from mirgecom.diffusion import DiffusionBoundary


class PrescribedBoundary(AVBoundary):
    """Boundary condition prescribes boundary soln with user-specified function.

    .. automethod:: __init__
    .. automethod:: boundary_pair
    """

    def __init__(self, userfunc):
        """Set the boundary function.

        Parameters
        ----------
        userfunc
            User function that prescribes the solution values on the exterior
            of the boundary. The given user function (*userfunc*) must take at
            least one parameter that specifies the coordinates at which to prescribe
            the solution.
        """
        self._userfunc = userfunc

    def _get_exterior_q(self, discr, dd, q, **kwargs):
        actx = q[0].array_context
        boundary_discr = discr.discr_from_dd(dd)
        nodes = thaw(actx, boundary_discr.nodes())
        return self._userfunc(nodes, **kwargs)

    # FIXME: Use dd instead of btag?
    def boundary_pair(self, discr, q, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        int_soln = discr.project("vol", btag, q)
        ext_soln = self._get_exterior_q(discr, as_dofdesc(btag), q, **kwargs)
        return TracePair(btag, interior=int_soln, exterior=ext_soln)

    def get_av_gradient_flux(self, discr, dd, q, **kwargs):  # noqa: D102
        q_tpair = TracePair(dd,
            interior=discr.project("vol", dd, q),
            exterior=self._get_exterior_q(discr, dd, q, **kwargs))
        from mirgecom.artificial_viscosity import av_gradient_flux
        return av_gradient_flux(discr, q_tpair)

    def get_av_flux(self, discr, dd, alpha_indicator, grad_q,
            **kwargs):  # noqa: D102
        alpha_indicator_int = discr.project("vol", dd, alpha_indicator)
        alpha_indicator_tpair = TracePair(dd,
            interior=alpha_indicator_int,
            exterior=alpha_indicator_int)
        grad_q_int = discr.project("vol", dd, grad_q)
        grad_q_tpair = TracePair(dd,
            interior=grad_q_int,
            exterior=grad_q_int)
        from mirgecom.artificial_viscosity import av_flux
        return av_flux(discr, alpha_indicator_tpair, grad_q_tpair)


class DummyBoundary(AVBoundary):
    """Boundary condition that assigns boundary-adjacent soln as the boundary solution.

    .. automethod:: boundary_pair
    """

    # FIXME: Use dd instead of btag?
    def boundary_pair(self, discr, q, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        dir_soln = discr.project("vol", btag, q)
        return TracePair(btag, interior=dir_soln, exterior=dir_soln)

    def get_av_gradient_flux(self, discr, dd, q, **kwargs):  # noqa: D102
        q_int = discr.project("vol", dd, q)
        q_tpair = TracePair(dd,
            interior=q_int,
            exterior=q_int)
        from mirgecom.artificial_viscosity import av_gradient_flux
        return av_gradient_flux(discr, q_tpair)

    def get_av_flux(self, discr, dd, alpha_indicator, grad_q,
            **kwargs):  # noqa: D102
        alpha_indicator_int = discr.project("vol", dd, alpha_indicator)
        alpha_indicator_tpair = TracePair(dd,
            interior=alpha_indicator_int,
            exterior=alpha_indicator_int)
        grad_q_int = discr.project("vol", dd, grad_q)
        grad_q_tpair = TracePair(dd,
            interior=grad_q_int,
            exterior=grad_q_int)
        from mirgecom.artificial_viscosity import av_flux
        return av_flux(discr, alpha_indicator_tpair, grad_q_tpair)


class AdiabaticSlipBoundary(AVBoundary):
    r"""Boundary condition implementing inviscid slip boundary.

    a.k.a. Reflective inviscid wall boundary

    This class implements an adiabatic reflective slip boundary given
    by
    $\mathbf{q^{+}} = [\rho^{-}, (\rho{E})^{-}, (\rho\vec{V})^{-}
    - 2((\rho\vec{V})^{-}\cdot\hat{\mathbf{n}}) \hat{\mathbf{n}}]$
    wherein the normal component of velocity at the wall is 0, and
    tangential components are preserved. These perfectly reflecting
    conditions are used by the forward-facing step case in
    [Hesthaven_2008]_, Section 6.6, and correspond to the characteristic
    boundary conditions described in detail in [Poinsot_1992]_.

    .. automethod:: boundary_pair
    """

    def _get_exterior_q(self, discr, dd, q):
        """Get the exterior solution on the boundary.

        The exterior solution is set such that there will be vanishing
        flux through the boundary, preserving mass, momentum (magnitude) and
        energy.
        rho_plus = rho_minus
        v_plus = v_minus - 2 * (v_minus . n_hat) * n_hat
        mom_plus = rho_plus * v_plus
        E_plus = E_minus
        """
        # Grab some boundary-relevant data
        dim = discr.dim
        cv = split_conserved(dim, q)
        actx = cv.mass.array_context

        # Grab a unit normal to the boundary
        nhat = thaw(actx, discr.normal(dd))

        # Get the interior/exterior solns
        int_soln = discr.project("vol", dd, q)
        int_cv = split_conserved(dim, int_soln)

        # Subtract out the 2*wall-normal component
        # of velocity from the velocity at the wall to
        # induce an equal but opposite wall-normal (reflected) wave
        # preserving the tangential component
        mom_normcomp = np.dot(int_cv.momentum, nhat)  # wall-normal component
        wnorm_mom = nhat * mom_normcomp  # wall-normal mom vec
        ext_mom = int_cv.momentum - 2.0 * wnorm_mom  # prescribed ext momentum

        # Form the external boundary solution with the new momentum
        bndry_soln = join_conserved(dim=dim, mass=int_cv.mass,
                                    energy=int_cv.energy,
                                    momentum=ext_mom,
                                    species_mass=int_cv.species_mass)

        return bndry_soln

    def _get_exterior_grad_q(self, discr, dd, grad_q):
        """Get the exterior grad(Q) on the boundary."""
        # Grab some boundary-relevant data
        num_equations, dim = grad_q.shape
        cv = split_conserved(dim, grad_q)
        actx = cv.mass[0].array_context

        # Grab a unit normal to the boundary
        normal = thaw(actx, discr.normal(dd))

        # Get the interior soln
        gradq_int = discr.project("vol", dd, grad_q)
        gradq_comp = split_conserved(dim, gradq_int)

        # Subtract 2*wall-normal component of q
        # to enforce q=0 on the wall
        s_mom_normcomp = np.outer(normal, np.dot(gradq_comp.momentum, normal))
        s_mom_flux = gradq_comp.momentum - 2*s_mom_normcomp

        # flip components to set a neumann condition
        return join_conserved(dim, mass=-gradq_comp.mass, energy=-gradq_comp.energy,
                              momentum=-s_mom_flux,
                              species_mass=-gradq_comp.species_mass)

    # FIXME: Use dd instead of btag?
    def boundary_pair(self, discr, q, btag, **kwargs):
        """Get the interior and exterior solution on the boundary."""
        bndry_soln = self._get_exterior_q(discr, as_dofdesc(btag), q)
        int_soln = discr.project("vol", btag, q)

        return TracePair(btag, interior=int_soln, exterior=bndry_soln)

    def get_av_gradient_flux(self, discr, dd, q, **kwargs):  # noqa: D102
        q_tpair = TracePair(dd,
            interior=discr.project("vol", dd, q),
            exterior=self._get_exterior_q(discr, dd, q))
        from mirgecom.artificial_viscosity import av_gradient_flux
        return av_gradient_flux(discr, q_tpair)

    def get_av_flux(self, discr, dd, alpha_indicator, grad_q,
            **kwargs):  # noqa: D102
        alpha_indicator_int = discr.project("vol", dd, alpha_indicator)
        alpha_indicator_tpair = TracePair(dd,
            interior=alpha_indicator_int,
            exterior=alpha_indicator_int)
        grad_q_tpair = TracePair(dd,
            interior=discr.project("vol", dd, grad_q),
            exterior=self._get_exterior_grad_q(discr, dd, grad_q))
        from mirgecom.artificial_viscosity import av_flux
        return av_flux(discr, alpha_indicator_tpair, grad_q_tpair)


class DirichletBoundary(DiffusionBoundary):
    r"""
    Dirichlet boundary condition.

    Enforces the boundary condition $u|_\Gamma = f$.

    For the diffusion operator, uses

    .. math::

                 u^+ &= 2 f - u^-

        (\nabla u)^+ &= (\nabla u)^-

    to compute boundary fluxes as shown in [Hesthaven_2008]_, Section 7.1.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) of $f$ along the boundary
        """
        self.value = value

    def get_diffusion_gradient_flux(self, discr, quad_tag, dd, u,
            **kwargs):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=2*self.value-u_int)
        from mirgecom.diffusion import diffusion_gradient_flux
        return diffusion_gradient_flux(discr, quad_tag, u_tpair)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u,
            **kwargs):  # noqa: D102
        alpha_int = discr.project("vol", dd, alpha)
        alpha_tpair = TracePair(dd, interior=alpha_int, exterior=alpha_int)
        grad_u_int = discr.project("vol", dd, grad_u)
        grad_u_tpair = TracePair(dd, interior=grad_u_int, exterior=grad_u_int)
        from mirgecom.diffusion import diffusion_flux
        return diffusion_flux(discr, quad_tag, alpha_tpair, grad_u_tpair)


class NeumannBoundary(DiffusionBoundary):
    r"""
    Neumann boundary condition.

    Enforces the boundary condition $(\nabla u \cdot \mathbf{\hat{n}})|_\Gamma = g$.

    For the diffusion operator, uses

    .. math::

        u^+ = u^-

    when computing the boundary fluxes for $\nabla u$, and uses

    .. math::

        (-\alpha \nabla u\cdot\mathbf{\hat{n}})|_\Gamma &=
            -\alpha^- (\nabla u\cdot\mathbf{\hat{n}})|_\Gamma

                                                        &= -\alpha^- g

    when computing the boundary fluxes for $\nabla \cdot (\alpha \nabla u)$.

    .. automethod:: __init__
    """

    def __init__(self, value):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        value: float or meshmode.dof_array.DOFArray
            the value(s) of $g$ along the boundary
        """
        self.value = value

    def get_diffusion_gradient_flux(self, discr, quad_tag, dd, u,
            **kwargs):  # noqa: D102
        u_int = discr.project("vol", dd, u)
        u_tpair = TracePair(dd, interior=u_int, exterior=u_int)
        from mirgecom.diffusion import diffusion_gradient_flux
        return diffusion_gradient_flux(discr, quad_tag, u_tpair)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u,
            **kwargs):  # noqa: D102
        dd_quad = dd.with_qtag(quad_tag)
        dd_allfaces_quad = dd_quad.with_dtag("all_faces")
        # Compute the flux directly instead of constructing an external grad_u value
        # (and the associated TracePair); this approach is simpler in the
        # spatially-varying alpha case (the other approach would result in a
        # grad_u_tpair that lives in the quadrature discretization; diffusion_flux
        # would need to be modified to accept such values).
        alpha_int_quad = discr.project("vol", dd_quad, alpha)
        value_quad = discr.project(dd, dd_quad, self.value)
        flux_quad = -alpha_int_quad*value_quad
        return discr.project(dd_quad, dd_allfaces_quad, flux_quad)


class AggregateBoundary(DiffusionBoundary, AVBoundary):
    """
    Combines multiple scalar boundaries into a single vector boundary.

    .. automethod:: __init__
    """

    def __init__(self, boundaries):
        """
        Initialize the boundary condition.

        Parameters
        ----------
        boundaries:
            a list or object array of boundaries
        """
        self.boundaries = boundaries.copy()

    def get_diffusion_gradient_flux(self, discr, quad_tag, dd, u,
            **kwargs):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_diffusion_gradient_flux(discr, quad_tag, dd, u[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return np.stack(component_fluxes, axis=0)

    def get_diffusion_flux(self, discr, quad_tag, dd, alpha, grad_u,
            **kwargs):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_diffusion_flux(discr, quad_tag, dd, alpha, grad_u[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return component_fluxes

    def get_av_gradient_flux(self, discr, dd, q, **kwargs):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_av_gradient_flux(discr, dd, q[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return np.stack(component_fluxes, axis=0)

    def get_av_flux(self, discr, dd, alpha_indicator, grad_q,
            **kwargs):  # noqa: D102
        component_fluxes = make_obj_array([
            bdry.get_av_flux(discr, dd, alpha_indicator, grad_q[i])
            for i, bdry in enumerate(self.boundaries)
            ])
        return component_fluxes
