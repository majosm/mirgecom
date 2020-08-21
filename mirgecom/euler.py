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

from dataclasses import dataclass

import numpy as np
from pytools.obj_array import (
    flat_obj_array,
    make_obj_array,
)
from meshmode.dof_array import thaw
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.eager import (
    interior_trace_pair,
    cross_rank_trace_pairs
)

from mirgecom.eos import IdealSingleGas

# from grudge.dt_finding import (
#    dt_geometric_factor,
#    dt_non_geometric_factor,
# )


__doc__ = r"""
This module is designed provide functions and utilities
useful for solving the Euler flow equations.

The Euler flow equations are:

.. :math::

    \partial_t \mathbf{Q} = -\nabla\cdot{\mathbf{F}} +
    (\mathbf{F}\cdot\hat{n})_\partial_{\Omega} + \mathbf{S}

where:
    state :math:`\mathbf{Q} = [\rho, \rho{E}, \rho\vec{V} ]`
    flux :math:`\mathbf{F} = [\rho\vec{V},(\rho{E} + p)\vec{V},
                (\rho(\vec{V}\otimes\vec{V}) + p*\mathbf{I})]`,
    domain boundary :math:`\partial_{\Omega}`,
    sources :math:`mathbf{S} =
                   [{(\partial_t{\rho})}_s, {(\partial_t{\rho{E}})}_s,
                    {(\partial_t{\rho\vec{V}})}_s]`

.. autofunction:: inviscid_operator
.. autofunction:: number_of_scalars
.. autofunction:: split_conserved
.. autofunction:: split_species
.. autofunction:: split_fields
.. autofunction:: get_inviscid_timestep
.. autofunction:: get_inviscid_cfl
.. autoclass:: ConservedVars
.. autoclass:: MassFractions
"""


@dataclass
class ConservedVars:
    r"""
    Class to resolve the canonical conserved quantities,
    (mass, energy, momentum) per unit volume =
    :math:`(\rho,\rhoE,\rho\vec{V})` from an agglomerated
    object array.

    .. attribute:: mass

        Mass per unit volume

    .. attribute:: energy

        Energy per unit volume

    .. attribute:: momentum

        Momentum vector per unit volume
    """
    mass: np.ndarray
    energy: np.ndarray
    momentum: np.ndarray


@dataclass
class MassFractions:
    r"""
    Class to pick off the species mass fractions
    (mass fractions) per unit volume =
    :math:`(\rhoY_{\alpha}) | 1 \le \alpha \le N_{species}`,
    from an agglomerated object array. :math:`N_{species}` is
    the number of mixture species.

    .. attribute:: mass

        Mass fraction per unit volume for each mixture species
    """
    massfractions: np.ndarray


def split_fields(ndim, q):
    """
    Method to spit out a list of named flow variables in
    an agglomerated flow solution. Useful for specifying
    named data arrays to helper functions (e.g. I/O).
    """
    qs = split_conserved(ndim, q)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum

    retlist = [
        ("mass", mass),
        ("energy", energy),
        ("momentum", mom),
    ]
    nscalar = number_of_scalars(ndim, q)
    if nscalar > 0:
        massfrac = split_species(ndim, q).massfraction
        retlist.append(("massfraction", massfrac))

    return retlist


def number_of_scalars(ndim, q):
    """
    Return the number of scalars or mixture species in a flow solution.
    """
    return len(q) - (ndim + 2)


def number_of_equations(ndim, q):
    """
    Return the number of equations (i.e. number of dofs) in the soln
    """
    return len(q) + number_of_scalars(ndim, q)


def split_conserved(dim, q):
    """
    Return a :class:`ConservedVars` that is the canonical conserved quantities,
    mass, energy, and momentum from the agglomerated object array representing
    the state, q.
    """
    return ConservedVars(mass=q[0], energy=q[1], momentum=q[2:2+dim])


def split_species(dim, q):
    """
    Return a :class:`MassFractions` object that represent the mixture species
    mass fractions from the agglomerated object array representing the state, q.
    """
    numscalar = number_of_scalars(dim, q)
    sindex = dim + 2
    return MassFractions(massfractions=q[sindex:sindex+numscalar])


def _inviscid_flux(discr, q, eos=IdealSingleGas()):
    r"""Computes the inviscid flux vectors from flow solution *q*

    The inviscid fluxes are
    :math:`(\rho\vec{V},(\rhoE+p)\vec{V},\rho(\vec{V}\otimes\vec{V})+p\mathbf{I})
    """
    ndim = discr.dim

    # q = [ rho rhoE rhoV ]
    qs = split_conserved(ndim, q)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum

    p = eos.pressure(q)

    # Fluxes:
    # [ rhoV (rhoE + p)V (rhoV.x.V + p*I) ]
    momflux = make_obj_array(
        [
            (mom[i] * mom[j] / mass + (p if i == j else 0))
            for i in range(ndim)
            for j in range(ndim)
        ]
    )
    massflux = mom * make_obj_array([1.0])
    energyflux = mom * make_obj_array([(energy + p) / mass])
    # scalarflux = mom * massfractions / mass

    return flat_obj_array(massflux, energyflux, momflux,)


def _get_wavespeed(dim, q, eos=IdealSingleGas()):
    """Returns the maximum wavespeed in for flow solution *q*"""
    qs = split_conserved(dim, q)
    mass = qs.mass
    mom = qs.momentum
    actx = mass.array_context

    v = mom * make_obj_array([1.0 / mass])

    sos = eos.sound_speed(q)
    return actx.np.sqrt(np.dot(v, v)) + sos


def _facial_flux(discr, q_tpair, eos=IdealSingleGas()):
    """Returns the flux across a face given the solution on both sides *q_tpair*"""
    dim = discr.dim

    qs = split_conserved(dim, q_tpair)
    mass = qs.mass
    energy = qs.energy
    mom = qs.momentum
    actx = qs.mass.int.array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    # Get inviscid fluxes [rhoV (rhoE + p)V (rhoV.x.V + p*I) ]
    #    qint = q_tpair.int
    #    qext = q_tpair.ext
    qint = flat_obj_array(mass.int, energy.int, mom.int)
    qext = flat_obj_array(mass.ext, energy.ext, mom.ext)

    # Jump in soln
    qjump = qext - qint

    flux_int = _inviscid_flux(discr, qint, eos)
    flux_ext = _inviscid_flux(discr, qext, eos)

    # Lax-Friedrichs/Rusanov after JSH/TW Nodal DG Methods, p. 209
    # DOI: 10.1007/978-0-387-72067-8
    flux_aver = (flux_int + flux_ext) * 0.5

    # wavespeeds = [ wavespeed_int, wavespeed_ext ]
    wavespeeds = [_get_wavespeed(dim, qint), _get_wavespeed(dim, qext)]

    lam = actx.np.maximum(*wavespeeds)
    lfr = qjump * make_obj_array([0.5 * lam])

    # Surface fluxes should be inviscid flux .dot. normal
    # rhoV .dot. normal
    # (rhoE + p)V  .dot. normal
    # (rhoV.x.V)_1 .dot. normal
    # (rhoV.x.V)_2 .dot. normal
    numeqns = number_of_equations(dim, qint)
    num_flux = flat_obj_array(
        [
            np.dot(flux_aver[(i * dim): ((i + 1) * dim)], normal)
            for i in range(numeqns)
        ]
    )

    # add Lax/Friedrichs jump penalty
    flux_weak = num_flux + lfr

    return discr.project(q_tpair.dd, "all_faces", flux_weak)


def inviscid_operator(
        discr, q, boundaries, t=0.0, eos=IdealSingleGas(),
):
    r"""
    RHS of the Euler flow equations

    Returns
    -------
    The right-hand-side of the Euler flow equations:

    :math:`\dot\mathbf{q} = \mathbf{S} - \nabla\cdot\mathbf{F} +
          (\mathbf{F}\cdot\hat{n})_\partial_{\Omega}`

    Parameters
    ----------
    q
        State array which expects at least the canonical conserved quantities
        (mass, energy, momentum) for the fluid at each point.

    boundaries
        Dictionary of boundary functions, one for each valid btag

    t
        Time

    eos
        class:EOS implementing the pressure and temperature functions for
        returning pressure and temperature as a function of the state q.
    """

    ndim = discr.dim

    vol_flux = _inviscid_flux(discr, q, eos)
    dflux = flat_obj_array(
        [
            discr.weak_div(vol_flux[(i * ndim): (i + 1) * ndim])
            for i in range(ndim + 2)
        ]
    )

    interior_face_flux = _facial_flux(
        discr, q_tpair=interior_trace_pair(discr, q), eos=eos
    )

    # Domain boundaries
    domain_boundary_flux = sum(
        _facial_flux(
            discr,
            q_tpair=boundaries[btag].boundary_pair(discr,
                                                   q,
                                                   t=t,
                                                   btag=btag,
                                                   eos=eos),
            eos=eos
        )
        for btag in boundaries
    )

    # Flux across partition boundaries
    partition_boundary_flux = sum(
        _facial_flux(discr, q_tpair=part_pair, eos=eos)
        for part_pair in cross_rank_trace_pairs(discr, q)
    )

    return discr.inverse_mass(
        dflux - discr.face_mass(interior_face_flux + domain_boundary_flux
                                + partition_boundary_flux)
    )


def get_inviscid_cfl(discr, q, dt, eos=IdealSingleGas()):
    """
    Routine calculates and returns CFL based on current state and timestep
    """
    wanted_dt = get_inviscid_timestep(discr, q, eos=eos)
    return dt / wanted_dt


def get_inviscid_timestep(discr, q, cfl=1.0, eos=IdealSingleGas()):
    """
    Routine (will) return the (local) maximum stable inviscid timestep.
    Currently, it's a hack waiting for the geometric_factor helpers port
    from grudge.
    """
    dim = discr.dim
    mesh = discr.mesh
    order = max([grp.order for grp in discr.discr_from_dd("vol").groups])
    nelements = mesh.nelements
    nel_1d = nelements ** (1.0 / (1.0 * dim))

    # This roughly reproduces the timestep AK used in wave toy
    dt = (1.0 - 0.25 * (dim - 1)) / (nel_1d * order ** 2)
    return cfl * dt

#    dt_ngf = dt_non_geometric_factor(discr.mesh)
#    dt_gf  = dt_geometric_factor(discr.mesh)
#    wavespeeds = _get_wavespeed(w,eos=eos)
#    max_v = clmath.max(wavespeeds)
#    return c*dt_ngf*dt_gf/max_v
