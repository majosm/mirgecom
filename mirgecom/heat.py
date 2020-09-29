r""":mod:`mirgecom.heat` computes the rhs of the heat equation.

Heat equation:

.. math::

    \partial_t \mathbf{u} = \alpha\nabla^2\mathbf{u}

.. autofunction:: heat_operator
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
from pytools.obj_array import (
    flat_obj_array, make_obj_array)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import thaw
from grudge.symbolic.primitives import TracePair
from grudge.eager import interior_trace_pair, cross_rank_trace_pairs

def _q_flux(discr, alpha, w_tpair):
    actx = w_tpair.int[0].array_context

    normal = thaw(actx, discr.normal(w_tpair.dd))

    flux_weak = math.sqrt(alpha)*w_tpair.avg*normal

    return discr.project(w_tpair.dd, "all_faces", flux_weak)


def _flux(discr, alpha, q_tpair):
    actx = q_tpair.int[0].array_context

    normal = thaw(actx, discr.normal(q_tpair.dd))

    flux_weak = math.sqrt(alpha)*make_obj_array([np.dot(q_tpair.avg, normal)])

    return discr.project(q_tpair.dd, "all_faces", flux_weak)


def heat_operator(discr, alpha, w):
    """Compute the RHS of the heat equation.

    Parameters
    ----------
    discr: grudge.eager.EagerDGDiscretization
        the discretization to use
    alpha: float
        the (constant) diffusivity
    w: numpy.ndarray
        an object array of DOF arrays, representing the state vector

    Returns
    -------
    numpy.ndarray
        an object array of DOF arrays, representing the ODE RHS
    """
    u = w[0]

    actx = u.array_context

    grad_u = discr.weak_grad(u)
    q_flux = (_q_flux(discr, alpha=alpha, w_tpair=interior_trace_pair(discr, w)) +
                sum(
                    _q_flux(discr, alpha=alpha, w_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, w)
                )
            )

    q = discr.inverse_mass(-math.sqrt(alpha)*grad_u + discr.face_mass(q_flux))

    return (
        discr.inverse_mass(
            make_obj_array([-math.sqrt(alpha)*discr.weak_div(q)])
            +  # noqa: W504
            discr.face_mass(
                _flux(discr, alpha=alpha, q_tpair=interior_trace_pair(discr, q))
                + sum(
                    _flux(discr, alpha=alpha, q_tpair=tpair)
                    for tpair in cross_rank_trace_pairs(discr, q))
                )
            )
        )
