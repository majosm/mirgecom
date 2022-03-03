"""Utilities and functions for symbolic code expressions.

.. autofunction:: diff
.. autofunction:: div
.. autofunction:: grad

.. autoclass:: EvaluationMapper
.. autofunction:: evaluate
"""

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

import numpy as np  # noqa
import numpy.linalg as la # noqa
from pytools.obj_array import make_obj_array
import pymbolic as pmbl
from pymbolic.mapper.evaluator import EvaluationMapper as BaseEvaluationMapper
import mirgecom.math as mm


def diff(var):
    """Return the symbolic derivative operator with respect to *var*."""
    from pymbolic.mapper.differentiator import DifferentiationMapper

    def func_map(arg_index, func, arg, allowed_nonsmoothness):
        if func == pmbl.var("sin"):
            return pmbl.var("cos")(*arg)
        elif func == pmbl.var("cos"):
            return -pmbl.var("sin")(*arg)
        elif func == pmbl.var("exp"):
            return pmbl.var("exp")(*arg)
        else:
            raise ValueError("Unrecognized function")

    return DifferentiationMapper(var, func_map=func_map)


def div(ambient_dim, func):
    """Return the symbolic divergence of *func*."""
    coords = pmbl.make_sym_vector("x", ambient_dim)

    def component_div(f):
        return sum(diff(coords[i])(f[i]) for i in range(ambient_dim))

    from grudge.op import _div_helper
    from pymbolic.primitives import Expression
    return _div_helper(ambient_dim, component_div, Expression, func)


def grad(ambient_dim, func, nested=False):
    """Return the symbolic *dim*-dimensional gradient of *func*."""
    coords = pmbl.make_sym_vector("x", ambient_dim)

    def component_grad(f):
        return make_obj_array([diff(coords[i])(f) for i in range(ambient_dim)])

    from grudge.op import _grad_helper
    from pymbolic.primitives import Expression
    return _grad_helper(
        ambient_dim, component_grad, Expression, func, nested=nested)


class EvaluationMapper(BaseEvaluationMapper):
    """Evaluates symbolic expressions given a mapping from variables to values.

    Inherits from :class:`pymbolic.mapper.evaluator.EvaluationMapper`.
    """

    def map_call(self, expr):
        """Map a symbolic code expression to actual function call."""
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        par, = expr.parameters
        return getattr(mm, expr.function.name)(self.rec(par))


def evaluate(expr, mapper_type=EvaluationMapper, **kwargs):
    """Evaluate a symbolic expression using a specified mapper."""
    mapper = mapper_type(kwargs)

    from arraycontext import rec_map_array_container
    return rec_map_array_container(mapper, expr)
