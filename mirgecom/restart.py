"""Provide some utilities for restarting simulations.

.. autofunction:: read_restart_data
.. autofunction:: write_restart_file
.. autoclass:: RESTART_TAG
.. autofunction:: memoize_from_restart

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

import pickle
from functools import wraps
from meshmode.dof_array import array_context_for_pickling


def read_restart_data(actx, filename):
    """Read the raw restart data dictionary from the given pickle restart file."""
    with array_context_for_pickling(actx):
        with open(filename, "rb") as f:
            return pickle.load(f)


def write_restart_file(actx, restart_data, filename, comm=None):
    """Pickle the simulation data into a file for use in restarting."""
    rank = 0
    if comm:
        rank = comm.Get_rank()
    if rank == 0:
        import os
        rst_dir = os.path.dirname(filename)
        if rst_dir:
            os.makedirs(rst_dir, exist_ok=True)
    if comm:
        comm.barrier()
    with array_context_for_pickling(actx):
        with open(filename, "wb") as f:
            pickle.dump(restart_data, f)


class RESTART_TAG:  # noqa: N801
    def __init__(self, tag):
        self.tag = tag


def memoize_from_restart(restart_data, result_spec):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if restart_data is None:
                return func(*args, **kwargs)
            else:
                return (
                    restart_data[spec.tag]
                    if isinstance(spec, RESTART_TAG)
                    else spec
                    for spec in result_spec)
        return wrapped_func

    return decorator
