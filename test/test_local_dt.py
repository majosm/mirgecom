"""Test time integrators."""

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
import logging
import pytest
import importlib

from mirgecom.integrators import (euler_step,
                                  lsrk54_step,
                                  lsrk144_step,
                                  rk4_step)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("integrator", "method_order"),
                         [(euler_step, 1),
                          (lsrk54_step, 4),
                          (lsrk144_step, 4),
                          (rk4_step, 4)])
def test_integration_order(integrator, method_order):
    """Test that time integrators have correct order using local time stepping.
    
    Run for dt as an array (and not as a float) but the order check is done
    one dt at a time.
    """
    def exact_soln(t):
        return np.exp(-t)

    def rhs(t, state):
        return -np.exp(-t)

    from pytools.convergence import EOCRecorder
    integrator_eoc = EOCRecorder()

    local_dt = np.asarray([0.5, 1.0, 1.5, 2.0])
    for i in range(0, local_dt.shape[0]):
        for refine in [1, 2, 4, 8]:
            dt = local_dt / refine
            t = 0*dt
            state = exact_soln(t)

            istep = 0

            while istep < 5:
                state = integrator(state, t, dt, rhs)
                t = t + dt       
                istep += 1

            error = np.abs(state[i] - exact_soln(t)[i]) / exact_soln(t)[i]
            integrator_eoc.add_data_point(dt[i], error)

        logger.info(f"Time Integrator EOC:\n = {integrator_eoc}")
        assert integrator_eoc.order_estimate() >= method_order - .01  
