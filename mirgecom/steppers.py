"""Helper functions for advancing a gas state.

.. autofunction:: advance_state
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

from logpyle import set_dt
from mirgecom.logging_quantities import set_sim_state


def advance_state(timestepper, checkpoint, state, logmgr=None):
    """
    Advance *state* in time.

    Parameters
    ----------
    timestepper
        Function that advances the state from one timestep to the next.
    checkpoint
        Function is user-defined and can be used to perform simulation status
        reporting, viz, and restart i/o. Returns a boolean indicating if the
        stepping should terminate.
    state
        Starting state that will be advanced by this stepper.

    Returns
    -------
    state
        The final state.
    """
    if state.step == 0:
        done = checkpoint(state)
        if done:
            raise ValueError("Checkpoint returned 'True' before stepping.")

    done = False

    while not done:

        if logmgr:
            logmgr.tick_before()
            t_before = state.time

        state = timestepper(state)

        done = checkpoint(state)

        if logmgr:
            set_dt(logmgr, state.time - t_before)
            set_sim_state(logmgr, state.fields)
            logmgr.tick_after()

    return state
