"""Provide some utilities for building simulation applications.

.. autofunction:: check_step
.. autofunction:: check_time
.. autofunction:: create_parallel_grid
.. autofunction:: get_sim_timestep
.. autofunction:: make_timestepper
.. autofunction:: sim_checkpoint
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

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np

from mirgecom.utils import get_containing_interval

from meshmode.dof_array import DOFArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """State for a given timestep.

    .. attribute:: step

        The current step number.

    .. attribute:: time

        The current time.

    .. attribute:: fields

        The state fields being evolved in time.

    """

    step: int
    time: float
    fields: Union[DOFArray, np.ndarray]


def check_step(step, interval):
    """
    Check step number against a user-specified interval.

    Utility is used typically for visualization.

    - Negative numbers mean 'never visualize'.
    - Zero means 'always visualize'.

    Useful for checking whether the current step is an output step,
    or anyting else that occurs on fixed intervals.
    """
    if interval == 0:
        return True
    elif interval < 0:
        return False
    elif step % interval == 0:
        return True
    return False


def check_time(t, interval, tol=1e-15):
    """Check simulation time against a user-specified interval."""
    _, t_interval_start, t_interval_end = get_containing_interval(
        0, interval, t)
    return t_interval_end-t <= tol or t-t_interval_start <= tol


def create_parallel_grid(comm, generate_grid):
    """Create and partition a grid.

    Create a grid with the user-supplied grid generation function
    *generate_grid*, partition the grid, and distribute it to every
    rank in the provided MPI communicator *comm*.

    Parameters
    ----------
    comm:
        MPI communicator over which to partition the grid
    generate_grid:
        Callable of zero arguments returning a :class:`meshmode.mesh.Mesh`.
        Will only be called on one (undetermined) rank.

    Returns
    -------
    local_mesh : :class:`meshmode.mesh.Mesh`
        The local partition of the the mesh returned by *generate_grid*.
    global_nelements : :class:`int`
        The number of elements in the serial grid
    """
    from meshmode.distributed import (
        MPIMeshDistributor,
        get_partition_by_pymetis,
    )
    num_parts = comm.Get_size()
    mesh_dist = MPIMeshDistributor(comm)
    global_nelements = 0

    if mesh_dist.is_mananger_rank():

        mesh = generate_grid()

        global_nelements = mesh.nelements

        part_per_element = get_partition_by_pymetis(mesh, num_parts)
        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    return local_mesh, global_nelements


def get_sim_timestep(state, dt_max, t_final=None, key_every=None):
    """Compute the timestep given a maximum value and various constraints."""
    if key_every is None:
        key_every = []

    if t_final is not None:
        key_every.append(t_final)

    dt = dt_max

    for key_dt in key_every:
        _, _, t_next_key = get_containing_interval(0, key_dt, state.time)
        dt = min(dt, t_next_key - state.time)

    return dt


def make_timestepper(get_timestep, field_stepper):
    """
    Create a function that advances the simulation from one step to the next.

    Parameters
    ----------
    get_timestep
        Function that returns the next timestep size given the current state.
    field_stepper
        Function that returns the next step's field values given the current field
        values, the current time, and the timestep size.

    Returns
    -------
    timestepper
        A function that takes the current step's state and returns the next step's
        state.
    """
    def timestepper(state):
        dt = get_timestep(state)
        return State(
            step=state.step + 1,
            time=state.time + dt,
            fields=field_stepper(state.fields, state.time, dt))

    return timestepper


def sim_checkpoint(state, weak=False, nsteps=None, t_final=None, nvis=None,
        vis_dt=None, write_vis=None, nrestart=None, write_restart=None):
    """Handle logic for basic checkpointing functionality."""
    done = False
    if nsteps is not None:
        done = state.step == nsteps
    if t_final is not None:
        done = done or state.time >= t_final

    if not weak:
        do_vis = done
        if nvis is not None:
            do_vis = do_vis or check_step(state.step, nvis)
        if vis_dt is not None:
            do_vis = do_vis or check_time(state.time, vis_dt)
        if do_vis and write_vis is not None:
            write_vis(state)

        do_restart = done
        if nrestart is not None:
            do_restart = do_vis or check_step(state.step, nrestart)
        if do_restart and write_restart is not None:
            write_restart(state)

    return done
