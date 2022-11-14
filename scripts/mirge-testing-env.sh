#!/bin/bash
#
# Applications may source this file for a set of environment
# variables to make it more convenient to exercise parallel
# mirgecom applications on various platforms.

# set -x

MIRGE_HOME=${1:-"${MIRGE_HOME}"}
if [[ -z "${MIRGE_HOME}" ]]; then
    MIRGE_HOME="."
fi
cd ${MIRGE_HOME}
MIRGE_HOME="$(pwd)"
cd -

MIRGE_PARALLEL_SPAWNER=""
MIRGE_MPI_EXEC="mpiexec"
PYOPENCL_TEST=""
PYOPENCL_CTX=""

if [[ $(hostname) == "porter" ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/run-gpus-generic.sh"
    PYOPENCL_TEST="port:nv"
    PYOPENCL_CTX="port:nv"

elif [[ $(hostname) == "lassen"* ]]; then
    MIRGE_PARALLEL_SPAWNER="bash ${MIRGE_HOME}/scripts/lassen-parallel-spawner.sh"
    PYOPENCL_TEST="port:tesla"
    PYOPENCL_CTX="port:tesla"
    MIRGE_MPI_EXEC="jsrun -g 1 -a 1"
fi

export MIRGE_HOME
export MIRGE_PARALLEL_SPAWNER
export MIRGE_MPI_EXEC
export PYOPENCL_TEST
export PYOPENCL_CTX
