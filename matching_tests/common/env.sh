#!/bin/bash
# Common environment for all matching tests.
#
# Sourced at the top of every <module>/run.sh:
#   source "${HERE}/../common/env.sh"
#
# Defines:
#   FC, FFLAGS, PYTHON, JSAM_ROOT, ATOL, RTOL
#   make_workdir <test_name>   — mkdir -p and echo the path
#   run_compare <label> <f_bin> <j_bin>

FC="${FC:-gfortran}"
FFLAGS="${FFLAGS:--O2 -fno-unsafe-math-optimizations}"
PYTHON="${PYTHON:-python}"
JSAM_ROOT="${JSAM_ROOT:-/glade/u/home/sabramian/jsam}"
ATOL="${ATOL:-5e-5}"
RTOL="${RTOL:-1e-4}"

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MT_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"

# Add jsam root to PYTHONPATH so "from common.bin_io import write_bin" and
# "from jsam.core..." both work without installing anything.
export PYTHONPATH="${MT_ROOT}:${JSAM_ROOT}:${PYTHONPATH:-}"

# make_workdir <name>
# Creates <matching_tests>/<name>/work/ and prints its absolute path.
make_workdir() {
    local name="$1"
    local wd="${MT_ROOT}/${name}/work"
    mkdir -p "$wd"
    echo "$wd"
}

# run_compare <label> <fortran_bin> <jsam_bin>
# Calls common/compare.py; inherits ATOL and RTOL from the environment.
run_compare() {
    local label="$1" f_bin="$2" j_bin="$3"
    "$PYTHON" -m common.compare "$label" "$f_bin" "$j_bin" "${ATOL}" "${RTOL}"
}
