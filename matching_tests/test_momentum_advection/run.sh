#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_momentum_advection.py.
#
# For every case:
#   1. Python fixture dumps (inputs.bin, jsam_out.bin).
#   2. Fortran kernel reads inputs.bin → fortran_out.bin.
#   3. Diff via common.compare.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_momentum_advection)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o mom_advection_driver

run_case() {
  local case="$1"
  local label="mom_advection.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./mom_advection_driver "$case"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

# _flux3 kernel unit tests
run_case flux3_positive
run_case flux3_negative
run_case flux3_zero
run_case flux3_constant

# Full advect_momentum — special cases where answer is trivially unchanged
run_case zero_velocity
run_case uniform_U
