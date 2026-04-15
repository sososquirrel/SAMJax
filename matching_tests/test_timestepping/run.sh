#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_timestepping.py
#
# For every numerical case in the python test we:
#   1. Re-use the python fixture to dump (inputs.bin, jsam_out.bin).
#   2. Run the Fortran kernel on inputs.bin -> fortran_out.bin.
#   3. Diff the two via common.compare at 4 decimal places.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_timestepping)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o timestepping_driver

run_case() {
  local mode="$1"
  local label="timestepping.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./timestepping_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case ab2_coefs
run_case ab2_step
