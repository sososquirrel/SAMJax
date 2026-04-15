#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_operators.py.
#
# For each case we:
#   1. Dump (inputs.bin, jsam_out.bin) from the Python side.
#   2. Run the Fortran kernel on inputs.bin → fortran_out.bin.
#   3. Diff the two via common.compare.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_operators)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o operators_driver

run_case() {
  local mode="$1"
  local label="operators.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./operators_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case divergence_zero_v_const
run_case divergence_linear_u
run_case laplacian_const
