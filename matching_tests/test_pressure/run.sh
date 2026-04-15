#!/bin/bash
# Fortran shadow of jsam/core/dynamics/pressure.py — press_rhs residual tests.
#
# All four cases compare press_rhs output (Fortran vs jsam).
# For zero/constant-p gradient cases the expected RHS is 0 or near-0.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_pressure)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o pressure_driver

run_case() {
  local mode="$1"
  local label="pressure.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./pressure_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case press_rhs_zero_velocity
run_case press_rhs_constant_U
run_case press_gradient_zero_p
run_case press_gradient_constant_p
