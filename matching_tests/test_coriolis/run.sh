#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_coriolis.py
#
# For each case:
#   1. dump_inputs.py writes inputs.bin + jsam_out.bin
#   2. Fortran driver reads inputs.bin → fortran_out.bin
#   3. common/compare.py diffs them

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_coriolis)"
cd "$WD"

# Build once
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o coriolis_driver

run_case() {
  local case_name="$1"
  local label="coriolis.${case_name}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case_name"
  ./coriolis_driver coriolis_tend
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case coriolis_tend.zero_velocity
run_case coriolis_tend.NH_positive_V
run_case coriolis_tend.polar_bc
