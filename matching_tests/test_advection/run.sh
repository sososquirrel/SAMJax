#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_advection.py.
#
# For every case:
#   1. Python fixture dumps (inputs.bin, jsam_out.bin).
#   2. Fortran kernel reads inputs.bin → fortran_out.bin.
#   3. Diff via common.compare.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_advection)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o advection_driver

run_case() {
  local case="$1"
  local label="advection.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./advection_driver "$case"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

# face_5th kernel unit tests
run_case face5_cn1
run_case face5_cn_neg1
run_case face5_cn0_linear

# Full advect_scalar — special cases where answer is trivially phi
run_case zero_velocity
run_case constant_field
