#!/bin/bash
# Fortran shadow of jsam/core/physics/sgs.py tests.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_sgs)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o sgs_driver

run_case() {
  local mode="$1"
  local label="sgs.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./sgs_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case shear_prod_zero_velocity
run_case shear_prod_uniform_u
run_case smag_zero_def2
run_case smag_positive_def2
run_case diffuse_scalar_uniform
