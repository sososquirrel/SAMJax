#!/bin/bash
# test_pressure_rhs_div — spherical divergence for pressure RHS
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-6}"
export RTOL="${RTOL:-1e-5}"

WD="$(make_workdir test_pressure_rhs_div)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o prhs_driver.o
$FC $FFLAGS prhs_driver.o -o prhs_driver

run_case() {
  local case="$1"
  local label="pressure_rhs.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./prhs_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case div_uniform_u
run_case div_linear_u
run_case div_v_with_ady
run_case div_w_with_rho
