#!/bin/bash
# test_diagnose_hmean — cos-lat + ady weighted horizontal mean
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-5}"
export RTOL="${RTOL:-1e-4}"

WD="$(make_workdir test_diagnose_hmean)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o hmean_driver.o
$FC $FFLAGS hmean_driver.o -o hmean_driver

run_case() {
  local case="$1"
  local label="diagnose_hmean.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./hmean_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case hmean_uniform
run_case hmean_lat_gradient
run_case hmean_polar_spike
run_case hmean_qv_qn_qp
