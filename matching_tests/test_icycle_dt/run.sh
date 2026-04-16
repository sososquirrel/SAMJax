#!/bin/bash
# test_icycle_dt — gSAM icycle subcycling vs jsam driver-level subcycling
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-3}"
export RTOL="${RTOL:-1e-2}"

WD="$(make_workdir test_icycle_dt)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o icycle_driver.o
$FC $FFLAGS icycle_driver.o -o icycle_driver

run_case() {
  local case="$1"
  local label="icycle_dt.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./icycle_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case icycle_dt_match
run_case icycle_3_vs_1
run_case icycle_ab_rotation
