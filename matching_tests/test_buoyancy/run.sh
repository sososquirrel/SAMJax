#!/bin/bash
# test_buoyancy — bit-level match of jsam _buoyancy_W (core/step.py)
#                 against gSAM SRC/buoyancy.f90 on identical fields.
#
# Uses SAMJax jsam — JSAM_ROOT is pinned to the SAMJax root below.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-5e-7}"
export RTOL="${RTOL:-5e-5}"

WD="$(make_workdir test_buoyancy)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o buoy_driver.o
$FC $FFLAGS buoy_driver.o -o buoy_driver

run_case() {
  local case="$1"
  local label="buoyancy.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./buoy_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case buoy_neutral
run_case buoy_qv_anom
run_case buoy_T_anom
run_case buoy_cloud
run_case buoy_native_dz
