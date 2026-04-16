#!/bin/bash
# test_w_courant_clip — W Courant number limiter
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-6}"
export RTOL="${RTOL:-1e-5}"

WD="$(make_workdir test_w_courant_clip)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o wclip_driver.o
$FC $FFLAGS wclip_driver.o -o wclip_driver

run_case() {
  local case="$1"
  local label="w_courant_clip.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./wclip_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case wclip_below_threshold
run_case wclip_above_threshold
run_case wclip_sponge_layer
run_case wclip_native_adzw
