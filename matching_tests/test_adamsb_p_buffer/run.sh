#!/bin/bash
# test_adamsb_p_buffer — pressure buffer rotation + adamsB over steps 1-3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-5e-6}"
export RTOL="${RTOL:-5e-5}"

WD="$(make_workdir test_adamsb_p_buffer)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o pb_driver.o
$FC $FFLAGS pb_driver.o -o pb_driver

run_case() {
  local case="$1"
  local label="adamsb_pbuf.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./pb_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case pbuf_step1_noop
run_case pbuf_linear_p
run_case pbuf_rotation_step2
run_case pbuf_gradient_metric
