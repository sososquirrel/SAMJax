#!/bin/bash
# test_sgs_euler_order — SGS momentum as non-AB Euler increment
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-5e-6}"
export RTOL="${RTOL:-5e-5}"

WD="$(make_workdir test_sgs_euler_order)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o sgsorder_driver.o
$FC $FFLAGS sgsorder_driver.o -o sgsorder_driver

run_case() {
  local case="$1"
  local label="sgs_euler_order.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./sgsorder_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case sgs_euler_zero
run_case sgs_euler_uniform_shear
run_case sgs_euler_post_advance
run_case sgs_euler_order_swap
