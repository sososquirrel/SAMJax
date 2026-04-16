#!/bin/bash
# test_fct_antidiff — Zalesak FCT antidiffusive flux limiter
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-5e-5}"
export RTOL="${RTOL:-1e-4}"

WD="$(make_workdir test_fct_antidiff)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o fct_driver.o
$FC $FFLAGS fct_driver.o -o fct_driver

run_case() {
  local case="$1"
  local label="fct_antidiff.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./fct_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case fct_uniform
run_case fct_step_function
run_case fct_cosine_bell
run_case fct_near_zero
