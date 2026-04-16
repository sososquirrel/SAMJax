#!/bin/bash
# test_twostep_oracle — 2-step integration vs gSAM oracle tensors
# This test does NOT compile Fortran — it's a pure Python comparison
# of the existing debug dump CSVs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_twostep_oracle)"
cd "$WD"

# No Fortran build needed — pure Python comparison
run_case() {
  local case="$1"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
}

run_case oracle_globals_step1
run_case oracle_globals_step2
run_case oracle_tensor_step1_adamsA
run_case oracle_summary
