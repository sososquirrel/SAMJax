#!/bin/bash
# test_adams — isolated matching test for gSAM adamsA.f90 / adamsB.f90
#              vs the corresponding jsam primitives
#              (jsam.core.dynamics.timestepping.ab_step and
#               jsam.core.dynamics.pressure.adams_b).
#
# For every case:
#   1. dump_inputs.py  constructs a small deterministic fixture,
#      calls jsam to compute the reference output, and writes
#      inputs.bin + jsam_out.bin.
#   2. driver.f90 (compiled once) reads inputs.bin, applies the gSAM
#      formula, and writes fortran_out.bin.
#   3. common.compare diffs the two.
#
# Uses /glade/u/home/sabramian/SAMJax/jsam — NOT the old jsam repo.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_adams)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o adams_driver

run_case() {
  local case="$1"
  local label="adams.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./adams_driver "$case"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

# adamsA — Adams-Bashforth 3 predictor (ab_step)
run_case adamsA_zero
run_case adamsA_ab3
run_case adamsA_diff

# adamsB — lagged pressure gradient correction (adams_b)
run_case adamsB_noop
run_case adamsB_const_p
