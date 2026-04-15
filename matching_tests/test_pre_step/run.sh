#!/bin/bash
# test_pre_step — saturation-function bit-level match
# ---------------------------------------------------
# Small-brick isolation of the QC/QI init mismatch in the IRMA debug500
# run at step 1 stage 0 (see scripts/compare_debug_globals.py output).
#
# The hypothesis this test is designed to exercise:
#   Every init-time QC/QI is ultimately a function of the four saturation
#   primitives below (applied to an ERA5 column). If the primitives
#   disagree, the downstream init certainly will.
#
#     gSAM:   SRC/sat.f90          (esatw, esati, qsatw, qsati)
#     jsam:   core/physics/slm/sat.py
#
# Each case:
#   1. dump_inputs.py writes inputs.bin (n, T(n), P(n)) and runs
#      jsam.core.physics.slm.sat at the same points → jsam_out.bin
#   2. driver.f90 (compiled once) reads inputs.bin, calls the copy-pasted
#      gSAM sat.f90 functions, writes fortran_out.bin
#   3. common.compare diffs the two blobs at the strict float32 level.
#
# IMPORTANT: uses /glade/u/home/sabramian/SAMJax/jsam (set via JSAM_ROOT),
# NOT the old /glade/u/home/sabramian/jsam.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

# Saturation kernels are pure pointwise arithmetic — expect very tight
# agreement. Any looser threshold would hide the bug this test exists
# to catch.
export ATOL="${ATOL:-1e-7}"
export RTOL="${RTOL:-1e-6}"

WD="$(make_workdir test_pre_step)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o sat_driver.o
$FC $FFLAGS sat_driver.o -o sat_driver

run_mode() {
  local mode="$1"
  local label="pre_step.sat.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./sat_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

# Per-function (localises any disagreement to esatw vs esati vs qsatw vs qsati).
run_mode esatw
run_mode esati
run_mode qsatw
run_mode qsati

# Whole-blob sanity.
run_mode all
