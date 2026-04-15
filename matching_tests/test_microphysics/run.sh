#!/bin/bash
# Fortran shadow of jsam/core/physics/microphysics.py — saturation functions.
#
# Tests qsatw and qsati (Buck 1981) matching.  The known qsat clamping bug
# (jsam uses max(p-es, 1e-3) instead of max(es, p-es)) means we compare
# the jsam formula output, not the "correct" formula.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_microphysics)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o microphysics_driver

run_case() {
  local mode="$1"
  local label="microphysics.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./microphysics_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case qsatw_at_20C_1000mb
run_case qsatw_monotone
run_case qsati_below_freezing
