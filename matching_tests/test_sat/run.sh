#!/bin/bash
# test_sat — Fortran shadow of jsam.core.physics.microphysics saturation.
# 4-decimal comparison of esatw, qsatw, dtqsatw across a (T,p) sweep that
# now includes 50 explicit low-pressure points where gSAM's
# `max(es, p-es)` clamp matters (195/5656 were failing before the fix).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_sat)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o sat_driver

run_case() {
  local mode="$1"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./sat_driver "$mode"
  run_compare "sat.${mode}" fortran_out.bin jsam_out.bin
}

for case in esatw qsatw dtqsatw; do
  run_case "$case"
done
