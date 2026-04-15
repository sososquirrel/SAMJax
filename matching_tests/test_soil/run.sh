#!/bin/bash
# Fortran shadow of jsam SLM soil / saturation / hydraulic code.
#
# Cases (all bit-close — pure scalar arithmetic, float32):
#   sat_qsatw   — saturation qv over liquid water (Magnus / IFS Cy47r3)
#   sat_qsati   — saturation qv over ice
#   cosby_1984  — Cosby 1984 soil hydraulic parameters from SAND/CLAY
#   fh_calc     — Clapp & Hornberger fractional humidity at soil surface

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_soil)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o soil_driver

run_case() {
  local mode="$1"
  local label="soil.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./soil_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case sat_qsatw
run_case sat_qsati
run_case cosby_1984
run_case fh_calc
