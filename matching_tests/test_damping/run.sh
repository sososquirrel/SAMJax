#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_damping.py
#
# For each case:
#   1. dump_inputs.py writes inputs.bin + jsam_out.bin (jsam pole_damping)
#   2. Fortran driver reads inputs.bin → fortran_out.bin (gSAM formula)
#   3. common/compare.py diffs them at 4 decimal places
#
# Note: u_max_phys=1e9 is passed to jsam so the physical cap never fires
# — both sides implement the same pure gSAM CFL-based umax formula.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_damping)"
cd "$WD"

# Build once
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o damping_driver

run_case() {
  local mode="$1"          # driver mode: pole_u or pole_v
  local case_name="$2"     # full case name: pole_u.polar_face_halved etc.
  local label="damping.${case_name}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case_name"
  ./damping_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case pole_u pole_u.polar_face_halved
run_case pole_u pole_u.equatorial_no_damp
run_case pole_u pole_u.high_lat_clip
run_case pole_v pole_v.polar_face_halved
