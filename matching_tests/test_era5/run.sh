#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"
WD="$(make_workdir test_era5)"
cd "$WD"
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o era5_driver
run_case() {
  local mode="$1"
  local atol="${2:-${ATOL}}"
  local rtol="${3:-${RTOL}}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./era5_driver "$mode"
  ATOL="$atol" RTOL="$rtol" run_compare "era5.${mode}" fortran_out.bin jsam_out.bin
}
run_case z_from_Z
run_case omega_to_w
run_case interp_pres
run_case stagger_u
run_case stagger_v
run_case stagger_w
run_case ref_column 1e-3 1e-3
