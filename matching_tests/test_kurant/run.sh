#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_kurant.py.
#
# For every numerical case in the python test we:
#   1. Re-use the python fixture to dump (inputs.bin, jsam_out.bin).
#   2. Run the Fortran kernel on inputs.bin → fortran_out.bin.
#   3. Diff the two via common.compare at 4 decimal places.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_kurant)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o kurant_driver

run_case() {
  local mode="$1" case="${2:-}"
  local label="kurant.${mode}${case:+.${case}}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode" $case
  ./kurant_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

# compute_cfl — one case per assertion in the python test.
for case in zero_velocity pure_zonal pure_vertical combines_axes latitude_narrowing; do
  run_case compute_cfl "$case"
done

# ab2_coefs — one batch covering all (nstep, dt_curr, dt_prev) the python test exercises.
run_case ab2_coefs
