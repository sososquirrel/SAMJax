#!/bin/bash
# Fortran shadow of jsam/core/physics/surface.py tests.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_surface)"
cd "$WD"

# Build once.
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o surface_driver

run_case() {
  local mode="$1"
  local label="surface.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./surface_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case qsat_w_at_300K
run_case qsat_monotone_in_T
run_case bulk_fluxes_warm_sst_shf
run_case bulk_fluxes_tau_opposes
