#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_lsforcing.py
#
# subsidence cases: zero_w, uniform_phi, magnitude
# ls_proc case:     dtls_magnitude

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_lsforcing)"
cd "$WD"

# Build once
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o lsforcing_driver

run_case() {
  local driver_mode="$1"
  local case_name="$2"
  local label="lsforcing.${case_name}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case_name"
  ./lsforcing_driver "$driver_mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case subsidence     subsidence.zero_w
run_case subsidence     subsidence.uniform_phi
run_case subsidence     subsidence.magnitude
run_case ls_proc_direct ls_proc_direct.dtls_magnitude
