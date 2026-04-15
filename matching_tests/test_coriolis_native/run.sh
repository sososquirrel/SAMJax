#!/bin/bash
# test_coriolis_native — same Fortran kernel as test_coriolis, driven
# by the real lat_720_dyvar metric (IRMALoader) on narrow latitude slices.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_coriolis_native)"
cd "$WD"

# Compile the EXISTING test_coriolis driver — we intentionally share
# the Fortran kernel so any upstream fix propagates to both tests.
DRIVER_SRC="${HERE}/../test_coriolis/driver.f90"
$FC $FFLAGS -c "$DRIVER_SRC" -o coriolis_driver.o
$FC $FFLAGS coriolis_driver.o -o coriolis_driver

run_case() {
  local case="$1"
  local label="coriolis_native.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./coriolis_driver coriolis_tend
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case native_equator
run_case native_midlat
run_case native_highlat
