#!/bin/bash
# test_advect_mom_native — same Fortran kernel as test_momentum_advection,
# driven by the real lat_720_dyvar metric (IRMALoader) on narrow latitude
# slices with the full 74-level non-uniform dz.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_advect_mom_native)"
cd "$WD"

# Compile the EXISTING test_momentum_advection driver — we intentionally
# share the Fortran kernel so any upstream fix propagates to both tests.
DRIVER_SRC="${HERE}/../test_momentum_advection/driver.f90"
$FC $FFLAGS -c "$DRIVER_SRC" -o mom_advection_driver.o
$FC $FFLAGS mom_advection_driver.o -o mom_advection_driver

run_case() {
  local case="$1"
  local label="advect_mom_native.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  # The existing driver takes a case name; we reuse its `uniform_U` mode
  # for both native cases because it expects the same binary layout
  # whether the metric is synthetic or native.
  ./mom_advection_driver uniform_U
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case native_uniform_U_eq
run_case native_uniform_U_mid
