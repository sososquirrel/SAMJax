#!/bin/bash
# test_static_energy_roundtrip — s = TABS + gamaz round-trip fidelity
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-5}"
export RTOL="${RTOL:-1e-4}"

WD="$(make_workdir test_static_energy_roundtrip)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o se_driver.o
$FC $FFLAGS se_driver.o -o se_driver

run_case() {
  local case="$1"
  local label="static_energy.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./se_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case roundtrip_identity
run_case roundtrip_strat
run_case roundtrip_cloudy
run_case gamaz_native
