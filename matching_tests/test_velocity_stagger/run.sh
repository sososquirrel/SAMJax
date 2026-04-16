#!/bin/bash
# test_velocity_stagger — verify C-grid staggering of mass-grid velocities
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

export ATOL="${ATOL:-1e-6}"
export RTOL="${RTOL:-1e-5}"

WD="$(make_workdir test_velocity_stagger)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o stagger_driver.o
$FC $FFLAGS stagger_driver.o -o stagger_driver

run_case() {
  local case="$1"
  local label="velocity_stagger.${case}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case"
  ./stagger_driver
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case stagger_periodic_u
run_case stagger_pole_v
run_case stagger_ground_w
