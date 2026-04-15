#!/bin/bash
# AB3 coefficient and step tests — Fortran vs jsam ab_coefs / ab_step.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir timestepping)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o timestepping_driver

run_case() {
  local mode="$1"
  local label="timestepping.${mode}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$mode"
  ./timestepping_driver "$mode"
  run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case ab3_coefs
run_case ab3_step
