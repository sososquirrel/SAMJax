#!/bin/bash
# test_grid_adzw — verify build_metric()'s dz_ref / adz / adzw on a
# synthetic stretched vertical grid against gSAM SRC/setgrid.f90.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_grid_adzw)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o adzw_driver

"$PYTHON" "${HERE}/dump_inputs.py"
./adzw_driver
run_compare "grid_adzw" fortran_out.bin jsam_out.bin
