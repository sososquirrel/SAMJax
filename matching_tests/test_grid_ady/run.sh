#!/bin/bash
# test_grid_ady — verify LatLonGrid.ady / dy_ref against gSAM setgrid.f90
# adyy formula on a synthetic non-uniform latitude band.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_grid_ady)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o ady_driver

"$PYTHON" "${HERE}/dump_inputs.py"
./ady_driver
run_compare "grid_ady" fortran_out.bin jsam_out.bin
