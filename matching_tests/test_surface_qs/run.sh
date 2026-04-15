#!/bin/bash
# test_surface_qs — verifies surface qs_sfc = 0.98 * qsatw(SST, p/100)
# matches gSAM SRC/sat.f90 + SRC/oceflx.f90 to 4 decimals on a synthetic
# (8 x 16) tropical SST/p tile.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_surface_qs)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o surface_qs_driver

"$PYTHON" "${HERE}/dump_inputs.py"
./surface_qs_driver
run_compare "surface_qs" fortran_out.bin jsam_out.bin
