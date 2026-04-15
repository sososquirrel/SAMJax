#!/bin/bash
# test_consts — verifies jsam.core.physics.microphysics constant exports
# match gSAM SRC/consts.f90 to 4 decimals.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_consts)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o consts_driver
./consts_driver

"$PYTHON" "${HERE}/dump_inputs.py"
run_compare "consts.all" fortran_out.bin jsam_out.bin
