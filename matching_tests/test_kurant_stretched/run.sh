#!/bin/bash
# test_kurant_stretched — verify jsam compute_cfl on a non-uniform vertical
# grid against gSAM SRC/kurant.f90.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_kurant_stretched)"
cd "$WD"

$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o kurant_stretched_driver

"$PYTHON" "${HERE}/dump_inputs.py"
./kurant_stretched_driver
run_compare "kurant_stretched" fortran_out.bin jsam_out.bin
