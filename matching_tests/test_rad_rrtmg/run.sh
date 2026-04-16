#!/bin/bash
# test_rad_rrtmg -- RRTMG SW+LW wrapper-vs-raw-f2py matching test.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_rad_rrtmg)"
cd "$WD"

"$PYTHON" "${HERE}/verify_co2_and_ozone.py"
"$PYTHON" "${HERE}/dump_inputs.py"
run_compare "rrtmg.sw_lw" fortran_out.bin jsam_out.bin
