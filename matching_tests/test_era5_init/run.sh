#!/bin/bash
# test_era5_init — compare jsam ERA5 initialization against gSAM debug dump
#
# No Fortran driver: gSAM reference comes from existing debug dump rank files.
# jsam side runs era5_init() from scratch and extracts the IRMA sub-region.
#
# Compares all 7 debug-dump fields: U, V, W, TABS, QC, QV, QI.
#
# Tolerances are deliberately loose because the two codes use different
# interpolation pipelines (gSAM reads pre-processed binary via read_field3D;
# jsam reads raw ERA5 netCDF via scipy + custom bilinear).  The test verifies
# structural agreement, not bit-level match.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JSAM_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../common/env.sh"

# Loose tolerances — init pipelines differ in interpolation method.
# These catch gross bugs (wrong field, flipped axes, unit errors) while
# allowing the ~1-5% differences from interpolation chain differences.
export ATOL="${ATOL:-1e-1}"
export RTOL="${RTOL:-5e-2}"

GSAM_DEBUG="${GSAM_DEBUG:-/glade/derecho/scratch/sabramian/gSAM_IRMA_500timesteps_DEBUG/debug}"

WD="$(make_workdir test_era5_init)"

echo "=== test_era5_init ==="
echo "  gSAM debug dir: $GSAM_DEBUG"
echo "  work dir:       $WD"
echo ""

# ── Step 1: Reconstruct gSAM IRMA box from rank files ──
echo "--- Step 1: reading gSAM debug dump ---"
"$PYTHON" "${HERE}/read_gsam_init.py" "$GSAM_DEBUG"
echo ""

# ── Step 2: Run jsam ERA5 init and extract IRMA box ──
echo "--- Step 2: running jsam ERA5 init ---"
"$PYTHON" "${HERE}/dump_jsam_init.py"
echo ""

# ── Step 3: Compare per field ──
echo "--- Step 3: comparing fields ---"
PASS=0
FAIL=0
for field in U V W TABS QC QV QI; do
    gsam_bin="${WD}/gsam_${field}.bin"
    jsam_bin="${WD}/jsam_${field}.bin"
    if [ ! -f "$gsam_bin" ] || [ ! -f "$jsam_bin" ]; then
        echo "  [SKIP] ${field}: missing bin file"
        continue
    fi
    if run_compare "era5_init.${field}" "$gsam_bin" "$jsam_bin"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Summary: ${PASS} passed, ${FAIL} failed ==="

# ── Step 4: Detailed per-field diagnostics ──
echo ""
echo "--- Step 4: detailed diagnostics ---"
"$PYTHON" "${HERE}/diagnose_init_diff.py" "$WD"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
