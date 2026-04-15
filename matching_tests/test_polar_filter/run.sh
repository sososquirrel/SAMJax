#!/bin/bash
# Fortran shadow of jsam/tests/unit/test_polar_filter.py
#
# NOTE: jsam uses a Fourier filter; gSAM uses a box smoother — they are
# NOT the same. This test uses a Fortran DFT implementation of the SAME
# Fourier filter as jsam, so it tests self-consistency (not gSAM parity).
# See TODO.md for details.
#
# Relaxed tolerance (1e-4 abs) because the brute-force DFT has more
# floating-point rounding than the FFTW-backed numpy/JAX rfft.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../common/env.sh"

WD="$(make_workdir test_polar_filter)"
cd "$WD"

# Build once
$FC $FFLAGS -c "${HERE}/driver.f90" -o driver.o
$FC $FFLAGS driver.o -o polar_filter_driver

run_case() {
  local case_name="$1"
  local label="polar_filter.${case_name}"
  "$PYTHON" "${HERE}/dump_inputs.py" "$case_name"
  ./polar_filter_driver polar_fourier_filter
  ATOL=1e-4 run_compare "$label" fortran_out.bin jsam_out.bin
}

run_case polar_fourier_filter.equator_unchanged
run_case polar_fourier_filter.high_mode_zeroed
run_case polar_fourier_filter.low_mode_preserved
