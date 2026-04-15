"""
Float32 binary comparator for matching tests.

Usage (from run.sh via run_compare shell function):
    python -m common.compare <label> <fortran_bin> <jsam_bin> [atol] [rtol]

Exits 0 on pass, 1 on fail.

Tolerances:
    atol  default 5e-5  (absolute)
    rtol  default 1e-4  (relative, relative to |jsam| value)

Pass criterion: for every element either
    |f - j| <= atol   OR   |f - j| / (|j| + 1e-30) <= rtol
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Make "python -m common.compare" work when invoked from a work/ subdir that
# does not have the matching_tests root on sys.path yet.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.bin_io import read_bin  # noqa: E402


def compare(
    label:   str,
    f_path:  str,
    j_path:  str,
    atol:    float = 5e-5,
    rtol:    float = 1e-4,
) -> int:
    """Return 0 on pass, 1 on fail."""
    try:
        f = read_bin(f_path)
        j = read_bin(j_path)
    except FileNotFoundError as e:
        print(f"FAIL [{label}]: missing file — {e}")
        return 1

    if f.shape != j.shape:
        print(f"FAIL [{label}]: shape mismatch fortran={f.shape} jsam={j.shape}")
        return 1

    abs_err = np.abs(f.astype(np.float64) - j.astype(np.float64))
    rel_err = abs_err / (np.abs(j.astype(np.float64)) + 1e-30)

    # Pass if EVERY element satisfies at least one tolerance.
    passed = (abs_err <= atol) | (rel_err <= rtol)

    if np.all(passed):
        print(f"PASS [{label}]: n={len(f)}  max_abs={abs_err.max():.2e}"
              f"  max_rel={rel_err.max():.2e}")
        return 0

    n_fail  = int((~passed).sum())
    bad_idx = np.argmax(~passed)
    print(
        f"FAIL [{label}]: {n_fail}/{len(f)} elements outside tolerance "
        f"(atol={atol:.0e} rtol={rtol:.0e})  "
        f"worst: fortran={f[bad_idx]:.6g} jsam={j[bad_idx]:.6g} "
        f"abs={abs_err[bad_idx]:.2e} rel={rel_err[bad_idx]:.2e}"
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python -m common.compare label f_bin j_bin [atol] [rtol]")
        sys.exit(2)
    label  = sys.argv[1]
    f_path = sys.argv[2]
    j_path = sys.argv[3]
    atol   = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-5
    rtol   = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
    sys.exit(compare(label, f_path, j_path, atol, rtol))
