"""Dump inputs and jsam outputs for test_icycle_dt.

Tests the AB3 tendency buffer rotation under gSAM-style icycle
subcycling vs jsam's driver-level Python rotation.

Cases
-----
icycle_dt_match     — 1 icycle, same dt. Both should match.
icycle_3_vs_1       — 3 icycles (gSAM) vs 1 step (jsam). Shows difference.
icycle_ab_rotation  — 3 icycles, compare buffer contents after rotation.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
SAMJAX_ROOT = MT_ROOT.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, str(SAMJAX_ROOT))

from common.bin_io import write_bin  # noqa: E402


def _build_state(case: str):
    n = 16
    np.random.seed(123)
    phi = np.random.randn(n).astype(np.float32) * 10.0

    if case == "icycle_dt_match":
        n_icycles = 1
        tend_vals = np.random.randn(n_icycles, n).astype(np.float32)
        dt_vals = np.array([4.81], dtype=np.float32)
    elif case in ("icycle_3_vs_1", "icycle_ab_rotation"):
        n_icycles = 3
        tend_vals = np.random.randn(n_icycles, n).astype(np.float32)
        dt_vals = np.array([4.81, 4.81, 4.81], dtype=np.float32)
    else:
        raise ValueError(f"unknown case: {case}")

    return n, n_icycles, phi, tend_vals, dt_vals


def _jsam_icycle(phi, tend_vals, dt_vals, n_icycles):
    """Simulate jsam's driver-level AB rotation."""
    phi = phi.copy().astype(np.float64)
    n = len(phi)

    # jsam stores nm1 and nm2 (not 3-slot circular)
    tend_nm1 = np.zeros(n, dtype=np.float64)
    tend_nm2 = np.zeros(n, dtype=np.float64)

    for ic in range(n_icycles):
        dt = float(dt_vals[ic])
        tend_n = tend_vals[ic].astype(np.float64)

        # AB coefficients
        if ic == 0:
            at, bt, ct = 1.0, 0.0, 0.0
        elif ic == 1:
            at, bt, ct = 1.5, -0.5, 0.0
        else:
            at, bt, ct = 23.0/12.0, -16.0/12.0, 5.0/12.0

        phi += dt * (at * tend_n + bt * tend_nm1 + ct * tend_nm2)

        # Python rotation: nm2 <- nm1, nm1 <- n
        tend_nm2 = tend_nm1.copy()
        tend_nm1 = tend_n.copy()

    return (phi.astype(np.float32),
            tend_nm1.astype(np.float32),
            tend_nm2.astype(np.float32))


def main() -> int:
    case = sys.argv[1]
    n, n_icycles, phi, tend_vals, dt_vals = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("ii", n, n_icycles))
        f.write(phi.astype(np.float32).tobytes())
        f.write(tend_vals.astype(np.float32).tobytes())
        f.write(dt_vals.astype(np.float32).tobytes())

    # jsam side
    phi_final, tend_nm1, tend_nm2 = _jsam_icycle(phi, tend_vals, dt_vals, n_icycles)

    # Match Fortran output layout: phi_final + tend(na) + tend(nb) + tend(nc)
    # After gSAM's 3 rotations: na points to the OLDEST slot.
    # jsam: tend_nm1 = most recent, tend_nm2 = second most recent
    # The mapping depends on how many rotations happened.
    # For comparison, output: phi + tend_nm1 + tend_nm2 + zeros
    # (The Fortran driver outputs the 3 slots in their current na/nb/nc order)
    #
    # NOTE: This test will SHOW the difference in buffer layout between
    # gSAM's circular rotation and jsam's Python swap. The phi_final
    # should match if the AB coefficients are correct.
    out = np.concatenate([
        phi_final,
        tend_nm1,
        tend_nm2,
        np.zeros(n, dtype=np.float32),  # jsam has no "nc" slot
    ])
    write_bin("jsam_out.bin", out)

    print(f"[icycle_dt] case={case}  n={n}  n_icycles={n_icycles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
