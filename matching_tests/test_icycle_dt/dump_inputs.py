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
    """Simulate gSAM's 3-slot circular buffer (main.f90 icycle rotation).

    gSAM main.f90 lines 353-356:
        nn=na; na=nc; nc=nb; nb=nn
    """
    phi = phi.copy().astype(np.float64)
    n = len(phi)
    slots = np.zeros((n, 3), dtype=np.float64)
    na, nb, nc = 0, 1, 2  # 0-indexed slot indices

    for ic in range(n_icycles):
        dt = float(dt_vals[ic])
        tend_n = tend_vals[ic].astype(np.float64)

        # AB coefficients: Euler -> AB2 -> AB3
        if ic == 0:
            at, bt, ct = 1.0, 0.0, 0.0
        elif ic == 1:
            at, bt, ct = 1.5, -0.5, 0.0
        else:
            at, bt, ct = 23.0/12.0, -16.0/12.0, 5.0/12.0

        slots[:, na] = tend_n
        phi += dt * (at * slots[:, na] + bt * slots[:, nb] + ct * slots[:, nc])

        # gSAM rotation: nn=na; na=nc; nc=nb; nb=nn
        na, nb, nc = nc, na, nb

    return (phi.astype(np.float32),
            slots[:, na].astype(np.float32),
            slots[:, nb].astype(np.float32),
            slots[:, nc].astype(np.float32))


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
    phi_final, tend_na, tend_nb, tend_nc = _jsam_icycle(phi, tend_vals, dt_vals, n_icycles)

    # Match Fortran output layout: phi_final + tend(na) + tend(nb) + tend(nc)
    out = np.concatenate([phi_final, tend_na, tend_nb, tend_nc])
    write_bin("jsam_out.bin", out)

    print(f"[icycle_dt] case={case}  n={n}  n_icycles={n_icycles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
