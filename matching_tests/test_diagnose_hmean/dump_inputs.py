"""Dump inputs and jsam outputs for test_diagnose_hmean.

Tests the cos-lat + ady weighted horizontal mean used in the diagnose
block (step.py:946-979). KNOWN BUG: jsam uses cos_lat only, missing
the ady stretching factor that gSAM includes.

Cases
-----
hmean_uniform       — uniform field, both methods agree
hmean_lat_gradient  — TABS increases poleward, ady weighting matters
hmean_polar_spike   — spike at poles, ady prevents over-weighting
hmean_qv_qn_qp     — test all 4 quantities: tabs0, qv0, qn0, qp0
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
    """Build synthetic fields with varying latitude structure."""
    nz, ny, nx = 4, 12, 8

    # Simulate lat_720_dyvar-like grid
    lat_deg = np.linspace(-85, 85, ny, dtype=np.float64)
    lat_rad = np.deg2rad(lat_deg)
    mu = np.cos(lat_rad).astype(np.float32)  # cos(lat)
    # ady: smaller at poles, larger at equator (dyvar grid)
    ady = (0.2 + 0.8 * mu).astype(np.float32)

    TABS = np.zeros((nz, ny, nx), dtype=np.float32)
    QV   = np.zeros((nz, ny, nx), dtype=np.float32)
    QC   = np.zeros((nz, ny, nx), dtype=np.float32)
    QI   = np.zeros((nz, ny, nx), dtype=np.float32)
    QR   = np.zeros((nz, ny, nx), dtype=np.float32)
    QS   = np.zeros((nz, ny, nx), dtype=np.float32)
    QG   = np.zeros((nz, ny, nx), dtype=np.float32)

    if case == "hmean_uniform":
        TABS[:] = 280.0
        QV[:] = 10e-3
    elif case == "hmean_lat_gradient":
        # TABS increases poleward: warm poles, cool equator
        for j in range(ny):
            TABS[:, j, :] = 250.0 + 30.0 * (1.0 - mu[j])
        QV[:] = 10e-3
    elif case == "hmean_polar_spike":
        TABS[:] = 200.0
        # Spike at polar rows
        TABS[:, 0, :] = 300.0
        TABS[:, 1, :] = 300.0
        TABS[:, -2, :] = 300.0
        TABS[:, -1, :] = 300.0
        QV[:] = 5e-3
    elif case == "hmean_qv_qn_qp":
        TABS[:] = 270.0
        QV[:] = 8e-3
        QC[:] = 0.5e-3
        QI[:] = 0.2e-3
        QR[:] = 0.1e-3
        QS[:] = 0.05e-3
        QG[:] = 0.02e-3
    else:
        raise ValueError(f"unknown case: {case}")

    return nz, ny, nx, mu, ady, TABS, QV, QC, QI, QR, QS, QG


def _jsam_hmean(nz, ny, nx, mu, ady, TABS, QV, QC, QI, QR, QS, QG):
    """Compute hmean using jsam's current (buggy) approach: cos_lat only."""
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    cos_lat = jnp.asarray(mu)
    # This is jsam's current code (step.py:955):
    _wgt = cos_lat / jnp.sum(cos_lat)

    def _hmean(field):
        return jnp.sum(jnp.mean(field, axis=2) * _wgt[None, :], axis=1)

    tabs_j = jnp.asarray(TABS)
    qv_j = jnp.asarray(QV)
    qc_j = jnp.asarray(QC)
    qi_j = jnp.asarray(QI)
    qr_j = jnp.asarray(QR)
    qs_j = jnp.asarray(QS)
    qg_j = jnp.asarray(QG)

    tabs0 = _hmean(tabs_j)
    q0    = _hmean(qv_j + qc_j + qi_j)
    qn0   = _hmean(qc_j + qi_j)
    qp0   = _hmean(qr_j + qs_j + qg_j)
    qv0   = q0 - qn0

    out = np.concatenate([
        np.asarray(tabs0, dtype=np.float32),
        np.asarray(qv0,   dtype=np.float32),
        np.asarray(qn0,   dtype=np.float32),
        np.asarray(qp0,   dtype=np.float32),
    ])
    return out


def main() -> int:
    case = sys.argv[1]
    nz, ny, nx, mu, ady, TABS, QV, QC, QI, QR, QS, QG = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(mu.astype(np.float32).tobytes())
        f.write(ady.astype(np.float32).tobytes())
        for a in (TABS, QV, QC, QI, QR, QS, QG):
            f.write(a.astype(np.float32).tobytes(order="C"))

    # jsam side (with current buggy cos_lat-only weights)
    out = _jsam_hmean(nz, ny, nx, mu, ady, TABS, QV, QC, QI, QR, QS, QG)
    write_bin("jsam_out.bin", out)

    print(f"[diagnose_hmean] case={case}  nz={nz}  ny={ny}  nx={nx}")
    if case != "hmean_uniform":
        # Show the difference between cos-only and cos*ady weighting
        wgt_cos = mu / mu.sum()
        wgt_full = (mu * ady) / (mu * ady).sum()
        tabs_zmean = TABS.mean(axis=2)  # (nz, ny)
        for k in range(min(2, nz)):
            t_cos = np.sum(tabs_zmean[k] * wgt_cos)
            t_full = np.sum(tabs_zmean[k] * wgt_full)
            print(f"  k={k}: tabs0_cos={t_cos:.4f}  tabs0_cos*ady={t_full:.4f}"
                  f"  diff={t_cos - t_full:.4f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
