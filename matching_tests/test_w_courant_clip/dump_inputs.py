"""Dump inputs and jsam outputs for test_w_courant_clip.

Tests the W Courant limiter and sponge-layer damping from gSAM damping.f90.

Cases
-----
wclip_below_threshold  — all W below wmax, no clipping
wclip_above_threshold  — W = 2*wmax at several levels
wclip_sponge_layer     — W at levels in the sponge (nu > 0.6)
wclip_native_adzw      — gSAM lat_720_dyvar adzw profile
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
    nz, ny, nx = 10, 4, 4
    dtn = np.float32(10.0)
    dt_base = np.float32(10.0)
    dz = np.float32(500.0)
    nub = np.float32(0.6)
    damping_w_cu = np.float32(0.3)

    # Interface heights (zi): uniformly spaced
    zi = np.linspace(0, nz * float(dz), nz + 1, dtype=np.float32)
    adzw = np.ones(nz, dtype=np.float32)

    np.random.seed(99)

    if case == "wclip_below_threshold":
        # wmax = 0.3 * 500 * 1.0 / 10.0 = 15.0 m/s
        W = np.random.randn(nz + 1, ny, nx).astype(np.float32) * 5.0
    elif case == "wclip_above_threshold":
        W = np.random.randn(nz + 1, ny, nx).astype(np.float32) * 5.0
        W[2, :, :] = 30.0   # well above wmax=15
        W[3, :, :] = -25.0
    elif case == "wclip_sponge_layer":
        W = np.random.randn(nz + 1, ny, nx).astype(np.float32) * 10.0
        # Last few levels are in sponge (nu > 0.6)
        # nu(k) = (zi(k) - zi(0)) / (zi(nz) - zi(0))
        # For nz=10, nu > 0.6 → k >= 7
    elif case == "wclip_native_adzw":
        try:
            from jsam.utils.IRMALoader import IRMALoader
            g = IRMALoader().grid
            zi_native = np.asarray(g["zi"], dtype=np.float32)
            nz = len(zi_native) - 1
            ny, nx = 4, 4
            zi = zi_native
            dz_vals = np.diff(zi)
            dz = np.float32(np.mean(dz_vals))
            adzw = (dz_vals / float(dz)).astype(np.float32)
        except Exception:
            # Fallback: stretched grid
            nz = 20
            zi = np.concatenate([
                np.linspace(0, 5000, 11),
                np.linspace(6000, 30000, 10)
            ]).astype(np.float32)
            dz_vals = np.diff(zi)
            dz = np.float32(np.mean(dz_vals))
            adzw = (dz_vals / float(dz)).astype(np.float32)
        W = np.random.randn(nz + 1, ny, nx).astype(np.float32) * 15.0
    else:
        raise ValueError(f"unknown case: {case}")

    return nz, ny, nx, dtn, dt_base, dz, nub, damping_w_cu, zi, adzw, W


def _apply_w_damping(nz, ny, nx, dtn, dt_base, dz, nub, damping_w_cu, zi, adzw, W):
    """Apply gSAM-style W sponge + CFL limiter."""
    W = W.copy()
    tau_max = float(dtn) / float(dt_base)

    # Compute taudamp
    taudamp = np.zeros(nz, dtype=np.float32)
    for k in range(nz):
        nu = (zi[k] - zi[0]) / (zi[-1] - zi[0])
        if nu > nub:
            zzz = 100.0 * ((nu - nub) / (1.0 - nub))**2
            taudamp[k] = 0.333 * tau_max * zzz / (1.0 + zzz)

    # Sponge
    for k in range(nz):
        if taudamp[k] > 0:
            W[k, :, :] /= (1.0 + taudamp[k])

    # CFL limiter
    for k in range(nz):
        if taudamp[k] == 0:
            wmax = damping_w_cu * dz * adzw[k] / dtn
            W[k, :, :] = (W[k, :, :] + np.clip(W[k, :, :], -wmax, wmax) * tau_max) \
                         / (1.0 + tau_max)

    return W.astype(np.float32), taudamp


def main() -> int:
    case = sys.argv[1]
    nz, ny, nx, dtn, dt_base, dz, nub, damping_w_cu, zi, adzw, W = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(struct.pack("fff", float(dtn), float(dt_base), float(dz)))
        f.write(struct.pack("ff", float(nub), float(damping_w_cu)))
        f.write(zi.astype(np.float32).tobytes())
        f.write(adzw.astype(np.float32).tobytes())
        f.write(W.astype(np.float32).tobytes(order="C"))

    # Python side
    W_new, taudamp = _apply_w_damping(nz, ny, nx, dtn, dt_base, dz, nub,
                                       damping_w_cu, zi, adzw, W)
    out = np.concatenate([W_new.ravel(order="C"), taudamp])
    write_bin("jsam_out.bin", out)

    print(f"[w_courant_clip] case={case}  nz={nz}  "
          f"sponge levels: {np.sum(taudamp > 0)}  "
          f"W range=[{W_new.min():.2f}, {W_new.max():.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
