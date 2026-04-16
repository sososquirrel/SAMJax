"""Dump inputs and jsam outputs for test_adamsb_p_buffer.

Tests pressure buffer rotation and adamsB lagged PGF correction.

Cases
-----
pbuf_step1_noop     — p_prev = p_pprev = 0 → adamsB is identity
pbuf_linear_p       — p = a*i + b*j + c*k, constant gradient
pbuf_rotation_step2 — simulated step-2 with real p_prev from step 1
pbuf_gradient_metric — non-uniform metric (rdy, rdz vary)
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
    nz, ny, nx = 4, 6, 8
    dt = np.float32(10.0)
    rdx = np.float32(1.0 / 25000.0)

    # Non-uniform rdy, rdz
    rdy = np.full(ny, 1.0 / 25000.0, dtype=np.float32)
    rdz = np.full(nz, 1.0 / 500.0, dtype=np.float32)

    np.random.seed(77)
    U = np.random.randn(nz, ny, nx + 1).astype(np.float32) * 5.0
    V = np.random.randn(nz, ny + 1, nx).astype(np.float32) * 3.0
    W = np.random.randn(nz + 1, ny, nx).astype(np.float32) * 0.5
    # Enforce BCs
    V[:, 0, :] = 0; V[:, -1, :] = 0
    W[0, :, :] = 0; W[-1, :, :] = 0

    if case == "pbuf_step1_noop":
        bt, ct = np.float32(0.0), np.float32(0.0)
        p_prev = np.zeros((nz, ny, nx), dtype=np.float32)
        p_pprev = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "pbuf_linear_p":
        bt, ct = np.float32(-0.5), np.float32(0.0)
        # p = 100*i + 50*j + 200*k
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        p_prev = (100.0 * ii + 50.0 * jj + 200.0 * kk).astype(np.float32).transpose(2, 1, 0)
        p_pprev = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "pbuf_rotation_step2":
        bt, ct = np.float32(-0.5), np.float32(0.0)
        p_prev = np.random.randn(nz, ny, nx).astype(np.float32) * 100.0
        p_pprev = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "pbuf_gradient_metric":
        bt, ct = np.float32(-16.0/12.0), np.float32(5.0/12.0)
        p_prev = np.random.randn(nz, ny, nx).astype(np.float32) * 100.0
        p_pprev = np.random.randn(nz, ny, nx).astype(np.float32) * 80.0
        # Non-uniform rdy (simulate ady variation)
        lat_deg = np.linspace(-60, 60, ny)
        rdy = (1.0 / (25000.0 * (0.3 + 0.7 * np.cos(np.deg2rad(lat_deg))))).astype(np.float32)
        rdz = (1.0 / np.linspace(100, 2000, nz)).astype(np.float32)
    else:
        raise ValueError(f"unknown case: {case}")

    return nz, ny, nx, dt, bt, ct, rdx, rdy, rdz, U, V, W, p_prev, p_pprev


def _jsam_adamsb(nz, ny, nx, dt, bt, ct, rdx, rdy, rdz, U, V, W, p_prev, p_pprev):
    """Apply adamsB correction jsam-style."""
    U = U.copy()
    V = V.copy()
    W = W.copy()

    # U correction (periodic x)
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):
                dpx = bt * (p_prev[k,j,i] - p_prev[k,j,i-1]) + \
                      ct * (p_pprev[k,j,i] - p_pprev[k,j,i-1])
                U[k,j,i] -= dt * dpx * rdx
            # Periodic: i=0 wraps
            dpx = bt * (p_prev[k,j,0] - p_prev[k,j,nx-1]) + \
                  ct * (p_pprev[k,j,0] - p_pprev[k,j,nx-1])
            U[k,j,0] -= dt * dpx * rdx
            U[k,j,nx] = U[k,j,0]

    # V correction (skip pole walls j=0, j=ny)
    for k in range(nz):
        for j in range(1, ny):
            for i in range(nx):
                dpy = bt * (p_prev[k,j,i] - p_prev[k,j-1,i]) + \
                      ct * (p_pprev[k,j,i] - p_pprev[k,j-1,i])
                V[k,j,i] -= dt * dpy * rdy[j]

    # W correction (skip ground k=0, lid k=nz)
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx):
                dpz = bt * (p_prev[k,j,i] - p_prev[k-1,j,i]) + \
                      ct * (p_pprev[k,j,i] - p_pprev[k-1,j,i])
                W[k,j,i] -= dt * dpz * rdz[k]

    return U.astype(np.float32), V.astype(np.float32), W.astype(np.float32)


def main() -> int:
    case = sys.argv[1]
    nz, ny, nx, dt, bt, ct, rdx, rdy, rdz, U, V, W, p_prev, p_pprev = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(struct.pack("ffff", float(dt), float(bt), float(ct), float(rdx)))
        f.write(rdy.astype(np.float32).tobytes())
        f.write(rdz.astype(np.float32).tobytes())
        f.write(U.astype(np.float32).tobytes(order="C"))
        f.write(V.astype(np.float32).tobytes(order="C"))
        f.write(W.astype(np.float32).tobytes(order="C"))
        f.write(p_prev.astype(np.float32).tobytes(order="C"))
        f.write(p_pprev.astype(np.float32).tobytes(order="C"))

    # jsam side
    U_new, V_new, W_new = _jsam_adamsb(nz, ny, nx, dt, bt, ct, rdx, rdy, rdz,
                                         U, V, W, p_prev, p_pprev)
    out = np.concatenate([
        U_new.ravel(order="C"),
        V_new.ravel(order="C"),
        W_new.ravel(order="C"),
    ])
    write_bin("jsam_out.bin", out)

    print(f"[adamsb_pbuf] case={case}  nz={nz}  ny={ny}  nx={nx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
