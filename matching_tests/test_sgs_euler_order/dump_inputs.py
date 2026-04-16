"""Dump inputs and jsam outputs for test_sgs_euler_order.

Tests that SGS momentum diffusion is applied as a non-AB Euler increment
AFTER the AB3 advance, matching gSAM's adamsA.f90 behaviour.

Cases
-----
sgs_euler_zero         — dudtd=0, pure AB3 step
sgs_euler_uniform_shear — constant dudtd, verify it adds linearly
sgs_euler_post_advance  — AB3 step then +dt*dudtd (correct order)
sgs_euler_order_swap    — put dudtd IN the AB3 buffer (wrong order, should differ)
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
    n = 32
    dt = np.float32(10.0)
    # AB3 constant-dt coefficients
    at = np.float32(23.0 / 12.0)
    bt = np.float32(-16.0 / 12.0)
    ct = np.float32(5.0 / 12.0)

    np.random.seed(55)
    phi = np.random.randn(n).astype(np.float32) * 10.0
    tend_na = np.random.randn(n).astype(np.float32) * 0.5
    tend_nb = np.random.randn(n).astype(np.float32) * 0.5
    tend_nc = np.random.randn(n).astype(np.float32) * 0.5

    if case == "sgs_euler_zero":
        dudtd = np.zeros(n, dtype=np.float32)
        mode = 0
    elif case == "sgs_euler_uniform_shear":
        dudtd = np.full(n, 0.1, dtype=np.float32)
        mode = 0
    elif case == "sgs_euler_post_advance":
        dudtd = np.random.randn(n).astype(np.float32) * 0.3
        mode = 0
    elif case == "sgs_euler_order_swap":
        dudtd = np.random.randn(n).astype(np.float32) * 0.3
        mode = 1  # WRONG order: dudtd inside AB3 buffer
    else:
        raise ValueError(f"unknown case: {case}")

    return n, dt, at, bt, ct, phi, tend_na, tend_nb, tend_nc, dudtd, mode


def _jsam_sgs_step(phi, dt, at, bt, ct, tend_na, tend_nb, tend_nc, dudtd, mode):
    """Compute the SGS step jsam-style."""
    phi = phi.astype(np.float64).copy()
    if mode == 0:
        # Correct: AB3 step + Euler SGS (gSAM adamsA pattern)
        phi += dt * (at * tend_na + bt * tend_nb + ct * tend_nc + dudtd)
    else:
        # Wrong: dudtd pollutes the AB3 buffer
        phi += dt * (at * (tend_na + dudtd) + bt * tend_nb + ct * tend_nc)
    return phi.astype(np.float32)


def main() -> int:
    case = sys.argv[1]
    n, dt, at, bt, ct, phi, tend_na, tend_nb, tend_nc, dudtd, mode = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("i", n))
        f.write(struct.pack("f", float(dt)))
        f.write(struct.pack("fff", float(at), float(bt), float(ct)))
        f.write(phi.astype(np.float32).tobytes())
        f.write(tend_na.astype(np.float32).tobytes())
        f.write(tend_nb.astype(np.float32).tobytes())
        f.write(tend_nc.astype(np.float32).tobytes())
        f.write(dudtd.astype(np.float32).tobytes())
        f.write(struct.pack("i", mode))

    # jsam side
    phi_new = _jsam_sgs_step(phi, dt, at, bt, ct, tend_na, tend_nb, tend_nc, dudtd, mode)
    write_bin("jsam_out.bin", phi_new)

    print(f"[sgs_euler_order] case={case}  n={n}  mode={'correct' if mode==0 else 'wrong'}")
    if case == "sgs_euler_order_swap":
        # Show how much the wrong order differs
        phi_correct = _jsam_sgs_step(phi, dt, at, bt, ct, tend_na, tend_nb, tend_nc, dudtd, 0)
        diff = np.abs(phi_new - phi_correct)
        print(f"  max |correct - wrong| = {diff.max():.4e}")
        print(f"  The factor: wrong has at*dudtd, correct has 1*dudtd.")
        print(f"  Excess = dt*(at-1)*dudtd = {dt*(at-1)*np.abs(dudtd).max():.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
