"""Dump inputs and jsam outputs for test_fct_antidiff.

Tests the FCT flux limiter in isolation on 1D periodic advection.

Cases
-----
fct_uniform        — uniform field, no limiting needed
fct_step_function  — sharp step, limiter must prevent oscillation
fct_cosine_bell    — smooth profile, limiter should be inactive
fct_near_zero      — near-zero field, tests positivity interaction
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
    """Build 1D periodic advection test case."""
    n = 64
    dx = np.float32(1000.0)   # 1 km cells
    dt = np.float32(5.0)      # CFL ~0.5

    if case == "fct_uniform":
        phi = np.ones(n, dtype=np.float32)
        u = np.full(n + 1, 100.0, dtype=np.float32)
    elif case == "fct_step_function":
        phi = np.zeros(n, dtype=np.float32)
        phi[n//4:3*n//4] = 1.0   # sharp step
        u = np.full(n + 1, 100.0, dtype=np.float32)
    elif case == "fct_cosine_bell":
        x = np.linspace(0, 2*np.pi, n, endpoint=False, dtype=np.float32)
        phi = 0.5 * (1.0 + np.cos(x - np.pi)).astype(np.float32)
        u = np.full(n + 1, 100.0, dtype=np.float32)
    elif case == "fct_near_zero":
        phi = np.full(n, 1e-8, dtype=np.float32)
        phi[n//2] = 1e-3
        u = np.full(n + 1, 100.0, dtype=np.float32)
    else:
        raise ValueError(f"unknown case: {case}")

    return n, phi, u, dt, dx


def _fct_1d(phi, u, dt, dx):
    """1D FCT with upwind + centered high-order (matching driver.f90)."""
    n = len(phi)
    cr = u * dt / dx

    # Low-order (upwind) flux
    f_low = np.zeros(n + 1, dtype=np.float32)
    f_high = np.zeros(n + 1, dtype=np.float32)
    for i in range(n + 1):
        im = (i - 1) % n
        ip = (i + 1) % n
        if u[i] >= 0:
            f_low[i] = u[i] * phi[im]
        else:
            f_low[i] = u[i] * phi[ip]
        f_high[i] = u[i] * 0.5 * (phi[im] + phi[ip])

    f_anti = f_high - f_low

    # Upwind update
    phi_up = phi - dt / dx * (f_low[1:] - f_low[:-1])

    # FCT limiting
    eps = 1e-30
    p_plus = np.zeros(n, dtype=np.float64)
    p_minus = np.zeros(n, dtype=np.float64)
    q_plus = np.zeros(n, dtype=np.float64)
    q_minus = np.zeros(n, dtype=np.float64)

    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        p_plus[i] = max(0, f_anti[i]) - min(0, f_anti[i + 1])
        p_minus[i] = max(0, f_anti[i + 1]) - min(0, f_anti[i])
        local_max = max(phi[im], phi[i], phi[ip])
        local_min = min(phi[im], phi[i], phi[ip])
        q_plus[i] = (local_max - phi_up[i]) * dx / dt
        q_minus[i] = (phi_up[i] - local_min) * dx / dt

    r_plus = np.where(p_plus > eps, np.minimum(1.0, q_plus / (p_plus + eps)), 1.0)
    r_minus = np.where(p_minus > eps, np.minimum(1.0, q_minus / (p_minus + eps)), 1.0)

    scale = np.zeros(n + 1, dtype=np.float64)
    for i in range(n + 1):
        im = (i - 1) % n
        ip = (i + 1) % n
        if f_anti[i] >= 0:
            scale[i] = min(r_plus[ip], r_minus[im])
        else:
            scale[i] = min(r_plus[im], r_minus[ip])

    phi_new = phi_up - dt / dx * (scale[1:] * f_anti[1:] - scale[:-1] * f_anti[:-1])
    return phi_new.astype(np.float32)


def main() -> int:
    case = sys.argv[1]
    n, phi, u, dt, dx = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("i", n))
        f.write(phi.astype(np.float32).tobytes())
        f.write(u.astype(np.float32).tobytes())
        f.write(struct.pack("ff", float(dt), float(dx)))

    # Python FCT (matching the Fortran driver)
    phi_new = _fct_1d(phi, u, dt, dx)
    write_bin("jsam_out.bin", phi_new)

    print(f"[fct_antidiff] case={case}  n={n}  "
          f"phi range=[{phi_new.min():.6e}, {phi_new.max():.6e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
