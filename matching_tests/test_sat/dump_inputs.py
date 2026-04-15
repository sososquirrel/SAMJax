"""Dump inputs and jsam-side answers for the saturation kernel test.

Modes:
   esatw    — sweep T  ∈ [180, 320] K
   qsatw    — sweep (T,p) on a Cartesian grid AND >=50 explicit
              (T,p) points with es>p-es (low-pressure regime,
              T ∈ [280,310] K, p ∈ [50,200] hPa).  Documented in README:
              195 of 5656 points were failing before microphysics.py
              switched to gSAM's `max(es, p-es)` clamp.
   dtqsatw  — same (T,p) sweep
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin  # noqa: E402

# Use the new jsam exports (guarded by the microphysics fix).
from jsam.core.physics.microphysics import (  # noqa: E402
    _esatw,
    qsatw,
    _dtqsatw,
)
import jax.numpy as jnp  # noqa: E402


def _sweep_TP():
    """Cartesian sweep covering both warm/cold and high/low pressure.

    Total = 56 T × 101 p = 5656 points (matches README count).  The
    `low_p` block below adds the 50 explicit points exercising the
    `max(es, p-es)` clamp.
    """
    T_grid = np.linspace(200.0, 320.0, 56, dtype=np.float32)
    p_grid = np.linspace(50.0,  1000.0, 101, dtype=np.float32)
    TT, PP = np.meshgrid(T_grid, p_grid, indexing="ij")
    return TT.ravel(), PP.ravel()


def _low_p_clamp_points():
    """50 points where es > p-es (gSAM clamp matters)."""
    Ts = np.linspace(280.0, 310.0, 10, dtype=np.float32)
    ps = np.linspace(50.0,  200.0,  5, dtype=np.float32)
    TT, PP = np.meshgrid(Ts, ps, indexing="ij")
    return TT.ravel(), PP.ravel()


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]

    if mode == "esatw":
        T = np.linspace(180.0, 320.0, 281, dtype=np.float32)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", T.size))
            f.write(T.tobytes())
        out = np.asarray(_esatw(jnp.asarray(T)), dtype=np.float32)
        write_bin(workdir / "jsam_out.bin", out)
        return 0

    if mode in ("qsatw", "dtqsatw"):
        T_full, P_full = _sweep_TP()
        T_low, P_low = _low_p_clamp_points()
        T = np.concatenate([T_full, T_low]).astype(np.float32)
        P = np.concatenate([P_full, P_low]).astype(np.float32)
        with open(workdir / "inputs.bin", "wb") as f:
            f.write(struct.pack("i", T.size))
            f.write(T.tobytes())
            f.write(P.tobytes())
        if mode == "qsatw":
            out = np.asarray(qsatw(jnp.asarray(T), jnp.asarray(P)), dtype=np.float32)
        else:
            out = np.asarray(_dtqsatw(jnp.asarray(T), jnp.asarray(P)), dtype=np.float32)
        write_bin(workdir / "jsam_out.bin", out)
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
