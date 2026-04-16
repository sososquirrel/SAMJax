"""Dump inputs and jsam outputs for test_velocity_stagger.

Cases
-----
stagger_periodic_u  — U field with known values, verify periodic wrap
stagger_pole_v      — V field, verify pole wall BCs (V=0 at j=0, j=ny)
stagger_ground_w    — W field, verify rigid-lid BCs (W=0 at k=0, k=nz)
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
    """Build small synthetic mass-grid velocity fields."""
    nz, ny, nx = 4, 6, 8
    np.random.seed(42)

    if case == "stagger_periodic_u":
        U_mass = np.random.randn(nz, ny, nx).astype(np.float32) * 10.0
        V_mass = np.zeros((nz, ny, nx), dtype=np.float32)
        W_mass = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "stagger_pole_v":
        U_mass = np.zeros((nz, ny, nx), dtype=np.float32)
        V_mass = np.random.randn(nz, ny, nx).astype(np.float32) * 5.0
        W_mass = np.zeros((nz, ny, nx), dtype=np.float32)
    elif case == "stagger_ground_w":
        U_mass = np.zeros((nz, ny, nx), dtype=np.float32)
        V_mass = np.zeros((nz, ny, nx), dtype=np.float32)
        W_mass = np.random.randn(nz, ny, nx).astype(np.float32) * 2.0
    else:
        raise ValueError(f"unknown case: {case}")

    return nz, ny, nx, U_mass, V_mass, W_mass


def _jsam_stagger(nz, ny, nx, U_mass, V_mass, W_mass):
    """Apply jsam-style staggering (from ModelState.from_gsam_nc pattern)."""
    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp

    # This is what jsam does in state.py:114-116
    U_stag = jnp.zeros((nz, ny, nx + 1)).at[:, :, :-1].set(jnp.asarray(U_mass))
    V_stag = jnp.zeros((nz, ny + 1, nx)).at[:, :-1, :].set(jnp.asarray(V_mass))
    W_stag = jnp.zeros((nz + 1, ny, nx)).at[:-1, :, :].set(jnp.asarray(W_mass))

    return (np.asarray(U_stag, dtype=np.float32),
            np.asarray(V_stag, dtype=np.float32),
            np.asarray(W_stag, dtype=np.float32))


def main() -> int:
    case = sys.argv[1]

    nz, ny, nx, U_mass, V_mass, W_mass = _build_state(case)

    # Write inputs.bin
    with open("inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(U_mass.astype(np.float32).tobytes(order="C"))
        f.write(V_mass.astype(np.float32).tobytes(order="C"))
        f.write(W_mass.astype(np.float32).tobytes(order="C"))

    # jsam side
    U_stag, V_stag, W_stag = _jsam_stagger(nz, ny, nx, U_mass, V_mass, W_mass)

    # Concatenate for comparison
    out = np.concatenate([
        U_stag.ravel(order="C"),
        V_stag.ravel(order="C"),
        W_stag.ravel(order="C"),
    ])
    write_bin("jsam_out.bin", out)

    n_u = nz * ny * (nx + 1)
    n_v = nz * (ny + 1) * nx
    n_w = (nz + 1) * ny * nx
    print(f"[velocity_stagger] case={case}  nz={nz}  ny={ny}  nx={nx}  "
          f"n_out={n_u + n_v + n_w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
