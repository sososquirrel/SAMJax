"""
Dump inputs and jsam outputs for test_polar_filter matching tests.

Strategy: Fortran implements the same Fourier filter as jsam (DFT-based),
NOT the gSAM box smoother. This tests self-consistency of the filter math.
See TODO.md for the mismatch note.

Cases:
  polar_fourier_filter.equator_unchanged — lat=0, all modes pass → output=input
  polar_fourier_filter.high_mode_zeroed  — lat=75°, mode m=6 (nx=16) → ≈0
  polar_fourier_filter.low_mode_preserved — lat=75°, mode m=1 → ≈input

inputs.bin:
  int32 nz, int32 ny, int32 nx
  float32 lat_rad(ny)
  float32 field(nz, ny, nx)   [C order]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin                                   # noqa: E402
from jsam.core.dynamics.polar_filter import polar_fourier_filter      # noqa: E402

WORKDIR = HERE / "work"


def _write_inputs(nz, ny, nx, lat_rad, field):
    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(lat_rad, dtype=np.float32).tobytes())
        f.write(np.asarray(field, dtype=np.float32).tobytes(order="C"))


def _jsam_filter(field, lat_rad, nx):
    out = polar_fourier_filter(
        jnp.array(field, dtype=jnp.float32),
        jnp.array(lat_rad, dtype=jnp.float32),
        nx,
    )
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    case = sys.argv[1]

    nx = 16

    if case == "polar_fourier_filter.equator_unchanged":
        # lat=0: m_max = floor(nx/2 * 1) = nx/2 → all modes pass → output=input
        nz, ny = 2, 1
        lat_rad = np.array([0.0], dtype=np.float32)
        # Random field with all harmonics excited
        rng = np.random.default_rng(42)
        field = rng.standard_normal((nz, ny, nx)).astype(np.float32)
        _write_inputs(nz, ny, nx, lat_rad, field)
        out = _jsam_filter(field, lat_rad, nx)
        write_bin(WORKDIR / "jsam_out.bin", out)

    elif case == "polar_fourier_filter.high_mode_zeroed":
        # lat=75°: cos(75°)≈0.259 → m_max = floor(8 * 0.259) = 2
        # Pure mode m=6 (> m_max=2) → should be zeroed
        nz, ny = 2, 1
        lat_rad = np.array([np.deg2rad(75.0)], dtype=np.float32)
        x = np.arange(nx) / nx
        row = np.cos(2 * np.pi * 6 * x).astype(np.float32)
        field = np.tile(row, (nz, ny, 1))
        _write_inputs(nz, ny, nx, lat_rad, field)
        out = _jsam_filter(field, lat_rad, nx)
        write_bin(WORKDIR / "jsam_out.bin", out)

    elif case == "polar_fourier_filter.low_mode_preserved":
        # lat=75°: m_max=2; mode m=1 (≤ m_max) → preserved
        nz, ny = 2, 1
        lat_rad = np.array([np.deg2rad(75.0)], dtype=np.float32)
        x = np.arange(nx) / nx
        row = np.cos(2 * np.pi * 1 * x).astype(np.float32)
        field = np.tile(row, (nz, ny, 1))
        _write_inputs(nz, ny, nx, lat_rad, field)
        out = _jsam_filter(field, lat_rad, nx)
        write_bin(WORKDIR / "jsam_out.bin", out)

    else:
        raise SystemExit(f"unknown case: {case}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
