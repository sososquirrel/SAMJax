"""Dump inputs and jsam outputs for test_operators matching tests.

Called from run.sh:
    python dump_inputs.py divergence_zero_v_const
    python dump_inputs.py divergence_linear_u
    python dump_inputs.py laplacian_const

Each invocation writes:
    work/inputs.bin    — fields + metric for the Fortran driver
    work/jsam_out.bin  — jsam-side answer
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin  # noqa: E402
from jsam.core.grid.latlon import LatLonGrid  # noqa: E402


def _make_grid() -> LatLonGrid:
    """Tiny grid matching the conftest tiny_grid fixture."""
    nz, ny, nx = 4, 8, 16
    lat  = np.linspace(-87.5, 87.5, ny)
    lon  = np.linspace(0.0, 337.5, nx)
    z    = np.array([100., 500., 1500., 4000.])
    zi   = np.array([0., 250., 1000., 2750., 5000.])
    rho  = np.array([1.2, 1.0, 0.8, 0.6])
    return LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)


def _write_inputs(workdir: Path, U, V, phi, dx: np.ndarray, dy: np.ndarray):
    """Write packed inputs.bin for the Fortran driver.

    Layout:
        int32  nz, ny, nx
        float32 U(nz, ny, nx+1)  C-order
        float32 V(nz, ny+1, nx)  C-order
        float32 phi(nz, ny, nx)  C-order
        float32 dx(ny)
        float32 dy(ny)
    """
    U   = np.asarray(U,   dtype=np.float32)
    V   = np.asarray(V,   dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    dx  = np.asarray(dx,  dtype=np.float32)
    dy  = np.asarray(dy,  dtype=np.float32)
    nz, ny, nx_plus1 = U.shape
    nx = nx_plus1 - 1
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(U.tobytes(order="C"))
        f.write(V.tobytes(order="C"))
        f.write(phi.tobytes(order="C"))
        f.write(dx.tobytes())
        f.write(dy.tobytes())


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]
    g = _make_grid()
    nz, ny, nx = g.nz, g.ny, g.nx

    dx = np.asarray(g.dx, dtype=np.float32)   # (ny,)
    dy = np.asarray(g.dy, dtype=np.float32)   # (ny,)

    if mode == "divergence_zero_v_const":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.ones((nz, ny + 1, nx))
        phi = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, phi, dx, dy)
        div = g.divergence(U, V)
        write_bin(workdir / "jsam_out.bin",
                  np.asarray(div, dtype=np.float32).ravel())
        return 0

    if mode == "divergence_linear_u":
        U = jnp.broadcast_to(
            jnp.arange(nx + 1, dtype=float)[None, None, :],
            (nz, ny, nx + 1),
        )
        V   = jnp.zeros((nz, ny + 1, nx))
        phi = jnp.zeros((nz, ny, nx))
        _write_inputs(workdir, U, V, phi, dx, dy)
        div = g.divergence(U, V)
        write_bin(workdir / "jsam_out.bin",
                  np.asarray(div, dtype=np.float32).ravel())
        return 0

    if mode == "laplacian_const":
        U   = jnp.zeros((nz, ny, nx + 1))
        V   = jnp.zeros((nz, ny + 1, nx))
        phi = jnp.ones((nz, ny, nx)) * 42.0
        _write_inputs(workdir, U, V, phi, dx, dy)
        lap = g.laplacian(phi)
        write_bin(workdir / "jsam_out.bin",
                  np.asarray(lap, dtype=np.float32).ravel())
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
