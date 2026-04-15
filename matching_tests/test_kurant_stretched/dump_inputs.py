"""test_kurant_stretched — kurant CFL on a non-uniform vertical grid.

Builds a tiny (nz=4, ny=8, nx=16) staggered W with a non-uniform
profile; instantiates a LatLonGrid with stretched zi; calls jsam
compute_cfl and the matching gSAM kurant.f90 snippet (via driver.f90)
and asserts agreement at 4 decimals.
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

from common.bin_io import write_bin                          # noqa: E402
from jsam.core.grid.latlon import LatLonGrid                 # noqa: E402
from jsam.core.dynamics.pressure import build_metric         # noqa: E402
from jsam.core.dynamics.kurant import compute_cfl            # noqa: E402


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = 4, 8, 16
    zi  = np.array([0.0, 100.0, 250.0, 600.0, 1300.0], dtype=np.float64)  # non-uniform
    z   = 0.5 * (zi[:-1] + zi[1:])
    rho = np.full(nz, 1.0)

    # 8-row tropical band (uniform 0.5° → ady ≈ 1.0 everywhere; OK because
    # the test is targeting the vertical stretch).  cos_lat varies slightly.
    lat = np.linspace(-2.0, 2.0, ny)
    lon = np.linspace(0.0, 7.5, nx)

    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid)

    # Construct W with a non-uniform profile (max at k=2) so the
    # stretched-grid factor matters.
    W = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    W[1, :, :] = 0.5
    W[2, :, :] = 1.0
    W[3, :, :] = 0.7
    W[4, :, :] = 0.2
    U = np.full((nz, ny, nx + 1), 5.0, dtype=np.float32)
    V = np.zeros((nz, ny + 1, nx), dtype=np.float32)

    dt = 30.0
    cfl_jsam = float(compute_cfl(jnp.asarray(U), jnp.asarray(V), jnp.asarray(W), metric, dt=dt))
    write_bin(workdir / "jsam_out.bin", np.array([cfl_jsam], dtype=np.float32))

    # Mass-centre staggered velocities to match driver.f90 expectations
    U_abs = np.maximum(np.abs(U[:, :, :-1]), np.abs(U[:, :, 1:])).astype(np.float32)
    V_abs = np.maximum(np.abs(V[:, :-1, :]), np.abs(V[:, 1:, :])).astype(np.float32)
    W_abs = np.maximum(np.abs(W[:-1, :, :]), np.abs(W[1:, :, :])).astype(np.float32)

    dx_ref  = float(metric["dx_lon"])
    dy_ref  = float(metric["dy_lat_ref"])
    dz_ref  = float(metric["dz_ref"])
    cos_lat = np.asarray(metric["cos_lat"], dtype=np.float32).reshape(-1)
    ady     = np.asarray(metric["ady"], dtype=np.float32).reshape(-1)
    adzw    = np.asarray(metric["adzw"], dtype=np.float32).reshape(-1)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(U_abs.tobytes(order="C"))
        f.write(V_abs.tobytes(order="C"))
        f.write(W_abs.tobytes(order="C"))
        f.write(struct.pack("fff", dx_ref, dy_ref, dz_ref))
        f.write(cos_lat.tobytes())
        f.write(ady.tobytes())
        f.write(adzw.tobytes())
        f.write(struct.pack("f", dt))

    return 0


if __name__ == "__main__":
    sys.exit(main())
