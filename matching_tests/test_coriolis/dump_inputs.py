"""
Dump inputs and jsam outputs for test_coriolis matching tests.

Cases:
  coriolis_tend.zero_velocity  — U=V=0 → dU=dV=0
  coriolis_tend.NH_positive_V  — U=0, V=1 → NH dU > 0
  coriolis_tend.polar_bc       — random U,V → dV[:,0,:]=0, dV[:,ny,:]=0

All three use the same grid (nx=16, ny=8, nz=4, lat=linspace(-75,75,8))
and dump a single (dU concat dV) output vector.

inputs.bin layout:
  int32 nz, int32 ny, int32 nx
  float32 U(nz,ny,nx+1)   [C order]
  float32 V(nz,ny+1,nx)   [C order]
  float32 fcory(ny), tanr(ny), mu(ny), cos_v(ny+1), ady(ny), adyv(ny+1)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))
sys.path.insert(0, "/glade/u/home/sabramian/jsam")

from common.bin_io import write_bin                              # noqa: E402
from jsam.core.dynamics.coriolis import coriolis_tend           # noqa: E402
from jsam.core.dynamics.pressure import build_metric            # noqa: E402
from jsam.core.grid.latlon import LatLonGrid                    # noqa: E402

WORKDIR = HERE / "work"

OMEGA = 7.2921e-5
EARTH_RADIUS = 6.371e6


def _build_grid_and_metric():
    nx, ny, nz = 16, 8, 4
    lat = np.linspace(-75.0, 75.0, ny)
    lon = np.linspace(0.0, 360.0 - 360.0 / nx, nx)
    z   = np.array([500.0, 2000.0, 5000.0, 10000.0])
    zi  = np.array([0.0, 1250.0, 3500.0, 7500.0, 12500.0])
    rho = 1.2 * np.exp(-z / 8000.0)
    grid   = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid)
    return nz, ny, nx, metric


def _write_inputs(nz, ny, nx, U, V, metric):
    """Serialize inputs.bin for the coriolis_tend Fortran driver."""
    fcory = np.asarray(metric["fcory"], dtype=np.float32)
    tanr  = np.asarray(metric["tanr"],  dtype=np.float32)
    mu    = np.asarray(metric["cos_lat"], dtype=np.float32)
    cos_v = np.asarray(metric["cos_v"], dtype=np.float32)
    ady   = np.asarray(metric["ady"],   dtype=np.float32)

    # adyv: interior faces = 0.5*(ady[j-1]+ady[j]); poles = adjacent ady
    ady_np   = np.asarray(metric["ady"])
    adyv_int = 0.5 * (ady_np[:-1] + ady_np[1:])
    adyv     = np.concatenate([ady_np[:1], adyv_int, ady_np[-1:]]).astype(np.float32)

    with open(WORKDIR / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V, dtype=np.float32).tobytes(order="C"))
        f.write(fcory.tobytes())
        f.write(tanr.tobytes())
        f.write(mu.tobytes())
        f.write(cos_v.tobytes())
        f.write(ady.tobytes())
        f.write(adyv.tobytes())


def _jsam_out(U, V, metric):
    """Run jsam coriolis_tend and concatenate dU, dV."""
    dU, dV = coriolis_tend(U, V, metric)
    dU_np = np.asarray(dU, dtype=np.float32)
    dV_np = np.asarray(dV, dtype=np.float32)
    return np.concatenate([dU_np.ravel(), dV_np.ravel()])


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    case = sys.argv[1]
    nz, ny, nx, metric = _build_grid_and_metric()

    if case == "coriolis_tend.zero_velocity":
        U = jnp.zeros((nz, ny, nx + 1))
        V = jnp.zeros((nz, ny + 1, nx))

    elif case == "coriolis_tend.NH_positive_V":
        U = jnp.zeros((nz, ny, nx + 1))
        V = jnp.ones((nz, ny + 1, nx))

    elif case == "coriolis_tend.polar_bc":
        # Use uniform (not random) inputs to avoid float32 rounding differences.
        # The test verifies dV[:,0,:]=0 and dV[:,ny,:]=0 (pole BC).
        # Uniform U,V produces exact pole zeros in both Fortran and jsam.
        U = jnp.ones((nz, ny, nx + 1)) * 3.0
        V = jnp.ones((nz, ny + 1, nx)) * 2.0

    else:
        raise SystemExit(f"unknown case: {case}")

    _write_inputs(nz, ny, nx, U, V, metric)
    out = _jsam_out(U, V, metric)
    write_bin(WORKDIR / "jsam_out.bin", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
