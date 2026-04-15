"""Dump inputs and jsam outputs for test_pressure matching tests.

Called from run.sh:
    python dump_inputs.py press_rhs_zero_velocity
    python dump_inputs.py press_rhs_constant_U
    python dump_inputs.py press_gradient_zero_p
    python dump_inputs.py press_gradient_constant_p

All four cases exercise press_rhs.  The gradient cases use zero or constant
pressure which implies RHS=0, so we compare RHS directly (residual test).

inputs.bin layout:
    int32  nz, ny, nx
    float32 U(nz, ny, nx+1)
    float32 V(nz, ny+1, nx)
    float32 W(nz+1, ny, nx)
    float32 rho(nz)
    float32 rhow(nz+1)
    float32 dz(nz)
    float32 dx                scalar (dx_lon equatorial, m)
    float32 dy(ny)            per-row (m)
    float32 imu(ny)           1/cos(lat_j)
    float32 cos_v(ny+1)
    float32 cos_lat(ny)
    float32 dt
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
from jsam.core.dynamics.pressure import build_metric, press_rhs  # noqa: E402


def _make_grid():
    nz, ny, nx = 4, 8, 16
    lat  = np.linspace(-87.5, 87.5, ny)
    lon  = np.linspace(0.0, 337.5, nx)
    z    = np.array([100., 500., 1500., 4000.])
    zi   = np.array([0., 250., 1000., 2750., 5000.])
    rho  = np.array([1.2, 1.0, 0.8, 0.6])
    return LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)


def _write_inputs(workdir, U, V, W, metric, dt):
    nz, ny, nx_p1 = np.asarray(U).shape
    nx = nx_p1 - 1
    rho     = np.asarray(metric["rho"],     dtype=np.float32)
    rhow    = np.asarray(metric["rhow"],    dtype=np.float32)
    dz      = np.asarray(metric["dz"],      dtype=np.float32)
    dx      = float(metric["dx_lon"])
    dy      = np.asarray(metric["dy_lat"],  dtype=np.float32)
    imu     = np.asarray(metric["imu"],     dtype=np.float32)
    cos_v   = np.asarray(metric["cos_v"],   dtype=np.float32)
    cos_lat = np.asarray(metric["cos_lat"], dtype=np.float32)
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(W, dtype=np.float32).tobytes(order="C"))
        f.write(rho.tobytes())
        f.write(rhow.tobytes())
        f.write(dz.tobytes())
        f.write(struct.pack("f", dx))
        f.write(dy.tobytes())
        f.write(imu.tobytes())
        f.write(cos_v.tobytes())
        f.write(cos_lat.tobytes())
        f.write(struct.pack("f", float(dt)))


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1]
    g    = _make_grid()
    m    = build_metric(g)
    nz, ny, nx = g.nz, g.ny, g.nx
    dt   = 10.0

    if mode == "press_rhs_zero_velocity":
        U = jnp.zeros((nz, ny, nx + 1))
        V = jnp.zeros((nz, ny + 1, nx))
        W = jnp.zeros((nz + 1, ny, nx))
        _write_inputs(workdir, U, V, W, m, dt)
        rhs = press_rhs(U, V, W, m, dt)
        write_bin(workdir / "jsam_out.bin", np.asarray(rhs, dtype=np.float32).ravel())
        return 0

    if mode == "press_rhs_constant_U":
        U = jnp.full((nz, ny, nx + 1), 10.0)
        V = jnp.zeros((nz, ny + 1, nx))
        W = jnp.zeros((nz + 1, ny, nx))
        _write_inputs(workdir, U, V, W, m, dt)
        rhs = press_rhs(U, V, W, m, dt)
        write_bin(workdir / "jsam_out.bin", np.asarray(rhs, dtype=np.float32).ravel())
        return 0

    if mode == "press_gradient_zero_p":
        # p=0 → no gradient correction → velocities unchanged → divergence = 0 initially
        U = jnp.zeros((nz, ny, nx + 1))
        V = jnp.zeros((nz, ny + 1, nx))
        W = jnp.zeros((nz + 1, ny, nx))
        _write_inputs(workdir, U, V, W, m, dt)
        rhs = press_rhs(U, V, W, m, dt)
        write_bin(workdir / "jsam_out.bin", np.asarray(rhs, dtype=np.float32).ravel())
        return 0

    if mode == "press_gradient_constant_p":
        # constant p → gradient = 0 → no velocity correction
        # Use divergence-free velocity to get RHS=0
        U = jnp.full((nz, ny, nx + 1), 5.0)
        V = jnp.zeros((nz, ny + 1, nx))
        W = jnp.zeros((nz + 1, ny, nx))
        _write_inputs(workdir, U, V, W, m, dt)
        rhs = press_rhs(U, V, W, m, dt)
        write_bin(workdir / "jsam_out.bin", np.asarray(rhs, dtype=np.float32).ravel())
        return 0

    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    sys.exit(main())
