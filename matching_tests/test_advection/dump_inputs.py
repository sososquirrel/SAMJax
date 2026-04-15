"""Re-uses fixtures from jsam/tests/unit/test_advection.py and dumps:
   - inputs.bin    (format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called from run.sh:
   python dump_inputs.py face5_cn1
   python dump_inputs.py face5_cn_neg1
   python dump_inputs.py face5_cn0_linear
   python dump_inputs.py zero_velocity
   python dump_inputs.py constant_field
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

from common.bin_io import write_bin                              # noqa: E402
from jsam.core.dynamics.advection import _face5, advect_scalar  # noqa: E402


# ---------------------------------------------------------------------------
# Shared grid + metric (matches adv_grid / adv_metric fixtures in test_advection.py)
# ---------------------------------------------------------------------------

def _make_adv_grid():
    from jsam.core.grid.latlon import LatLonGrid
    nx, ny, nz = 32, 16, 16
    lat = np.linspace(-0.4, 0.4, ny)
    lon = np.linspace(0.0, 0.9, nx)
    z   = np.linspace(200.0, 12000.0, nz)
    dz  = np.diff(np.linspace(0.0, 14000.0, nz + 1))
    zi  = np.linspace(0.0, 14000.0, nz + 1)
    rho = 1.2 * np.exp(-z / 8000.0)
    return LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)


def _make_adv_metric(grid):
    from jsam.core.dynamics.pressure import build_metric
    return build_metric(grid)


# ---------------------------------------------------------------------------
# face5 formula cases
# ---------------------------------------------------------------------------

def _dump_face5(case: str, workdir: Path) -> None:
    """Dump 6 stencil values + cn for one face5 sub-case."""
    if case == "face5_cn1":
        x = jnp.arange(10, dtype=float)
        f = np.array([float(x[i]) for i in range(6)], dtype=np.float32)
        cn = np.float32(1.0)
        expected = float(x[2])   # f_im1
    elif case == "face5_cn_neg1":
        x = jnp.arange(10, dtype=float)
        f = np.array([float(x[i]) for i in range(6)], dtype=np.float32)
        cn = np.float32(-1.0)
        expected = float(x[3])   # f_i
    elif case == "face5_cn0_linear":
        i = jnp.arange(-3, 3, dtype=float)
        fv = 3.0 * i + 1.0
        f = np.array([float(v) for v in fv], dtype=np.float32)
        cn = np.float32(0.0)
        expected = float(0.5 * (fv[2] + fv[3]))   # midpoint
    else:
        raise ValueError(f"unknown face5 case: {case}")

    # Compute jsam answer
    jsam_result = float(_face5(
        jnp.array(float(f[0])), jnp.array(float(f[1])), jnp.array(float(f[2])),
        jnp.array(float(f[3])), jnp.array(float(f[4])), jnp.array(float(f[5])),
        jnp.array(float(cn))
    ))

    # Write inputs: 6 float32 stencil values, then float32 cn
    with open(workdir / "inputs.bin", "wb") as fh:
        fh.write(f.tobytes())
        fh.write(struct.pack("f", float(cn)))

    write_bin(workdir / "jsam_out.bin", np.array([jsam_result], dtype=np.float32))


# ---------------------------------------------------------------------------
# Full advect_scalar cases — write phi, U, V, W, metric, dt
# ---------------------------------------------------------------------------

def _write_full_inputs(phi, U, V, W, metric, dt: float, workdir: Path) -> None:
    nz, ny, nx = phi.shape
    dx = float(metric["dx_lon"])
    dy = np.asarray(metric["dy_lat"], dtype=np.float32)   # (ny,)
    dz = np.asarray(metric["dz"],     dtype=np.float32)   # (nz,)
    rho  = np.asarray(metric["rho"],  dtype=np.float32)   # (nz,)
    rhow = np.asarray(metric["rhow"], dtype=np.float32)   # (nz+1,)
    imu  = np.asarray(metric["imu"],  dtype=np.float32)   # (ny,)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(phi, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(U,   dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V,   dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(W,   dtype=np.float32).tobytes(order="C"))
        f.write(struct.pack("f", dx))
        f.write(dy.tobytes())
        f.write(dz.tobytes())
        f.write(rho.tobytes())
        f.write(rhow.tobytes())
        f.write(imu.tobytes())
        f.write(struct.pack("f", float(dt)))


def _dump_zero_velocity(workdir: Path) -> None:
    import jax
    grid   = _make_adv_grid()
    metric = _make_adv_metric(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    key = jax.random.PRNGKey(0)
    phi = jax.random.normal(key, (nz, ny, nx))
    U = jnp.zeros((nz, ny, nx + 1))
    V = jnp.zeros((nz, ny + 1, nx))
    W = jnp.zeros((nz + 1, ny, nx))
    dt = 10.0

    _write_full_inputs(phi, U, V, W, metric, dt, workdir)
    phi_new = advect_scalar(phi, U, V, W, metric, dt=dt)
    write_bin(workdir / "jsam_out.bin",
              np.asarray(phi_new, dtype=np.float32).ravel(order="C"))


def _dump_constant_field(workdir: Path) -> None:
    grid   = _make_adv_grid()
    metric = _make_adv_metric(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    phi = jnp.ones((nz, ny, nx)) * 3.14
    U = jnp.full((nz, ny, nx + 1), 5.0)
    V = jnp.zeros((nz, ny + 1, nx))
    W = jnp.zeros((nz + 1, ny, nx))
    dt = 10.0

    _write_full_inputs(phi, U, V, W, metric, dt, workdir)
    phi_new = advect_scalar(phi, U, V, W, metric, dt=dt)
    write_bin(workdir / "jsam_out.bin",
              np.asarray(phi_new, dtype=np.float32).ravel(order="C"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    case = sys.argv[1]

    if case in ("face5_cn1", "face5_cn_neg1", "face5_cn0_linear"):
        _dump_face5(case, workdir)
        return 0

    if case == "zero_velocity":
        _dump_zero_velocity(workdir)
        return 0

    if case == "constant_field":
        _dump_constant_field(workdir)
        return 0

    raise SystemExit(f"unknown case: {case}")


if __name__ == "__main__":
    sys.exit(main())
