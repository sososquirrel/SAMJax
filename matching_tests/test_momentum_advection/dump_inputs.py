"""Re-uses fixtures from jsam/tests/unit/test_momentum_advection.py and dumps:
   - inputs.bin    (format the Fortran driver expects)
   - jsam_out.bin  (jsam-side answer for the same case)

Called from run.sh:
   python dump_inputs.py flux3_positive
   python dump_inputs.py flux3_negative
   python dump_inputs.py flux3_zero
   python dump_inputs.py flux3_constant
   python dump_inputs.py zero_velocity
   python dump_inputs.py uniform_U
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
from jsam.core.dynamics.advection import _flux3, advect_momentum      # noqa: E402


# ---------------------------------------------------------------------------
# Shared grid + metric (matches mom_grid / mom_metric fixtures)
# ---------------------------------------------------------------------------

def _make_mom_grid():
    from jsam.core.grid.latlon import LatLonGrid
    nx, ny, nz = 32, 16, 16
    lat = np.linspace(-0.4, 0.4, ny)
    lon = np.linspace(0.0, 0.9, nx)
    z   = np.linspace(200.0, 12000.0, nz)
    zi  = np.linspace(0.0, 14000.0, nz + 1)
    rho = 1.2 * np.exp(-z / 8000.0)
    return LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)


def _make_mom_metric(grid):
    from jsam.core.dynamics.pressure import build_metric
    return build_metric(grid)


# ---------------------------------------------------------------------------
# flux3 scalar cases
# ---------------------------------------------------------------------------

def _dump_flux3(case: str, workdir: Path) -> None:
    """Dump 4 stencil values + u_adv for one _flux3 sub-case."""
    if case == "flux3_positive":
        # Linear field, u_adv > 0  (matches TestFlux3.test_positive_flow_linear_field)
        x = jnp.arange(-1, 3, dtype=float)
        fv = 2.0 * x + 1.0
        u_adv = jnp.array(3.0)
    elif case == "flux3_negative":
        # Linear field, u_adv < 0  (matches TestFlux3.test_negative_flow_linear_field)
        x = jnp.arange(-1, 3, dtype=float)
        fv = 2.0 * x + 1.0
        u_adv = jnp.array(-2.0)
    elif case == "flux3_zero":
        # u_adv = 0  (matches TestFlux3.test_zero_velocity)
        fv = jnp.array([1.0, 3.0, 2.0, 4.0])
        u_adv = jnp.array(0.0)
    elif case == "flux3_constant":
        # Constant field, u_adv > 0 (use positive; constant result u*C)
        fv = jnp.full(4, 5.0)
        u_adv = jnp.array(3.0)
    else:
        raise ValueError(f"unknown flux3 case: {case}")

    f = np.array([float(v) for v in fv], dtype=np.float32)
    u_f = np.float32(float(u_adv))

    jsam_result = float(_flux3(
        jnp.array(float(f[0])), jnp.array(float(f[1])),
        jnp.array(float(f[2])), jnp.array(float(f[3])),
        jnp.array(float(u_f))
    ))

    # Write inputs: 4 float32 stencil values, then float32 u_adv
    with open(workdir / "inputs.bin", "wb") as fh:
        fh.write(f.tobytes())
        fh.write(struct.pack("f", float(u_f)))

    write_bin(workdir / "jsam_out.bin", np.array([jsam_result], dtype=np.float32))


# ---------------------------------------------------------------------------
# Full advect_momentum cases
# ---------------------------------------------------------------------------

def _write_full_mom_inputs(U, V, W, metric, dt: float, workdir: Path) -> None:
    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1

    dx     = float(metric["dx_lon"])
    dy_ref = float(metric["dy_lat_ref"])
    dz     = np.asarray(metric["dz"], dtype=np.float32)   # (nz,)
    dz_ref = float(dz[0])
    ady    = np.asarray(metric["ady"],  dtype=np.float32)  # (ny,)
    rho    = np.asarray(metric["rho"],  dtype=np.float32)  # (nz,)
    rhow   = np.asarray(metric["rhow"], dtype=np.float32)  # (nz+1,)
    mu     = np.asarray(metric["cos_lat"], dtype=np.float32)  # (ny,)
    muv    = np.asarray(metric["cos_v"],   dtype=np.float32)  # (ny+1,)

    # adz = dz / dz_ref, adzw at w-faces
    adz_arr = (dz / dz_ref).astype(np.float32)
    adz_int = 0.5 * (adz_arr[:-1] + adz_arr[1:])
    adzw    = np.concatenate([adz_arr[:1], adz_int, adz_arr[-1:]]).astype(np.float32)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, nx))
        f.write(np.asarray(U,  dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V,  dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(W,  dtype=np.float32).tobytes(order="C"))
        f.write(struct.pack("f", dx))
        f.write(struct.pack("f", dy_ref))
        f.write(struct.pack("f", dz_ref))
        f.write(ady.tobytes())
        f.write(adz_arr.tobytes())
        f.write(adzw.tobytes())
        f.write(rho.tobytes())
        f.write(rhow.tobytes())
        f.write(mu.tobytes())
        f.write(muv.tobytes())
        f.write(struct.pack("f", float(dt)))


def _jsam_mom_out(U_new, V_new, W_new) -> np.ndarray:
    """Concatenate U_new, V_new, W_new into a single flat float32 array (C order)."""
    return np.concatenate([
        np.asarray(U_new, dtype=np.float32).ravel(order="C"),
        np.asarray(V_new, dtype=np.float32).ravel(order="C"),
        np.asarray(W_new, dtype=np.float32).ravel(order="C"),
    ])


def _dump_zero_velocity(workdir: Path) -> None:
    grid   = _make_mom_grid()
    metric = _make_mom_metric(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    U = jnp.zeros((nz, ny, nx + 1))
    V = jnp.zeros((nz, ny + 1, nx))
    W = jnp.zeros((nz + 1, ny, nx))
    dt = 10.0

    _write_full_mom_inputs(U, V, W, metric, dt, workdir)
    U_new, V_new, W_new = advect_momentum(U, V, W, metric, dt=dt)
    write_bin(workdir / "jsam_out.bin", _jsam_mom_out(U_new, V_new, W_new))


def _dump_uniform_U(workdir: Path) -> None:
    grid   = _make_mom_grid()
    metric = _make_mom_metric(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    U = jnp.full((nz, ny, nx + 1), 5.0)
    V = jnp.zeros((nz, ny + 1, nx))
    W = jnp.zeros((nz + 1, ny, nx))
    dt = 10.0

    _write_full_mom_inputs(U, V, W, metric, dt, workdir)
    U_new, V_new, W_new = advect_momentum(U, V, W, metric, dt=dt)
    write_bin(workdir / "jsam_out.bin", _jsam_mom_out(U_new, V_new, W_new))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    case = sys.argv[1]

    if case in ("flux3_positive", "flux3_negative", "flux3_zero", "flux3_constant"):
        _dump_flux3(case, workdir)
        return 0

    if case == "zero_velocity":
        _dump_zero_velocity(workdir)
        return 0

    if case == "uniform_U":
        _dump_uniform_U(workdir)
        return 0

    raise SystemExit(f"unknown case: {case}")


if __name__ == "__main__":
    sys.exit(main())
