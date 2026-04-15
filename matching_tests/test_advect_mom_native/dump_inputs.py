"""Native-grid fixture for test_advect_mom_native.

Uses the real gSAM lat_720_dyvar latitude grid (IRMALoader) for a
narrow latitude slice and the full 74-level non-uniform `dz`. Calls
jsam advect_momentum and writes inputs.bin in the same layout as
test_momentum_advection/dump_inputs.py so the existing driver.f90
can be reused unchanged.

Usage:
    python dump_inputs.py <case>
    case ∈ { native_uniform_U_eq, native_uniform_U_mid }
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


# A narrow lat band (~10 rows) at each target latitude. Using the
# full 74-level z grid is expensive to iterate over by hand but cheap
# at this ny, nx, and it's the point of the test — we want to exercise
# the non-uniform dz weights.
CASES = {
    "native_uniform_U_eq":  dict(j_lo=355, j_hi=365, U_const=5.0),
    "native_uniform_U_mid": dict(j_lo=540, j_hi=550, U_const=5.0),
}

NX = 8
DT = 10.0


def _build_case(case: str):
    params = CASES[case]

    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    from jsam.utils.IRMALoader import IRMALoader
    from jsam.core.grid.latlon import LatLonGrid
    from jsam.core.dynamics.pressure import build_metric
    from jsam.core.dynamics.advection import advect_momentum

    g = IRMALoader().grid
    lat_full = np.asarray(g["lat"])
    z   = np.asarray(g["z"])
    dz  = np.asarray(g["dz"])

    j_lo, j_hi = params["j_lo"], params["j_hi"]
    lat = lat_full[j_lo:j_hi]
    ny = lat.size
    lon = np.linspace(0.0, 360.0 - 360.0 / NX, NX)
    zi  = np.concatenate([[0.0], z + 0.5 * dz[0]])   # rough; build_metric regenerates
    rho = 1.2 * np.exp(-z / 8000.0)

    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid)
    nz = grid.nz

    U = jnp.full((nz, ny, NX + 1), float(params["U_const"]))
    V = jnp.zeros((nz, ny + 1, NX))
    W = jnp.zeros((nz + 1, ny, NX))

    U_new, V_new, W_new = advect_momentum(U, V, W, metric, dt=DT)
    return ny, nz, U, V, W, U_new, V_new, W_new, metric


def _write_inputs(workdir: Path, ny: int, nz: int, U, V, W, metric):
    """Matches test_momentum_advection/_write_full_mom_inputs layout."""
    dx     = float(metric["dx_lon"])
    dy_ref = float(metric["dy_lat_ref"])
    dz     = np.asarray(metric["dz"], dtype=np.float32)
    dz_ref = float(dz[0])
    ady    = np.asarray(metric["ady"],  dtype=np.float32)
    rho    = np.asarray(metric["rho"],  dtype=np.float32)
    rhow   = np.asarray(metric["rhow"], dtype=np.float32)
    mu     = np.asarray(metric["cos_lat"], dtype=np.float32)
    muv    = np.asarray(metric["cos_v"],   dtype=np.float32)

    adz_arr = (dz / dz_ref).astype(np.float32)
    adz_int = 0.5 * (adz_arr[:-1] + adz_arr[1:])
    adzw    = np.concatenate([adz_arr[:1], adz_int, adz_arr[-1:]]).astype(np.float32)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", nz, ny, NX))
        f.write(np.asarray(U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(W, dtype=np.float32).tobytes(order="C"))
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
        f.write(struct.pack("f", float(DT)))


def main() -> int:
    case = sys.argv[1]
    if case not in CASES:
        raise SystemExit(f"unknown case: {case}")

    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    ny, nz, U, V, W, U_new, V_new, W_new, metric = _build_case(case)
    _write_inputs(workdir, ny, nz, U, V, W, metric)

    jsam_out = np.concatenate([
        np.asarray(U_new, dtype=np.float32).ravel(order="C"),
        np.asarray(V_new, dtype=np.float32).ravel(order="C"),
        np.asarray(W_new, dtype=np.float32).ravel(order="C"),
    ])
    write_bin(workdir / "jsam_out.bin", jsam_out)
    print(f"[advect_mom_native] case={case}  nz={nz}  ny={ny}  nx={NX}  "
          f"n_out={jsam_out.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
