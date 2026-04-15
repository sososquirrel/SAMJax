"""Native-grid fixture for test_coriolis_native.

Uses the real gSAM lat_720_dyvar latitude grid (via IRMALoader) to
build a narrow latitude band, and calls jsam.core.dynamics.coriolis
.coriolis_tend on a small 3D state. Inputs.bin layout matches the
existing test_coriolis/driver.f90 (which we link against from run.sh).

Usage:
    python dump_inputs.py <case>
    case ∈ { native_equator, native_midlat, native_highlat }
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


# Latitude row ranges (0-based, inclusive-exclusive) into the lat_720_dyvar
# grid. The gSAM grid has 720 rows from ~-89.4° to ~+89.4°.
CASES = {
    "native_equator": dict(j_lo=355, j_hi=365, U_const=5.0, V_const=3.0),
    "native_midlat":  dict(j_lo=540, j_hi=550, U_const=10.0, V_const=0.0),
    "native_highlat": dict(j_lo=685, j_hi=695, U_const=3.0,  V_const=1.0),
}

NZ = 4
NX = 8


def _build_case(case: str):
    params = CASES[case]

    import jax
    jax.config.update("jax_enable_x64", False)
    import jax.numpy as jnp
    from jsam.utils.IRMALoader import IRMALoader
    from jsam.core.grid.latlon import LatLonGrid
    from jsam.core.dynamics.pressure import build_metric
    from jsam.core.dynamics.coriolis import coriolis_tend

    g = IRMALoader().grid
    lat_full = np.asarray(g["lat"])
    j_lo, j_hi = params["j_lo"], params["j_hi"]
    lat = lat_full[j_lo:j_hi]
    ny = lat.size

    # Small longitude span (native dx doesn't matter for coriolis — dU
    # depends on ady/adyv in y only; dV depends on imuv).
    lon = np.linspace(0.0, 360.0 - 360.0 / NX, NX)
    z   = np.array([500.0, 2000.0, 5000.0, 10000.0])
    zi  = np.array([0.0, 1250.0, 3500.0, 7500.0, 12500.0])
    rho = 1.2 * np.exp(-z / 8000.0)

    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)
    metric = build_metric(grid)

    # Constant U, V state; W not used by coriolis_tend in this path.
    U = jnp.full((NZ, ny, NX + 1), float(params["U_const"]))
    V = jnp.full((NZ, ny + 1, NX), float(params["V_const"]))
    # Pole rows: gSAM enforces V=0 at the poles; for a non-polar slice
    # this BC doesn't fire, but coriolis_tend still respects it at the
    # slice edges (j=0 and j=ny faces) so we leave them as the constant.
    W = jnp.zeros((NZ + 1, ny, NX))

    dU, dV, _dW = coriolis_tend(U, V, W, metric)
    return (ny, U, V, metric, dU, dV)


def _write_inputs(workdir: Path, ny: int, U, V, metric):
    fcory = np.asarray(metric["fcory"], dtype=np.float32)
    tanr  = np.asarray(metric["tanr"],  dtype=np.float32)
    mu    = np.asarray(metric["cos_lat"], dtype=np.float32)
    cos_v = np.asarray(metric["cos_v"], dtype=np.float32)
    ady   = np.asarray(metric["ady"],   dtype=np.float32)

    ady_np   = np.asarray(metric["ady"])
    adyv_int = 0.5 * (ady_np[:-1] + ady_np[1:])
    adyv     = np.concatenate([ady_np[:1], adyv_int, ady_np[-1:]]).astype(np.float32)

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("iii", NZ, ny, NX))
        f.write(np.asarray(U, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(V, dtype=np.float32).tobytes(order="C"))
        f.write(fcory.tobytes())
        f.write(tanr.tobytes())
        f.write(mu.tobytes())
        f.write(cos_v.tobytes())
        f.write(ady.tobytes())
        f.write(adyv.tobytes())


def main() -> int:
    case = sys.argv[1]
    if case not in CASES:
        raise SystemExit(f"unknown case: {case}")

    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    ny, U, V, metric, dU, dV = _build_case(case)
    _write_inputs(workdir, ny, U, V, metric)

    jsam_out = np.concatenate([
        np.asarray(dU, dtype=np.float32).ravel(order="C"),
        np.asarray(dV, dtype=np.float32).ravel(order="C"),
    ])
    write_bin(workdir / "jsam_out.bin", jsam_out)
    print(f"[coriolis_native] case={case}  ny={ny}  n_out={jsam_out.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
