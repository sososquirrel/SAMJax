"""test_grid_ady — verify LatLonGrid.ady / dy_ref against gSAM setgrid.f90.

Generates a synthetic non-uniform 8-row latitude band (mimicking the
middle of lat_720_dyvar) and:

  - converts latv → metres via R_earth (matching gSAM setgrid.f90:222
    `yv_gl = latv_gl * deg2rad * rad_earth`),
  - dumps yv_meters to inputs.bin so the Fortran driver applies the
    gSAM adyy/dy formulas verbatim,
  - asks jsam LatLonGrid for `ady` and `dy_ref`,
  - writes them as a single float32 vector for compare.py.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin                      # noqa: E402
from jsam.core.grid.latlon import LatLonGrid, EARTH_RADIUS  # noqa: E402

# Non-uniform mass-row latitudes (8 rows) — mimics the middle of
# lat_720_dyvar where row spacing varies smoothly with latitude.
LAT = np.array(
    [10.20, 10.62, 11.10, 11.65, 12.30, 13.05, 13.95, 15.00],
    dtype=np.float64,
)


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    ny = LAT.size                               # 8 mass rows
    lat = LAT

    # Build a tiny LatLonGrid.  Vertical structure is irrelevant.
    z   = np.array([10.0, 30.0])
    zi  = np.array([0.0, 20.0, 40.0])
    rho = np.array([1.2, 1.1])
    lon = np.array([0.0, 1.0])
    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=rho)

    # Use jsam's reconstructed v-faces so both sides see identical
    # interfaces.  Fortran-side driver works in metres (gSAM yv_gl is
    # `latv * deg2rad * rad_earth`, setgrid.f90:222).
    latv = np.asarray(grid.lat_v, dtype=np.float64)
    yv_m = np.deg2rad(latv) * EARTH_RADIUS

    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", ny))
        f.write(yv_m.astype(np.float32).tobytes())

    # jsam values.
    dy_ref = float(grid.dy_ref)
    ady    = np.asarray(grid.ady, dtype=np.float32)

    out = np.concatenate([np.array([dy_ref], dtype=np.float32), ady])
    write_bin(workdir / "jsam_out.bin", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
