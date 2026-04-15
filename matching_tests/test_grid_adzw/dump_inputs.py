"""test_grid_adzw — dump synthetic stretched zi to Fortran and ask jsam
build_metric() for the same dz_ref / adz / adzw values.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MT_ROOT = HERE.parent
sys.path.insert(0, str(MT_ROOT))

from common.bin_io import write_bin  # noqa: E402

from jsam.core.grid.latlon import LatLonGrid  # noqa: E402
from jsam.core.dynamics.pressure import build_metric  # noqa: E402

# Synthetic stretched interface profile (10 interfaces → 9 full cells)
ZI = np.array(
    [0.0, 100.0, 250.0, 500.0, 900.0, 1400.0, 2000.0, 2700.0, 3500.0, 4400.0],
    dtype=np.float64,
)


def main() -> int:
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    nz = ZI.size - 1                          # 9 full cells
    z  = 0.5 * (ZI[:-1] + ZI[1:])             # cell centres
    rho = np.full(nz, 1.0)                    # arbitrary; not used for adz/adzw

    # Tiny lat/lon grid — only the vertical structure matters here.
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([0.0, 1.0])

    grid = LatLonGrid(lat=lat, lon=lon, z=z, zi=ZI, rho=rho)
    metric = build_metric(grid)

    dz_ref = float(metric["dz_ref"])
    adz    = np.asarray(metric["adz"], dtype=np.float32).reshape(-1)
    adzw   = np.asarray(metric["adzw"], dtype=np.float32).reshape(-1)

    out = np.concatenate([
        np.array([dz_ref], dtype=np.float32),
        adz.astype(np.float32),
        adzw.astype(np.float32),
    ])

    # Write inputs.bin (int32 nz, float32 zi[nz+1])
    with open(workdir / "inputs.bin", "wb") as f:
        f.write(struct.pack("i", nz))
        f.write(ZI.astype(np.float32).tobytes())

    write_bin(workdir / "jsam_out.bin", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
