"""
era5_binary.py — Reader for the actual gSAM ERA5 init binary.

The binary written by gSAM's ERA5 pre-processing tool has this layout
(Fortran unformatted, little-endian, 4-byte record markers):

  Record 1:  nx1, ny1, nz1          (3 × int32)
  Record 2:  zin(nz1)               (nz1 × float32) global-mean heights, ascending
  Record 3:  presin(nz1)            (nz1 × float32) pressure levels, ascending (hPa)

  Then, for each of the 9 fields in order [U, V, W, TABS, QV, QCL, QCI, QPL, QPI]:
    Record A:  lonr(nx1)            (nx1 × float32) source longitudes
    Record B:  latr(ny1)            (ny1 × float32) source latitudes
    Record C:  zr(nz1)              (nz1 × float32) source heights  (same as zin)
    Record D:  pr(nz1)              (nz1 × float32) source pressures (hPa)
    Records E1..Enz1:  slab_k(ny1, nx1) (ny1*nx1 × float32) one per vertical level

The horizontal layout is S→N (latr ascending) and 0→360°.
The vertical layout is ascending height (index 0 ≈ 1000 hPa surface, index 36 ≈ 1 hPa top).

W in the binary is OMEGA (Pa/s).  gSAM applies the omega→w conversion after loading.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_rec(fh, dtype: np.dtype, n: int | None = None) -> np.ndarray:
    """Read one Fortran unformatted record, return as 1-D array."""
    rec_len = struct.unpack('<I', fh.read(4))[0]
    raw = fh.read(rec_len)
    end_len = struct.unpack('<I', fh.read(4))[0]
    if rec_len != end_len:
        raise IOError(f"Fortran record length mismatch: {rec_len} != {end_len}")
    arr = np.frombuffer(raw, dtype=dtype)
    if n is not None and len(arr) != n:
        raise IOError(f"Expected {n} elements, got {len(arr)}")
    return arr


def read_gsam_init_binary(bin_path: str | Path) -> dict:
    """
    Read a gSAM ERA5 init binary and return all fields at ERA5 native resolution.

    Parameters
    ----------
    bin_path : path to init_era5_YYYYMMDDHHH_GLOBAL.bin

    Returns
    -------
    dict with keys:
        'nx', 'ny', 'nz'    : int — source grid dimensions (1440, 721, 37)
        'lon'               : (nx,) float32 — source longitudes [°]
        'lat'               : (ny,) float32 — source latitudes  [°], ascending S→N
        'zin'               : (nz,) float32 — global-mean heights [m], ascending
        'presin'            : (nz,) float32 — pressure levels [hPa], ascending
        'U', 'V', 'W'       : (nz, ny, nx) float32 — U/V (m/s), W = OMEGA (Pa/s)
        'TABS'              : (nz, ny, nx) float32 — temperature [K]
        'QV', 'QCL', 'QCI' : (nz, ny, nx) float32 — mixing ratios [kg/kg]
        'QPL', 'QPI'        : (nz, ny, nx) float32 — precip mixing ratios [kg/kg]

    All 3-D fields are in ascending-height, S→N order (index 0 = surface level).
    """
    bin_path = Path(bin_path)
    field_names = ['U', 'V', 'W', 'TABS', 'QV', 'QCL', 'QCI', 'QPL', 'QPI']

    with open(bin_path, 'rb') as fh:
        # ── Header ────────────────────────────────────────────────────────────
        dims = _read_rec(fh, np.dtype('<i4'), 3)
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

        zin    = _read_rec(fh, np.dtype('<f4'), nz)   # ascending heights [m]
        presin = _read_rec(fh, np.dtype('<f4'), nz)   # ascending pressures [hPa]

        # ── Fields ────────────────────────────────────────────────────────────
        result = {
            'nx': nx, 'ny': ny, 'nz': nz,
            'zin': zin, 'presin': presin,
        }

        for fname in field_names:
            lonr = _read_rec(fh, np.dtype('<f4'), nx)
            latr = _read_rec(fh, np.dtype('<f4'), ny)
            zr   = _read_rec(fh, np.dtype('<f4'), nz)
            pr   = _read_rec(fh, np.dtype('<f4'), nz)

            slabs = np.empty((nz, ny, nx), dtype=np.float32)
            for k in range(nz):
                slab = _read_rec(fh, np.dtype('<f4'), ny * nx)
                slabs[k] = slab.reshape(ny, nx)

            if fname == 'U':
                result['lon'] = lonr
                result['lat'] = latr
                result['zr']  = zr    # same as zin; kept per-field for safety
                result['pr']  = pr

            result[fname] = slabs

    return result
