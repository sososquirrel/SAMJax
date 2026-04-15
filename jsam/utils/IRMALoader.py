"""
IRMA reference data loader.

Lazily opens the gSAM IRMA 10-day simulation output as xarray datasets.
All heavy computation is deferred — call .compute() only when needed.

Files: /glade/derecho/scratch/sabramian/gSAM_IRMA_10days/OUT_NC/
  - 81  × 3D_atm.nc  (every 3h,  Sep 5-15 2017)
  - 480 × 2D_atm.nc  (every 30min)
"""

from __future__ import annotations
from pathlib import Path
from functools import cached_property

import numpy as np
import netCDF4 as nc
import xarray as xr

IRMA_NC_DIR = Path("/glade/derecho/scratch/sabramian/gSAM_IRMA_10days/OUT_NC")


class IRMALoader:
    """
    Lazy reference data loader for the gSAM IRMA 10-day run.

    Usage:
        loader = IRMALoader()
        ds3d = loader.ds3d     # xr.Dataset, lazy
        ds2d = loader.ds2d     # xr.Dataset, lazy
        grid = loader.grid     # dict of grid arrays
    """

    def __init__(self, nc_dir: Path = IRMA_NC_DIR):
        self.nc_dir = Path(nc_dir)
        if not self.nc_dir.exists():
            raise FileNotFoundError(f"IRMA output not found: {nc_dir}")

    # ── File lists ────────────────────────────────────────────────────────────

    @cached_property
    def files_3d(self) -> list[Path]:
        return sorted(self.nc_dir.glob("*.3D_atm.nc"))

    @cached_property
    def files_2d(self) -> list[Path]:
        return sorted(self.nc_dir.glob("*.2D_atm.nc"))

    # ── Lazy xarray datasets ──────────────────────────────────────────────────

    @cached_property
    def ds3d(self) -> xr.Dataset:
        """All 3D output, lazy. Dims: (time, z, lat, lon)."""
        return xr.open_mfdataset(
            [str(f) for f in self.files_3d],
            combine="nested",
            concat_dim="time",
            chunks={"time": 1, "z": 74, "lat": 180, "lon": 360},
            decode_times=False,
        )

    @cached_property
    def ds2d(self) -> xr.Dataset:
        """All 2D output, lazy. Dims: (time, lat, lon)."""
        return xr.open_mfdataset(
            [str(f) for f in self.files_2d],
            combine="nested",
            concat_dim="time",
            chunks={"time": 10, "lat": 720, "lon": 1440},
            decode_times=False,
        )

    # ── Grid metadata (from first 3D file, always cheap to load) ─────────────

    @cached_property
    def grid(self) -> dict:
        """Static grid arrays from the first 3D file."""
        with nc.Dataset(self.files_3d[0]) as ds:
            return {
                "lat":  np.array(ds.variables["lat"][:]),    # (720,)
                "lon":  np.array(ds.variables["lon"][:]),    # (1440,)
                "z":    np.array(ds.variables["z"][:]),      # (74,)  m
                "zi":   np.array(ds.variables["zi"][:]),     # (74,)  m
                "p":    np.array(ds.variables["p"][:]),      # (74,)  hPa
                "pi":   np.array(ds.variables["pi"][:]),     # (74,)  hPa
                "rho":  np.array(ds.variables["rho"][:]),    # (74,)  kg/m³
                "rhoi": np.array(ds.variables["rhoi"][:]),   # (74,)  kg/m³
                "dz":   np.array(ds.variables["dz"][:]),     # (74,)  m
                "wgt":  np.array(ds.variables["wgt"][:]),    # (720,) cos(lat) weights
            }

    @property
    def lat(self): return self.grid["lat"]
    @property
    def lon(self): return self.grid["lon"]
    @property
    def z(self): return self.grid["z"]
    @property
    def p(self): return self.grid["p"]
    @property
    def rho(self): return self.grid["rho"]
    @property
    def cos_lat(self): return self.grid["wgt"]

    # ── Convenience: time axis ────────────────────────────────────────────────

    @cached_property
    def times_3d(self) -> np.ndarray:
        """Time coordinate of 3D files in hours from simulation start."""
        t = []
        for f in self.files_3d:
            with nc.Dataset(f) as ds:
                t.append(float(ds.variables["time"][0]))
        return np.array(t)   # days

    @cached_property
    def times_2d(self) -> np.ndarray:
        """Time coordinate of 2D files in days from simulation start."""
        t = []
        for f in self.files_2d:
            with nc.Dataset(f) as ds:
                t.append(float(ds.variables["time"][0]))
        return np.array(t)

    # ── Single-file loaders (for fast unit tests) ─────────────────────────────

    def load_3d_at(self, time_idx: int) -> dict[str, np.ndarray]:
        """Load all 3D fields from one time snapshot."""
        vars3d = ["U", "V", "W", "TABS", "QV", "QC", "QI", "QR", "QS", "QG", "TKH", "QRAD", "ZdBZ"]
        with nc.Dataset(self.files_3d[time_idx]) as ds:
            return {v: np.array(ds.variables[v][0]) for v in vars3d if v in ds.variables}

    def load_2d_at(self, time_idx: int) -> dict[str, np.ndarray]:
        """Load all 2D fields from one time snapshot."""
        vars2d = ["Prec", "PW", "CAPE", "CIN", "PSFC", "VOR850", "VOR200",
                  "U850", "V850", "U200", "V200", "LHF", "SHF", "LWNT", "SWNT"]
        with nc.Dataset(self.files_2d[time_idx]) as ds:
            return {v: np.array(ds.variables[v][0]) for v in vars2d if v in ds.variables}

    # ── TC centre tracking ────────────────────────────────────────────────────

    @cached_property
    def tc_track(self) -> dict[str, np.ndarray]:
        """
        Estimate TC centre at each 2D output time by finding
        the minimum of PSFC in the Atlantic/Caribbean domain
        (5–35°N, 255–310°E).

        Returns dict with keys: 'time', 'lat_c', 'lon_c', 'psfc_min'
        """
        lat, lon = self.lat, self.lon
        lat_mask = (lat >= 5) & (lat <= 35)
        lon_mask = (lon >= 255) | (lon <= 10)   # Atlantic

        lats_c, lons_c, pmin = [], [], []
        for f in self.files_2d:
            with nc.Dataset(f) as ds:
                psfc = np.array(ds.variables["PSFC"][0])   # (lat, lon)

            sub = psfc[np.ix_(lat_mask, lon_mask)]
            j, i = np.unravel_index(np.argmin(sub), sub.shape)
            lats_c.append(lat[lat_mask][j])
            lons_c.append(lon[lon_mask][i])
            pmin.append(sub[j, i])

        return {
            "time":     self.times_2d,
            "lat_c":    np.array(lats_c),
            "lon_c":    np.array(lons_c),
            "psfc_min": np.array(pmin),
        }
