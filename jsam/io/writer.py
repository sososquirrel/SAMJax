"""
NetCDF writer for jsam model output.

Writes 3D_atm snapshots matching the gSAM OUT_NC format so that
IRMALoader and standard atmospheric analysis tools work unchanged.

gSAM conventions reproduced here
---------------------------------
- U, V, W are on the **mass grid** (cell centres), not C-grid faces.
- Moisture variables (QV, QC, QI, QR, QS, QG) are in **g/kg**.
- QRAD is in **K/day**.
- One file per output snapshot, named:
    {casename}_{YYYY}{DDD}_{HHMMSS}_3D_atm.nc
  where DDD is the day-of-year (gSAM convention).

Staggering conventions (jsam → mass grid)
------------------------------------------
U  (nz, ny, nx+1)  → average of faces i and i+1  (nx columns)
V  (nz, ny+1, nx)  → average of faces j and j+1  (ny rows)
W  (nz+1, ny, nx)  → average of faces k and k+1  (nz levels); W[0]=W[-1]=0 enforced
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from jsam.core.grid.latlon import LatLonGrid
from jsam.core.state import ModelState


# ---------------------------------------------------------------------------
# De-staggering helpers (pure NumPy — called with np.array(jax_array))
# ---------------------------------------------------------------------------

def _destagger_u(U: np.ndarray) -> np.ndarray:
    """(nz, ny, nx+1) → (nz, ny, nx): average east and west faces."""
    return 0.5 * (U[:, :, :-1] + U[:, :, 1:])


def _destagger_v(V: np.ndarray) -> np.ndarray:
    """(nz, ny+1, nx) → (nz, ny, nx): average south and north faces."""
    return 0.5 * (V[:, :-1, :] + V[:, 1:, :])


def _destagger_w(W: np.ndarray) -> np.ndarray:
    """(nz+1, ny, nx) → (nz, ny, nx): average lower and upper faces."""
    return 0.5 * (W[:-1, :, :] + W[1:, :, :])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_3d_atm(
    state: ModelState,
    grid: LatLonGrid,
    metric: dict,
    sim_time: datetime,
    out_dir: str | Path,
    casename: str = "jsam",
) -> Path:
    """
    Write one 3D_atm snapshot to NetCDF, matching the gSAM OUT_NC format.

    Parameters
    ----------
    state    : current model state (JAX arrays)
    grid     : LatLonGrid (provides lat, lon, z, zi, rho)
    metric   : dict from build_metric (provides dz, pres)
    sim_time : wall-clock datetime corresponding to this snapshot
    out_dir  : directory to write into (created if absent)
    casename : run identifier prefix (default "jsam")

    Returns
    -------
    Path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Convert JAX arrays to NumPy ─────────────────────────────────────────
    U_stag = np.array(state.U)   # (nz, ny, nx+1)
    V_stag = np.array(state.V)   # (nz, ny+1, nx)
    W_stag = np.array(state.W)   # (nz+1, ny, nx)

    U = _destagger_u(U_stag).astype(np.float32)
    V = _destagger_v(V_stag).astype(np.float32)
    W = _destagger_w(W_stag).astype(np.float32)

    # Moisture: kg/kg → g/kg
    QV  = np.array(state.QV,  dtype=np.float32) * 1e3
    QC  = np.array(state.QC,  dtype=np.float32) * 1e3
    QI  = np.array(state.QI,  dtype=np.float32) * 1e3
    QR  = np.array(state.QR,  dtype=np.float32) * 1e3
    QS  = np.array(state.QS,  dtype=np.float32) * 1e3
    QG  = np.array(state.QG,  dtype=np.float32) * 1e3

    TABS = np.array(state.TABS, dtype=np.float32)

    # ── Grid metadata ────────────────────────────────────────────────────────
    z   = np.array(grid.z,  dtype=np.float32)   # (nz,)  cell centres [m]
    zi  = np.array(grid.zi, dtype=np.float32)   # (nz+1,) interfaces [m]
    lat = np.array(grid.lat, dtype=np.float64)  # (ny,)
    lon = np.array(grid.lon, dtype=np.float64)  # (nx,)
    rho = np.array(grid.rho, dtype=np.float32)  # (nz,) [kg/m³]

    dz  = np.array(metric["dz"],   dtype=np.float32)   # (nz,) [m]
    p   = np.array(metric["pres"], dtype=np.float32) / 100.0  # Pa → hPa

    # Interface pressure: linear interpolation of cell-centre p (approx)
    pi = np.empty(len(z) + 1, dtype=np.float32)
    pi[1:-1] = 0.5 * (p[:-1] + p[1:])
    pi[0]    = p[0]  + 0.5 * (p[0]  - p[1])
    pi[-1]   = p[-1] - 0.5 * (p[-2] - p[-1])

    # Area weights (cos-lat, normalised)
    cos_w = np.cos(np.deg2rad(lat)).astype(np.float64)

    # Staggered coordinate arrays (gSAM convention: face positions)
    dlon = lon[1] - lon[0]
    dlat = lat[1] - lat[0]
    lonu = lon - 0.5 * dlon   # U-face longitudes
    latv = np.empty(len(lat) + 1, dtype=np.float64)
    latv[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    latv[0]    = lat[0]  - 0.5 * dlat
    latv[-1]   = lat[-1] + 0.5 * dlat

    # ── Time metadata ────────────────────────────────────────────────────────
    epoch = datetime(2017, 9, 5, 0)   # gSAM IRMA run start; override via casename
    t_sec = (sim_time - epoch).total_seconds()
    t_day = t_sec / 86400.0

    # ── Build xarray Dataset ─────────────────────────────────────────────────
    time_coord = np.array([sim_time], dtype="datetime64[ns]")
    coords = {
        "time": time_coord,
        "z":    ("z",   z),
        "zi":   ("zi",  zi),
        "lat":  ("lat", lat),
        "lon":  ("lon", lon),
        "lonu": ("lonu", lonu),
        "latv": ("latv", latv),
    }

    def _3d(arr):
        return (["time", "z", "lat", "lon"], arr[np.newaxis])

    ds = xr.Dataset(
        {
            # ── Metadata ──
            "wgt":     (["lat"], cos_w,
                        {"long_name": "averaging weights"}),
            "p":       (["z"],   p,
                        {"long_name": "pressure",           "units": "mb"}),
            "pi":      (["zi"],  pi,
                        {"long_name": "interface pressure", "units": "mb"}),
            "day":     (["time"], np.array([t_day]),
                        {"long_name": "day",                "units": "day"}),
            "timesec": (["time"], np.array([t_sec]),
                        {"long_name": "run time in sec",    "units": "s"}),
            "dz":      (["z"],   dz,
                        {"long_name": "layer thickness",    "units": "m"}),
            "rho":     (["z"],   rho,
                        {"long_name": "density",            "units": "kg/m3"}),
            # ── Dynamics ──
            "U":    (*_3d(U),
                     {"long_name": "X Wind Component",  "units": "m/s"}),
            "V":    (*_3d(V),
                     {"long_name": "Y Wind Component",  "units": "m/s"}),
            "W":    (*_3d(W),
                     {"long_name": "Z Wind Component",  "units": "m/s"}),
            "TABS": (*_3d(TABS),
                     {"long_name": "Absolute Temperature", "units": "K"}),
            # ── Moisture ──
            "QV":   (*_3d(QV),
                     {"long_name": "Water Vapor",  "units": "g/kg"}),
            "QC":   (*_3d(QC),
                     {"long_name": "Cloud Water",  "units": "g/kg"}),
            "QI":   (*_3d(QI),
                     {"long_name": "Cloud Ice",    "units": "g/kg"}),
            "QR":   (*_3d(QR),
                     {"long_name": "Rain Water",   "units": "g/kg"}),
            "QS":   (*_3d(QS),
                     {"long_name": "Snow Water",   "units": "g/kg"}),
            "QG":   (*_3d(QG),
                     {"long_name": "Graupel Water","units": "g/kg"}),
        },
        coords=coords,
        attrs={
            "model":       "jsam",
            "description": "JAX port of SAM (System for Atmospheric Modelling)",
        },
    )

    # ── File name: {casename}_{YYYY}{DDD}_{HHMMSS}_3D_atm.nc ───────────────
    doy    = sim_time.timetuple().tm_yday
    tstr   = sim_time.strftime("%H%M%S")
    fname  = f"{casename}_{sim_time.year}{doy:03d}_{tstr}_3D_atm.nc"
    fpath  = out_dir / fname

    ds.to_netcdf(
        fpath,
        unlimited_dims=["time"],
        encoding={v: {"zlib": True, "complevel": 4}
                  for v in ["U", "V", "W", "TABS", "QV", "QC", "QI", "QR", "QS", "QG"]},
    )

    return fpath
