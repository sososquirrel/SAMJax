"""
ERA5 initialization and large-scale forcing for jsam.

Reads ERA5 0.25° pressure-level and surface analyses from NCAR RDA (ds633.0 / d633000),
reprojects onto the jsam lat-lon C-grid, and returns ModelState + forcing objects.

NCAR RDA path convention
------------------------
Pressure-level (daily files, one per variable, 24 hourly snapshots each):
    {rda_root}/e5.oper.an.pl/{YYYYMM}/
        e5.oper.an.pl.{code_var}.ll025{sc|uv}.{YYYYMMDD}00_{YYYYMMDD}23.nc

Surface (monthly files):
    {rda_root}/e5.oper.an.sfc/{YYYYMM}/
        e5.oper.an.sfc.{code_var}.ll025sc.{YYYYMM}0100_{YYYYMM}{ndays}23.nc

ERA5 native layout (pressure-level files)
------------------------------------------
  dims : (time=24, level=37, latitude=721, longitude=1440)
  level     : 1 → 1000 hPa, top → bottom  (index 0 = highest altitude)
  latitude  : 90 → -90°  N → S
  longitude : 0 → 359.75°

Unit conversions applied here
------------------------------
  Z (m²/s²)    → height  z = Z / _G               [m]
  OMEGA (Pa/s) → w        = -OMEGA * Rd * T / (p * g)  [m/s]
  Q, CLWC, CIWC           → already kg/kg, loaded as-is
  T                        → already K, loaded as-is
"""

from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.interpolate
import xarray as xr
import jax.numpy as jnp

from jsam.core.grid.latlon import LatLonGrid
from jsam.core.physics.lsforcing import LargeScaleForcing
from jsam.core.state import ModelState

# ---------------------------------------------------------------------------
# Constants (ERA5 / gSAM convention)
# ---------------------------------------------------------------------------
_G  = 9.79764  # m/s²   matches gSAM consts.f90 ggr (ERA5 Z→z conversion uses 9.80665 natively; the 0.07% difference is negligible over a single 37-level reprojection)
_RD = 287.04   # J/(kg K)  gas constant for dry air

#: Default path to NCAR RDA ERA5 archive on GLADE.
RDA_ROOT = "/glade/campaign/collections/rda/data/d633000"

# ERA5 pressure-level variable registry
#   key → (code_var_filename_fragment, grid_type, netcdf_varname)
_PL_VARS: dict[str, tuple[str, str, str]] = {
    "Z":     ("128_129_z",    "sc", "Z"),      # geopotential   m²/s²
    "T":     ("128_130_t",    "sc", "T"),      # temperature    K
    "U":     ("128_131_u",    "uv", "U"),      # zonal wind     m/s
    "V":     ("128_132_v",    "uv", "V"),      # merid. wind    m/s
    "OMEGA": ("128_135_w",    "sc", "W"),      # omega          Pa/s
    "Q":     ("128_133_q",    "sc", "Q"),      # spec. humidity kg/kg
    "CLWC":  ("128_246_clwc", "sc", "CLWC"),  # cloud liquid   kg/kg
    "CIWC":  ("128_247_ciwc", "sc", "CIWC"),  # cloud ice      kg/kg
}

# ERA5 standard pressure levels (37), hPa, top → bottom (native file order)
_ERA5_P_HPA: np.ndarray = np.array([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775,
    800, 825, 850, 875, 900, 925, 950, 975, 1000,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _pl_path(
    dt: datetime,
    code_var: str,
    grid_type: str = "sc",
    rda_root: str = RDA_ROOT,
) -> Path:
    """
    Return the path to a daily ERA5 pressure-level file.

    Parameters
    ----------
    dt        : the date (hour is irrelevant; the file covers all 24 UTC hours)
    code_var  : fragment like ``"128_130_t"``
    grid_type : ``"sc"`` for scalars, ``"uv"`` for wind components
    """
    ym = dt.strftime("%Y%m")
    d  = dt.strftime("%Y%m%d")
    fname = f"e5.oper.an.pl.{code_var}.ll025{grid_type}.{d}00_{d}23.nc"
    return Path(rda_root) / "e5.oper.an.pl" / ym / fname


def _sfc_path(
    year: int,
    month: int,
    code_var: str,
    rda_root: str = RDA_ROOT,
) -> Path:
    """
    Return the path to a monthly ERA5 surface file.

    Parameters
    ----------
    code_var : fragment like ``"128_034_sstk"``
    """
    ym    = f"{year:04d}{month:02d}"
    ndays = calendar.monthrange(year, month)[1]
    fname = f"e5.oper.an.sfc.{code_var}.ll025sc.{ym}0100_{ym}{ndays:02d}23.nc"
    return Path(rda_root) / "e5.oper.an.sfc" / ym / fname


# ---------------------------------------------------------------------------
# ERA5 grid metadata
# ---------------------------------------------------------------------------

def era5_latlon() -> tuple[np.ndarray, np.ndarray]:
    """
    ERA5 0.25° grid coordinates (native order).

    Returns
    -------
    lat : (721,)  90 → -90°  N→S
    lon : (1440,) 0 → 359.75°
    """
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.linspace(0.0, 359.75, 1440)
    return lat, lon


def era5_p_pa() -> np.ndarray:
    """37 ERA5 standard pressure levels in Pa, top→bottom (native file order)."""
    return _ERA5_P_HPA * 100.0


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _read_pl_snapshot(
    dt: datetime,
    var_key: str,
    rda_root: str = RDA_ROOT,
) -> np.ndarray:
    """
    Load one ERA5 pressure-level variable at a single UTC time.

    Returns
    -------
    arr : (37, 721, 1440)  float64
        Native ERA5 order: level top→bottom, latitude N→S.
    """
    code_var, grid_type, nc_var = _PL_VARS[var_key]
    path = _pl_path(dt, code_var, grid_type, rda_root)
    with xr.open_dataset(path) as ds:
        return ds[nc_var].isel(time=dt.hour).values.astype(np.float64)


# ---------------------------------------------------------------------------
# Unit-conversion helpers (pure NumPy, testable without I/O)
# ---------------------------------------------------------------------------

def omega_to_w(
    omega: np.ndarray,
    T: np.ndarray,
    p_pa: np.ndarray,
) -> np.ndarray:
    """
    Convert ERA5 omega (Pa/s) → vertical velocity w (m/s).

    w = -omega / (rho * g) = -omega * Rd * T / (p * g)

    Parameters
    ----------
    omega : (..., nlev, ...) Pa/s  (any shape, p_pa broadcast on ``lev_axis``)
    T     : same shape as omega    K
    p_pa  : (nlev,) Pa             pressure levels (must broadcast with omega/T)
    """
    p_bc = p_pa.reshape((-1,) + (1,) * (omega.ndim - 1))  # broadcast over trailing dims
    return -omega * _RD * T / (p_bc * _G)


# ---------------------------------------------------------------------------
# Interpolation helpers (pure NumPy/scipy, testable without I/O)
# ---------------------------------------------------------------------------

def interp_pressure(
    field: np.ndarray,
    p_src_pa: np.ndarray,
    p_tgt_pa: np.ndarray,
) -> np.ndarray:
    """
    Vertically interpolate a 3-D ERA5 field from pressure levels to target pressures.

    Uses 1-D linear interpolation along axis 0 (scipy ``interp1d`` with axis
    broadcasting — a single call handles all (lat, lon) columns at once).
    Extrapolates linearly beyond the source range (model top is ~15 hPa, ERA5
    top is 1 hPa, so in practice no extrapolation occurs for IRMA-like grids).

    Parameters
    ----------
    field    : (nz_src, ...) float64   field on ERA5 pressure levels
    p_src_pa : (nz_src,) Pa            source pressure levels (must be monotone)
    p_tgt_pa : (nz_tgt,) Pa            target model pressure levels

    Returns
    -------
    out : (nz_tgt, ...) float64
    """
    # Clamp target pressures to source range — do NOT extrapolate.
    # The model surface can exceed the ERA5 bottom level (1000 hPa), and
    # extrapolating QV below 1000 hPa gives unphysically high values that
    # crash the saturation adjustment Newton iteration.
    p_tgt_clamped = np.clip(p_tgt_pa, p_src_pa[0], p_src_pa[-1])
    f = scipy.interpolate.interp1d(
        p_src_pa, field, axis=0,
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    return f(p_tgt_clamped)


def interp_horiz(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """
    Bilinear horizontal interpolation from the ERA5 grid to the model grid.

    Both grids must be uniformly spaced (as ERA5 0.25° and gSAM grids are).
    All vertical levels are processed simultaneously via NumPy fancy indexing —
    no Python loop over levels.  Longitude is periodic; latitude is clamped.

    Parameters
    ----------
    field   : (nz, ny_src, nx_src) float64
    src_lat : (ny_src,) degrees, ascending (S→N), uniformly spaced
    src_lon : (nx_src,) degrees, uniformly spaced
    tgt_lat : (ny_tgt,) degrees, ascending
    tgt_lon : (nx_tgt,) degrees

    Returns
    -------
    out : (nz, ny_tgt, nx_tgt) float64
    """
    ny_src, nx_src = len(src_lat), len(src_lon)
    dlat = (src_lat[-1] - src_lat[0]) / (ny_src - 1)
    dlon = (src_lon[-1] - src_lon[0]) / (nx_src - 1)

    # Fractional indices for each target coordinate
    j_frac = (tgt_lat - src_lat[0]) / dlat          # (ny_t,)
    i_frac = (tgt_lon - src_lon[0]) / dlon           # (nx_t,)

    # Latitude: clamp to valid range
    j_frac = np.clip(j_frac, 0.0, ny_src - 1.0 - 1e-10)

    # Longitude: wrap periodically
    i_frac = np.mod(i_frac, float(nx_src))

    j0 = np.floor(j_frac).astype(np.intp)            # (ny_t,)
    j1 = np.minimum(j0 + 1, ny_src - 1)
    i0 = np.floor(i_frac).astype(np.intp)            # (nx_t,)
    i1 = (i0 + 1) % nx_src

    wj = (j_frac - j0)[:, None]  # (ny_t, 1)  weight toward j1
    wi = (i_frac - i0)[None, :]  # (1, nx_t)  weight toward i1

    # Reshape index arrays for (ny_t, nx_t) broadcast
    j0_2d = j0[:, None]   # (ny_t, 1)
    j1_2d = j1[:, None]
    i0_2d = i0[None, :]   # (1, nx_t)
    i1_2d = i1[None, :]

    # field[:, j_2d, i_2d] broadcasts to (nz, ny_t, nx_t)
    return (
        (1 - wj) * (1 - wi) * field[:, j0_2d, i0_2d]
        + (1 - wj) *      wi  * field[:, j0_2d, i1_2d]
        +      wj  * (1 - wi) * field[:, j1_2d, i0_2d]
        +      wj  *      wi  * field[:, j1_2d, i1_2d]
    )


# ---------------------------------------------------------------------------
# C-grid staggering helpers (pure NumPy)
# ---------------------------------------------------------------------------

def stagger_u(u_mass: np.ndarray) -> np.ndarray:
    """
    Interpolate u (nz, ny, nx) on mass grid → U (nz, ny, nx+1) on east faces.

    U[..., i] = (u[..., i] + u[..., i+1]) / 2  (i = 0 … nx-1, periodic)
    U[..., nx] = U[..., 0]  (periodic duplicate)
    """
    u_east = np.roll(u_mass, -1, axis=2)                  # u shifted one cell east
    U = np.empty((*u_mass.shape[:2], u_mass.shape[2] + 1), dtype=u_mass.dtype)
    U[:, :, :-1] = 0.5 * (u_mass + u_east)
    U[:, :, -1]  = U[:, :, 0]                             # periodic east = west
    return U


def stagger_v(v_mass: np.ndarray) -> np.ndarray:
    """
    Interpolate v (nz, ny, nx) on mass grid → V (nz, ny+1, nx) on north faces.

    V[..., j, :] = (v[..., j-1, :] + v[..., j, :]) / 2  (j = 1 … ny-1)
    V[..., 0, :] = V[..., ny, :] = 0  (polar / wall BCs)
    """
    V = np.zeros((*v_mass.shape[:2], v_mass.shape[2]), dtype=v_mass.dtype)
    # temporary shape: we need (nz, ny+1, nx)
    out = np.zeros((v_mass.shape[0], v_mass.shape[1] + 1, v_mass.shape[2]), dtype=v_mass.dtype)
    out[:, 1:-1, :] = 0.5 * (v_mass[:, :-1, :] + v_mass[:, 1:, :])
    # poles stay zero
    return out


def stagger_w(w_mass: np.ndarray) -> np.ndarray:
    """
    Interpolate w (nz, ny, nx) on mass grid → W (nz+1, ny, nx) on top faces.

    W[k] = (w[k-1] + w[k]) / 2  (k = 1 … nz-1)
    W[0] = W[nz] = 0  (rigid-lid BCs)
    """
    out = np.zeros((w_mass.shape[0] + 1, *w_mass.shape[1:]), dtype=w_mass.dtype)
    out[1:-1] = 0.5 * (w_mass[:-1] + w_mass[1:])
    return out


# ---------------------------------------------------------------------------
# Domain-mean profile helper
# ---------------------------------------------------------------------------

def _domain_mean_profile(
    field: np.ndarray,          # (37, ny_era5, nx_era5) top→bottom, N→S
    era5_lat_sn: np.ndarray,    # (721,) S→N
    era5_lon: np.ndarray,       # (1440,)
    model_lat: np.ndarray,      # (ny,) domain extent S→N
    model_lon: np.ndarray,      # (nx,) domain extent
) -> np.ndarray:
    """Return the area-weighted horizontal mean over the model domain, shape (37,)."""
    lat_mask = (era5_lat_sn >= model_lat[0]) & (era5_lat_sn <= model_lat[-1])
    lon_mask = (era5_lon    >= model_lon[0]) & (era5_lon    <= model_lon[-1])
    sub = field[:, lat_mask, :][:, :, lon_mask]     # (37, ny_sub, nx_sub)

    # Area-weighted mean: weight by cos(lat)
    cos_w = np.cos(np.deg2rad(era5_lat_sn[lat_mask]))  # (ny_sub,)
    w2d = cos_w[np.newaxis, :, np.newaxis]              # broadcast over (37, _, nx_sub)
    return (sub * w2d).sum(axis=(1, 2)) / (w2d * np.ones_like(sub)).sum(axis=(1, 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _gsam_reference_column(
    z:      np.ndarray,   # (nz,)   model cell-centre heights (m)
    zi:     np.ndarray,   # (nz+1,) model interface heights   (m)
    tabs0:  np.ndarray,   # (nz,)   domain-mean absolute temperature (K)
    pres0:  float,        # surface reference pressure (hPa)
    pres_seed: np.ndarray | None = None,   # (nz,) log-interp seed pressure (hPa)
) -> dict:
    """
    Build the gSAM anelastic reference column EXACTLY as setdata.f90:427-466
    does in the dolatlon branch.

    Follows the Fortran sequence:

        presr(1) = (pres0/1000)**(rgas/cp)
        presi(1) = pres0
        do k=1,nzm
          prespot(k)  = (1000/pres(k))**(rgas/cp)
          t0(k)       = tabs0(k)*prespot(k)               ! potential temperature
          presr(k+1)  = presr(k) - g/cp/t0(k)*(zi(k+1)-zi(k))
          presi(k+1)  = 1000*presr(k+1)**(cp/rgas)
          pres(k)     = exp( log(presi(k)) +
                             log(presi(k+1)/presi(k)) *
                             (z(k)-zi(k)) / (zi(k+1)-zi(k)) )
        end do
        rho(k)  = (presi(k)-presi(k+1))/(zi(k+1)-zi(k)) / g * 100
        rhow(k) = (rho(k-1)*adz(k)+rho(k)*adz(k-1))/(adz(k)+adz(k-1))  [interior]
        rhow(1)  = 2*rho(1)  - rhow(2)
        rhow(nz) = 2*rho(nzm)- rhow(nzm)

    The loop has an implicit dependence `pres(k)` appearing inside
    `prespot(k)` before being updated at the end of the same iteration. gSAM
    resolves this by seeding `pres(k)` with a log-linear interpolation of
    the init-file pressure levels (setdata.f90:360-367).  We mirror that
    with *pres_seed* (hPa).  Two refinement sweeps are then enough for
    bit-stability in float64.

    Returns a dict with keys rho, rhow, pres, presi, t0 (all numpy
    arrays; pressure in hPa, density in kg/m³).
    """
    RGAS = 287.04
    CP   = 1004.64
    GGR  = 9.79764       # gSAM consts.f90

    nz  = len(z)
    assert len(zi) == nz + 1

    dz  = np.diff(zi)                      # (nz,)  — same as gSAM dz*adz(k)
    adz = dz / dz[0]                       # (nz,)  gSAM adz(k) (arbitrary ref)

    # Seed pres(k) — gSAM log-linear interpolation of init-file levels.
    if pres_seed is None:
        # Fall back to a dry-isothermal guess (300 K) if caller gave none.
        pres = pres0 * np.exp(-GGR * z / (RGAS * 300.0))
    else:
        pres = np.array(pres_seed, dtype=np.float64)

    presi = np.zeros(nz + 1, dtype=np.float64)
    presr = np.zeros(nz + 1, dtype=np.float64)
    t0    = np.zeros(nz,     dtype=np.float64)

    # Two refinement sweeps — gSAM does exactly one, but iterating twice
    # absorbs the initial seed mismatch; the update is a ≤0.1 hPa tweak.
    for _sweep in range(2):
        presr[0] = (pres0 / 1000.0) ** (RGAS / CP)
        presi[0] = pres0
        for k in range(nz):
            prespot_k = (1000.0 / pres[k]) ** (RGAS / CP)
            t0[k]     = tabs0[k] * prespot_k
            presr[k + 1] = presr[k] - GGR / CP / t0[k] * (zi[k + 1] - zi[k])
            presi[k + 1] = 1000.0 * presr[k + 1] ** (CP / RGAS)
            # gSAM log-linear back-fill of pres(k)
            pres[k] = np.exp(
                np.log(presi[k])
                + np.log(presi[k + 1] / presi[k])
                * (z[k] - zi[k]) / (zi[k + 1] - zi[k])
            )

    # Density from layer pressure drop (gSAM setdata.f90:465).
    # presi is hPa → ×100 to convert (presi[k]-presi[k+1]) to Pa, then divide
    # by g·dz to get kg/m³.  gSAM writes this as "/ggr*100." which is the
    # same thing.
    rho = (presi[:-1] - presi[1:]) / (zi[1:] - zi[:-1]) / GGR * 100.0  # (nz,)

    # rhow — interior faces adz cross-weighted (setdata.f90:475-477);
    # bottom/top linear extrapolation (lines 483-484).
    rhow = np.zeros(nz + 1, dtype=np.float64)
    rhow[1:-1] = (rho[:-1] * adz[1:] + rho[1:] * adz[:-1]) \
                 / (adz[:-1] + adz[1:])
    rhow[0]  = 2.0 * rho[0]  - rhow[1]
    rhow[-1] = 2.0 * rho[-1] - rhow[-2]

    return {
        "rho":   rho.astype(np.float64),
        "rhow":  rhow.astype(np.float64),
        "pres":  pres.astype(np.float64),
        "presi": presi.astype(np.float64),
        "t0":    t0.astype(np.float64),
    }


def era5_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    dt: datetime,
    rda_root: str = RDA_ROOT,
) -> LatLonGrid:
    """
    Build a :class:`~jsam.core.grid.latlon.LatLonGrid` whose base-state ``rho``
    profile is built with gSAM's hydrostatic sequence from setdata.f90:427-466.

    Sequence (matches gSAM exactly, dolatlon branch):

      1. Compute domain-mean ERA5 temperature at model heights → ``tabs0(k)``
      2. Seed pres(k) by log-linear interpolation of ERA5 pressure levels → z.
      3. Pick pres0 as domain-mean ERA5 surface pressure.
      4. Run ``_gsam_reference_column`` which reproduces the hydrostatic
         potential-temperature integration and derives rho/rhow/presi.

    Parameters
    ----------
    lat, lon : 1-D arrays defining the target model horizontal grid (degrees)
    z, zi    : 1-D arrays defining the target model vertical grid (metres)
    dt       : ERA5 analysis time (UTC); only date + hour matter
    """
    era5_lat_ns, era5_lon = era5_latlon()
    era5_lat_sn = era5_lat_ns[::-1]                       # S→N for masking

    # Load T and Z (geopotential) from ERA5 (top→bottom, N→S)
    T_pl = _read_pl_snapshot(dt, "T", rda_root)           # (37, 721, 1440)
    Z_pl = _read_pl_snapshot(dt, "Z", rda_root)           # (37, 721, 1440)

    # ERA5 N→S → flip to S→N for masking
    T_sn = T_pl[:, ::-1, :]
    Z_sn = Z_pl[:, ::-1, :]

    p_pa = era5_p_pa()                                    # (37,) top→bottom

    # Domain-mean T and z profiles (top→bottom ordering preserved)
    T_mean = _domain_mean_profile(T_sn, era5_lat_sn, era5_lon, lat, lon)   # (37,)
    z_mean = _domain_mean_profile(Z_sn, era5_lat_sn, era5_lon, lat, lon) / _G

    # ── 1. tabs0(k) on the model vertical grid (ascending z) ───────────────
    z_bt = z_mean[::-1]              # (37,) increasing height
    T_bt = T_mean[::-1]              # (37,)
    tabs0_model = np.interp(z, z_bt, T_bt, left=T_bt[0], right=T_bt[-1])

    # ── 2. Seed pres(k) — gSAM setdata.f90:360-368 log-linear interp ───────
    p_hpa_bt = (p_pa[::-1] / 100.0)    # (37,) Pa → hPa, ascending z
    # Log-linear (gSAM exp(log(.)+log(./.) *...)) ≡ interp in log-p vs z.
    pres_seed_hpa = np.exp(
        np.interp(z, z_bt, np.log(p_hpa_bt),
                  left=np.log(p_hpa_bt[0]), right=np.log(p_hpa_bt[-1]))
    )

    # ── 3. pres0 — domain-mean ERA5 surface pressure (hPa) ─────────────────
    # gSAM setdata.f90:368 derives pres0 by log-interp of presin to z=0.
    pres0_hpa = float(np.exp(
        np.interp(0.0, z_bt, np.log(p_hpa_bt),
                  left=np.log(p_hpa_bt[0]), right=np.log(p_hpa_bt[-1]))
    ))

    # ── 4. Hydrostatic reference column ────────────────────────────────────
    ref = _gsam_reference_column(
        z=z, zi=zi, tabs0=tabs0_model,
        pres0=pres0_hpa, pres_seed=pres_seed_hpa,
    )

    return LatLonGrid(lat=lat, lon=lon, z=z, zi=zi, rho=ref["rho"])


def era5_state(
    grid: LatLonGrid,
    metric: dict,
    dt: datetime,
    rda_root: str = RDA_ROOT,
) -> ModelState:
    """
    Initialise a :class:`~jsam.core.state.ModelState` from ERA5 pressure-level
    analysis at time *dt*.

    Steps
    -----
    1. Load T, U, V, OMEGA, Q, CLWC, CIWC from ERA5.
    2. Convert OMEGA → w (m/s) on ERA5 pressure levels.
    3. Flip lat N→S → S→N.
    4. Vertically interpolate from ERA5 37 p-levels to model p-levels
       (from ``metric["pres"]``), using ``scipy.interpolate.interp1d``
       — a single call handles all horizontal columns at once.
    5. Bilinearly interpolate to model lat/lon (per vertical level).
    6. Stagger U, V, W to C-grid faces.

    QR = QS = QG = TKE = 0  (ERA5 has no precipitating species or TKE).
    """
    era5_lat_ns, era5_lon = era5_latlon()
    era5_lat_sn = era5_lat_ns[::-1]   # S→N, ascending

    p_src = era5_p_pa()               # (37,) Pa, top→bottom (increasing pressure)
    # model pressure levels: decreasing from surface to top
    pres_model = np.array(metric["pres"])  # (nz,) Pa

    def _load_remap(var_key: str, clamp_zero: bool = False) -> np.ndarray:
        """Load one ERA5 variable, flip S→N, vert-interp, horiz-interp."""
        raw = _read_pl_snapshot(dt, var_key, rda_root)[:, ::-1, :]   # → S→N
        vi  = interp_pressure(raw, p_src, pres_model)
        del raw
        hi = interp_horiz(vi, era5_lat_sn, era5_lon, grid.lat, grid.lon)
        del vi
        if clamp_zero:
            hi = np.maximum(hi, 0.0)
        return hi

    # ── T and OMEGA are loaded together so we can compute w before remapping ──
    T_tb     = _read_pl_snapshot(dt, "T",     rda_root)    # (37, 721, 1440)
    OMEGA_tb = _read_pl_snapshot(dt, "OMEGA", rda_root)
    W_tb     = omega_to_w(OMEGA_tb, T_tb, p_src)
    del OMEGA_tb

    T_hi = interp_horiz(
        interp_pressure(T_tb[:, ::-1, :], p_src, pres_model),
        era5_lat_sn, era5_lon, grid.lat, grid.lon,
    )
    W_hi = interp_horiz(
        interp_pressure(W_tb[:, ::-1, :], p_src, pres_model),
        era5_lat_sn, era5_lon, grid.lat, grid.lon,
    )
    del T_tb, W_tb

    # ── Remaining variables one at a time ─────────────────────────────────────
    U_hi    = _load_remap("U")
    V_hi    = _load_remap("V")
    Q_hi    = _load_remap("Q",    clamp_zero=True)
    CLWC_hi = _load_remap("CLWC", clamp_zero=True)
    CIWC_hi = _load_remap("CIWC", clamp_zero=True)

    # ── C-grid staggering ──────────────────────────────────────────────────────
    U_stag = stagger_u(U_hi)   # (nz, ny, nx+1)
    V_stag = stagger_v(V_hi)   # (nz, ny+1, nx)
    W_stag = stagger_w(W_hi)   # (nz+1, ny, nx)

    nz, ny, nx = len(grid.z), len(grid.lat), len(grid.lon)

    return ModelState(
        U=jnp.array(U_stag),
        V=jnp.array(V_stag),
        W=jnp.array(W_stag),            # ERA5 omega → w (m/s), staggered to W-faces
        TABS=jnp.array(T_hi),
        QV=jnp.array(Q_hi),
        QC=jnp.array(CLWC_hi),
        QI=jnp.array(CIWC_hi),
        QR=jnp.zeros((nz, ny, nx)),
        QS=jnp.zeros((nz, ny, nx)),
        QG=jnp.zeros((nz, ny, nx)),
        TKE=jnp.zeros((nz, ny, nx)),
        p_prev =jnp.zeros((nz, ny, nx)),
        p_pprev=jnp.zeros((nz, ny, nx)),
        nstep=jnp.int32(0),
        time=jnp.float64(0.0),
    )


def era5_sst(
    grid: LatLonGrid,
    dt: datetime,
    rda_root: str = RDA_ROOT,
) -> jnp.ndarray:
    """
    Load ERA5 sea-surface temperature (SSTK, code 034) onto the model horizontal
    grid.  Returns a (ny, nx) JAX array in K.

    The monthly surface file contains 24 × ndays hourly snapshots starting at
    {YYYYMM}01 00UTC; the time index is ``(dt.day - 1) * 24 + dt.hour``.
    """
    path = _sfc_path(dt.year, dt.month, "128_034_sstk", rda_root)
    t_idx = (dt.day - 1) * 24 + dt.hour
    with xr.open_dataset(path) as ds:
        sst_ns = ds["SSTK"].isel(time=t_idx).values.astype(np.float64)  # (721, 1440)

    era5_lat_ns, era5_lon = era5_latlon()
    era5_lat_sn = era5_lat_ns[::-1]
    sst_sn = sst_ns[::-1, :]   # flip to S→N

    # Bilinear interpolation to model grid (single 2-D field)
    lon_ext = np.append(era5_lon, era5_lon[0] + 360.0)
    sst_ext = np.concatenate([sst_sn, sst_sn[:, :1]], axis=1)

    interp = scipy.interpolate.RegularGridInterpolator(
        (era5_lat_sn, lon_ext), sst_ext,
        method="linear", bounds_error=False, fill_value=None,
    )
    lat_pts = np.repeat(grid.lat, len(grid.lon))
    lon_pts = np.tile(grid.lon, len(grid.lat))
    pts = np.column_stack([lat_pts, lon_pts])
    sst = interp(pts).reshape(len(grid.lat), len(grid.lon))

    return jnp.array(sst)


def era5_ls_forcing(
    grid: LatLonGrid,
    dts: list[datetime],
    rda_root: str = RDA_ROOT,
) -> LargeScaleForcing:
    """
    Build a :class:`~jsam.core.physics.lsforcing.LargeScaleForcing` from ERA5
    omega over a sequence of analysis times.

    ``dtls`` and ``dqls`` (horizontal advective tendencies) are set to zero —
    they can be computed offline from ERA5 if needed.  ``wsub`` is derived from
    the domain-mean ERA5 omega profile:
        w = -omega * Rd * T / (p * g)

    Parameters
    ----------
    dts : list of :class:`datetime`, one per forcing snapshot.
          Simulation time is measured in seconds from ``dts[0]``.
    """
    era5_lat_ns, era5_lon = era5_latlon()
    era5_lat_sn = era5_lat_ns[::-1]
    p_src = era5_p_pa()      # (37,) Pa top→bottom

    t_seconds = np.array([(dt - dts[0]).total_seconds() for dt in dts])
    wsub_all  = np.empty((len(dts), len(grid.z)), dtype=np.float64)

    for i, dt in enumerate(dts):
        T_tb     = _read_pl_snapshot(dt, "T",     rda_root)
        OMEGA_tb = _read_pl_snapshot(dt, "OMEGA", rda_root)
        Z_tb     = _read_pl_snapshot(dt, "Z",     rda_root)

        # flip to S→N
        T_sn     = T_tb[:, ::-1, :]
        OMEGA_sn = OMEGA_tb[:, ::-1, :]
        Z_sn     = Z_tb[:, ::-1, :]

        # domain-mean profiles
        T_mean     = _domain_mean_profile(T_sn,     era5_lat_sn, era5_lon, grid.lat, grid.lon)
        OMEGA_mean = _domain_mean_profile(OMEGA_sn, era5_lat_sn, era5_lon, grid.lat, grid.lon)
        z_mean     = _domain_mean_profile(Z_sn,     era5_lat_sn, era5_lon, grid.lat, grid.lon) / _G

        # omega → w (1-D, already domain mean)
        w_mean = -OMEGA_mean * _RD * T_mean / (p_src * _G)  # (37,)

        # flip to ascending z for np.interp
        z_bt = z_mean[::-1]
        w_bt = w_mean[::-1]

        wsub_all[i] = np.interp(grid.z, z_bt, w_bt, left=w_bt[0], right=w_bt[-1])

    nz    = len(grid.z)
    ntime = len(dts)
    return LargeScaleForcing(
        dtls=jnp.zeros((ntime, nz)),
        dqls=jnp.zeros((ntime, nz)),
        wsub=jnp.array(wsub_all),
        z_prof=jnp.array(grid.z, dtype=jnp.float64),
        t_prof=jnp.array(t_seconds),
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def era5_init(
    lat: np.ndarray,
    lon: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    dt: datetime,
    ls_forcing_times: list[datetime] | None = None,
    rda_root: str = RDA_ROOT,
    polar_filter: bool = True,
) -> dict:
    """
    One-call ERA5 initialisation: builds the grid, metric, state, SST, and
    optionally a large-scale forcing time series.

    Parameters
    ----------
    lat, lon : 1-D arrays, target model horizontal grid (degrees)
    z, zi    : 1-D arrays, target model vertical grid (metres)
    dt       : ERA5 analysis time for the initial condition
    ls_forcing_times : list of :class:`datetime` for large-scale forcing
                       snapshots; ``None`` → skip (forcing fields not returned)
    rda_root : path to NCAR RDA ERA5 archive

    Returns
    -------
    A dict with keys:

    ``"grid"``
        :class:`~jsam.core.grid.latlon.LatLonGrid`
    ``"metric"``
        dict returned by :func:`~jsam.core.dynamics.pressure.build_metric`
    ``"state"``
        :class:`~jsam.core.state.ModelState`
    ``"sst"``
        JAX array (ny, nx) K
    ``"ls_forcing"``
        :class:`~jsam.core.physics.lsforcing.LargeScaleForcing` or ``None``

    Example
    -------
    >>> from datetime import datetime
    >>> from jsam.io.era5 import era5_init
    >>> from tests.validation.irma_loader import IRMALoader
    >>> loader = IRMALoader()
    >>> g = loader.grid
    >>> out = era5_init(
    ...     lat=g["lat"], lon=g["lon"], z=g["z"], zi=g["zi"],
    ...     dt=datetime(2017, 9, 5, 0),
    ... )
    >>> state, grid, sst = out["state"], out["grid"], out["sst"]
    """
    from jsam.core.dynamics.pressure import build_metric

    grid   = era5_grid(lat, lon, z, zi, dt, rda_root)
    metric = build_metric(grid, polar_filter=polar_filter)
    state  = era5_state(grid, metric, dt, rda_root)
    sst    = era5_sst(grid, dt, rda_root)

    ls_forcing = None
    if ls_forcing_times is not None:
        ls_forcing = era5_ls_forcing(grid, ls_forcing_times, rda_root)

    return {
        "grid":       grid,
        "metric":     metric,
        "state":      state,
        "sst":        sst,
        "ls_forcing": ls_forcing,
    }
