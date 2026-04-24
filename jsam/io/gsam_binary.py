"""
Readers for gSAM Fortran-unformatted binary files.

Two distinct categories of reader:

  2D surface files (readsurface.f90 format)
  ------------------------------------------
  - sst_*.bin, terrain_*.bin, landmask_*.bin, landtype_*.bin, lai_*.bin,
    snow_*.bin, snowt_*.bin
  - Format: nobs → nx1 → ny1 → days[nobs] → fld[nx1,ny1] × nobs
  - Returned as (nobs, ny_gl, nx_gl) in (lat, lon) order

  3D init file (init_era5_*.bin)
  --------------------------------
  - Written by the ERA5→gSAM pre-processing scripts.
  - Format: (nx1,ny1,nz1) → zin[nz1] → presin[nz1] →
    per-variable: lonr→latr→zr→pr → fld[nx1,ny1] × nz1
  - Variables: u, v, omega, tabs, qv, qcl, qci, qpl, qpi
  - Interpolated to the gSAM model grid via the exact logic of
    gSAM SRC/read_field3D.f90 (bilinear in lon/lat, linear in height).
  - Returns (nz, ny, nx) in jsam axis order.

Binary format: native-endian (little-endian x86_64) Fortran unformatted
sequential.  Every record is bracketed by two 4-byte length markers.

Grid convention (load_gsam_init output)
----------------------------------------
  lon  : (nx_gl,)       cell-centre longitudes   [degrees east, 0..359.75]
  lat  : (ny_gl,)       cell-centre latitudes    [degrees, non-uniform]
  lonu : (nx_gl+1,)     u-staggered longitudes   [degrees, -0.125..359.875]
  latv : (ny_gl+1,)     v-staggered latitudes    [degrees, -90..90]
  z    : (nzm,)         cell-centre heights      [m]
  zi   : (nzm+1,)       cell-interface heights   [m]
  adz  : (nzm,)         (zi[k+1]-zi[k]) / dz_ref
  pres : (nzm,)         reference pressure       [hPa]

3D field shapes (jsam convention: nz first):
  U    : (nzm, ny_gl, nx_gl+1)   m/s
  V    : (nzm, ny_gl+1, nx_gl)   m/s
  W    : (nzm+1, ny_gl, nx_gl)   m/s  (0 at k=0 and k=nzm; omega converted)
  TABS : (nzm, ny_gl, nx_gl)     K
  QV   : (nzm, ny_gl, nx_gl)     kg/kg
  QC   : (nzm, ny_gl, nx_gl)     kg/kg
  QI   : (nzm, ny_gl, nx_gl)     kg/kg
  QR   : (nzm, ny_gl, nx_gl)     kg/kg  (qpl in gSAM)
  QS   : (nzm, ny_gl, nx_gl)     kg/kg  (qpi in gSAM)
"""

from __future__ import annotations

import numpy as np
from typing import BinaryIO


# ── Physical constants (gSAM values) ────────────────────────────────────────
_RGAS = 287.04    # J / (kg·K)
_CP   = 1004.0    # J / (kg·K)
_GGR  = 9.81      # m / s²

# ---------------------------------------------------------------------------
# Low-level Fortran record I/O
# ---------------------------------------------------------------------------

def _read_record(f: BinaryIO, dtype, count: int) -> np.ndarray:
    """Read one Fortran unformatted record: 4-byte len, payload, 4-byte len.

    Works with any file-like object (real file or BytesIO).
    """
    rl1_raw = f.read(4)
    if not rl1_raw:
        raise EOFError("unexpected EOF reading record marker")
    rl1 = int(np.frombuffer(rl1_raw, dtype='<i4')[0])
    payload = f.read(rl1)
    rl2 = int(np.frombuffer(f.read(4), dtype='<i4')[0])
    assert rl1 == rl2, f"record marker mismatch: {rl1} vs {rl2}"
    expected = count * np.dtype(dtype).itemsize
    assert rl1 == expected, f"record length {rl1} != expected {expected}"
    return np.frombuffer(payload, dtype=dtype)


def _skip_record(f: BinaryIO) -> None:
    """Skip one Fortran unformatted record."""
    rl = int(np.fromfile(f, dtype='<i4', count=1)[0])
    f.seek(rl, 1)
    f.read(4)


# ---------------------------------------------------------------------------
# 2D surface file readers  (readsurface.f90 format)
# ---------------------------------------------------------------------------

def read_readsurface_field(path: str, field_dtype=np.float32):
    """Read a 2D surface binary file (readsurface.f90 format).

    Returns
    -------
    days  : (nobs,)          float32  — fractional day of each snapshot
    field : (nobs, ny, nx)   float32  — data in (lat, lon) order
    """
    with open(path, 'rb') as f:
        nobs = int(_read_record(f, '<i4', 1)[0])
        nx1  = int(_read_record(f, '<i4', 1)[0])
        ny1  = int(_read_record(f, '<i4', 1)[0])
        days = _read_record(f, '<f4', nobs).astype(np.float32)
        field = np.empty((nobs, ny1, nx1), dtype=np.float32)
        for i in range(nobs):
            # Fortran fld(nx,ny) column-major → reshape(ny,nx) in C-order
            # gives arr[j_lat, i_lon] = fld(i+1, j+1).
            raw = _read_record(f, '<f4', nx1 * ny1)
            field[i] = raw.reshape(ny1, nx1)
    if field_dtype != np.float32:
        field = field.astype(field_dtype)
    return days, field


def read_landtype(path: str) -> np.ndarray:
    """Read landtype_*.bin → (ny_gl, nx_gl) int32."""
    _, fld = read_readsurface_field(path)
    return np.rint(fld[0]).astype(np.int32)


def read_landmask(path: str) -> np.ndarray:
    """Read landmask_*.bin → (ny_gl, nx_gl) int32."""
    _, fld = read_readsurface_field(path)
    return np.rint(fld[0]).astype(np.int32)


def read_lai_monthly(path: str) -> np.ndarray:
    """Read lai_*.bin → (12, ny_gl, nx_gl) float32."""
    _, fld = read_readsurface_field(path)
    assert fld.shape[0] == 12, f"expected 12 LAI months, got {fld.shape[0]}"
    return fld


def read_snow(path: str) -> np.ndarray:
    """Read snow_*.bin → (ny_gl, nx_gl) float32 (snow depth, m)."""
    _, fld = read_readsurface_field(path)
    return fld[0]


def read_snowt(path: str) -> np.ndarray:
    """Read snowt_*.bin → (ny_gl, nx_gl) float32 (snow temperature, K)."""
    _, fld = read_readsurface_field(path)
    return fld[0]


def read_soil_sand_clay(path: str):
    """Read soil_*.bin → (sand, clay) each (ny_gl, nx_gl) float32 (%)."""
    with open(path, 'rb') as f:
        nx1  = int(_read_record(f, '<i4', 1)[0])
        ny1  = int(_read_record(f, '<i4', 1)[0])
        clay = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
        sand = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
    return sand, clay


def read_soil_init(path: str):
    """Read soil_init_*.bin.

    Returns
    -------
    zsoil : (nsoil1,)               float32 — layer-centre depths [m]
    soilt : (nsoil1, ny_gl, nx_gl)  float32 — soil temperature [K]
    soilw : (nsoil1, ny_gl, nx_gl)  float32 — soil moisture
    """
    with open(path, 'rb') as f:
        nsoil1 = int(_read_record(f, '<i4', 1)[0])
        nx1    = int(_read_record(f, '<i4', 1)[0])
        ny1    = int(_read_record(f, '<i4', 1)[0])
        zsoil  = _read_record(f, '<f4', nsoil1).astype(np.float32)
        soilt  = np.empty((nsoil1, ny1, nx1), dtype=np.float32)
        soilw  = np.empty((nsoil1, ny1, nx1), dtype=np.float32)
        for k in range(nsoil1):
            soilt[k] = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
        for k in range(nsoil1):
            soilw[k] = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
    return zsoil, soilt, soilw


# ---------------------------------------------------------------------------
# Grid construction — mirrors gSAM setgrid.f90 for the IRMA readlat case
# ---------------------------------------------------------------------------

def read_lat_dyvar(
    path: str = "/glade/u/home/sabramian/gSAM1.8.7/GRIDS/lat_720_dyvar",
) -> np.ndarray:
    """Read gSAM dyvar ASCII lat file → (ny_gl,) float64 degrees.

    File has one entry per line with 3 whitespace-separated columns; only the
    first column (latitude) is used, matching setgrid.f90 ``read(1,*) lat(j)``.
    """
    vals = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                vals.append(float(parts[0]))
    return np.array(vals, dtype=np.float64)


def read_grd(
    path: str = "/glade/u/home/sabramian/gSAM1.8.7/CASES/IRMA/grd",
) -> tuple:
    """Read gSAM ASCII grd file (new-style: zi interface heights).

    The first entry is zi[0] = 0.0; subsequent lines give zi[1..nz-1].
    Matches setgrid.f90 "new style with grd specifying height of grid interfaces".

    Returns
    -------
    zi   : (nz,)   float64 — interface heights [m], where nz = nzm+1
    z    : (nzm,)  float64 — cell-centre heights = 0.5*(zi[:-1]+zi[1:])
    adz  : (nzm,)  float64 — (zi[k+1]-zi[k]) / dz_ref
    dz   : float            — reference thickness = zi[1]-zi[0]
    """
    zi_list = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                zi_list.append(float(parts[0]))
    zi  = np.array(zi_list, dtype=np.float64)
    dz  = float(zi[1] - zi[0])
    z   = 0.5 * (zi[:-1] + zi[1:])
    adz = (zi[1:] - zi[:-1]) / dz
    return zi, z, adz, dz


def build_gsam_grid(
    gsam_root: str = "/glade/u/home/sabramian/gSAM1.8.7",
    nx_gl: int = 1440,
    ny_gl: int = 720,
) -> dict:
    """Build the gSAM IRMA model grid arrays, mirroring setgrid.f90.

    Reads GRIDS/lat_720_dyvar and CASES/IRMA/grd, then computes lon, lonu,
    latv following the exact formulas in setgrid.f90.

    Returns dict with keys: lon, lat, lonu, latv, z, zi, adz, dz, nzm,
    nx_gl, ny_gl.
    """
    import os
    lat_file = os.path.join(gsam_root, "GRIDS", "lat_720_dyvar")
    grd_file = os.path.join(gsam_root, "CASES", "IRMA", "grd")

    lat = read_lat_dyvar(lat_file)          # (ny_gl,)
    zi, z, adz, dz = read_grd(grd_file)    # zi: (nzm+1,)
    nzm = len(z)

    # Scalar longitudes — exact port of setparm.f90 + setgrid.f90 for the
    # doglobal=T, readlat=T, dolatlon=T case:
    #   setparm:  dlon = 360._8/nx_gl            (float64)
    #             dx   = dlon*deg2rad*rad_earth   (float64 rhs → stored as real/float32)
    #   setgrid:  lon_gl(i) = 0 + float64(dx_f32)*(i-1)/rad_earth*rad2deg
    #
    # The float32 truncation of dx shifts lon_gl by ~4e-6 deg per cell, making
    # lon_gl(1041) = 259.9999963 < 260.0 → i_min_gl=1042 matches the oracle.
    _rad_earth = np.float64(6371229.0)
    _rad2deg   = np.float64(180.0) / np.float64(np.pi)
    _deg2rad   = np.float64(np.pi) / np.float64(180.0)
    _dlon_f64  = np.float64(360.0) / np.float64(nx_gl)
    _dx_f32    = np.float32(_dlon_f64 * _deg2rad * _rad_earth)   # gSAM: real(dx)
    lon = np.float64(_dx_f32) * np.arange(nx_gl, dtype=np.float64) / _rad_earth * _rad2deg

    # u-staggered longitudes (setgrid.f90 lines 335-336):
    #   lonu_gl(i)      = lon_gl(i)      - 0.5*(lon_gl(2)-lon_gl(1))
    #   lonu_gl(nx_gl+1)= lon_gl(nx_gl)  + 0.5*(lon_gl(2)-lon_gl(1))
    _dlon_out = lon[1] - lon[0]   # actual spacing after float32 truncation
    lonu = np.empty(nx_gl + 1, dtype=np.float64)
    lonu[:nx_gl] = lon - 0.5 * _dlon_out
    lonu[nx_gl]  = lon[-1] + 0.5 * _dlon_out

    # v-staggered latitudes (setgrid.f90 lines 199-222, doglobal=.true.):
    #   latv_gl(1)       = -90  (exact)
    #   latv_gl(j)       = 0.5*(lat(j)+lat(j-1))  for j=2..ny_gl
    #   latv_gl(ny_gl+1) = +90  (exact)
    latv = np.empty(ny_gl + 1, dtype=np.float64)
    latv[0]       = -90.0
    latv[1:ny_gl] = 0.5 * (lat[:-1] + lat[1:])
    latv[ny_gl]   = 90.0

    return dict(lon=lon, lat=lat, lonu=lonu, latv=latv,
                z=z, zi=zi, adz=adz, dz=dz, nzm=nzm,
                nx_gl=nx_gl, ny_gl=ny_gl)


# ---------------------------------------------------------------------------
# 3D init file reader — exact port of gSAM SRC/read_field3D.f90
# ---------------------------------------------------------------------------

def _read_field3d(
    f: BinaryIO,
    nx_src: int,
    ny_src: int,
    nz_src: int,
    lon_tgt: np.ndarray,
    lat_tgt: np.ndarray,
    z_tgt: np.ndarray,
    pres_tgt: np.ndarray,
) -> np.ndarray:
    """Read one 3D field and interpolate to target grid.  Exact port of gSAM read_field3D.f90."""
    lonr, latr, zr, pr, src = _read_raw_slabs(f, nx_src, ny_src, nz_src)
    wts = _build_interp_weights(lonr, latr, zr, pr,
                                lon_tgt, lat_tgt, z_tgt, pres_tgt,
                                nx_src, ny_src, nz_src)
    return _apply_interp(src, wts)


def _read_raw_slabs(
    f: BinaryIO,
    nx_src: int,
    ny_src: int,
    nz_src: int,
) -> tuple:
    """Read one field's coord records + data slabs from f (file or BytesIO).

    Returns (lonr, latr, zr, pr, src) where src is (nz_src, ny_src, nx_src) float32.
    """
    lonr = np.array(_read_record(f, '<f4', nx_src), dtype=np.float64)
    latr = np.array(_read_record(f, '<f4', ny_src), dtype=np.float64)
    zr   = np.array(_read_record(f, '<f4', nz_src), dtype=np.float64)
    pr   = np.array(_read_record(f, '<f4', nz_src), dtype=np.float64)
    src  = np.empty((nz_src, ny_src, nx_src), dtype=np.float32)
    for k in range(nz_src):
        src[k] = _read_record(f, '<f4', nx_src * ny_src).reshape(ny_src, nx_src)
    return lonr, latr, zr, pr, src


def _build_interp_weights(
    lonr: np.ndarray,
    latr: np.ndarray,
    zr: np.ndarray,
    pr: np.ndarray,
    lon_tgt: np.ndarray,
    lat_tgt: np.ndarray,
    z_tgt: np.ndarray,
    pres_tgt: np.ndarray,
    nx_src: int,
    ny_src: int,
    nz_src: int,
) -> dict:
    """Compute bilinear + vertical interpolation weights (no data copying)."""
    lonr_ext = np.empty(nx_src + 2)
    lonr_ext[1:nx_src + 1] = lonr
    lonr_ext[0]           = 2 * lonr[0]  - lonr[1]
    lonr_ext[nx_src + 1]  = 2 * lonr[-1] - lonr[-2]

    latr_ext = np.empty(ny_src + 2)
    latr_ext[1:ny_src + 1] = latr
    latr_ext[0]           = 2 * latr[0]  - latr[1]
    latr_ext[ny_src + 1]  = 2 * latr[-1] - latr[-2]

    m_idx = np.searchsorted(lonr_ext, lon_tgt, side='right') - 1
    m_idx = np.clip(m_idx, 0, nx_src)
    x = (lon_tgt - lonr_ext[m_idx]) / (lonr_ext[m_idx + 1] - lonr_ext[m_idx])

    n_idx = np.searchsorted(latr_ext, lat_tgt, side='right') - 1
    n_idx = np.clip(n_idx, 0, ny_src)
    y = (lat_tgt - latr_ext[n_idx]) / (latr_ext[n_idx + 1] - latr_ext[n_idx])

    di0 = (m_idx - 1) % nx_src
    di1 = m_idx % nx_src
    dj0 = (n_idx - 1) % ny_src
    dj1 = n_idx % ny_src

    use_height = (float(zr[0]) != -999.0)
    if use_height:
        kk_arr  = np.clip(np.searchsorted(zr, z_tgt, side='right') - 1, 0, nz_src - 2).astype(np.intp)
        wgt_arr = np.clip((z_tgt - zr[kk_arr]) / (zr[kk_arr + 1] - zr[kk_arr]), 0.0, 1.0)
    else:
        log_pr   = np.log(pr)
        log_ptgt = np.log(pres_tgt)
        kk_arr   = np.clip(np.searchsorted(-pr, -pres_tgt, side='right') - 1, 0, nz_src - 2).astype(np.intp)
        wgt_arr  = (log_ptgt - log_pr[kk_arr]) / (log_pr[kk_arr + 1] - log_pr[kk_arr])

    return dict(x=x, y=y, di0=di0, di1=di1, dj0=dj0, dj1=dj1,
                kk_arr=kk_arr, wgt_arr=wgt_arr)


def _apply_interp(src: np.ndarray, wts: dict) -> np.ndarray:
    """Interpolate one field (nz_src, ny_src, nx_src) → (nz_t, ny_t, nx_t) float32."""
    di0, di1, dj0, dj1 = wts['di0'], wts['di1'], wts['dj0'], wts['dj1']
    x, y, kk_arr, wgt_arr = wts['x'], wts['y'], wts['kk_arr'], wts['wgt_arr']

    src_lo = src[kk_arr];  src_hi = src[kk_arr + 1]
    wy1 = (1.0 - y)[None, :, None];  wy0 = y[None, :, None]
    wx1 = (1.0 - x)[None, None, :];  wx0 = x[None, None, :]
    wz1 = (1.0 - wgt_arr)[:, None, None];  wz0 = wgt_arr[:, None, None]

    slo_j0 = src_lo[:, dj0, :];  slo_j1 = src_lo[:, dj1, :]
    shi_j0 = src_hi[:, dj0, :];  shi_j1 = src_hi[:, dj1, :]
    bilin_lo = wy1 * (wx1 * slo_j0[:, :, di0] + wx0 * slo_j0[:, :, di1]) + \
               wy0 * (wx1 * slo_j1[:, :, di0] + wx0 * slo_j1[:, :, di1])
    bilin_hi = wy1 * (wx1 * shi_j0[:, :, di0] + wx0 * shi_j0[:, :, di1]) + \
               wy0 * (wx1 * shi_j1[:, :, di0] + wx0 * shi_j1[:, :, di1])
    return (wz1 * bilin_lo + wz0 * bilin_hi).astype(np.float32)


def _apply_interp_batch(src_batch: np.ndarray, wts: dict) -> np.ndarray:
    """Interpolate a batch (nvars, nz_src, ny_src, nx_src) → (nvars, nz_t, ny_t, nx_t) float32.

    Computes bilinear weights once for all variables sharing the same target grid.
    """
    di0, di1, dj0, dj1 = wts['di0'], wts['di1'], wts['dj0'], wts['dj1']
    x, y, kk_arr, wgt_arr = wts['x'], wts['y'], wts['kk_arr'], wts['wgt_arr']

    src_lo = src_batch[:, kk_arr, :, :]   # (nv, nz_t, ny_src, nx_src)
    src_hi = src_batch[:, kk_arr + 1, :, :]

    wy1 = (1.0 - y)[None, None, :, None]  # (1, 1, ny_t, 1)
    wy0 = y[None, None, :, None]
    wx1 = (1.0 - x)[None, None, None, :]  # (1, 1, 1, nx_t)
    wx0 = x[None, None, None, :]
    wz1 = (1.0 - wgt_arr)[None, :, None, None]  # (1, nz_t, 1, 1)
    wz0 = wgt_arr[None, :, None, None]

    slo_j0 = src_lo[:, :, dj0, :];  slo_j1 = src_lo[:, :, dj1, :]
    shi_j0 = src_hi[:, :, dj0, :];  shi_j1 = src_hi[:, :, dj1, :]
    bilin_lo = wy1 * (wx1 * slo_j0[:, :, :, di0] + wx0 * slo_j0[:, :, :, di1]) + \
               wy0 * (wx1 * slo_j1[:, :, :, di0] + wx0 * slo_j1[:, :, :, di1])
    bilin_hi = wy1 * (wx1 * shi_j0[:, :, :, di0] + wx0 * shi_j0[:, :, :, di1]) + \
               wy0 * (wx1 * shi_j1[:, :, :, di0] + wx0 * shi_j1[:, :, :, di1])
    return (wz1 * bilin_lo + wz0 * bilin_hi).astype(np.float32)


# ---------------------------------------------------------------------------
# Pressure profile on model grid  (mirrors setdata.f90 lines 361-369)
# ---------------------------------------------------------------------------

def _compute_pres(
    z: np.ndarray,
    zin: np.ndarray,
    presin: np.ndarray,
) -> tuple:
    """Log-linear interpolation of ERA5 pressures to gSAM model heights.

    Exactly matches setdata.f90 lines 361-369.

    Returns
    -------
    pres  : (nzm,) float64  [hPa]
    pres0 : float            surface pressure [hPa]
    """
    nzm  = len(z)
    nz1  = len(zin)
    pres = np.empty(nzm, dtype=np.float64)
    for k in range(nzm):
        if z[k] <= zin[0]:
            m = 0
        else:
            m = min(int(np.searchsorted(zin, z[k], side='right')) - 1, nz1 - 2)
        pres[k] = np.exp(
            np.log(presin[m]) +
            (np.log(presin[m + 1]) - np.log(presin[m])) /
            (zin[m + 1] - zin[m]) * (z[k] - zin[m])
        )
    pres0 = np.exp(
        np.log(presin[0]) +
        (np.log(presin[1]) - np.log(presin[0])) /
        (zin[1] - zin[0]) * (0.0 - zin[0])
    )
    return pres, float(pres0)


# ---------------------------------------------------------------------------
# Omega → w conversion  (mirrors setdata.f90 lines 487-495)
# ---------------------------------------------------------------------------

def _omega_to_w(
    omega: np.ndarray,
    pres: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    adz: np.ndarray,
    tabs: np.ndarray,
    pres0: float,
    lat: np.ndarray | None = None,
) -> np.ndarray:
    """Convert omega (Pa/s at interfaces) to w (m/s).

    Replicates setdata.f90 lines 464-495 for dolatlon=.true.:

      1. Compute presi via hydrostatic integration using tabs0 (horizontal
         mean temperature weighted by cos(lat)).
      2. rho[k] = (presi[k]-presi[k+1]) / (zi[k+1]-zi[k]) / GGR * 100
      3. rhow[k] = (rho[k-1]*adz[k]+rho[k]*adz[k-1]) / (adz[k]+adz[k-1])
      4. Loop k = nzm-1 down to 1 (Fortran k=nzm..2):
           w[k] = -(adz[k]*omega[k]+adz[k-1]*omega[k-1])
                  / (adz[k]+adz[k-1]) / (rhow[k]*GGR)

    Parameters
    ----------
    omega  : (nzm+1, ny, nx) — pressure velocity [Pa/s]; 0 at k=0 and k=nzm
    pres   : (nzm,)  [hPa]
    z      : (nzm,)  [m]
    zi     : (nzm+1,)[m]
    adz    : (nzm,)
    tabs   : (nzm, ny, nx) [K]
    pres0  : float  [hPa]
    lat    : (ny,) latitudes in degrees — used for cos(lat) weighting of tabs0,
             matching gSAM diagnose.f90. If None, falls back to uniform weights.

    Returns
    -------
    w : (nzm+1, ny, nx) float32  [m/s]
    """
    nzm = len(z)

    # Horizontal mean temperature weighted by cos(lat), matching gSAM's
    # diagnose.f90 which uses wgt(j,k) = mu(j)*ady(j)/sums where mu=cos(lat).
    # For a uniform-spacing lat-lon grid ady is constant and cancels, leaving
    # tabs0[k] = sum_j[cos(lat_j) * mean_i(tabs[k,j,:])] / sum_j[cos(lat_j)].
    tabs_f64 = tabs.astype(np.float64)   # (nzm, ny, nx)
    if lat is not None:
        cos_lat = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))  # (ny,)
        cos_lat_norm = cos_lat / cos_lat.sum()
        zonal_mean = tabs_f64.mean(axis=2)                               # (nzm, ny)
        tabs0 = np.einsum('j,kj->k', cos_lat_norm, zonal_mean)          # (nzm,)
    else:
        tabs0 = tabs_f64.mean(axis=(1, 2))  # (nzm,) — unweighted fallback

    # Hydrostatic presi (setdata.f90 lines 427-436)
    presi    = np.empty(nzm + 1, dtype=np.float64)
    presi[0] = pres0
    presr    = (pres0 / 1000.0) ** (_RGAS / _CP)
    for k in range(nzm):
        prespot  = (1000.0 / pres[k]) ** (_RGAS / _CP)
        t0k      = tabs0[k] * prespot              # potential temperature
        presr    = presr - _GGR / _CP / t0k * (zi[k + 1] - zi[k])
        presi[k + 1] = 1000.0 * presr ** (_CP / _RGAS)

    # rho at cell centres (setdata.f90 line 465)
    rho = (presi[:-1] - presi[1:]) / (zi[1:] - zi[:-1]) / _GGR * 100.0  # kg/m³

    # rhow at interfaces (dolatlon branch, setdata.f90 lines 475-476)
    rhow = np.empty(nzm + 1, dtype=np.float64)
    for k in range(1, nzm):
        rhow[k] = (rho[k - 1] * adz[k] + rho[k] * adz[k - 1]) / (adz[k] + adz[k - 1])
    rhow[0]   = 2.0 * rho[0]       - rhow[1]
    rhow[nzm] = 2.0 * rho[nzm - 1] - rhow[nzm - 1]

    # Omega → w (setdata.f90 lines 488-493):
    # Loop Fortran k=nzm..2 → Python k_py=nzm-1..1 (top-to-bottom)
    w = omega.astype(np.float64)
    for k_py in range(nzm - 1, 0, -1):
        denom   = (adz[k_py] + adz[k_py - 1]) * rhow[k_py] * _GGR
        w[k_py] = -(adz[k_py] * w[k_py] + adz[k_py - 1] * w[k_py - 1]) / denom
    w[0]   = 0.0
    w[nzm] = 0.0
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Hydrostatic pressure recomputation  (setdata.f90 lines 426-436)
# ---------------------------------------------------------------------------

def _hydrostatic_pres_recompute(
    pres0_hpa: float,
    pres_first_hpa: np.ndarray,   # (nzm,) first-guess pres from log-interp [hPa]
    tabs0: np.ndarray,            # (nzm,) cos-lat-weighted horizontal mean TABS [K]
    z: np.ndarray,                # (nzm,) cell-centre heights [m]
    zi: np.ndarray,               # (nzm+1,) interface heights [m]
) -> np.ndarray:                  # (nzm,) cell-centre pressure [hPa]
    """Recompute the reference pressure profile hydrostatically.

    Replicates gSAM setdata.f90 lines 426-436, called after diagnose() sets
    tabs0.  The result is what gSAM stores in pres(k) and then in pp(:,:,k)
    (line 469) before calling micro_init() → cloud().

    Uses the first-guess pres (log-interpolated from ERA5) only for the Exner
    function prespot; the actual profile is integrated from pres0 using the
    mean temperature sounding tabs0.
    """
    nzm = len(z)
    presi = np.empty(nzm + 1, dtype=np.float64)
    presi[0] = float(pres0_hpa)
    presr = (pres0_hpa / 1000.0) ** (_RGAS / _CP)
    for k in range(nzm):
        prespot = (1000.0 / float(pres_first_hpa[k])) ** (_RGAS / _CP)
        t0k = float(tabs0[k]) * prespot
        presr = presr - _GGR / _CP / t0k * (float(zi[k + 1]) - float(zi[k]))
        presi[k + 1] = 1000.0 * presr ** (_CP / _RGAS)
    # Cell-centre pressure: log-linear interpolation between interfaces
    # (setdata.f90 lines 433-435)
    pres_new = np.exp(
        np.log(presi[:-1]) +
        np.log(presi[1:] / presi[:-1]) * (z - zi[:-1]) / (zi[1:] - zi[:-1])
    )
    return pres_new.astype(np.float64)


# ---------------------------------------------------------------------------
# Saturation-adjustment helpers (numpy scalars — for init only)
# ---------------------------------------------------------------------------

_EPS      = np.float64(0.622)
_LV       = np.float64(2.501e6)
_LF       = np.float64(0.337e6)
_LS       = np.float64(2.834e6)
_CP_MICRO = np.float64(1004.64)
_FAC_COND = _LV / _CP_MICRO
_FAC_FUS  = _LF / _CP_MICRO
_FAC_SUB  = _LS / _CP_MICRO


def _np_esatw(T):
    return np.float64(6.1121) * np.exp(np.float64(17.502) * (T - np.float64(273.16))
                                       / (T - np.float64(32.19)))

def _np_esati(T):
    return np.float64(6.1121) * np.exp(np.float64(22.587) * (T - np.float64(273.16))
                                       / (T + np.float64(0.7)))

def _np_qsatw(T, p_mb):
    es = _np_esatw(T)
    return _EPS * es / np.maximum(es, p_mb - es)

def _np_qsati(T, p_mb):
    es = _np_esati(T)
    return _EPS * es / np.maximum(es, p_mb - es)

def _np_dtqsatw(T, p_mb):
    es = _np_esatw(T)
    dte = es * np.float64(17.502) * (np.float64(273.16) - np.float64(32.19)) / (T - np.float64(32.19)) ** 2
    return _EPS * dte / (p_mb - es) * (np.float64(1.0) + es / (p_mb - es))

def _np_dtqsati(T, p_mb):
    es = _np_esati(T)
    dte = es * np.float64(22.587) * (np.float64(273.16) + np.float64(0.7)) / (T + np.float64(0.7)) ** 2
    return _EPS * dte / (p_mb - es) * (np.float64(1.0) + es / (p_mb - es))


# ---------------------------------------------------------------------------
# micro_set phase-partitioning (MICRO_SAM1MOM/microphysics.f90)
# ---------------------------------------------------------------------------

def apply_micro_set(
    TABS: np.ndarray,
    QV: np.ndarray,
    QC: np.ndarray,
    QI: np.ndarray,
    QR: np.ndarray,
    QS: np.ndarray,
    pres: np.ndarray,
) -> None:
    """Apply gSAM micro_set phase-partitioning in-place.

    Replicates MICRO_SAM1MOM/microphysics.f90 subroutine micro_set().
    Called once after init to repartition QC/QI (cloud) and QR/QS (precip)
    based on temperature thresholds, and clip QV >= 0.

    Modifies QV, QC, QI, QR, QS in-place.

    Parameters
    ----------
    TABS : (nzm, ny, nx)  absolute temperature [K]
    QV   : (nzm, ny, nx)  water vapour [kg/kg]
    QC   : (nzm, ny, nx)  cloud liquid [kg/kg]
    QI   : (nzm, ny, nx)  cloud ice [kg/kg]
    QR   : (nzm, ny, nx)  rain [kg/kg]
    QS   : (nzm, ny, nx)  snow [kg/kg]
    pres : (nzm,)         reference pressure [hPa]
    """
    # SAM1MOM thresholds (micro_params.f90)
    tbgmin = np.float32(253.16)
    tbgmax = np.float32(273.16)
    tprmin = np.float32(268.16)
    tprmax = np.float32(283.16)
    a_bg   = np.float32(1.0) / (tbgmax - tbgmin)
    a_pr   = np.float32(1.0) / (tprmax - tprmin)

    np.clip(QV, 0.0, None, out=QV)

    for k in range(TABS.shape[0]):
        # Cloud liquid/ice partition
        qq = np.maximum(np.float32(0.0), QC[k] + QI[k])
        if pres[k] < 50.0:
            qq[:] = np.float32(0.0)
        om_bg = np.clip((TABS[k] - tbgmin) * a_bg, np.float32(0.0), np.float32(1.0))
        QC[k] = qq * om_bg
        QI[k] = qq * (np.float32(1.0) - om_bg)

        # Precip rain/snow partition
        qq_pr = np.maximum(np.float32(0.0), QR[k] + QS[k])
        om_pr = np.clip((TABS[k] - tprmin) * a_pr, np.float32(0.0), np.float32(1.0))
        QR[k] = qq_pr * om_pr
        QS[k] = qq_pr * (np.float32(1.0) - om_pr)


# ---------------------------------------------------------------------------
# Cloud saturation adjustment (cloud.f90 + micro_diagnose)
# ---------------------------------------------------------------------------

def apply_cloud_satadj(
    TABS: np.ndarray,
    QV: np.ndarray,
    QC: np.ndarray,
    QI: np.ndarray,
    QR: np.ndarray,
    QS: np.ndarray,
    pres: np.ndarray,
    z: np.ndarray,
) -> None:
    """Apply gSAM MICRO_SAM1MOM cloud saturation adjustment in-place.

    Replicates setdata.f90 line 500:
        call micro_init()  →  call cloud()  +  call micro_diagnose()
    followed by the global diagnose() call (setdata.f90 line 552) which resets
    TABS from the conserved static energy:
        tabs = t - gamaz + fac_cond*(qcl+qpl) + fac_sub*(qci+qpi)

    MICRO_SAM1MOM cloud.f90 uses q = qv + qcl + qci (total non-precipitating water).
    Cloud ice is included in q (gSAM micro_init line 206) and repartitioned by micro_diagnose().

    tabs_dry = t - gamaz = TABS - fac_cond*(QC+QR) - fac_sub*(QI+QS) is the
    conserved quantity.  Final TABS is diagnosed as:
        TABS_out = tabs_dry + fac_cond*(QC_new+QR_new) + fac_sub*(QI_new+QS_new)

    Modifies TABS, QV, QC, QI, QR, QS in-place.

    At initialisation pp(:,:,k) = pres(k) (setdata.f90:469), so reference
    pressure is used for all qsat calls.
    """
    # gSAM cloud.f90 uses Fortran REAL (float32) throughout the Newton loop.
    F = np.float32
    tbgmin = F(253.16); tbgmax = F(273.16)
    tprmin = F(268.16); tprmax = F(283.16)
    fac_cond = F(_FAC_COND); fac_fus = F(_FAC_FUS); fac_sub = F(_FAC_SUB)
    eps      = F(0.622)
    a_bg = F(1.0) / (tbgmax - tbgmin)
    b_bg = tbgmin * a_bg
    a_pr = F(1.0) / (tprmax - tprmin)
    b_pr = tprmin * a_pr

    # Broadcast pres as (nzm,1,1); keep float32 to match gSAM.
    p_mb = pres.astype(F)[:, None, None]

    qcl = QC.astype(F); qci = QI.astype(F)
    qpl = QR.astype(F); qpi = QS.astype(F)

    # t - gamaz: conserved static energy (post-micro_set condensates, matching gSAM micro_set line 297).
    tabs_dry = TABS.astype(F) - fac_cond * (qcl + qpl) - fac_sub * (qci + qpi)
    q  = np.maximum(F(0.0), QV.astype(F) + qcl + qci)
    qp = np.maximum(F(0.0), qpl + qpi)

    # rh_homo computed once from original TABS (cloud.f90 lines 48-52), float32.
    tabs_orig = TABS.astype(F)
    rh_homo = np.where(
        (tabs_orig < F(235.0)) & (qci < F(1e-8)),
        F(2.583) - tabs_orig / F(207.8),
        F(1.0),
    )

    # Inline float32 saturation functions (mirrors gSAM sat.f90 in REAL precision).
    def _esatw32(T): return F(6.1121) * np.exp(F(17.502) * (T - F(273.16)) / (T - F(32.19)))
    def _esati32(T): return F(6.1121) * np.exp(F(22.587) * (T - F(273.16)) / (T + F(0.7)))
    def _qsatw32(T, p):
        es = _esatw32(T); return eps * es / np.maximum(es, p - es)
    def _qsati32(T, p):
        es = _esati32(T); return eps * es / np.maximum(es, p - es)
    def _dtqsatw32(T, p):
        es = _esatw32(T)
        dte = es * F(17.502) * (F(273.16) - F(32.19)) / (T - F(32.19)) ** 2
        return eps * dte / (p - es) * (F(1.0) + es / (p - es))
    def _dtqsati32(T, p):
        es = _esati32(T)
        dte = es * F(22.587) * (F(273.16) + F(0.7)) / (T + F(0.7)) ** 2
        return eps * dte / (p - es) * (F(1.0) + es / (p - es))

    # cloud.f90 lines 58-89: initial estimate to decide whether condensation is possible.
    # fac1 = fac_cond + (1+bp)*fac_fus;  fac2 = fac_fus*ap  (cloud.f90 lines 32-33)
    fac1 = fac_cond + (F(1.0) + b_pr) * fac_fus
    fac2 = fac_fus * a_pr
    tabs1_est = (tabs_dry + fac1 * qp) / (F(1.0) + fac2 * qp)
    # Phase-dependent qsatt at the estimate (cloud.f90 lines 62-84).
    om_est  = np.clip(a_bg * tabs1_est - b_bg, F(0.0), F(1.0))
    qsatt_est = np.where(
        tabs1_est >= tbgmax, _qsatw32(tabs1_est, p_mb),
        np.where(tabs1_est <= tbgmin,
                 _qsati32(tabs1_est, p_mb) * rh_homo,
                 om_est * _qsatw32(tabs1_est, p_mb) + (F(1.0) - om_est) * _qsati32(tabs1_est, p_mb) * rh_homo))
    # Only run Newton where condensation is possible (cloud.f90 line 89).
    condensing = q > qsatt_est

    # cloud.f90 line 92: "better initial guess — use previous temperature".
    # tabs2 = ERA5 tabs (saved before line 57 overwrites tabs with t-gamaz).
    tabs1     = tabs_orig.copy()          # tabs2 in cloud.f90
    converged = ~condensing               # non-condensing cells already done

    for _ in range(100):
        om      = np.clip(a_bg * tabs1 - b_bg, F(0.0), F(1.0))
        omp     = np.clip(a_pr * tabs1 - b_pr, F(0.0), F(1.0))
        lstarn  = fac_cond + (F(1.0) - om) * fac_fus
        dlstarn = np.where((tabs1 > tbgmin) & (tabs1 < tbgmax), a_bg * fac_fus, F(0.0))
        lstarp  = fac_cond + (F(1.0) - omp) * fac_fus
        dlstarp = np.where((tabs1 > tprmin) & (tabs1 < tprmax), a_pr * fac_fus, F(0.0))
        qsati_v  = _qsati32(tabs1, p_mb)
        # Mixed-phase Newton loop has no rh_homo (cloud.f90 line 115-116).
        qsat     = np.where(tabs1 >= tbgmax, _qsatw32(tabs1, p_mb),
                   np.where(tabs1 <= tbgmin, qsati_v * rh_homo,
                            om * _qsatw32(tabs1, p_mb) + (F(1.0) - om) * qsati_v))
        dqsati_v = _dtqsati32(tabs1, p_mb)
        dqsat    = np.where(tabs1 >= tbgmax, _dtqsatw32(tabs1, p_mb),
                   np.where(tabs1 <= tbgmin, dqsati_v * rh_homo,
                            om * _dtqsatw32(tabs1, p_mb) + (F(1.0) - om) * dqsati_v))
        sat_excess = q - qsat
        fff        = tabs_dry - tabs1 + lstarn * sat_excess + lstarp * qp
        dfff       = dlstarn * sat_excess + dlstarp * qp - lstarn * dqsat - F(1.0)
        dtabs      = -fff / dfff
        tabs1      = np.where(converged, tabs1, tabs1 + dtabs)
        newly_conv = np.abs(dtabs) < F(0.001)
        converged  = converged | newly_conv
        if converged.all():
            break

    om_f     = np.clip(a_bg * tabs1 - b_bg, F(0.0), F(1.0))
    qsati_vf = _qsati32(tabs1, p_mb)
    qsat_f   = np.where(tabs1 >= tbgmax, _qsatw32(tabs1, p_mb),
               np.where(tabs1 <= tbgmin, qsati_vf * rh_homo,
                        om_f * _qsatw32(tabs1, p_mb) + (F(1.0) - om_f) * qsati_vf))
    # Non-condensing cells: qn=0 (cloud.f90 line 144).
    qn = np.where(condensing, np.maximum(F(0.0), q - qsat_f), F(0.0))

    # micro_diagnose: zero out qn <= 0.001*qsatt (microphysics.f90 lines 360-369).
    qn = np.where(qn > F(0.001) * qsat_f, qn, F(0.0))

    qcl_new = qn * om_f
    qci_new = qn * (F(1.0) - om_f)
    omp_f   = np.clip((tabs1 - tprmin) / (tprmax - tprmin), F(0.0), F(1.0))
    qpl_new = qp * omp_f
    qpi_new = qp * (F(1.0) - omp_f)

    # diagnose(): TABS = t - gamaz + fac_cond*(qcl+qpl) + fac_sub*(qci+qpi).
    # Use float64 for the final TABS reconstruction to avoid rounding the energy balance.
    tabs_out = tabs_dry.astype(np.float64) + _FAC_COND * qcl_new + _FAC_COND * qpl_new \
                                           + _FAC_SUB  * qci_new + _FAC_SUB  * qpi_new

    TABS[:] = tabs_out.astype(np.float32)
    QV[:]   = (q - qn).astype(np.float32)
    QC[:]   = qcl_new.astype(np.float32)
    QI[:]   = qci_new.astype(np.float32)
    QR[:]   = qpl_new.astype(np.float32)
    QS[:]   = qpi_new.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level 3D initialisation reader
# ---------------------------------------------------------------------------

def read_init3d(
    init_file: str,
    grid: dict,
    convert_omega: bool = True,
) -> dict:
    """Read gSAM init_era5_*.bin and interpolate to the gSAM model grid.

    Replicates setdata.f90 lines 349-413 (readinit=.true. branch) with the
    exact interpolation from read_field3D.f90.

    Variable order in the file (matches gSAM setdata.f90):
      u, v, omega, tabs, qv, qcl, qci, qpl, qpi

    Parameters
    ----------
    init_file     : path to init_era5_*.bin
    grid          : dict from build_gsam_grid()
    convert_omega : True → convert omega (Pa/s) to w (m/s) via hydrostatic
                    pressure profile (matches setdata.f90 lines 487-495).

    Returns
    -------
    dict with keys (jsam (nz, ny, nx) convention):
      U    (nzm, ny_gl, nx_gl+1)   [m/s]
      V    (nzm, ny_gl+1, nx_gl)   [m/s]
      W    (nzm+1, ny_gl, nx_gl)   [m/s or Pa/s if convert_omega=False]
      TABS (nzm, ny_gl, nx_gl)     [K]
      QV   (nzm, ny_gl, nx_gl)     [kg/kg]
      QC   (nzm, ny_gl, nx_gl)     [kg/kg]
      QI   (nzm, ny_gl, nx_gl)     [kg/kg]
      QR   (nzm, ny_gl, nx_gl)     [kg/kg]  (=qpl)
      QS   (nzm, ny_gl, nx_gl)     [kg/kg]  (=qpi)
      pres (nzm,)                  [hPa]
      pres0                        [hPa]
    """
    lon  = grid['lon']     # (nx_gl,)
    lonu = grid['lonu']    # (nx_gl+1,)
    lat  = grid['lat']     # (ny_gl,)
    latv = grid['latv']    # (ny_gl+1,)
    z    = grid['z']       # (nzm,)
    zi   = grid['zi']      # (nzm+1,)
    adz  = grid['adz']     # (nzm,)
    nzm  = grid['nzm']
    ny_gl = grid['ny_gl']
    nx_gl = grid['nx_gl']

    # ── Read entire file at once, then parse from memory ─────────────────────
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor
    import time as _time

    _t0_io = _time.time()
    with open(init_file, 'rb') as _fdisk:
        _raw = _fdisk.read()
    print(f"  file read in {_time.time()-_t0_io:.1f}s ({len(_raw)/1e9:.2f} GB)", flush=True)

    f = BytesIO(_raw)
    del _raw  # free raw bytes; BytesIO holds its own copy

    # ── File header ──────────────────────────────────────────────────────────
    hdr               = _read_record(f, '<i4', 3)
    nx_src, ny_src, nz_src = int(hdr[0]), int(hdr[1]), int(hdr[2])
    zin    = np.array(_read_record(f, '<f4', nz_src), dtype=np.float64)
    presin = np.array(_read_record(f, '<f4', nz_src), dtype=np.float64)

    print(f"  init: nx={nx_src} ny={ny_src} nz={nz_src}  "
          f"zin=[{zin[0]:.1f}..{zin[-1]:.1f}]m  "
          f"presin=[{presin[0]:.1f}..{presin[-1]:.1f}]hPa")

    pres, pres0 = _compute_pres(z, zin, presin)
    log_pres_at_zi = np.interp(
        zi[1:nzm],
        np.concatenate([[0.0], z]),
        np.log(np.concatenate([[pres0], pres])),
    )
    pres_zi = np.exp(log_pres_at_zi)

    # ── Read all 9 raw fields sequentially from in-memory buffer ─────────────
    _t0_read = _time.time()
    _names = ['u', 'v', 'omega', 'tabs', 'qv', 'qcl', 'qci', 'qpl', 'qpi']
    _raws = []
    for _name in _names:
        print(f"  {_name} ...", flush=True)
        _raws.append(_read_raw_slabs(f, nx_src, ny_src, nz_src))
    f.close()
    print(f"  slabs parsed in {_time.time()-_t0_read:.1f}s", flush=True)

    # ── Compute interpolation weights (one set for each unique target grid) ───
    _wts_u  = _build_interp_weights(*_raws[0][:4], lonu, lat,       z,        pres,    nx_src, ny_src, nz_src)
    _wts_v  = _build_interp_weights(*_raws[1][:4], lon,  latv,      z,        pres,    nx_src, ny_src, nz_src)
    _wts_om = _build_interp_weights(*_raws[2][:4], lon,  lat,       zi[1:nzm], pres_zi, nx_src, ny_src, nz_src)
    _wts_sh = _build_interp_weights(*_raws[3][:4], lon,  lat,       z,        pres,    nx_src, ny_src, nz_src)

    # Stack the 6 shared-target fields into one batch array
    _src_batch = np.stack([_raws[i][4] for i in range(3, 9)], axis=0)  # (6, nz_src, ny_src, nx_src)

    # ── Interpolate in parallel: u / v / omega / batch(tabs…qpi) ─────────────
    _t0_interp = _time.time()
    with ThreadPoolExecutor(max_workers=4) as _pool:
        _fu  = _pool.submit(_apply_interp,       _raws[0][4], _wts_u)
        _fv  = _pool.submit(_apply_interp,       _raws[1][4], _wts_v)
        _fom = _pool.submit(_apply_interp,       _raws[2][4], _wts_om)
        _fsh = _pool.submit(_apply_interp_batch, _src_batch,  _wts_sh)
        u_raw     = _fu.result()
        v_raw     = _fv.result()
        omega_raw = _fom.result()
        _batch    = _fsh.result()   # (6, nz_t, ny_t, nx_t)
    print(f"  interpolation done in {_time.time()-_t0_interp:.1f}s", flush=True)

    tabs, qv, qcl, qci, qpl, qpi = (_batch[i] for i in range(6))
    del _raws, _src_batch, _batch

    # ── Assemble W: 0 at surface and top, omega at interior interfaces ────────
    # gSAM: w(:,:,1)=0, w(:,:,nz)=0, w(:,:,2:nzm)=omega_raw
    # jsam: W[0]=0,      W[nzm]=0,    W[1:nzm]=omega_raw
    W = np.zeros((nzm + 1, ny_gl, nx_gl), dtype=np.float32)
    W[1:nzm] = omega_raw

    if convert_omega:
        print("  omega→w ...", flush=True)
        W = _omega_to_w(W, pres, z, zi, adz, tabs, pres0, lat=lat)

    return dict(
        U=u_raw,
        V=v_raw,
        W=W,
        TABS=tabs,
        QV=qv,
        QC=qcl,
        QI=qci,
        QR=qpl,
        QS=qpi,
        pres=pres,
        pres0=float(pres0),
    )


# ---------------------------------------------------------------------------
# Convenience loader: grid + all 3D fields + key surface fields
# ---------------------------------------------------------------------------

def load_gsam_init(
    gsam_root: str = "/glade/u/home/sabramian/gSAM1.8.7",
    nx_gl: int = 1440,
    ny_gl: int = 720,
    convert_omega: bool = True,
) -> dict:
    """Load the complete gSAM IRMA initial state from the binary files.

    Reads (relative to gsam_root):
      CASES/IRMA/grd
      GRIDS/lat_720_dyvar
      GLOBAL_DATA/BIN_D/init_era5_2017090500_GLOBAL.bin
      GLOBAL_DATA/BIN_D/sst_2017090400-2017091500_1440x720_dyvar_era5.bin
      GLOBAL_DATA/BIN_D/terrain_1440x720_dyvar.bin
      GLOBAL_DATA/BIN_D/landmask_1440x720_dyvar.bin

    Returns
    -------
    dict with:
      grid      : dict (lon, lat, lonu, latv, z, zi, adz, dz, pres, pres0)
      U,V,W     : staggered velocity arrays
      TABS,QV,QC,QI,QR,QS : thermodynamic fields
      sst_days  : (nobs,) float32 — fractional day of each SST snapshot
      sst       : (nobs, ny_gl, nx_gl) float32 — SST [K]
      terrain   : (ny_gl, nx_gl) float32 — terrain height [m]
      landmask  : (ny_gl, nx_gl) int32   — land mask (0=ocean, 1=land)
    """
    import os, pickle, time, hashlib
    bin_d = os.path.join(gsam_root, "GLOBAL_DATA", "BIN_D")

    # ── Disk cache ────────────────────────────────────────────────────────────
    _init_bin = os.path.join(bin_d, "init_era5_2017090500_GLOBAL.bin")
    _cache_dir = os.environ.get("JSAM_INIT_CACHE_DIR",
                                "/glade/work/sabramian/jsam_init_cache")
    try:
        _mtime = f"{os.path.getmtime(_init_bin):.0f}"
        _key   = hashlib.md5(
            f"{_mtime}_{nx_gl}_{ny_gl}_{int(convert_omega)}".encode()
        ).hexdigest()
        _cache_file = os.path.join(_cache_dir, f"gsam_init_{_key}.pkl")
        if False and os.path.exists(_cache_file):  # cache load disabled
            print("Loading gSAM binary initial state from cache ...")
            t0 = time.time()
            with open(_cache_file, 'rb') as _f:
                _result = pickle.load(_f)
            print(f"  Cache loaded in {time.time() - t0:.1f}s  ({_cache_file})")
            return _result
    except Exception as _e:
        _cache_file = None
        print(f"  [init cache] skipped: {_e}")

    print("Building gSAM grid ...")
    grid = build_gsam_grid(gsam_root, nx_gl=nx_gl, ny_gl=ny_gl)

    init_file = os.path.join(bin_d, "init_era5_2017090500_GLOBAL.bin")
    print(f"Reading 3D init: {init_file}")
    fields = read_init3d(init_file, grid, convert_omega=convert_omega)

    # Recompute pres hydrostatically from mean-T sounding, matching gSAM
    # setdata.f90 lines 426-436 (called after diagnose(), before micro_init()).
    # This gives the same pres that gSAM's cloud() sees via pp(:,:,k)=pres(k).
    _cos_lat = np.cos(np.deg2rad(np.asarray(grid['lat'], dtype=np.float64)))
    _cos_lat_norm = _cos_lat / _cos_lat.sum()
    _tabs0 = np.einsum('j,kj->k', _cos_lat_norm,
                       fields['TABS'].astype(np.float64).mean(axis=2))
    _pres_hydro = _hydrostatic_pres_recompute(
        float(fields['pres0']), fields['pres'].astype(np.float64),
        _tabs0, np.asarray(grid['z'], np.float64),
        np.asarray(grid['zi'], np.float64),
    )
    print(f"  pres hydrostatic: [{_pres_hydro[0]:.2f}..{_pres_hydro[-1]:.3f}] hPa"
          f"  (log-interp was [{fields['pres'][0]:.2f}..{fields['pres'][-1]:.3f}])")
    fields['pres'] = _pres_hydro.astype(np.float32)

    print("  micro_set: repartitioning QC/QI/QR/QS, clipping QV ...")
    apply_micro_set(
        fields['TABS'], fields['QV'], fields['QC'],
        fields['QI'],   fields['QR'], fields['QS'],
        fields['pres'],
    )

    # gSAM micro_set() sets t = tabs + gamaz - fac_cond*(qcl_ms+qpl_ms) - fac_sub*(qci_ms+qpi_ms)
    # using the POST-repartitioned condensates.  diagnose() then sets:
    #   tabs = t - gamaz + fac_cond*(qcl_ms+qpl_ms) + fac_sub*(qci_ms+qpi_ms) = TABS_ERA5
    # micro_set does not modify TABS, so fields['TABS'] is still TABS_ERA5 — no reassignment needed.
    # _t_minus_gamaz must use post-micro_set condensates to match gSAM's t.
    _t_minus_gamaz = (fields['TABS'].astype(np.float64)
                      - _FAC_COND * (fields['QC'].astype(np.float64) + fields['QR'].astype(np.float64))
                      - _FAC_SUB  * (fields['QI'].astype(np.float64) + fields['QS'].astype(np.float64)))

    # Load terrain before satadj so we can mask below-terrain cells.
    terrain_file = os.path.join(bin_d, "terrain_1440x720_dyvar.bin")
    print(f"Reading terrain: {terrain_file}")
    _, terrain_all = read_readsurface_field(terrain_file)
    terrain = terrain_all[0]   # (ny, nx)  float32  m

    # gSAM micro_init() (readinit branch, nrestart=0):
    #   q = qv+qcl+qci;  qp = qpl+qpi
    #   call cloud()          ← Newton satadj, skipping k < k_terra
    #   call micro_diagnose() ← repartitions qn→qcl/qci with 0.001*qsatt threshold
    #
    # Replicate this exactly: run satadj, then apply terrain masking (cloud.f90
    # skips below-terrain cells via `if(k.lt.k_terra(i,j)) cycle`; micro_diagnose
    # then zeros qcl/qci there since qn==0).
    print("  cloud satadj: Newton saturation adjustment + micro_diagnose ...")

    # Save state before satadj for terrain-cell restoration.
    _TABS_before = fields['TABS'].copy()          # (nzm, ny, nx)
    _q_total     = (fields['QV'] + fields['QC'] + fields['QI']).copy()  # qv+qcl+qci total water
    _QR_before   = fields['QR'].copy()
    _QS_before   = fields['QS'].copy()

    apply_cloud_satadj(
        fields['TABS'], fields['QV'], fields['QC'],
        fields['QI'],   fields['QR'], fields['QS'],
        fields['pres'], grid['z'],
    )

    # Terrain masking: restore below-terrain cells to pre-satadj state, matching
    # gSAM cloud.f90 `if(k.lt.k_terra) cycle` + micro_diagnose qn==0 → qcl=qci=0, qv=q.
    _z = np.asarray(grid['z'], dtype=np.float32)
    _above = (_z[:, None, None] > terrain[None, :, :])   # True = above terrain
    # gSAM setdata.f90 line 552: after micro_diagnose zeroes qcl/qci for terrain cells,
    # diagnose() sets tabs = tabs_dry + fac_cond*qpl + fac_sub*qpi  (no cloud terms).
    # _TABS_before includes fac_cond*QC_ms + fac_sub*QI_ms which is wrong for terrain cells.
    _TABS_terrain = (
        _t_minus_gamaz
        + _FAC_COND * _QR_before.astype(np.float64)
        + _FAC_SUB  * _QS_before.astype(np.float64)
    ).astype(np.float32)
    fields['TABS'] = np.where(_above, fields['TABS'], _TABS_terrain)
    fields['QV']   = np.where(_above, fields['QV'],   _q_total)
    fields['QC']   = np.where(_above, fields['QC'],   np.float32(0.0))
    fields['QI']   = np.where(_above, fields['QI'],   np.float32(0.0))
    fields['QR']   = np.where(_above, fields['QR'],   _QR_before)
    fields['QS']   = np.where(_above, fields['QS'],   _QS_before)

    sst_file = os.path.join(
        bin_d, "sst_2017090400-2017091500_1440x720_dyvar_era5.bin"
    )
    print(f"Reading SST: {sst_file}")
    sst_days, sst = read_readsurface_field(sst_file)

    lm_file = os.path.join(bin_d, "landmask_1440x720_dyvar.bin")
    print(f"Reading landmask: {lm_file}")
    landmask = read_landmask(lm_file)

    out = dict(grid=grid)
    out.update(fields)
    out['sst_days'] = sst_days
    out['sst']      = sst
    out['terrain']  = terrain
    out['landmask'] = landmask
    # Also attach pres/pres0 on the grid dict for downstream convenience
    grid['pres']  = fields['pres']
    grid['pres0'] = fields['pres0']

    if _cache_file is not None:
        try:
            os.makedirs(_cache_dir, exist_ok=True)
            with open(_cache_file, 'wb') as _f:
                pickle.dump(out, _f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Init state cached → {_cache_file}")
        except Exception as _e:
            print(f"  [init cache] write failed: {_e}")

    return out


# ---------------------------------------------------------------------------
# Terrain mask construction — mirrors gSAM terrain.f90 setterrain()
# ---------------------------------------------------------------------------

def build_terrain_masks(
    elevation: np.ndarray,
    z: np.ndarray,
) -> dict:
    """Build 3-D terrain masks (terrau, terrav, terraw) from a 2-D elevation
    field and the 1-D cell-centre height vector.

    Mirrors gSAM terrain.f90 setterrain() lines 369-707, restricted to the
    doterrain=.true., readterr=.true. global-run code path (no wall BCs, no
    pit/peak repair — those are for small-domain LES cases).

    Parameters
    ----------
    elevation : (ny, nx) float  — terrain height above sea-level [m]
    z         : (nzm,)   float  — cell-centre heights [m]

    Returns
    -------
    dict with numpy arrays:
      terra  : (nzm, ny,   nx)   — 1 above terrain, 0 inside terrain
      terrau : (nzm, ny,   nx+1) — min(terra[k,j,i-1], terra[k,j,i]), periodic x
      terrav : (nzm, ny+1, nx)   — min(terra[k,j-1,i], terra[k,j,i]), clamped y
      terraw : (nzm+1, ny, nx)   — 0 at both faces of any terrain cell, 1 elsewhere
    """
    ny, nx = elevation.shape
    nzm = len(z)

    # ── terra: scalar-cell mask ──────────────────────────────────────────────
    # gSAM terrain.f90 lines 369-378:
    #   if(elevation(i,j) >= z(k))  terra(i,j,k) = 0.
    # Convention: terra defaults to 1 (above terrain).
    terra = np.ones((nzm, ny, nx), dtype=np.float32)
    for k in range(nzm):
        terra[k] = np.where(elevation >= z[k], 0.0, 1.0)

    # ── terrau: U-face mask (min of the two neighbouring scalar cells) ───────
    # gSAM terrain.f90 lines 699-707 (1-indexed):
    #   terrau(i,j,k) = min(terra(i-1,j,k), terra(i,j,k))
    # jsam U has nx+1 x-faces (periodic):
    #   face i ∈ [0, nx] lies between scalar column (i-1)%nx and i%nx.
    terrau = np.ones((nzm, ny, nx + 1), dtype=np.float32)
    # terra shifted right by 1 in x (periodic): terra[:, :, (i-1)%nx]
    terra_xm1 = np.roll(terra, shift=1, axis=2)  # terra[k,j,(i-1)%nx]
    # Faces 0..nx-1
    terrau[:, :, :nx] = np.minimum(terra_xm1, terra)
    # Face nx is the periodic wrap-around: between cell nx-1 and cell 0
    terrau[:, :, nx] = terrau[:, :, 0]

    # ── terrav: V-face mask (min of the two neighbouring scalar cells) ───────
    # gSAM terrain.f90 lines 699-707 (1-indexed):
    #   terrav(i,j,k) = min(terra(i,j-YES3D,k), terra(i,j,k))
    # jsam V has ny+1 y-faces:
    #   face j ∈ [0, ny] lies between scalar row j-1 and j.
    # At the poles (j=0 and j=ny), gSAM uses 0-latidude boundary conditions;
    # for the global grid we clamp: pole face inherits the adjacent interior row.
    terrav = np.ones((nzm, ny + 1, nx), dtype=np.float32)
    # Interior faces 1..ny-1: min(terra[k,j-1,i], terra[k,j,i])
    terrav[:, 1:ny, :] = np.minimum(terra[:, :-1, :], terra[:, 1:, :])
    # South-pole face (j=0): clamp to southernmost row
    terrav[:, 0, :] = terra[:, 0, :]
    # North-pole face (j=ny): clamp to northernmost row
    terrav[:, ny, :] = terra[:, -1, :]

    # ── terraw: W-face mask ──────────────────────────────────────────────────
    # gSAM terrain.f90 lines 688-697 (1-indexed):
    #   if(terra(i,j,k) < 1):
    #     terraw(i,j,k)   = 0   ! bottom face of scalar cell k
    #     terraw(i,j,k+1) = 0   ! top    face of scalar cell k
    # jsam W has nzm+1 faces (0..nzm); W[k] is between scalar cells k-1 and k.
    # terra[k] < 1  ⟹  terraw[k]=0  AND  terraw[k+1]=0.
    terraw = np.ones((nzm + 1, ny, nx), dtype=np.float32)
    below_terrain = terra < 1.0   # (nzm, ny, nx) bool
    # Zero the bottom face (k) and top face (k+1) for every terrain cell.
    # Use np.where on sliced views to avoid chained-indexing writeback issues.
    terraw[:-1] = np.where(below_terrain, 0.0, terraw[:-1])
    terraw[1:]  = np.where(below_terrain, 0.0, terraw[1:])

    return dict(terra=terra, terrau=terrau, terrav=terrav, terraw=terraw)


# ---------------------------------------------------------------------------
# Horizontal regridding utility (kept for backward compatibility)
# ---------------------------------------------------------------------------

def interp_horiz_dyvar(field, src_lat, src_lon, tgt_lat, tgt_lon, method='bilinear'):
    """Bilinear/nearest interpolation from a non-uniform-lat, uniform-lon
    source grid to any target lat/lon grid.  Longitude is treated as periodic.

    Parameters
    ----------
    field   : (..., ny_src, nx_src)
    src_lat : (ny_src,) strictly monotone
    src_lon : (nx_src,) strictly monotone, uniform degrees
    tgt_lat : (ny_tgt,)
    tgt_lon : (nx_tgt,)
    method  : 'bilinear' or 'nearest'

    Returns
    -------
    (..., ny_tgt, nx_tgt)
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
    ny_src  = src_lat.size
    nx_src  = src_lon.size

    lat_desc     = src_lat[0] > src_lat[-1]
    src_lat_asc  = src_lat[::-1] if lat_desc else src_lat

    jy = np.searchsorted(src_lat_asc, tgt_lat) - 1
    jy = np.clip(jy, 0, ny_src - 2)
    y0 = src_lat_asc[jy]
    y1 = src_lat_asc[jy + 1]
    fy = np.clip((tgt_lat - y0) / (y1 - y0), 0.0, 1.0)
    if lat_desc:
        ja = (ny_src - 1) - jy
        jb = (ny_src - 1) - (jy + 1)
        wa, wb = fy, 1.0 - fy
    else:
        ja, jb = jy + 1, jy
        wa, wb = fy, 1.0 - fy

    dlon    = (src_lon[-1] - src_lon[0]) / (nx_src - 1)
    fx_full = ((tgt_lon - src_lon[0]) % 360.0) / dlon
    ix      = np.floor(fx_full).astype(np.int64) % nx_src
    fx      = fx_full - np.floor(fx_full)
    ix1     = (ix + 1) % nx_src

    if method == 'nearest':
        j_near = np.where(wa >= 0.5, ja, jb)
        i_near = np.where(fx >= 0.5, ix1, ix)
        return field[..., j_near[:, None], i_near[None, :]]

    if method != 'bilinear':
        raise ValueError(f"unknown method: {method!r}")

    Ja  = ja[:, None];  Jb  = jb[:, None]
    Ix0 = ix[None, :];  Ix1 = ix1[None, :]
    wa_b = wa[:, None]; wb_b = wb[:, None]
    fx_b = fx[None, :]

    fld = field.astype(np.float64, copy=False)
    out = (wb_b * ((1 - fx_b) * fld[..., Jb, Ix0] + fx_b * fld[..., Jb, Ix1]) +
           wa_b * ((1 - fx_b) * fld[..., Ja, Ix0] + fx_b * fld[..., Ja, Ix1]))
    return out.astype(np.float32 if field.dtype == np.float32 else out.dtype)
