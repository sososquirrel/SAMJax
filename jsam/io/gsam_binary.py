"""
Readers for gSAM Fortran-unformatted binary files in GLOBAL_DATA/BIN_D/.

All outputs are numpy arrays at the native gSAM grid (1440 lon x 720 lat,
lat axis first in returned arrays, i.e. shape (ny_gl, nx_gl)). Use
`interp_horiz_dyvar` to regrid onto the jsam LatLonGrid.

The Fortran files are unformatted sequential: each record has a 4-byte
little-endian length marker before and after the payload. gSAM was compiled
on little-endian x86_64 so all dtypes here are explicitly little-endian.
"""
import numpy as np


def _read_record(f, dtype, count):
    """Read a single Fortran unformatted record: 4-byte len, payload, 4-byte len."""
    rl1 = np.fromfile(f, dtype='<i4', count=1)
    if rl1.size == 0:
        raise EOFError("unexpected EOF reading record marker")
    rl1 = int(rl1[0])
    data = np.fromfile(f, dtype=dtype, count=count)
    rl2 = int(np.fromfile(f, dtype='<i4', count=1)[0])
    assert rl1 == rl2, f"record marker mismatch: {rl1} vs {rl2}"
    assert rl1 == count * np.dtype(dtype).itemsize, (
        f"rec len mismatch: header says {rl1}, expected {count * np.dtype(dtype).itemsize}"
    )
    return data


def read_readsurface_field(path, field_dtype=np.float32):
    """Read a file in readsurface.f90 format.

    Returns (days, field) where
        days  is shape (nobs,) float32
        field is shape (nobs, ny_gl, nx_gl) with dtype=field_dtype

    The underlying storage is always real32; for landtype/landmask callers
    pass field_dtype=np.float32 here and then cast via np.rint.astype(int32).
    """
    with open(path, 'rb') as f:
        nobs = int(_read_record(f, '<i4', 1)[0])
        nx1 = int(_read_record(f, '<i4', 1)[0])
        ny1 = int(_read_record(f, '<i4', 1)[0])
        days = _read_record(f, '<f4', nobs).astype(np.float32)
        field = np.empty((nobs, ny1, nx1), dtype=np.float32)
        for i in range(nobs):
            raw = _read_record(f, '<f4', nx1 * ny1)
            # Fortran stores as (nx_gl, ny_gl) column-major => reshape (ny, nx)
            field[i] = raw.reshape(ny1, nx1)
    if field_dtype != np.float32:
        field = field.astype(field_dtype)
    return days, field


def read_landtype(path):
    """Read landtype_*.bin -> (ny_gl, nx_gl) int32."""
    _, fld = read_readsurface_field(path)
    return np.rint(fld[0]).astype(np.int32)


def read_landmask(path):
    """Read landmask_*.bin -> (ny_gl, nx_gl) int32."""
    _, fld = read_readsurface_field(path)
    return np.rint(fld[0]).astype(np.int32)


def read_lai_monthly(path):
    """Read lai_*.bin -> (12, ny_gl, nx_gl) float32 (monthly climatology)."""
    _, fld = read_readsurface_field(path)
    assert fld.shape[0] == 12, f"expected 12 monthly LAI slices, got {fld.shape[0]}"
    return fld


def read_snow(path):
    """Read snow_*.bin -> (ny_gl, nx_gl) float32 (snow depth [m])."""
    _, fld = read_readsurface_field(path)
    return fld[0]


def read_snowt(path):
    """Read snowt_*.bin -> (ny_gl, nx_gl) float32 (snow surface temperature [K])."""
    _, fld = read_readsurface_field(path)
    return fld[0]


def read_soil_sand_clay(path):
    """Read soil_*.bin -> (SAND, CLAY) each (ny_gl, nx_gl) float32 (% content).

    Fortran file order is CLAY then SAND (see init_soil_tw); we return SAND
    first as a matter of convention.
    """
    with open(path, 'rb') as f:
        nx1 = int(_read_record(f, '<i4', 1)[0])
        ny1 = int(_read_record(f, '<i4', 1)[0])
        clay = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
        sand = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
    return sand, clay


def read_soil_init(path):
    """Read soil_init_*.bin.

    Returns
        zsoil  (nsoil1,) float32 - layer centre depths
        soilt  (nsoil1, ny_gl, nx_gl) float32 - soil temperature [K]
        soilw  (nsoil1, ny_gl, nx_gl) float32 - soil wetness / vol. content
    """
    with open(path, 'rb') as f:
        nsoil1 = int(_read_record(f, '<i4', 1)[0])
        nx1 = int(_read_record(f, '<i4', 1)[0])
        ny1 = int(_read_record(f, '<i4', 1)[0])
        zsoil = _read_record(f, '<f4', nsoil1).astype(np.float32)
        soilt = np.empty((nsoil1, ny1, nx1), dtype=np.float32)
        soilw = np.empty((nsoil1, ny1, nx1), dtype=np.float32)
        for k in range(nsoil1):
            soilt[k] = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
        for k in range(nsoil1):
            soilw[k] = _read_record(f, '<f4', nx1 * ny1).reshape(ny1, nx1)
    return zsoil, soilt, soilw


def read_lat_dyvar(path="/glade/u/home/sabramian/gSAM1.8.7/GRIDS/lat_720_dyvar"):
    """Read the 720-value ASCII lat array used by gSAM dyvar grid.

    The file has 3 whitespace-separated columns: latitude, index, weight.
    We return the first column as a float64 ndarray of shape (720,).
    """
    data = np.loadtxt(path)
    if data.ndim == 2:
        lat = data[:, 0]
    else:
        lat = data
    return lat.astype(np.float64)


# ---------------------------------------------------------------------------
# Horizontal regridding
# ---------------------------------------------------------------------------


def interp_horiz_dyvar(field, src_lat, src_lon, tgt_lat, tgt_lon, method='bilinear'):
    """Bilinear/nearest interpolation from a non-uniform-lat, uniform-lon
    source grid to any target lat/lon grid. Longitude axis is treated as
    periodic.

    Parameters
    ----------
    field : ndarray, shape (ny_src, nx_src) or (..., ny_src, nx_src)
    src_lat : (ny_src,) strictly monotone, may be non-uniform
    src_lon : (nx_src,) strictly monotone, assumed uniform in degrees
    tgt_lat : (ny_tgt,)
    tgt_lon : (nx_tgt,)
    method  : 'bilinear' or 'nearest'

    Returns
    -------
    ndarray of shape (..., ny_tgt, nx_tgt), matching field.dtype for 'nearest'
    and float32/float64 (upcast) for 'bilinear'.
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
    ny_src = src_lat.size
    nx_src = src_lon.size

    # Determine latitude direction so searchsorted works on an ascending array.
    lat_desc = src_lat[0] > src_lat[-1]
    if lat_desc:
        src_lat_asc = src_lat[::-1]
    else:
        src_lat_asc = src_lat

    # ---- latitude fractional indices into src_lat_asc ----
    jy = np.searchsorted(src_lat_asc, tgt_lat) - 1
    jy = np.clip(jy, 0, ny_src - 2)
    y0 = src_lat_asc[jy]
    y1 = src_lat_asc[jy + 1]
    fy = (tgt_lat - y0) / (y1 - y0)
    fy = np.clip(fy, 0.0, 1.0)  # clamp outside src range
    if lat_desc:
        # map ascending index back to original descending index
        j0_orig = (ny_src - 1) - jy          # corresponds to y0 (small lat)
        j1_orig = (ny_src - 1) - (jy + 1)    # corresponds to y1 (large lat)
        # We want the pair (j_low, j_high) and fy w.r.t. j_low in original ordering.
        # In ascending: value at jy is small lat, at jy+1 is larger lat, fy in [0,1].
        # In descending indexing, the larger lat sits at the SMALLER original index.
        # Keep two indices and a weight for the second:
        ja = j1_orig  # index whose lat is y1 (bigger lat)
        jb = j0_orig  # index whose lat is y0 (smaller lat)
        # target lat interpolation: val = (1-fy)*y0 + fy*y1  (ascending sense)
        # In the original array: val = fy*field[ja] + (1-fy)*field[jb]
        wa = fy
        wb = 1.0 - fy
    else:
        ja = jy + 1
        jb = jy
        wa = fy
        wb = 1.0 - fy

    # ---- longitude fractional indices (periodic, uniform) ----
    dlon = (src_lon[-1] - src_lon[0]) / (nx_src - 1)
    lon0 = src_lon[0]
    # fractional index in the periodic source grid
    fx_full = ((tgt_lon - lon0) % 360.0) / dlon
    ix = np.floor(fx_full).astype(np.int64) % nx_src
    fx = fx_full - np.floor(fx_full)
    ix1 = (ix + 1) % nx_src

    leading_shape = field.shape[:-2]
    ny_tgt = tgt_lat.size
    nx_tgt = tgt_lon.size

    if method == 'nearest':
        # Snap to nearest lat/lon corner.
        j_near = np.where(wa >= 0.5, ja, jb)         # (ny_tgt,)
        i_near = np.where(fx >= 0.5, ix1, ix)        # (nx_tgt,)
        J = j_near[:, None]
        I = i_near[None, :]
        out = field[..., J, I]
        return out

    if method != 'bilinear':
        raise ValueError(f"unknown method: {method!r}")

    # Bilinear: gather the four corners with numpy advanced indexing.
    # Build 2D index arrays of shape (ny_tgt, nx_tgt).
    Ja = ja[:, None]
    Jb = jb[:, None]
    Ix0 = ix[None, :]
    Ix1 = ix1[None, :]

    # Weights
    wa_b = wa[:, None]            # weight for "upper" lat row (ja)
    wb_b = wb[:, None]            # weight for "lower" lat row (jb)
    fx_b = fx[None, :]
    one_minus_fx = 1.0 - fx_b

    f = field.astype(np.float64, copy=False)
    v00 = f[..., Jb, Ix0]  # lat jb, lon ix
    v01 = f[..., Jb, Ix1]  # lat jb, lon ix+1
    v10 = f[..., Ja, Ix0]  # lat ja, lon ix
    v11 = f[..., Ja, Ix1]  # lat ja, lon ix+1

    out = (wb_b * (one_minus_fx * v00 + fx_b * v01)
           + wa_b * (one_minus_fx * v10 + fx_b * v11))
    return out.astype(np.float32 if field.dtype == np.float32 else out.dtype)
