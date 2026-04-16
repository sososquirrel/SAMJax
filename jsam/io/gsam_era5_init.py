"""
gsam_era5_init.py — gSAM-exact ERA5 initialisation from raw netCDF.

Replicates gSAM setdata.f90 ERA5 readinit pipeline reading directly from
ERA5 netCDF on the NCAR RDA (d633000), instead of the pre-processed binary
``init_era5_YYYYMMDDHHH_GLOBAL.bin``.

Four key differences vs :func:`~jsam.io.era5.era5_state`:

1. **Height-based linear vertical interpolation** — matches gSAM
   ``read_field3D.f90`` (``zr(1) != -999`` branch):
   ``wgt = max(0, min(1, (z_k - zr[kk]) / (zr[kk+1] - zr[kk])))``.
   The source height coordinate is the GLOBAL area-weighted mean of ERA5
   geopotential / g, matching the ``zin`` column pre-computed when gSAM's
   init binary was created.

2. **Correct omega → w conversion order** — gSAM interpolates omega to
   interface heights ``zi(2:nzm)`` *first*, then applies the adz-weighted
   formula (setdata.f90:487-495):
   ``w(k) = -(adz(k)*omega(k) + adz(k-1)*omega(k-1)) / (adz(k)+adz(k-1)) / (rhow(k)*ggr)``
   looping ``k = nzm .. 2`` (descending).  Because each update uses the
   *original* omega values at k and k-1 (not yet overwritten), the loop is
   equivalent to a single vectorised pass and is implemented as such.

3. **micro_set post-processing** — replicates
   ``SRC/MICRO_SAM1MOM/microphysics.f90:265-302``.  QCL/QCI are repartitioned
   by temperature; QV is clamped to ≥ 0.  (Precipitation arrays QPL/QPI/QPG
   are zero for ERA5 init and are not modified.)

4. **Direct stagger interpolation** — U is interpolated to the stagger
   longitude positions ``lonu = lon + dlon/2`` (periodic); V is interpolated
   to the stagger latitude positions ``lat_v`` (midpoints, ±90 at poles).
   This matches gSAM's ``xx_u/yy_v`` grid vectors rather than the
   mass-average approximation used in :func:`~jsam.io.era5.era5_state`.

Usage
-----
Replace the ``era5_state`` call inside :func:`~jsam.io.era5.era5_init` with::

    from jsam.io.gsam_era5_init import era5_state_gsam
    state = era5_state_gsam(grid, metric, dt, rda_root)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import jax.numpy as jnp

from jsam.core.grid.latlon import LatLonGrid
from jsam.core.state import ModelState
from jsam.io.era5 import (
    RDA_ROOT,
    _G,
    _read_pl_snapshot,
    era5_latlon,
    interp_horiz,
)

# ---------------------------------------------------------------------------
# gSAM physical constants (consts.f90)
# ---------------------------------------------------------------------------
_GGR      = 9.79764          # m/s²       gravity
_CP       = 1004.64          # J/(kg K)   specific heat of dry air
_LCOND    = 2.501e6          # J/kg       latent heat of condensation
_LSUB     = 2.834e6          # J/kg       latent heat of sublimation
_FAC_COND = _LCOND / _CP     # = 2490.7…  K / (kg/kg)
_FAC_SUB  = _LSUB  / _CP     # = 2820.0…  K / (kg/kg)

# ---------------------------------------------------------------------------
# micro_set temperature thresholds (micro_params.f90)
# ---------------------------------------------------------------------------
_TBGMIN = 253.16   # K   min T for cloud liquid (pure ice below)
_TBGMAX = 273.16   # K   max T for cloud liquid (pure liquid above)
_TPRMIN = 268.16   # K   min T for rain
_TPRMAX = 283.16   # K   max T for snow/graupel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _global_mean_height(
    Z_tb_ns: np.ndarray,
    era5_lat_ns: np.ndarray,
) -> np.ndarray:
    """
    Compute global area-weighted mean geopotential height at each ERA5 level.

    This reproduces the ``zin`` column stored in gSAM's pre-processed binary.

    Parameters
    ----------
    Z_tb_ns   : (37, 721, 1440) m²/s²  ERA5 geopotential, top→bottom, N→S
    era5_lat_ns : (721,) degrees N→S

    Returns
    -------
    zr : (37,) m  global mean height per pressure level, **top→bottom** order
         (zr[0] ≈ 47 500 m = 1 hPa level; zr[36] ≈ 97 m = 1000 hPa level)
    """
    h = Z_tb_ns / _G                                # (37, 721, 1440)  metres
    cos_w = np.cos(np.deg2rad(era5_lat_ns))         # (721,) N→S weights
    w3d   = cos_w[None, :, None]                    # (1, 721, 1)
    total_w = (w3d * np.ones_like(h)).sum(axis=(1, 2))   # (37,)
    zr = (h * w3d).sum(axis=(1, 2)) / total_w           # (37,) top→bottom
    return zr


def _interp_height_3d(
    slabs_bt: np.ndarray,   # (nz_src, ny_src, nx_src)  ascending height order
    zr_bt:    np.ndarray,   # (nz_src,) ascending mean heights of source slabs
    z_tgt:    np.ndarray,   # (nz_tgt,) target heights (ascending)
) -> np.ndarray:
    """
    Height-based linear vertical interpolation matching gSAM ``read_field3D``.

    Implements (per model level k):

        kk  = last source level with  zr[kk] <= z_tgt[k]
        wgt = max(0, min(1, (z_tgt[k] - zr[kk]) / (zr[kk+1] - zr[kk])))
        out[k] = (1-wgt) * slabs[kk] + wgt * slabs[kk+1]

    Extrapolation is by clamping (wgt = 0 below the bottom slab, wgt = 1
    above the top slab), matching gSAM's ``max(0, min(1, …))`` guard.

    Parameters
    ----------
    slabs_bt : (nz_src, ny_src, nx_src)  source field, *bottom-first* (index 0 =
               surface ~1000 hPa, index nz_src-1 = top ~1 hPa)
    zr_bt    : (nz_src,) ascending mean heights of the source levels [m]
    z_tgt    : (nz_tgt,) target heights [m], must be ascending

    Returns
    -------
    out : (nz_tgt, ny_src, nx_src) float64
    """
    nz_src = len(zr_bt)

    # Binary-search for lower-bracket index in zr_bt for each target level.
    # searchsorted(..., 'right') - 1 gives the last index with zr_bt[kk] <= z_tgt[k].
    kk = np.searchsorted(zr_bt, z_tgt, side='right') - 1
    kk = np.clip(kk, 0, nz_src - 2)   # ensure valid [kk, kk+1] pair

    dz  = zr_bt[kk + 1] - zr_bt[kk]   # (nz_tgt,)
    wgt = np.where(dz > 0.0,
                   (z_tgt - zr_bt[kk]) / dz,
                   0.0)
    wgt = np.clip(wgt, 0.0, 1.0)       # (nz_tgt,)

    # Fancy indexing → (nz_tgt, ny_src, nx_src)
    lo = slabs_bt[kk]           # lower bracket slab
    hi = slabs_bt[kk + 1]       # upper bracket slab
    w  = wgt[:, None, None]     # broadcast weight

    return (1.0 - w) * lo + w * hi


def _micro_set(
    tabs:     np.ndarray,   # (nzm, ny, nx) K        absolute temperature
    qcl:      np.ndarray,   # (nzm, ny, nx) kg/kg    cloud liquid (input)
    qci:      np.ndarray,   # (nzm, ny, nx) kg/kg    cloud ice   (input)
    qv:       np.ndarray,   # (nzm, ny, nx) kg/kg    water vapour (input)
    pres_hpa: np.ndarray,   # (nzm,) hPa             model pressure levels
    gamaz:    np.ndarray,   # (nzm,) K               ggr*z/cp
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicate gSAM micro_set (MICRO_SAM1MOM/microphysics.f90:265-302).

    Repartitions QCL/QCI based on temperature; clamps QV ≥ 0.
    Precipitation arrays (QPL/QPI/QPG) are all zero for ERA5 init so their
    partitioning is a no-op and is omitted.

    Returns
    -------
    (qcl_new, qci_new, qv_new, t_static)
    where ``t_static`` is the liquid-water static energy:
        t = tabs + gamaz - fac_cond*(qcl+qpl) - fac_sub*(qci+qpi)
    """
    a_bg = 1.0 / (_TBGMAX - _TBGMIN)    # 1/20

    qv_new = np.maximum(0.0, qv)

    # Liquid fraction (0 = all ice, 1 = all liquid)
    om = np.clip((tabs - _TBGMIN) * a_bg, 0.0, 1.0)   # (nzm, ny, nx)

    # Total condensate (ERA5 CLWC + CIWC)
    qq = np.maximum(0.0, qcl + qci)                    # (nzm, ny, nx)

    # Above 50 hPa: force qq = 0 (gSAM: "no cloud above 50 mb")
    above_50 = (pres_hpa < 50.0)[:, None, None]        # (nzm, 1, 1) broadcast
    qq = np.where(above_50, 0.0, qq)

    qcl_new = qq * om
    qci_new = qq * (1.0 - om)

    # Liquid-water static energy t = tabs + ggr/cp*z - fac_cond*qcl - fac_sub*qci
    # (qpl = qpi = 0, so only cloud terms appear)
    t_static = tabs + gamaz[:, None, None] - _FAC_COND * qcl_new - _FAC_SUB * qci_new

    return qcl_new, qci_new, qv_new, t_static


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def era5_state_gsam(
    grid:     LatLonGrid,
    metric:   dict,
    dt:       datetime,
    rda_root: str = RDA_ROOT,
) -> ModelState:
    """
    Initialise a :class:`~jsam.core.state.ModelState` from ERA5 netCDF,
    replicating the gSAM setdata.f90 ERA5 readinit pipeline exactly.

    Parameters
    ----------
    grid     : :class:`~jsam.core.grid.latlon.LatLonGrid` — must have ``zi``
               of length ``nz+1`` (not truncated to nzm).
    metric   : dict from :func:`~jsam.core.dynamics.pressure.build_metric`
               — provides model pressure levels (key ``"pres"``, Pa).
    dt       : ERA5 analysis datetime (UTC).
    rda_root : path to NCAR RDA ERA5 archive (d633000).

    Returns
    -------
    :class:`~jsam.core.state.ModelState`
    """
    era5_lat_ns, era5_lon = era5_latlon()   # (721,) N→S,  (1440,) 0→359.75
    era5_lat_sn = era5_lat_ns[::-1]         # (721,) S→N

    nzm = len(grid.z)          # 74
    ny  = len(grid.lat)        # 720
    nx  = len(grid.lon)        # 1440
    zi  = np.asarray(grid.zi)  # (75,) m  interface heights
    z   = np.asarray(grid.z)   # (74,) m  cell-centre heights

    # adz(k) = (zi[k+1]-zi[k]) / 40 m   — matches gSAM setgrid.f90 adz(k)
    dz_ref = zi[1] - zi[0]               # = 40.0 m
    adz    = np.diff(zi) / dz_ref        # (74,) adz(1:nzm) in gSAM notation

    # gamaz(k) = ggr * z(k) / cp
    gamaz = _GGR / _CP * z              # (74,)

    # ── Base-state densities at interfaces ────────────────────────────────────
    # Matches gSAM setdata.f90:475-484 (dolatlon branch).
    rho  = np.asarray(grid.rho)         # (74,) kg/m³
    rhow = np.zeros(nzm + 1, dtype=np.float64)
    # Interior interfaces k=1..73 (Python), Fortran rhow(2:nzm):
    rhow[1:-1] = (rho[:-1] * adz[1:] + rho[1:] * adz[:-1]) / (adz[:-1] + adz[1:])
    rhow[0]    = 2.0 * rho[0]  - rhow[1]
    rhow[-1]   = 2.0 * rho[-1] - rhow[-2]

    # Model pressure in hPa (needed for micro_set 50 hPa threshold)
    pres_hpa = np.array(metric["pres"]) / 100.0   # (74,) hPa

    # ── Load ERA5 pressure-level fields ───────────────────────────────────────
    print(f"[era5_state_gsam] Loading ERA5 fields for {dt.isoformat()} ...")

    Z_tb     = _read_pl_snapshot(dt, "Z",     rda_root)   # (37,721,1440) top→bottom, N→S
    T_tb     = _read_pl_snapshot(dt, "T",     rda_root)
    U_tb     = _read_pl_snapshot(dt, "U",     rda_root)
    V_tb     = _read_pl_snapshot(dt, "V",     rda_root)
    OMEGA_tb = _read_pl_snapshot(dt, "OMEGA", rda_root)
    Q_tb     = _read_pl_snapshot(dt, "Q",     rda_root)
    CLWC_tb  = _read_pl_snapshot(dt, "CLWC", rda_root)
    CIWC_tb  = _read_pl_snapshot(dt, "CIWC", rda_root)

    # ── Global mean height profile (= gSAM init-binary zin) ──────────────────
    zr_tb = _global_mean_height(Z_tb, era5_lat_ns)   # (37,) top→bottom
    zr_bt = zr_tb[::-1].copy()                        # (37,) ascending (surface→top)
    print(f"[era5_state_gsam] zin range: {zr_bt[0]:.1f} m .. {zr_bt[-1]:.1f} m")
    del Z_tb

    # ── Reorder to surface→top, S→N ───────────────────────────────────────────
    def _bt_sn(f):
        """Flip (37,721,1440) top→bottom N→S → bottom→top S→N."""
        return f[::-1, ::-1, :].copy()

    T_bt     = _bt_sn(T_tb);     del T_tb
    U_bt     = _bt_sn(U_tb);     del U_tb
    V_bt     = _bt_sn(V_tb);     del V_tb
    OMEGA_bt = _bt_sn(OMEGA_tb); del OMEGA_tb
    Q_bt     = _bt_sn(Q_tb);     del Q_tb
    CLWC_bt  = _bt_sn(CLWC_tb);  del CLWC_tb
    CIWC_bt  = _bt_sn(CIWC_tb);  del CIWC_tb

    # ── Height-based vertical interpolation ───────────────────────────────────
    print("[era5_state_gsam] Height-based vertical interpolation ...")

    # Scalar fields at mass-grid heights z(k)
    T_vi    = _interp_height_3d(T_bt,    zr_bt, z)                     # (74,721,1440)
    Q_vi    = np.maximum(0.0, _interp_height_3d(Q_bt,    zr_bt, z))
    CLWC_vi = np.maximum(0.0, _interp_height_3d(CLWC_bt, zr_bt, z))
    CIWC_vi = np.maximum(0.0, _interp_height_3d(CIWC_bt, zr_bt, z))

    # U at mass-grid heights (horizontal stagger applied later)
    U_vi = _interp_height_3d(U_bt, zr_bt, z)    # (74,721,1440)

    # V at mass-grid heights (horizontal stagger applied later)
    V_vi = _interp_height_3d(V_bt, zr_bt, z)    # (74,721,1440)

    # omega interpolated to interface heights zi(2:nzm) = zi[1:74] (73 values)
    zi_iface  = zi[1:nzm]                                              # (73,) m
    OMEGA_vi  = _interp_height_3d(OMEGA_bt, zr_bt, zi_iface)           # (73,721,1440)

    del T_bt, U_bt, V_bt, OMEGA_bt, Q_bt, CLWC_bt, CIWC_bt

    # ── Horizontal interpolation ───────────────────────────────────────────────
    print("[era5_state_gsam] Bilinear horizontal interpolation ...")

    # Scalar fields: mass-grid lat/lon
    TABS_m  = interp_horiz(T_vi,    era5_lat_sn, era5_lon, grid.lat, grid.lon)
    QV_m    = interp_horiz(Q_vi,    era5_lat_sn, era5_lon, grid.lat, grid.lon)
    QCL_m   = interp_horiz(CLWC_vi, era5_lat_sn, era5_lon, grid.lat, grid.lon)
    QCI_m   = interp_horiz(CIWC_vi, era5_lat_sn, era5_lon, grid.lat, grid.lon)
    del T_vi, Q_vi, CLWC_vi, CIWC_vi

    # U: interpolate to stagger longitudes lonu = lon + dlon/2 (periodic)
    dlon  = float(grid.lon[1] - grid.lon[0])          # 0.25°
    lonu  = (np.asarray(grid.lon) + 0.5 * dlon) % 360.0   # (nx,)
    U_hi  = interp_horiz(U_vi, era5_lat_sn, era5_lon, grid.lat, lonu)  # (74,ny,nx)
    del U_vi

    # V: interpolate to stagger latitudes lat_v = (ny+1,)
    lat_v = np.asarray(grid.lat_v)                    # (ny+1,) includes ±90°
    V_hi  = interp_horiz(V_vi, era5_lat_sn, era5_lon, lat_v, grid.lon)  # (74,ny+1,nx)
    del V_vi

    # omega: horizontal interp to mass-grid lat/lon
    OMEGA_hi = interp_horiz(OMEGA_vi, era5_lat_sn, era5_lon, grid.lat, grid.lon)  # (73,ny,nx)
    del OMEGA_vi

    # ── omega → w conversion  (gSAM setdata.f90:487-495) ─────────────────────
    #
    #   do k = nzm, 2, -1          ! Fortran k = 74..2
    #     w(:,:,k) = -(adz(k)*w(:,:,k) + adz(k-1)*w(:,:,k-1)) &
    #                /(adz(k)+adz(k-1)) / (rhow(k)*ggr)
    #   end do
    #   w(:,:,1)  = 0.
    #   w(:,:,nz) = 0.
    #
    # Python mapping: Fortran k → Python W[k-1].
    # Each update at Fortran k uses w(:,:,k) and w(:,:,k-1) — the *original*
    # omega at those levels (k-1 is never modified before k is processed in a
    # descending loop).  The entire loop is therefore a single vectorised pass.
    #
    # W_arr indices:  W_arr[0]     = surface  (zi[0] = 0 m)
    #                 W_arr[1:74]  = converted w at zi[1:74]
    #                 W_arr[74]    = top       (zi[74])
    W_arr = np.zeros((nzm + 1, ny, nx), dtype=np.float64)
    W_arr[1:nzm] = OMEGA_hi   # (73, ny, nx) — raw omega before conversion
    del OMEGA_hi

    # Vectorised update (k_py = 1..73, corresponding to Fortran k = 2..74):
    #   W[k] = -(adz[k]*W[k] + adz[k-1]*W[k-1]) / (adz[k]+adz[k-1]) / (rhow[k]*ggr)
    # All right-hand-side quantities are the *original* omega values.
    adz_k   = adz[1:nzm, None, None]       # (73,1,1) — adz(2:nzm) Fortran
    adz_km1 = adz[0:nzm-1, None, None]     # (73,1,1) — adz(1:nzm-1) Fortran
    rhow_k  = rhow[1:nzm, None, None]      # (73,1,1) — rhow(2:nzm) Fortran
    W_arr[1:nzm] = (
        -(adz_k * W_arr[1:nzm] + adz_km1 * W_arr[0:nzm-1])
        / (adz_k + adz_km1)
        / (rhow_k * _GGR)
    )
    # W_arr[0] and W_arr[nzm] remain 0 (surface / rigid-lid BCs)

    # ── U stagger: append periodic east column ────────────────────────────────
    # gSAM U(nx+1, ny, nzm): the (nx+1)-th face is the east face of the last
    # cell = west face of the first cell (periodic).
    U_stag = np.empty((nzm, ny, nx + 1), dtype=np.float64)
    U_stag[:, :, :-1] = U_hi
    U_stag[:, :,  -1] = U_hi[:, :, 0]    # periodic duplicate
    del U_hi

    # ── micro_set post-processing ──────────────────────────────────────────────
    print("[era5_state_gsam] Applying micro_set ...")
    QCL_new, QCI_new, QV_new, _ = _micro_set(
        TABS_m, QCL_m, QCI_m, QV_m, pres_hpa, gamaz
    )

    # ── Build ModelState ──────────────────────────────────────────────────────
    print("[era5_state_gsam] Building ModelState ...")
    return ModelState(
        U    = jnp.array(U_stag,   dtype=jnp.float64),
        V    = jnp.array(V_hi,     dtype=jnp.float64),
        W    = jnp.array(W_arr,    dtype=jnp.float64),
        TABS = jnp.array(TABS_m,   dtype=jnp.float64),
        QV   = jnp.array(QV_new,   dtype=jnp.float64),
        QC   = jnp.array(QCL_new,  dtype=jnp.float64),
        QI   = jnp.array(QCI_new,  dtype=jnp.float64),
        QR   = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        QS   = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        QG   = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        TKE  = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        p_prev  = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        p_pprev = jnp.zeros((nzm, ny, nx), dtype=jnp.float64),
        nstep = jnp.int32(0),
        time  = jnp.float64(0.0),
    )
