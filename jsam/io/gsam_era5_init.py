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
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jsam.core.grid.latlon import LatLonGrid
from jsam.core.state import ModelState
from jsam.io.era5 import (
    RDA_ROOT,
    _G,
    _ERA5_P_HPA,
    _read_pl_snapshot,
    era5_latlon,
    interp_horiz,
    _gsam_reference_column,
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

def era5_state_from_gsam_init_binary(
    bin_path: Path | str,
    grid: LatLonGrid,
    metric: dict,
    ug: float = 0.0,
    vg: float = 0.0,
) -> ModelState:
    """
    Initialise ModelState from the actual gSAM ERA5 init binary.

    Reads the raw ERA5-resolution fields from
    ``init_era5_YYYYMMDDHHH_GLOBAL.bin`` at float32 precision (exactly as
    gSAM does), then applies the same height-based vertical interpolation,
    bilinear horizontal interpolation, omega→w conversion, and micro_set
    post-processing as :func:`era5_state_gsam`.

    Using the binary rather than ERA5 netCDF eliminates any float32/float64
    discrepancy in the source ``zr`` heights, giving bit-identical
    interpolation results to gSAM.

    Parameters
    ----------
    bin_path : Path or str
        Path to ``init_era5_YYYYMMDDHHH_GLOBAL.bin``  (gSAM format).
    grid : :class:`~jsam.core.grid.latlon.LatLonGrid`
    metric : dict from build_metric  (provides ``"pres"`` in Pa)

    Returns
    -------
    :class:`~jsam.core.state.ModelState`
    """
    from jsam.io.era5_binary import read_gsam_init_binary

    bin_path = Path(bin_path)
    print(f"[era5_state_from_gsam_init_binary] Loading from {bin_path} ...")

    data = read_gsam_init_binary(bin_path)

    nzm = len(grid.z)
    ny  = len(grid.lat)
    nx  = len(grid.lon)
    zi  = np.asarray(grid.zi)
    z   = np.asarray(grid.z)

    # adz(k) = dz(k) / dz_ref
    dz_ref = zi[1] - zi[0]
    adz    = np.diff(zi) / dz_ref   # (nzm,)

    # gamaz(k) = ggr/cp * z(k)
    gamaz = _GGR / _CP * z          # (nzm,)

    # pres_hpa for micro_set 50-hPa threshold (from metric)
    pres_hpa = np.array(metric["pres"]) / 100.0   # (nzm,)

    # Source grid from binary (float32, S→N, ascending height)
    lonr = data['lon'].astype(np.float64)   # (1440,)
    latr = data['lat'].astype(np.float64)   # (721,)
    zr   = data['zr'].astype(np.float64)    # (37,) ascending heights [m]

    # ── Vertical interpolation (height-based, matching gSAM read_field3D) ──
    print("[era5_state_from_gsam_init_binary] Height-based vertical interpolation ...")

    # Scalar fields at mass-grid heights z[k]
    T_vi    = _interp_height_3d(data['TABS'].astype(np.float64), zr, z)
    Q_vi    = np.maximum(0.0, _interp_height_3d(data['QV'].astype(np.float64),  zr, z))
    CLWC_vi = np.maximum(0.0, _interp_height_3d(data['QCL'].astype(np.float64), zr, z))
    CIWC_vi = np.maximum(0.0, _interp_height_3d(data['QCI'].astype(np.float64), zr, z))
    U_vi    = _interp_height_3d(data['U'].astype(np.float64),    zr, z)
    V_vi    = _interp_height_3d(data['V'].astype(np.float64),    zr, z)

    # Omega at interface heights zi[1:nzm] = zi(2:nzm) in Fortran (73 values)
    # gSAM setdata.f90:392: read_field3D(…w…,zi(2:nzm),…,nzm-1)
    zi_iface = zi[1:nzm]
    OMEGA_vi = _interp_height_3d(data['W'].astype(np.float64), zr, zi_iface)  # (73,721,1440)

    # ── Horizontal interpolation ──────────────────────────────────────────────
    print("[era5_state_from_gsam_init_binary] Bilinear horizontal interpolation ...")

    TABS_m  = interp_horiz(T_vi,    latr, lonr, grid.lat, grid.lon)
    QV_m    = interp_horiz(Q_vi,    latr, lonr, grid.lat, grid.lon)
    QCL_m   = interp_horiz(CLWC_vi, latr, lonr, grid.lat, grid.lon)
    QCI_m   = interp_horiz(CIWC_vi, latr, lonr, grid.lat, grid.lon)
    del T_vi, Q_vi, CLWC_vi, CIWC_vi

    # U: stagger longitude lonu = lon - dlon/2 (gSAM setgrid.f90 C-grid west face)
    dlon  = float(grid.lon[1] - grid.lon[0])
    lonu  = (np.asarray(grid.lon) - 0.5 * dlon) % 360.0
    U_hi  = interp_horiz(U_vi, latr, lonr, grid.lat, lonu)
    del U_vi

    # V: stagger latitude lat_v
    lat_v = np.asarray(grid.lat_v)
    V_hi  = interp_horiz(V_vi, latr, lonr, lat_v, grid.lon)
    del V_vi

    # Omega at interface heights → mass-grid lat/lon
    OMEGA_hi = interp_horiz(OMEGA_vi, latr, lonr, grid.lat, grid.lon)  # (73, ny, nx)
    del OMEGA_vi

    # ── micro_set (before rho recompute, same as gSAM diagnose order) ─────────
    print("[era5_state_from_gsam_init_binary] Applying micro_set ...")
    QCL_new, QCI_new, QV_new, _ = _micro_set(
        TABS_m, QCL_m, QCI_m, QV_m, pres_hpa, gamaz
    )
    # Save original TABS for (a) Newton starting point in satadj and
    # (b) tabs0_diag for hydrostatic recompute (gSAM diagnose uses pre-satadj TABS).
    TABS_m_presatadj = TABS_m.copy()

    # ── Recompute rho/rhow from actual TABS (gSAM diagnose + pres recompute) ──
    # gSAM setdata.f90:424-485: after micro_set, calls diagnose() which sets
    # tabs0(k) = area-weighted horiz mean of tabs, then recomputes presi/pres/rho/rhow.
    # This is the rho used in the omega→w conversion at lines 487-495.
    print("[era5_state_from_gsam_init_binary] Recomputing rho from diagnosed tabs0 ...")

    # tabs0: area-weighted horizontal mean of TABS_m (matches gSAM diagnose.f90)
    cos_lat = np.cos(np.deg2rad(np.asarray(grid.lat)))
    wgt = cos_lat / cos_lat.sum()
    tabs0_diag = np.sum(np.mean(TABS_m, axis=2) * wgt[None, :], axis=1)  # (nzm,)

    # pres0 and pres_seed from binary's zin/presin (float32, same as gSAM uses)
    zin_f32    = data['zin']    # (37,) float32, ascending heights
    presin_f32 = data['presin'] # (37,) float32, ascending hPa
    # pres0: log-linear extrapolation to z=0 (below surface level zin[0]≈97m)
    pres0_bin = float(np.exp(
        float(np.log(presin_f32[0]))
        + (float(np.log(presin_f32[1])) - float(np.log(presin_f32[0])))
          / (float(zin_f32[1]) - float(zin_f32[0]))
          * (0.0 - float(zin_f32[0]))
    ))
    # pres_seed: log-linear interp of presin at zin to model heights z
    kk = np.searchsorted(zin_f32.astype(np.float64), z, side='right') - 1
    kk = np.clip(kk, 0, len(zin_f32) - 2)
    ln_p_seed = (np.log(presin_f32[kk].astype(np.float64))
                 + (np.log(presin_f32[kk + 1].astype(np.float64))
                    - np.log(presin_f32[kk].astype(np.float64)))
                   / (zin_f32[kk + 1].astype(np.float64) - zin_f32[kk].astype(np.float64))
                   * (z - zin_f32[kk].astype(np.float64)))
    pres_seed = np.exp(ln_p_seed)

    ref = _gsam_reference_column(z=z, zi=zi, tabs0=tabs0_diag,
                                  pres0=pres0_bin, pres_seed=pres_seed)
    rho  = ref['rho']
    rhow = np.zeros(nzm + 1, dtype=np.float64)
    rhow[1:-1] = (rho[:-1] * adz[1:] + rho[1:] * adz[:-1]) / (adz[:-1] + adz[1:])
    rhow[0]    = 2.0 * rho[0]  - rhow[1]
    rhow[-1]   = 2.0 * rho[-1] - rhow[-2]

    # ── Omega → W conversion (setdata.f90:487-495) ────────────────────────────
    # W_arr[1:nzm] holds omega at interface heights zi[1:nzm] (73 levels).
    # gSAM formula (descending k=nzm..2, vectorised as single pass):
    #   w(k) = -(adz(k)*omega(k) + adz(k-1)*omega(k-1)) / (adz(k)+adz(k-1)) / (rhow(k)*ggr)
    # Python k_py = 1..73: omega(k)=W_arr[k_py] at zi[k_py], omega(k-1)=W_arr[k_py-1]
    # W_arr[0] = 0 (surface BC), W_arr[74] = 0 (top BC).
    W_arr = np.zeros((nzm + 1, ny, nx), dtype=np.float64)
    W_arr[1:nzm] = OMEGA_hi   # (73, ny, nx) — omega at zi[1:74]
    del OMEGA_hi

    adz_k   = adz[1:nzm, None, None]    # adz[1..73] = Fortran adz(2..nzm)
    adz_km1 = adz[0:nzm-1, None, None]  # adz[0..72] = Fortran adz(1..nzm-1)
    rhow_k  = rhow[1:nzm, None, None]   # rhow[1..73] = Fortran rhow(2..nzm)
    W_arr[1:nzm] = (
        -(adz_k * W_arr[1:nzm] + adz_km1 * W_arr[0:nzm-1])
        / (adz_k + adz_km1)
        / (rhow_k * _GGR)
    )

    # ── micro_init saturation adjustment (gSAM setdata.f90 → micro_init → cloud → micro_diagnose)
    # gSAM sequence: micro_set (415) → diagnose (424) → hydrostatic pres recompute →
    #   omega→w (487) → micro_init (530) → cloud.f90 (satadj) → micro_diagnose.
    # This evaporates oversaturated condensate to thermodynamic equilibrium,
    # reducing QCL_max from ~2.81e-3 to ~7.67e-4.
    #
    # gSAM cloud.f90:
    #   tabs2 = tabs(i,j,k)              — save original TABS as Newton starting point
    #   tabs(i,j,k) = t - gamaz          — tabs_dry = TABS - fac_cond*qcl - fac_sub*qci
    #   q = qv + qcl + qci               — total non-precip water
    #   Newton (niter<100) using pp(i,j,k)=pres(k) (hydrostatic 1D reference, hPa)
    #   → qn = max(0, q - qsat)          — condensate at equilibrium
    #   micro_diagnose: qcl=qn*om, qci=qn*(1-om), qv=q-qn
    print("[era5_state_from_gsam_init_binary] Applying saturation adjustment (micro_init) ...")
    from jsam.core.physics.microphysics import satadj as _satadj, MicroParams as _MicroParams
    _micro_params = _MicroParams()
    _zeros_3d = jnp.zeros((nzm, ny, nx), dtype=jnp.float64)
    # gSAM cloud.f90 uses pp(i,j,k) = pres(k) — the hydrostatically RECOMPUTED pressure
    # from diagnose.f90, not the original metric pressure. Patch metric with ref['pres'] (Pa).
    _metric_satadj = {**metric, "pres": ref['pres'] * 100.0}   # hPa → Pa
    _TABS_sa, _QV_sa, _QC_sa, _QI_sa = _satadj(
        jnp.array(TABS_m,   dtype=jnp.float64),
        jnp.array(QV_new,   dtype=jnp.float64),
        jnp.array(QCL_new,  dtype=jnp.float64),
        jnp.array(QCI_new,  dtype=jnp.float64),
        _zeros_3d, _zeros_3d, _zeros_3d,   # QR=QS=QG=0 for ERA5 init
        metric=_metric_satadj,
        params=_micro_params,
        n_iter=100,
        # gSAM cloud.f90: Newton starting point = original TABS (before tabs=t-gamaz overwrite)
        tabs_guess=jnp.array(TABS_m_presatadj, dtype=jnp.float64),
    )
    TABS_m  = np.array(_TABS_sa)
    QV_new  = np.array(_QV_sa)
    QCL_new = np.array(_QC_sa)
    QCI_new = np.array(_QI_sa)
    del _TABS_sa, _QV_sa, _QC_sa, _QI_sa, _zeros_3d, TABS_m_presatadj, _metric_satadj

    # ── U stagger: periodic east column ──────────────────────────────────────
    U_stag = np.empty((nzm, ny, nx + 1), dtype=np.float64)
    U_stag[:, :, :-1] = U_hi
    U_stag[:, :,  -1] = U_hi[:, :, 0]
    del U_hi

    # ── Fix 8.1: subtract domain translation velocities (setdata.f90:455-456) ─
    # gSAM: u(i,j,k) -= ug;  v(i,j,k) -= vg
    # ug and vg default to 0 in params.f90 and are not set for IRMA.
    if ug != 0.0:
        U_stag -= ug
    if vg != 0.0:
        V_hi   -= vg

    # ── Build ModelState ──────────────────────────────────────────────────────
    print("[era5_state_from_gsam_init_binary] Building ModelState ...")
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


def era5_state_gsam(
    grid:     LatLonGrid,
    metric:   dict,
    dt:       datetime,
    rda_root: str = RDA_ROOT,
    ug:       float = 0.0,
    vg:       float = 0.0,
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

    # U: interpolate to stagger longitudes lonu = lon - dlon/2 (gSAM setgrid.f90:335)
    dlon  = float(grid.lon[1] - grid.lon[0])          # 0.25°
    lonu  = (np.asarray(grid.lon) - 0.5 * dlon) % 360.0   # (nx,)
    U_hi  = interp_horiz(U_vi, era5_lat_sn, era5_lon, grid.lat, lonu)  # (74,ny,nx)
    del U_vi

    # V: interpolate to stagger latitudes lat_v = (ny+1,)
    lat_v = np.asarray(grid.lat_v)                    # (ny+1,) includes ±90°
    V_hi  = interp_horiz(V_vi, era5_lat_sn, era5_lon, lat_v, grid.lon)  # (74,ny+1,nx)
    del V_vi

    # omega: horizontal interp to mass-grid lat/lon
    OMEGA_hi = interp_horiz(OMEGA_vi, era5_lat_sn, era5_lon, grid.lat, grid.lon)  # (73,ny,nx)
    del OMEGA_vi

    # ── micro_set (gSAM setdata.f90:415) ─────────────────────────────────────
    # Called before diagnose/pres recompute; uses metric pres_hpa as seed.
    print("[era5_state_gsam] Applying micro_set ...")
    QCL_new, QCI_new, QV_new, _ = _micro_set(
        TABS_m, QCL_m, QCI_m, QV_m, pres_hpa, gamaz
    )
    # Save original TABS for (a) Newton starting point in satadj (gSAM cloud.f90 tabs2) and
    # (b) hydrostatic recompute (gSAM diagnose uses pre-satadj TABS, micro_init comes later).
    TABS_m_presatadj = TABS_m.copy()

    # ── Fix 8.2: hydrostatic pressure recompute (setdata.f90:424-484) ────────
    # gSAM: diagnose() sets tabs0 = area-weighted domain mean of tabs (424),
    # then hydrostatically recomputes presi/pres/rho/rhow (427-436, 465, 476-484).
    # This must happen BEFORE omega→w (487-495) because rhow enters the formula.
    print("[era5_state_gsam] Hydrostatic pressure recompute from diagnosed tabs0 ...")
    cos_lat_w = np.cos(np.deg2rad(np.asarray(grid.lat)))   # (ny,)
    wgt       = cos_lat_w / cos_lat_w.sum()
    tabs0     = np.sum(np.mean(TABS_m, axis=2) * wgt[None, :], axis=1)  # (nzm,)

    # pres0: log-linear extrapolation of ERA5 pressure levels to z=0
    # (mirrors gSAM setdata.f90:368 applied to the global-mean ERA5 columns).
    p_hpa_bt  = _ERA5_P_HPA[::-1]    # (37,) ascending hPa (surface→top)
    pres0_hpa = float(np.exp(
        np.interp(0.0, zr_bt, np.log(p_hpa_bt),
                  left=np.log(p_hpa_bt[0]), right=np.log(p_hpa_bt[-1]))
    ))
    # pres_seed: log-interp of ERA5 levels at model cell-centre heights
    pres_seed_hpa = np.exp(
        np.interp(z, zr_bt, np.log(p_hpa_bt),
                  left=np.log(p_hpa_bt[0]), right=np.log(p_hpa_bt[-1]))
    )
    ref = _gsam_reference_column(z=z, zi=zi, tabs0=tabs0,
                                  pres0=pres0_hpa, pres_seed=pres_seed_hpa)
    # Replace rhow with the hydrostatically consistent values (setdata.f90:476-484)
    rhow = ref['rhow']    # (nzm+1,)

    # ── omega → w conversion  (gSAM setdata.f90:487-495) ─────────────────────
    #   do k = nzm, 2, -1          ! Fortran k = 74..2
    #     w(:,:,k) = -(adz(k)*w(:,:,k) + adz(k-1)*w(:,:,k-1)) &
    #                /(adz(k)+adz(k-1)) / (rhow(k)*ggr)
    #   end do
    # Uses hydrostatically recomputed rhow from Fix 8.2 above.
    W_arr = np.zeros((nzm + 1, ny, nx), dtype=np.float64)
    W_arr[1:nzm] = OMEGA_hi   # (73, ny, nx) — raw omega before conversion
    del OMEGA_hi

    adz_k   = adz[1:nzm, None, None]       # (73,1,1) — adz(2:nzm) Fortran
    adz_km1 = adz[0:nzm-1, None, None]     # (73,1,1) — adz(1:nzm-1) Fortran
    rhow_k  = rhow[1:nzm, None, None]      # (73,1,1) — rhow(2:nzm) Fortran
    W_arr[1:nzm] = (
        -(adz_k * W_arr[1:nzm] + adz_km1 * W_arr[0:nzm-1])
        / (adz_k + adz_km1)
        / (rhow_k * _GGR)
    )
    # W_arr[0] and W_arr[nzm] remain 0 (surface / rigid-lid BCs)

    # ── micro_init saturation adjustment (gSAM setdata.f90 → micro_init → cloud → micro_diagnose)
    # gSAM sequence: micro_set (415) → diagnose/pres recompute (424) → omega→w (487)
    #   → micro_init (530) → cloud.f90 (Newton satadj) → micro_diagnose.
    # Matches the same step in era5_state_from_gsam_init_binary; see that function for rationale.
    print("[era5_state_gsam] Applying saturation adjustment (micro_init) ...")
    from jsam.core.physics.microphysics import satadj as _satadj, MicroParams as _MicroParams
    _micro_params = _MicroParams()
    _zeros_3d = jnp.zeros((nzm, ny, nx), dtype=jnp.float64)
    # gSAM cloud.f90 uses pres(k) after hydrostatic recompute (diagnose.f90)
    _metric_satadj = {**metric, "pres": ref['pres'] * 100.0}   # hPa → Pa
    _TABS_sa, _QV_sa, _QC_sa, _QI_sa = _satadj(
        jnp.array(TABS_m,           dtype=jnp.float64),
        jnp.array(QV_new,           dtype=jnp.float64),
        jnp.array(QCL_new,          dtype=jnp.float64),
        jnp.array(QCI_new,          dtype=jnp.float64),
        _zeros_3d, _zeros_3d, _zeros_3d,   # QR=QS=QG=0 for ERA5 init
        metric=_metric_satadj,
        params=_micro_params,
        n_iter=100,
        # gSAM cloud.f90: Newton starting point = original TABS (before tabs=t-gamaz overwrite)
        tabs_guess=jnp.array(TABS_m_presatadj, dtype=jnp.float64),
    )
    TABS_m  = np.array(_TABS_sa)
    QV_new  = np.array(_QV_sa)
    QCL_new = np.array(_QC_sa)
    QCI_new = np.array(_QI_sa)
    del _TABS_sa, _QV_sa, _QC_sa, _QI_sa, _zeros_3d, TABS_m_presatadj, _metric_satadj

    # ── U stagger: append periodic east column ────────────────────────────────
    U_stag = np.empty((nzm, ny, nx + 1), dtype=np.float64)
    U_stag[:, :, :-1] = U_hi
    U_stag[:, :,  -1] = U_hi[:, :, 0]    # periodic duplicate
    del U_hi

    # ── Fix 8.1: subtract domain translation velocities (setdata.f90:455-456) ─
    # gSAM: u(i,j,k) -= ug;  v(i,j,k) -= vg
    # ug and vg default to 0 in params.f90; not set for IRMA.
    if ug != 0.0:
        U_stag -= ug
    if vg != 0.0:
        V_hi   -= vg

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


_GSAM_BIN_ROOT = "/glade/u/home/sabramian/gSAM1.8.7/GLOBAL_DATA/BIN_D"


def era5_state_gsam_with_caching(
    grid: LatLonGrid,
    metric: dict,
    dt: datetime,
    rda_root: str = RDA_ROOT,
    cache_dir: Path | str | None = None,
    gsam_bin_root: str = _GSAM_BIN_ROOT,
) -> ModelState:
    """
    Initialise ModelState from the gSAM ERA5 init binary when available,
    otherwise fall back to computing from ERA5 netCDF.

    Priority order:
    1. gSAM init binary  (``init_era5_YYYYMMDDHHH_GLOBAL.bin`` in gsam_bin_root)
       — uses float32 source data identical to what gSAM itself reads, giving
       bit-for-bit matching initialization.
    2. ERA5 netCDF       (computed by :func:`era5_state_gsam`)

    Parameters
    ----------
    grid         : :class:`~jsam.core.grid.latlon.LatLonGrid`
    metric       : dict from build_metric
    dt           : ERA5 analysis datetime
    rda_root     : path to NCAR RDA ERA5 netCDF archive
    cache_dir    : unused (kept for API compatibility)
    gsam_bin_root: directory containing gSAM init binaries

    Returns
    -------
    :class:`~jsam.core.state.ModelState`
    """
    bin_name = f"init_era5_{dt.strftime('%Y%m%d%H')}_GLOBAL.bin"

    # Priority 1: gSAM init binary (exact match to gSAM initialization)
    gsam_bin = Path(gsam_bin_root) / bin_name
    if gsam_bin.exists():
        print(f"[era5_state_gsam_with_caching] Using gSAM binary: {gsam_bin}")
        return era5_state_from_gsam_init_binary(gsam_bin, grid, metric)

    # Priority 2: compute from ERA5 netCDF
    print(f"[era5_state_gsam_with_caching] gSAM binary not found at {gsam_bin}, "
          f"computing from ERA5 netCDF ...")
    return era5_state_gsam(grid, metric, dt, rda_root)
