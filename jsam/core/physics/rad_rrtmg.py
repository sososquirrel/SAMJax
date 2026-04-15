"""
RRTMG longwave radiation for jsam — batched-column wrapper around the
f2py-built gSAM RRTMG_LW extension module.

Build artefact expected at ``/glade/work/sabramian/jsam_rrtmg_build/``
(see ``jsam_rrtmg_build/build.sh`` and ``test_one_column.py`` for the
Day 1 smoke test of the underlying ``.so``).

Design
------
RRTMG is column-independent, so we vectorize naively: flatten
(ny, nx) into a single ``ncol`` axis, call the Fortran wrapper once
per radiation step with a (ncol, nlay) batch, then reshape back to
(nz, ny, nx). JAX integration is via :func:`jax.pure_callback`.

Clouds use RRTMG's default cloud optics (``inflglw=2, iceflglw=3,
liqflglw=1``) with LWP/IWP and effective radii computed per layer
from ``state.QC`` / ``state.QI``. Liquid Re is 14 µm over open ocean
(CAM default). Ice Re comes from the CAM hexagonal-column retab
lookup table (180–274 K).

Surface temperature is per column (``ny, nx``). Ozone is still an
analytic Gaussian profile — real gSAM ``o3file`` parsing is deferred.
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jsam.core.state import ModelState


# ---------------------------------------------------------------------------
# Locate the f2py-built extension (jsam_rrtmg_lw.*.so)
# ---------------------------------------------------------------------------

_RRTMG_BUILD_DIR = "/glade/work/sabramian/jsam_rrtmg_build"

def _import_rrtmg_lw():
    if _RRTMG_BUILD_DIR not in sys.path:
        sys.path.insert(0, _RRTMG_BUILD_DIR)
    import jsam_rrtmg_lw  # noqa: F401
    return jsam_rrtmg_lw

_LW = None
_LW_INITIALIZED = False


def _ensure_initialized(cpdair: float) -> None:
    """One-time call to ``rrtmg_lw_ini`` — loads spectral k-distribution tables."""
    global _LW, _LW_INITIALIZED
    if _LW is None:
        _LW = _import_rrtmg_lw()
    if not _LW_INITIALIZED:
        _LW.rrtmg_lw_init.rrtmg_lw_ini(cpdair)
        _LW_INITIALIZED = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadRRTMGConfig:
    """
    Static (non-JAX) configuration for the RRTMG_LW driver.

    Trace gas VMRs are uniform in space/time for now. Clouds use
    RRTMG's built-in parameterization (``inflglw=2``) with per-column
    LWP/IWP and effective radii derived from QC/QI.
    """
    cpdair:    float = 1004.64   # J/(kg·K)  matches gSAM consts.f90
    emis:      float = 0.98      # band-independent surface emissivity (gSAM emis_water)
    co2_vmr:   float = 400e-6
    ch4_vmr:   float = 1.8e-6
    n2o_vmr:   float = 320e-9
    o2_vmr:    float = 0.209
    cfc11_vmr: float = 0.0
    cfc12_vmr: float = 0.0
    cfc22_vmr: float = 0.0
    ccl4_vmr:  float = 0.0


# ---------------------------------------------------------------------------
# Interface pressures and ozone profile (derived once from the metric)
# ---------------------------------------------------------------------------

_G_GRAV = 9.79764                # m/s²  gSAM consts
_P_SURF_DEFAULT = 101325.0       # Pa
_MW_AIR = 28.97
_MW_H2O = 18.02

# CAM ice effective radius lookup table (hexagonal columns, T = 180..274 K)
# From gSAM SRC/RAD_RRTM/cam_rad_parameterizations.f90
_ICE_RE_TMIN = 180.0
_ICE_RETAB = np.array([
    5.92779, 6.26422, 6.61973, 6.99539, 7.39234, 7.81177, 8.25496, 8.72323,
    9.21800, 9.74075, 10.2930, 10.8765, 11.4929, 12.1440, 12.8317, 13.5581,
    14.2319, 15.0351, 15.8799, 16.7674, 17.6986, 18.6744, 19.6955, 20.7623,
    21.8757, 23.0364, 24.2452, 25.5034, 26.8125, 27.7895, 28.6450, 29.4167,
    30.1088, 30.7306, 31.2943, 31.8151, 32.3077, 32.7870, 33.2657, 33.7540,
    34.2601, 34.7892, 35.3442, 35.9255, 36.5316, 37.1602, 37.8078, 38.4720,
    39.1508, 39.8442, 40.5552, 41.2912, 42.0635, 42.8876, 43.7863, 44.7853,
    45.9170, 47.2165, 48.7221, 50.4710, 52.4980, 54.8315, 57.4898, 60.4785,
    63.7898, 65.5604, 71.2885, 75.4113, 79.7368, 84.2351, 88.8833, 93.6658,
    98.5739, 103.603, 108.752, 114.025, 119.424, 124.954, 130.630, 136.457,
    142.446, 148.608, 154.956, 161.503, 168.262, 175.248, 182.473, 189.952,
    197.699, 205.728, 214.055, 222.694, 231.661, 240.971, 250.639,
], dtype=np.float64)                         # (95,)

_LIQ_RE_OCEAN = 14.0                         # µm, CAM default for open ocean


def build_plev_hpa(metric: dict, p_surf_Pa: float = _P_SURF_DEFAULT) -> np.ndarray:
    """Hydrostatic interface pressures (hPa), shape (nz+1,), sfc → TOA."""
    rho = np.asarray(metric["rho"])
    dz  = np.asarray(metric["dz"])
    nz  = rho.shape[0]
    p_face_Pa = np.empty(nz + 1, dtype=np.float64)
    p_face_Pa[0] = p_surf_Pa
    for k in range(nz):
        p_face_Pa[k + 1] = p_face_Pa[k] - rho[k] * _G_GRAV * dz[k]
    return p_face_Pa / 100.0


def analytic_o3_vmr(play_hpa: np.ndarray) -> np.ndarray:
    """Gaussian ozone profile peaking near 10 hPa — replace with real climo."""
    p = np.asarray(play_hpa, dtype=np.float64)
    return 8e-6 * np.exp(-0.5 * (np.log(p / 10.0) / 1.2) ** 2)


# ---------------------------------------------------------------------------
# gSAM o3file binary parser  (unformatted sequential, little-endian)
# ---------------------------------------------------------------------------

_MWO3 = 47.998
_O3_MMR_TO_VMR = _MW_AIR / _MWO3   # 0.60342...


def _read_fortran_record(f) -> Optional[bytes]:
    """Read one Fortran sequential unformatted record (head/tail 4-byte marker)."""
    head = f.read(4)
    if len(head) < 4:
        return None
    n = struct.unpack("<i", head)[0]
    data = f.read(n)
    tail = struct.unpack("<i", f.read(4))[0]
    if n != tail:
        raise IOError(f"Fortran record head/tail mismatch: {n} vs {tail}")
    return data


def _skip_record(f) -> None:
    head = f.read(4)
    n = struct.unpack("<i", head)[0]
    f.seek(n, 1)
    f.read(4)


@dataclass
class GSAMOzoneClimo:
    """One time slice of a gSAM o3file, held as a 3-D (nz, ny, nx) VMR field."""
    lons:     np.ndarray       # (nxr,) deg (may include negatives after flip)
    lats:     np.ndarray       # (nyr,) deg
    pres_hpa: np.ndarray       # (nzr,) hPa
    o3_vmr:   np.ndarray       # (nzr, nyr, nxr) vmr (post 0.6034 mult)

    @classmethod
    def from_file(cls, path: str, day: Optional[float] = None) -> "GSAMOzoneClimo":
        with open(path, "rb") as f:
            hdr = _read_fortran_record(f)
            nx1, ny1, nz1 = struct.unpack("<iii", hdr)
            nobs = struct.unpack("<i", _read_fortran_record(f))[0]
            days = np.frombuffer(_read_fortran_record(f), dtype="<f4").copy()

            # Pick nearest time slice
            if day is None:
                nn = 0
            else:
                nn = int(np.clip(np.searchsorted(days, day) - 1, 0, nobs - 1))

            # Skip nn full slices (4 coord records + nz1 field records each)
            for _ in range(nn):
                for _ in range(4 + nz1):
                    _skip_record(f)

            lonr = np.frombuffer(_read_fortran_record(f), dtype="<f4").astype(np.float64)
            latr = np.frombuffer(_read_fortran_record(f), dtype="<f4").astype(np.float64)
            zr   = np.frombuffer(_read_fortran_record(f), dtype="<f4").astype(np.float64)  # noqa: F841
            pr   = np.frombuffer(_read_fortran_record(f), dtype="<f4").astype(np.float64)

            fld = np.empty((nz1, ny1, nx1), dtype=np.float64)
            for k in range(nz1):
                rec = _read_fortran_record(f)
                # Fortran order: fld(1:nxr,1:nyr) → (nyr, nxr) in C-contig
                fld[k] = np.frombuffer(rec, dtype="<f4").reshape(ny1, nx1).astype(np.float64)

        # Mass mixing ratio → vmr
        o3_vmr = fld * _O3_MMR_TO_VMR
        return cls(lons=lonr, lats=latr, pres_hpa=pr, o3_vmr=o3_vmr)

    def to_columns(
        self,
        lats_deg: np.ndarray,   # (ncol,)
        lons_deg: np.ndarray,   # (ncol,)
        play_hpa: np.ndarray,   # (nz,)  target layer pressures (sfc→TOA)
    ) -> np.ndarray:
        """Bilinear horizontal + log-p vertical interpolation → (ncol, nz) vmr."""
        lats_deg = np.asarray(lats_deg, dtype=np.float64)
        lons_deg = np.asarray(lons_deg, dtype=np.float64)
        play_hpa = np.asarray(play_hpa, dtype=np.float64)

        # Wrap lons into the source [min, max] range. Source is typically 0-360.
        src_lon_min, src_lon_max = self.lons[0], self.lons[-1]
        lon = lons_deg.copy()
        if src_lon_min >= 0.0:
            lon = np.mod(lon, 360.0)
        else:
            lon = ((lon + 180.0) % 360.0) - 180.0
        lon = np.clip(lon, src_lon_min, src_lon_max)
        lat = np.clip(lats_deg, self.lats[0], self.lats[-1])

        i = np.clip(np.searchsorted(self.lons, lon) - 1, 0, len(self.lons) - 2)
        j = np.clip(np.searchsorted(self.lats, lat) - 1, 0, len(self.lats) - 2)
        dx = (lon - self.lons[i]) / (self.lons[i + 1] - self.lons[i])
        dy = (lat - self.lats[j]) / (self.lats[j + 1] - self.lats[j])

        # (nzr, ncol) horizontal slab by gather
        f00 = self.o3_vmr[:, j,     i]      # (nzr, ncol)
        f10 = self.o3_vmr[:, j,     i + 1]
        f01 = self.o3_vmr[:, j + 1, i]
        f11 = self.o3_vmr[:, j + 1, i + 1]
        slab = ((1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f10
                + (1 - dx) * dy * f01 + dx * dy * f11)   # (nzr, ncol)

        # Source pressure ordering: gSAM binary goes TOA→sfc (low→high pres) as k increases.
        # Use log-p linear interpolation along the source axis per column.
        src_lnp = np.log(self.pres_hpa)
        order = np.argsort(src_lnp)                      # ascending lnp
        src_lnp_s = src_lnp[order]
        slab_s = slab[order, :]                          # (nzr, ncol)

        tgt_lnp = np.log(play_hpa)                       # (nz,)
        # Interpolate each column independently via numpy vectorization
        ncol = slab_s.shape[1]
        out = np.empty((ncol, tgt_lnp.size), dtype=np.float64)
        for c in range(ncol):
            out[c] = np.interp(tgt_lnp, src_lnp_s, slab_s[:, c])
        return np.clip(out, 0.0, None)


def _ice_re_from_T(tlay: np.ndarray) -> np.ndarray:
    """CAM hexagonal-column Re_ice(T). Shape-preserving."""
    t = np.asarray(tlay, dtype=np.float64)
    idx = (t - (_ICE_RE_TMIN - 1.0)).astype(np.int64)
    idx = np.clip(idx, 1, 94)
    frac = t - np.floor(t)
    r0 = _ICE_RETAB[idx - 1]
    r1 = _ICE_RETAB[idx]
    return r0 * (1.0 - frac) + r1 * frac


# ---------------------------------------------------------------------------
# Batched-column NumPy driver
# ---------------------------------------------------------------------------

def _rrtmg_lw_numpy(
    play_hpa:  np.ndarray,   # (ncol, nlay)
    plev_hpa:  np.ndarray,   # (ncol, nlay+1)
    tlay:      np.ndarray,   # (ncol, nlay)
    tlev:      np.ndarray,   # (ncol, nlay+1)
    tsfc:      np.ndarray,   # (ncol,)
    h2ovmr:    np.ndarray,   # (ncol, nlay)
    o3vmr:     np.ndarray,   # (ncol, nlay)
    cldfr:     np.ndarray,   # (ncol, nlay)
    cicewp:    np.ndarray,   # (ncol, nlay)  g/m²
    cliqwp:    np.ndarray,   # (ncol, nlay)  g/m²
    reice:     np.ndarray,   # (ncol, nlay)  µm
    reliq:     np.ndarray,   # (ncol, nlay)  µm
    cfg:       RadRRTMGConfig,
) -> np.ndarray:
    """Call the f2py-built RRTMG_LW for one batch of columns."""
    _ensure_initialized(cfg.cpdair)
    ncol, nlay = play_hpa.shape

    def _f(val):
        return np.full((ncol, nlay), float(val), dtype=np.float64)

    co2    = _f(cfg.co2_vmr)
    ch4    = _f(cfg.ch4_vmr)
    n2o    = _f(cfg.n2o_vmr)
    o2     = _f(cfg.o2_vmr)
    cfc11  = _f(cfg.cfc11_vmr)
    cfc12  = _f(cfg.cfc12_vmr)
    cfc22  = _f(cfg.cfc22_vmr)
    ccl4   = _f(cfg.ccl4_vmr)

    emis   = np.full((ncol, 16), float(cfg.emis), dtype=np.float64)

    # Clear-sky taucld/tauaer placeholders (RRTMG ignores them when
    # inflglw=2 and uses its own optics derived from c(i|l)cewp + re).
    taucld = np.zeros((16, ncol, nlay), dtype=np.float64)
    tauaer = np.zeros((ncol, nlay, 16), dtype=np.float64)

    icld = 1                     # on, maximum-random overlap
    idrv = 0
    inflglw  = 2                 # compute optics from cwp & re
    iceflglw = 3                 # Fu (2007) ice scheme
    liqflglw = 1                 # Hu & Stamnes liquid

    uflx, dflx, hr, uflxc, dflxc, hrc, _du1, _du2 = (
        _LW.rrtmg_lw_rad_nomcica.rrtmg_lw(
            ncol, nlay, icld, idrv,
            np.ascontiguousarray(play_hpa), np.ascontiguousarray(plev_hpa),
            np.ascontiguousarray(tlay),     np.ascontiguousarray(tlev),
            np.ascontiguousarray(tsfc),
            np.ascontiguousarray(h2ovmr),   np.ascontiguousarray(o3vmr),
            co2, ch4, n2o, o2,
            cfc11, cfc12, cfc22, ccl4,
            emis,
            inflglw, iceflglw, liqflglw,
            np.ascontiguousarray(cldfr),
            taucld,
            np.ascontiguousarray(cicewp), np.ascontiguousarray(cliqwp),
            np.ascontiguousarray(reice),  np.ascontiguousarray(reliq),
            tauaer,
        )
    )
    return (
        np.asarray(hr, dtype=np.float64),        # (ncol, nlay) K/day
        np.asarray(dflx, dtype=np.float64),      # (ncol, nlay+1) W/m² down-welling
    )


# ---------------------------------------------------------------------------
# Host-side tendency: state → dTABS_dt (K/s)
# ---------------------------------------------------------------------------

_CHUNK_NCOL = 4096


def _compute_dTABS_dt_host(
    TABS_host:  np.ndarray,     # (nz, ny, nx)
    QV_host:    np.ndarray,     # (nz, ny, nx)  kg/kg
    QC_host:    np.ndarray,     # (nz, ny, nx)  kg/kg
    QI_host:    np.ndarray,     # (nz, ny, nx)  kg/kg
    sst_host:   np.ndarray,     # (ny, nx) K
    play_hpa:   np.ndarray,     # (nz,)  layer pressure (hPa) sfc→TOA
    plev_hpa:   np.ndarray,     # (nz+1,) interface pressure (hPa) sfc→TOA
    o3vmr:      np.ndarray,     # (nz,) or (ncol, nz)
    cfg:        RadRRTMGConfig,
) -> np.ndarray:                # (nz, ny, nx) K/s
    nz, ny, nx = TABS_host.shape
    ncol = ny * nx

    # (nz, ny, nx) → (ncol, nz)
    TABS_col = TABS_host.reshape(nz, ncol).T
    QV_col   = QV_host.reshape(nz, ncol).T
    QC_col   = QC_host.reshape(nz, ncol).T
    QI_col   = QI_host.reshape(nz, ncol).T
    tsfc_col = sst_host.reshape(ncol).astype(np.float64)
    # Land fallback: ERA5 SST is NaN over land — use lowest-layer TABS.
    _nan_sfc = ~np.isfinite(tsfc_col)
    if _nan_sfc.any():
        tsfc_col = np.where(_nan_sfc, TABS_col[:, 0], tsfc_col)

    h2o_vmr  = QV_col * (_MW_AIR / _MW_H2O)

    play = np.broadcast_to(play_hpa[None, :], (ncol, nz)).copy()
    plev = np.broadcast_to(plev_hpa[None, :], (ncol, nz + 1)).copy()
    if o3vmr.ndim == 1:
        o3v = np.broadcast_to(o3vmr[None, :], (ncol, nz)).copy()
    else:
        assert o3vmr.shape == (ncol, nz), f"o3vmr shape {o3vmr.shape} != {(ncol, nz)}"
        o3v = np.ascontiguousarray(o3vmr, dtype=np.float64)

    # Layer mass (kg/m²) from interface pressure thickness: dp/g
    # plev is sfc→TOA so dp = plev[k]-plev[k+1] > 0
    dp_hpa = plev[:, :-1] - plev[:, 1:]                        # (ncol, nz)
    layerMass_kg = 100.0 * dp_hpa / _G_GRAV                     # kg/m²

    # gSAM convention: LWP, IWP in g/m²
    cliqwp = QC_col * 1e3 * layerMass_kg                        # (ncol, nz)
    cicewp = QI_col * 1e3 * layerMass_kg

    # Binary cloud fraction: 1 where either phase has mass, else 0
    cldfr  = np.where((cliqwp > 0.0) | (cicewp > 0.0), 1.0, 0.0)

    # Effective radii
    reliq  = np.where(cliqwp > 0.0, _LIQ_RE_OCEAN, 0.0)         # (ncol, nz) µm
    reice_all = _ice_re_from_T(TABS_col)
    reice  = np.where(cicewp > 0.0, reice_all, 0.0)

    # Clamp ice re to RRTMG's valid range for iceflglw=3 (Fu): 5..131 µm
    reice = np.clip(reice, 5.0, 131.0)
    # Liq re valid range for liqflglw=1 (Hu & Stamnes): 2.5..60 µm
    reliq = np.clip(reliq, 2.5, 60.0)

    # Interface temperatures. Bottom interface = surface skin T (gSAM rad.f90
    # line 444: interfaceT(:,1) = sstxy + t00). Top interface = linear extrap
    # (gSAM line 447: 2*layerT(nzm+1) - interfaceT(nzm+1)). tsfc (skin) is
    # passed separately too; RRTMG uses tsfc for surface Planck and tlev[0]
    # for the surface-interface flux calc — gSAM ties them together via SST.
    tlev = np.empty((ncol, nz + 1), dtype=np.float64)
    tlev[:, 1:-1] = 0.5 * (TABS_col[:, :-1] + TABS_col[:, 1:])
    tlev[:, 0]    = tsfc_col
    tlev[:, -1]   = 2.0 * TABS_col[:, -1] - tlev[:, -2]

    # --- Extra top layer (match gSAM rad.f90 rad_driver) ---------------------
    # gSAM passes RRTMG an extra fictitious layer extending from the model-top
    # interface down to ~1e-4 hPa. Without it our real top layer absorbs the
    # entire stratospheric flux divergence into a thin slab and cooling is
    # exaggerated by ~3x. Properties: isothermal with top real layer, no
    # clouds, h2o/o3 copied from top real layer.
    nlay_rad = nz + 1
    plev_ext = np.empty((ncol, nlay_rad + 1), dtype=np.float64)
    plev_ext[:, :nz + 1] = plev
    plev_ext[:, nz + 1]  = np.minimum(1.0e-4, 0.25 * plev[:, nz])

    play_ext = np.empty((ncol, nlay_rad), dtype=np.float64)
    play_ext[:, :nz] = play
    play_ext[:, nz]  = 0.5 * plev_ext[:, nz]         # midpoint of extra layer

    tlay_ext = np.empty((ncol, nlay_rad), dtype=np.float64)
    tlay_ext[:, :nz] = TABS_col
    tlay_ext[:, nz]  = TABS_col[:, -1]

    tlev_ext = np.empty((ncol, nlay_rad + 1), dtype=np.float64)
    tlev_ext[:, :nz + 1] = tlev
    tlev_ext[:, nz + 1]  = TABS_col[:, -1]           # extra layer top interface

    h2o_ext = np.empty((ncol, nlay_rad), dtype=np.float64)
    h2o_ext[:, :nz] = h2o_vmr
    h2o_ext[:, nz]  = h2o_vmr[:, -1]

    o3_ext = np.empty((ncol, nlay_rad), dtype=np.float64)
    o3_ext[:, :nz] = o3v
    o3_ext[:, nz]  = o3v[:, -1]

    cldfr_ext  = np.zeros((ncol, nlay_rad), dtype=np.float64)
    cldfr_ext[:, :nz] = cldfr
    cicewp_ext = np.zeros((ncol, nlay_rad), dtype=np.float64)
    cicewp_ext[:, :nz] = cicewp
    cliqwp_ext = np.zeros((ncol, nlay_rad), dtype=np.float64)
    cliqwp_ext[:, :nz] = cliqwp
    reice_ext  = np.zeros((ncol, nlay_rad), dtype=np.float64)
    reice_ext[:, :nz] = reice
    reliq_ext  = np.zeros((ncol, nlay_rad), dtype=np.float64)
    reliq_ext[:, :nz] = reliq

    hr_K_per_day = np.empty((ncol, nz), dtype=np.float64)
    lwds_col = np.empty((ncol,), dtype=np.float64)
    tsfc64 = tsfc_col.astype(np.float64)

    for i0 in range(0, ncol, _CHUNK_NCOL):
        i1 = min(i0 + _CHUNK_NCOL, ncol)
        hr_ext, dflx_ext = _rrtmg_lw_numpy(
            play_hpa=play_ext[i0:i1], plev_hpa=plev_ext[i0:i1],
            tlay=tlay_ext[i0:i1],     tlev=tlev_ext[i0:i1],
            tsfc=tsfc64[i0:i1],
            h2ovmr=h2o_ext[i0:i1], o3vmr=o3_ext[i0:i1],
            cldfr=cldfr_ext[i0:i1],
            cicewp=cicewp_ext[i0:i1], cliqwp=cliqwp_ext[i0:i1],
            reice=reice_ext[i0:i1],   reliq=reliq_ext[i0:i1],
            cfg=cfg,
        )
        # Drop the extra top layer; jsam only advances the nz real layers.
        hr_K_per_day[i0:i1] = hr_ext[:, :nz]
        # Surface down-welling LW = dflx at sfc interface (index 0).
        lwds_col[i0:i1] = dflx_ext[:, 0]

    # Safety clip: physically-realised LW heating should be within ±50 K/day
    # (stratospheric emission can reach ~-12 K/d, tropospheric cloud tops ~-25).
    # Coarser than ~1° grid boxes can produce unphysically concentrated
    # LWP/IWP + tiny effective radii that drive RRTMG into nonsense; cap
    # before it destabilises the dynamics. No-op on realistic native grids.
    np.clip(hr_K_per_day, -50.0, 50.0, out=hr_K_per_day)

    dTABS_dt_col = hr_K_per_day / 86400.0
    lwds_2d = lwds_col.reshape(ny, nx).astype(np.float64)
    return dTABS_dt_col.T.reshape(nz, ny, nx), lwds_2d


def _compute_dTABS_dt_only_host(*args, **kwargs):
    """Back-compat wrapper: returns heating rates only (drops lwds)."""
    dTABS_dt, _ = _compute_dTABS_dt_host(*args, **kwargs)
    return dTABS_dt


# ---------------------------------------------------------------------------
# JAX-facing top-level step
# ---------------------------------------------------------------------------

def build_o3vmr_for_metric(
    metric: dict,
    climo:  Optional[GSAMOzoneClimo] = None,
) -> np.ndarray:
    """
    Build the per-column ozone VMR array that ``rad_rrtmg_proc`` consumes.

    Returns (nz,) if ``climo`` is None (analytic fallback) or
    (ncol, nz) with ncol = ny*nx when ``climo`` is supplied and the metric
    carries ``lat`` / ``lon`` (deg) arrays of shape (ny,) / (nx,).
    """
    plev_hpa = build_plev_hpa(metric)
    play_hpa = 0.5 * (plev_hpa[:-1] + plev_hpa[1:])
    if climo is None:
        return analytic_o3_vmr(play_hpa)

    if "lat_deg" in metric and "lon_deg" in metric:
        lat_deg = np.asarray(metric["lat_deg"])
        lon_deg = np.asarray(metric["lon_deg"])
    elif "lat_rad" in metric and "lon_rad" in metric:
        lat_deg = np.rad2deg(np.asarray(metric["lat_rad"]))
        lon_deg = np.rad2deg(np.asarray(metric["lon_rad"]))
    else:
        raise KeyError("metric needs lat_deg/lon_deg (or lat_rad/lon_rad) for o3 climo")

    ny = lat_deg.size
    nx = lon_deg.size
    lat_col = np.broadcast_to(lat_deg[:, None], (ny, nx)).reshape(-1)
    lon_col = np.broadcast_to(lon_deg[None, :], (ny, nx)).reshape(-1)
    return climo.to_columns(lat_col, lon_col, play_hpa)   # (ncol, nz)


def compute_qrad_rrtmg(
    state:   ModelState,
    metric:  dict,
    config:  RadRRTMGConfig,
    sst:     jax.Array,          # (ny, nx)  K — per-column surface temperature
    o3vmr:   Optional[np.ndarray] = None,
) -> jax.Array:                  # (nz, ny, nx) K/s heating rate
    """
    Compute RRTMG_LW radiative heating rates without applying them.

    Returns ``dTABS/dt`` in K/s — call ``TABS += dt * qrad`` every step to
    apply, matching gSAM's ``t(i,j,k) = t(i,j,k) + qrad(i,j,k)*dtn`` pattern
    (gSAM recomputes qrad every ``nrad`` steps but applies it every step).

    Clouds use ``state.QC`` / ``state.QI`` via RRTMG's built-in optics.
    Surface temperature is per-column. ``o3vmr`` may be a ``(nz,)`` column
    profile or a ``(ncol, nz)`` per-column field; if ``None`` it falls back
    to the analytic Gaussian.
    """
    plev_hpa = build_plev_hpa(metric)                               # (nz+1,)
    play_hpa = 0.5 * (plev_hpa[:-1] + plev_hpa[1:])                 # (nz,)
    if o3vmr is None:
        o3vmr_arr = analytic_o3_vmr(play_hpa)                       # (nz,)
    else:
        o3vmr_arr = np.asarray(o3vmr, dtype=np.float64)

    nz, ny, nx = state.TABS.shape

    def _host_callback(TABS_np, QV_np, QC_np, QI_np, sst_np):
        return _compute_dTABS_dt_only_host(
            TABS_host=np.asarray(TABS_np, dtype=np.float64),
            QV_host  =np.asarray(QV_np,   dtype=np.float64),
            QC_host  =np.asarray(QC_np,   dtype=np.float64),
            QI_host  =np.asarray(QI_np,   dtype=np.float64),
            sst_host =np.asarray(sst_np,  dtype=np.float64),
            play_hpa =play_hpa.astype(np.float64),
            plev_hpa =plev_hpa.astype(np.float64),
            o3vmr    =o3vmr_arr,
            cfg=config,
        ).astype(np.float32)

    out_shape = jax.ShapeDtypeStruct((nz, ny, nx), jnp.float32)
    return jax.pure_callback(
        _host_callback, out_shape,
        state.TABS, state.QV, state.QC, state.QI, sst,
        vmap_method="sequential",
    )


def compute_qrad_and_lwds_rrtmg(
    state:   ModelState,
    metric:  dict,
    config:  RadRRTMGConfig,
    sst:     jax.Array,          # (ny, nx) K
    o3vmr:   Optional[np.ndarray] = None,
) -> tuple[jax.Array, jax.Array]:
    """Same as :func:`compute_qrad_rrtmg` but also returns the surface
    down-welling LW flux ``lwds`` as an ``(ny, nx)`` array (W/m²), needed
    by the SLM radiative balance.
    """
    plev_hpa = build_plev_hpa(metric)
    play_hpa = 0.5 * (plev_hpa[:-1] + plev_hpa[1:])
    if o3vmr is None:
        o3vmr_arr = analytic_o3_vmr(play_hpa)
    else:
        o3vmr_arr = np.asarray(o3vmr, dtype=np.float64)

    nz, ny, nx = state.TABS.shape

    def _host_callback(TABS_np, QV_np, QC_np, QI_np, sst_np):
        dTABS_dt, lwds = _compute_dTABS_dt_host(
            TABS_host=np.asarray(TABS_np, dtype=np.float64),
            QV_host  =np.asarray(QV_np,   dtype=np.float64),
            QC_host  =np.asarray(QC_np,   dtype=np.float64),
            QI_host  =np.asarray(QI_np,   dtype=np.float64),
            sst_host =np.asarray(sst_np,  dtype=np.float64),
            play_hpa =play_hpa.astype(np.float64),
            plev_hpa =plev_hpa.astype(np.float64),
            o3vmr    =o3vmr_arr,
            cfg=config,
        )
        return (
            dTABS_dt.astype(np.float32),
            lwds.astype(np.float32),
        )

    out_shapes = (
        jax.ShapeDtypeStruct((nz, ny, nx), jnp.float32),
        jax.ShapeDtypeStruct((ny, nx),     jnp.float32),
    )
    return jax.pure_callback(
        _host_callback, out_shapes,
        state.TABS, state.QV, state.QC, state.QI, sst,
        vmap_method="sequential",
    )


def rad_rrtmg_proc(
    state:   ModelState,
    metric:  dict,
    config:  RadRRTMGConfig,
    dt:      float,
    sst:     jax.Array,          # (ny, nx)  K — per-column surface temperature
    o3vmr:   Optional[np.ndarray] = None,
) -> ModelState:
    """
    One RRTMG_LW step. Computes heating rates and applies ``dTABS/dt · dt``
    to ``state.TABS`` in a single call (legacy interface; prefer
    ``compute_qrad_rrtmg`` + per-step application to match gSAM timing).
    """
    dTABS_dt = compute_qrad_rrtmg(state, metric, config, sst, o3vmr=o3vmr)
    return ModelState(
        U=state.U, V=state.V, W=state.W,
        TABS=state.TABS + dt * dTABS_dt,
        QV=state.QV, QC=state.QC, QI=state.QI,
        QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE, p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
