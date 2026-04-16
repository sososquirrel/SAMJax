"""1-moment bulk microphysics: saturation adjustment, precipitation processes (Kessler),
upwind sedimentation with FCT limiter. Phase partition by T-dependent weights."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as _scipy_gamma

from jsam.core.state import ModelState


# Constants (gSAM)
G_GRAV   = 9.79764
CP       = 1004.64
CPV      = 1870.0
CPW      = 3991.86795711963
RV       = 461.5
RGAS     = 287.04
EPS      = 0.622
LV       = 2.501e6
LF       = 0.337e6
LS       = 2.834e6
FAC_COND = LV / CP
FAC_FUS  = LF / CP
FAC_SUB  = LS / CP
THERCO   = 2.40e-2
DIFFELQ  = 2.21e-5
MUELQ    = 1.717e-5
RAD_EARTH  = 6371229.0
SIGMA_SB   = 5.670373e-8
EMIS_WATER = 0.98
PI         = float(np.pi)


@dataclass(frozen=True)
class MicroParams:
    """SAM1MOM parameters (gSAM defaults)."""
    tbgmin: float = 253.16
    tbgmax: float = 273.16
    tprmin: float = 268.16
    tprmax: float = 283.16
    tgrmin: float = 223.16
    tgrmax: float = 283.16
    qcw0:     float = 1.0e-3
    qci0:     float = 1.0e-4
    alphaelq: float = 1.0e-3
    betaelq:  float = 1.0e-3
    rhor: float = 1000.0
    rhos: float = 100.0
    rhog: float = 400.0
    nzeror: float = 8.0e6
    nzeros: float = 3.0e6
    nzerog: float = 4.0e6
    a_rain: float = 842.0
    b_rain: float = 0.8
    a_snow: float = 4.84
    b_snow: float = 0.25
    a_grau: float = 94.5
    b_grau: float = 0.5
    erccoef: float = 1.0
    esccoef: float = 1.0
    esicoef: float = 0.1
    egccoef: float = 1.0
    egicoef: float = 0.1
    qp_threshold: float = 1.0e-12
    icefall_fudge: float = 1.0
    gamma_rave:    float = 1.0
    donograupel: bool = False
    do_ice_fall: bool = True
    # Khairoutdinov-Kogan (2000) autoconversion
    # Note: gSAM IRMA config (prm_debug500) uses doKKauto=.true. for better agreement
    # with observations; defaulting to True to match oracle behavior
    doKKauto: bool = True
    doKKaccr: bool = False
    Nc_land:  float = 300.0   # cloud droplet concentration over land  [cm^-3]
    Nc_ocn:   float = 50.0    # cloud droplet concentration over ocean [cm^-3]
    do_scale_dependence_of_autoconv: bool = True


def _gamma_coefs(p: MicroParams) -> dict:
    """Gamma function values derived from terminal-velocity exponents.

    gSAM declares these as real*4 (float32), so we truncate to float32
    precision to match (micro_params.f90:82).
    """
    def _gam32(x):
        return float(np.float32(_scipy_gamma(x)))

    return {
        "gamr1": _gam32(3 + p.b_rain),
        "gamr2": _gam32((5 + p.b_rain) / 2),
        "gams1": _gam32(3 + p.b_snow),
        "gams2": _gam32((5 + p.b_snow) / 2),
        "gamg1": _gam32(3 + p.b_grau),
        "gamg2": _gam32((5 + p.b_grau) / 2),
    }


def _esatw(tabs: jax.Array) -> jax.Array:
    """Saturation vapor pressure over water (mb, Buck 1981)."""
    return 6.1121 * jnp.exp(17.502 * (tabs - 273.16) / (tabs - 32.19))

def _esati(tabs: jax.Array) -> jax.Array:
    """Saturation vapor pressure over ice (mb, Buck 1981)."""
    return 6.1121 * jnp.exp(22.587 * (tabs - 273.16) / (tabs + 0.7))

def qsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over water (kg/kg)."""
    es = _esatw(tabs)
    return EPS * es / jnp.maximum(es, pres_mb - es)

def qsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over ice (kg/kg)."""
    es = _esati(tabs)
    return EPS * es / jnp.maximum(es, pres_mb - es)

def _dtqsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsatw)/d(tabs)."""
    es = _esatw(tabs)
    a1, T0, T1 = 17.502, 273.16, 32.19
    dtesatw = es * a1 * (T0 - T1) / (tabs - T1) ** 2
    return 0.622 * dtesatw / (pres_mb - es) * (1.0 + es / (pres_mb - es))

def _dtqsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsati)/d(tabs)."""
    es = _esati(tabs)
    a1, T0, T1 = 22.587, 273.16, -0.7
    dtesati = es * a1 * (T0 - T1) / (tabs - T1) ** 2
    return 0.622 * dtesati / (pres_mb - es) * (1.0 + es / (pres_mb - es))


# ---------------------------------------------------------------------------
# Saturation adjustment  (port of cloud.f90)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params", "n_iter"))
def satadj(TABS: jax.Array, QV: jax.Array, QC: jax.Array, QI: jax.Array,
           QR: jax.Array, QS: jax.Array, QG: jax.Array, metric: dict,
           params: MicroParams, n_iter: int = 20,
           tabs_dry_override: "jax.Array | None" = None,
           tabs_guess: "jax.Array | None" = None) -> tuple:
    """Newton-iteration saturation adjustment; returns (TABS_new, QV_new, QC_new, QI_new)."""
    pres_mb  = metric["pres"][:, None, None] / 100.0
    gamaz_3d = metric["gamaz"][:, None, None]

    a_bg = 1.0 / (params.tbgmax - params.tbgmin)
    b_bg = params.tbgmin * a_bg
    a_pr = 1.0 / (params.tprmax - params.tprmin)
    b_pr = params.tprmin * a_pr
    a_gr = 0.0 if params.donograupel else 1.0 / (params.tgrmax - params.tgrmin)

    if tabs_dry_override is not None:
        tabs_dry = tabs_dry_override
    else:
        omp0 = jnp.clip((TABS - params.tprmin) * a_pr, 0.0, 1.0)
        qp_liq = (QR + QS + QG) * omp0
        qp_ice = (QR + QS + QG) * (1.0 - omp0)
        tabs_dry = TABS - FAC_COND * (QC + qp_liq) - FAC_SUB * (QI + qp_ice)

    q  = QV + QC + QI   # total non-precip water
    qp = QR + QS + QG   # total precip water

    def _newton(tabs1):
        om = jnp.clip(a_bg * tabs1 - b_bg, 0.0, 1.0)    # cloud liquid fraction

        lstarn  = FAC_COND + (1.0 - om) * FAC_FUS
        dlstarn = jnp.where(
            (tabs1 > params.tbgmin) & (tabs1 < params.tbgmax),
            a_bg * FAC_FUS, 0.0,
        )

        omp_ = jnp.clip(a_pr * tabs1 - b_pr, 0.0, 1.0)
        lstarp  = FAC_COND + (1.0 - omp_) * FAC_FUS
        dlstarp = jnp.where(
            (tabs1 > params.tprmin) & (tabs1 < params.tprmax),
            a_pr * FAC_FUS, 0.0,
        )

        rh_homo = jnp.where(
            (tabs1 < 235.0) & (QI < 1.0e-8),
            2.583 - tabs1 / 207.8,
            1.0,
        )

        qsati_homo = qsati(tabs1, pres_mb) * rh_homo
        qsat = jnp.where(
            tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, qsati_homo,
                om * qsatw(tabs1, pres_mb) + (1.0 - om) * qsati_homo,
            ),
        )
        drh_homo = jnp.where(
            (tabs1 < 235.0) & (QI < 1.0e-8),
            -1.0 / 207.8,
            0.0,
        )
        dtqsati_homo = (_dtqsati(tabs1, pres_mb) * rh_homo
                        + qsati(tabs1, pres_mb) * drh_homo)
        dqsat = jnp.where(
            tabs1 >= params.tbgmax, _dtqsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, dtqsati_homo,
                om * _dtqsatw(tabs1, pres_mb) + (1.0 - om) * dtqsati_homo,
            ),
        )

        sat_excess = q - qsat
        fff  = tabs_dry - tabs1 + lstarn * sat_excess + lstarp * qp
        dfff = dlstarn * sat_excess + dlstarp * qp - lstarn * dqsat - 1.0
        return tabs1 - fff / dfff

    # Initial guess: current temperature (matches gSAM cloud.f90 tabs1=tabs).
    # The alternative tabs_dry+FAC_COND*q overshoots to ~490 K for q>>qsat,
    # where es >> pres and the Newton step is O(0.001 K) — never converges
    # in 20 iterations.  Starting from TABS itself (near the true root) avoids
    # the ill-conditioned high-T regime entirely.
    # F11: when tabs_guess is provided (= physical TABS from previous satadj),
    # use it as the Newton start, matching gSAM cloud.f90: tabs1 = tabs2.
    tabs1 = (tabs_guess if tabs_guess is not None else TABS) + 0.0

    # Fixed Newton iterations (unrolled at trace time → no Python control-flow overhead)
    for _ in range(n_iter):
        tabs1 = _newton(tabs1)

    # Final condensate and phase partitioning
    om_f = jnp.clip(a_bg * tabs1 - b_bg, 0.0, 1.0)
    # C12 fix: apply rh_homo to final qsat (consistent with Newton iteration)
    rh_homo_f = jnp.where(
        (tabs1 < 235.0) & (QI < 1.0e-8),
        2.583 - tabs1 / 207.8,
        1.0,
    )
    qsati_homo_f = qsati(tabs1, pres_mb) * rh_homo_f
    qsat_f = jnp.where(
        tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
        jnp.where(
            tabs1 <= params.tbgmin, qsati_homo_f,
            om_f * qsatw(tabs1, pres_mb) + (1.0 - om_f) * qsati_homo_f,
        ),
    )
    qn_new = jnp.maximum(0.0, q - qsat_f)

    # For undersaturated air (qn_new==0) the Newton result is wrong because the
    # iteration found a spurious root.  Override tabs1 to the no-condensation
    # temperature: tabs_dry + lstarp*qp  (matching gSAM cloud.f90 logic where
    # the Newton block is skipped entirely when q <= qsatt).
    omp_f = jnp.clip(
        (tabs1 - params.tprmin) / (params.tprmax - params.tprmin), 0.0, 1.0
    )
    lstarp_f = FAC_COND + (1.0 - omp_f) * FAC_FUS
    tabs_no_cond = tabs_dry + lstarp_f * qp
    tabs1 = jnp.where(qn_new > 0.0, tabs1, tabs_no_cond)

    return tabs1, q - qn_new, qn_new * om_f, qn_new * (1.0 - om_f)


# ---------------------------------------------------------------------------
# Evaporation coefficients  (the TABS-dependent part of precip_init.f90)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params",))
def _compute_evap_coefs(
    TABS:   jax.Array,   # (nz, ny, nx)
    metric: dict,
    params: MicroParams,
) -> tuple:
    """
    Compute the six evaporation coefficient arrays that depend on TABS.

    gSAM computes these inside precip_init() which is called every 10 steps
    (microphysics.f90:328).  Returning them as a plain tuple lets micro_proc
    cache and reuse them for the intervening 9 steps.

    Returns: (evapr1, evapr2, evaps1, evaps2, evapg1, evapg2)  each (nz,ny,nx)
    """
    rho     = metric["rho"][:, None, None]
    pres_mb = metric["pres"][:, None, None] / 100.0
    p = params

    gcoefs = _gamma_coefs(p)
    gamr2  = gcoefs["gamr2"]
    gams2  = gcoefs["gams2"]
    gamg2  = gcoefs["gamg2"]

    pratio = jnp.sqrt(1.29 / rho)

    rrr1 = 393.0 / (TABS + 120.0) * (TABS / 273.0) ** 1.5
    rrr2 = (TABS / 273.0) ** 1.94 * (1000.0 / pres_mb)
    estw = 100.0 * _esatw(TABS)
    esti = 100.0 * _esati(TABS)

    c1r = (LV / (TABS * RV) - 1.0) * LV / (THERCO * rrr1 * TABS)
    c2r = RV * TABS / (DIFFELQ * rrr2 * estw)
    evapr1 = (0.78 * 2.0 * np.pi * p.nzeror / jnp.sqrt(np.pi * p.rhor * p.nzeror * rho)
              / (c1r + c2r))
    evapr2 = (0.31 * 2.0 * np.pi * p.nzeror * gamr2 * 0.89
              * jnp.sqrt(p.a_rain / (MUELQ * rrr1))
              / (np.pi * p.rhor * p.nzeror) ** ((5 + p.b_rain) / 8.0) / (c1r + c2r)
              * rho ** ((1 + p.b_rain) / 8.0) * jnp.sqrt(pratio))

    c1s = (LS / (TABS * RV) - 1.0) * LS / (THERCO * rrr1 * TABS)
    c2s = RV * TABS / (DIFFELQ * rrr2 * esti)
    evaps1 = (0.65 * 4.0 * p.nzeros / jnp.sqrt(np.pi * p.rhos * p.nzeros * rho)
              / (c1s + c2s))
    evaps2 = (0.49 * 4.0 * p.nzeros * gams2 * jnp.sqrt(p.a_snow / (MUELQ * rrr1))
              / (np.pi * p.rhos * p.nzeros) ** ((5 + p.b_snow) / 8.0) / (c1s + c2s)
              * rho ** ((1 + p.b_snow) / 8.0) * jnp.sqrt(pratio))

    evapg1 = (0.65 * 4.0 * p.nzerog / jnp.sqrt(np.pi * p.rhog * p.nzerog * rho)
              / (c1s + c2s))
    evapg2 = (0.49 * 4.0 * p.nzerog * gamg2 * jnp.sqrt(p.a_grau / (MUELQ * rrr1))
              / (np.pi * p.rhog * p.nzerog) ** ((5 + p.b_grau) / 8.0) / (c1s + c2s)
              * rho ** ((1 + p.b_grau) / 8.0) * jnp.sqrt(pratio))

    return evapr1, evapr2, evaps1, evaps2, evapg1, evapg2


# Module-level cache: stores evap coefs and the nstep they were computed at.
# Refreshed every 10 steps to match gSAM's precip_init call frequency.
_evap_coef_cache: dict = {"nstep": -999, "coefs": None}


# ---------------------------------------------------------------------------
# Precipitation processes  (port of precip_proc.f90)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params",))
def precip_proc(
    TABS: jax.Array,   # (nz, ny, nx)  K  — after satadj
    QC:   jax.Array,   # cloud liquid
    QI:   jax.Array,   # cloud ice
    QR:   jax.Array,   # rain
    QS:   jax.Array,   # snow
    QG:   jax.Array,   # graupel
    QV:   jax.Array,   # water vapour  (needed for evaporation)
    metric: dict,
    params: MicroParams,
    dt: float,
    evap_coefs: tuple | None = None,  # pre-computed (evapr1,evapr2,evaps1,evaps2,evapg1,evapg2)
    landmask: "jax.Array | None" = None,  # (ny, nx) bool/int; used for KK Ncc
) -> tuple:  # (TABS_new, QV_new, QC_new, QI_new, QR_new, QS_new, QG_new)
    """
    Microphysical source/sink terms: autoconversion, accretion, and evaporation.

    Does NOT include sedimentation (handled by precip_fall).
    All arithmetic is implicit to keep condensate non-negative.

    Port of gSAM MICRO_SAM1MOM/precip_proc.f90.
    """
    rho     = metric["rho"][:, None, None]     # (nz,1,1) kg/m³
    pres_mb = metric["pres"][:, None, None] / 100.0   # (nz,1,1) mb

    p = params
    a_bg = 1.0 / (p.tbgmax - p.tbgmin)
    a_pr = 1.0 / (p.tprmax - p.tprmin)
    b_pr = p.tprmin * a_pr
    a_gr = 0.0 if p.donograupel else 1.0 / (p.tgrmax - p.tgrmin)
    b_gr = p.tgrmin * a_gr

    # Phase weight scalars (temperature-dependent)
    omn = jnp.clip((TABS - p.tbgmin) * a_bg, 0.0, 1.0)   # cloud liquid fraction
    omp = jnp.clip((TABS - p.tprmin) * a_pr, 0.0, 1.0)   # rain fraction of qp
    omg = jnp.clip((TABS - p.tgrmin) * a_gr, 0.0, 1.0)   # graupel fraction of snow+grau

    qcc = QC   # cloud liquid (= QC since omn already applied by satadj)
    qii = QI   # cloud ice
    qp  = QR + QS + QG

    # Bulk precip split
    qrr = qp * omp
    qss = qp * (1.0 - omp) * (1.0 - omg)
    qgg = qp * (1.0 - omp) * omg

    pratio = jnp.sqrt(1.29 / rho)   # density correction for terminal velocity

    # Powers for accretion coefficients
    powr1 = (3.0 + p.b_rain) / 4.0
    pows1 = (3.0 + p.b_snow) / 4.0
    powg1 = (3.0 + p.b_grau) / 4.0
    powr2 = (5.0 + p.b_rain) / 8.0
    pows2 = (5.0 + p.b_snow) / 8.0
    powg2 = (5.0 + p.b_grau) / 8.0

    gcoefs = _gamma_coefs(p)
    gamr1 = gcoefs["gamr1"]
    gams1 = gcoefs["gams1"]
    gamg1 = gcoefs["gamg1"]

    # ── Accretion coefficients (precip_init.f90) ──────────────────────────────
    # Rain
    accrrc = (0.25 * np.pi * p.nzeror * p.a_rain * gamr1 * pratio /
              (np.pi * p.rhor * p.nzeror / rho) ** powr1 * p.erccoef)
    # Snow
    coef1_snow = (0.25 * np.pi * p.nzeros * p.a_snow * gams1 * pratio /
                  (np.pi * p.rhos * p.nzeros / rho) ** pows1)
    accrsc = coef1_snow * p.esccoef
    accrsi = coef1_snow * p.esicoef
    # Graupel
    coef1_grau = (0.25 * np.pi * p.nzerog * p.a_grau * gamg1 * pratio /
                  (np.pi * p.rhog * p.nzerog / rho) ** powg1)
    accrgc = coef1_grau * p.egccoef
    accrgi = coef1_grau * p.egicoef

    # ── Ice auto-conversion fudge factor ─────────────────────────────────────
    # coefice = exp(0.025*(T-273.15)) for T < 273.15, else 0
    coefice = jnp.where(TABS < 273.15, jnp.exp(0.025 * (TABS - 273.15)), 0.0)

    # ── Evaporation coefficients (precip_init.f90) ───────────────────────────
    # Use cached coefs if provided (recomputed every 10 steps by micro_proc,
    # matching gSAM's mod(nstep,10).eq.0 call to precip_init).
    if evap_coefs is not None:
        evapr1, evapr2, evaps1, evaps2, evapg1, evapg2 = evap_coefs
    else:
        evapr1, evapr2, evaps1, evaps2, evapg1, evapg2 = _compute_evap_coefs(
            TABS, metric, params,
        )

    qn = qcc + qii   # total cloud condensate

    # ── Branch 1: autoconversion + accretion (when qn > 0) ──────────────────
    # Cloud water autoconversion
    if p.doKKauto:
        # Khairoutdinov-Kogan (2000) parameterization (precip_proc.f90:95-119)
        # Ncc depends on land/ocean; KK uses zero threshold (qcw0 = 0 in gSAM)
        qcw0_eff = 0.0
        if landmask is not None:
            # landmask: (ny, nx) bool/int; broadcast to (1, ny, nx)
            _lm = jnp.asarray(landmask, dtype=jnp.float32)[None, :, :]
            Ncc = _lm * p.Nc_land + (1.0 - _lm) * p.Nc_ocn
        else:
            Ncc = p.Nc_ocn  # all-ocean fallback
        # KK rate: power 1.47 not 2.47 because autor is multiplied by qcc implicitly
        autor = 1350.0 * jnp.maximum(qcc, 0.0) ** 1.47 / (Ncc ** 1.79 + 1e-30)
        # Scale-dependence factor: dx must be in km; factor = min(1, 100/(dx_km*mu))^2
        # At <100 km effective resolution (typical global), factor = 1 (no change)
        dx_km = metric["dx_lon"] / 1000.0      # convert m → km
        mu_2d = metric["cos_lat"][None, :, None]  # (1, ny, 1)
        eff_dx_km = dx_km * mu_2d + 1e-30
        if p.do_scale_dependence_of_autoconv:
            scale_fac = jnp.minimum(1.0, 100.0 / eff_dx_km) ** 2
        else:
            scale_fac = jnp.maximum(1.0, (0.001 * eff_dx_km) ** 2)
        autor = autor * scale_fac
    else:
        # Standard Kessler parameterization
        qcw0_eff = p.qcw0
        autor = jnp.where(qcc > p.qcw0, p.alphaelq, 0.0)

    # Cloud ice autoconversion
    autos = jnp.where(qii > p.qci0, p.betaelq * coefice, 0.0)

    # Accretion rates
    accrr  = jnp.where(omp > 0.001, accrrc * qrr ** powr1, 0.0)
    accrcs = jnp.where((omp < 0.999) & (omg < 0.999), accrsc * qss ** pows1, 0.0)
    accris = jnp.where((omp < 0.999) & (omg < 0.999),
                       accrsi * coefice * qss ** pows1, 0.0)
    accrcg = jnp.where((omp < 0.999) & (omg > 0.001), accrgc * qgg ** powg1, 0.0)
    accrig = jnp.where((omp < 0.999) & (omg > 0.001),
                       accrgi * coefice * qgg ** powg1, 0.0)

    # Implicit scheme for qcc and qii (precip_proc.f90 line 186-187)
    qcc_imp = (qcc + dt * autor * qcw0_eff) / (1.0 + dt * (accrr + accrcs + accrcg + autor))
    qii_imp = (qii + dt * autos * p.qci0) / (1.0 + dt * (accris + accrig + autos))

    dq_conv = dt * (accrr * qcc_imp + autor * (qcc_imp - qcw0_eff)
                    + (accris + accrig) * qii_imp
                    + (accrcs + accrcg) * qcc_imp
                    + autos * (qii_imp - p.qci0))
    dq_conv = jnp.minimum(dq_conv, qn)   # can't remove more than available

    # ── Branch 2: evaporation outside cloud (qn=0, qp > threshold) ──────────
    qsat_evap = jnp.where(
        TABS >= p.tbgmax, qsatw(TABS, pres_mb),
        jnp.where(TABS <= p.tbgmin, qsati(TABS, pres_mb),
                  omn * qsatw(TABS, pres_mb) + (1.0 - omn) * qsati(TABS, pres_mb)),
    )
    subsatfrac = QV / (qsat_evap + 1e-30) - 1.0   # negative when subsaturated

    dq_evap_r = jnp.where(omp > 0.001,
                           (evapr1 * jnp.sqrt(qrr + 1e-30) +
                            evapr2 * (qrr + 1e-30) ** powr2) * subsatfrac, 0.0)
    dq_evap_s = jnp.where((omp < 0.999) & (omg < 0.999),
                           (evaps1 * jnp.sqrt(qss + 1e-30) +
                            evaps2 * (qss + 1e-30) ** pows2) * subsatfrac, 0.0)
    dq_evap_g = jnp.where((omp < 0.999) & (omg > 0.001),
                           (evapg1 * jnp.sqrt(qgg + 1e-30) +
                            evapg2 * (qgg + 1e-30) ** powg2) * subsatfrac, 0.0)
    dq_evap = jnp.maximum(-0.5 * qp, (dq_evap_r + dq_evap_s + dq_evap_g) * dt)

    # ── Select branch: conv if qn > 0, evap if qp>threshold and qn=0 ─────────
    has_cloud  = qn > 0.0
    has_precip_no_cloud = (qp > p.qp_threshold) & (qn <= 0.0)

    dq = jnp.where(has_cloud, dq_conv,
                   jnp.where(has_precip_no_cloud, dq_evap, 0.0))

    # D20 fix: tiny qp fully evaporated when qn=0 (gSAM precip_proc.f90:227-231)
    tiny_evap = (qn <= 0.0) & (qp > 0.0) & (qp <= p.qp_threshold)
    dq = jnp.where(tiny_evap, -qp, dq)

    # Apply net change: dq > 0 → cloud→precip;  dq < 0 → precip→vapour
    qp_raw = qp + dq
    qp_new = jnp.maximum(0.0, qp_raw)
    # D19 fix: mass conservation — clipped qp mass returned to vapor
    clip_mass = qp_raw - qp_new   # negative when clipping occurred
    q_new  = jnp.maximum(0.0, (QV + qn) - dq - clip_mass)   # q = QV + qn + recovered mass

    # Re-partition updated q and qp
    qv_new  = jnp.maximum(0.0, q_new - jnp.maximum(0.0, q_new - qsat_evap))
    qn_new  = q_new - qv_new
    qc_new  = qn_new * omn
    qi_new  = qn_new * (1.0 - omn)

    qr_new = qp_new * omp
    qs_new = qp_new * (1.0 - omp) * (1.0 - omg)
    qg_new = qp_new * (1.0 - omp) * omg

    # Temperature tendency from latent heat exchange  (ΔT = fac * Δq)
    lfac_n  = FAC_COND * omn + FAC_SUB * (1.0 - omn)
    lfac_p  = FAC_COND * omp + FAC_SUB * (1.0 - omp)
    dTABS = (lfac_p * (qp_new - qp)   # latent heat from precip change
             - lfac_n * (qn_new - qn)) # latent heat from condensate change
    TABS_new = TABS + dTABS

    return TABS_new, qv_new, qc_new, qi_new, qr_new, qs_new, qg_new


# ---------------------------------------------------------------------------
# Precipitation sedimentation  (port of precip_fall.f90 — upwind scheme)
# ---------------------------------------------------------------------------

def _fall_col_one(
    qp_col:   jax.Array,   # (nz,)  mixing ratio (kg/kg)
    rho_col:  jax.Array,   # (nz,)  kg/m³
    dz_col:   jax.Array,   # (nz,)  m  layer thickness
    v_coef:   float,       # pre-computed: a*gamma(4+b)/6 / (pi*rho_hyd*N0)^(b/4)
    c_exp:    float,       # b/4
    vt_max:   float,       # velocity cap (m/s): 9 rain, 2 snow, 10 graupel
    dt:       float,
) -> tuple:  # (qp_new, cfl_max)
    """
    MPDATA sedimentation for one species in one column with adaptive subcycling.

    Matches gSAM precip_fall.f90: upwind pass + anti-diffusive correction
    with non-oscillatory (FCT) flux limiting.

    Adaptive subcycling (gSAM precip_fall.f90:109-118): if CFL > 0.9,
    use nprec = ceil(CFL/0.9) substeps to maintain stability.

    Terminal velocity follows gSAM microphysics.f90:484-486 and
    precip_fall.f90:54,97:
      term_vel = v_coef * (rho*qp)^c_exp            (microphysics.f90)
      vt = sqrt(1.29/rho) * term_vel                 (precip_fall.f90 rhofac)
    where v_coef = a * gamma(4+b)/6 / (pi*rho_hyd*N0)^c_exp.

    Returns (qp_new, cfl_max).
    """
    qp_safe = jnp.maximum(qp_col, 0.0)
    rhofac = jnp.sqrt(1.29 / rho_col)

    # Terminal velocity (m/s, positive downward)
    # gSAM: term_vel_qp = min(vt_max, v_coef*(rho*qp)^c_exp)
    # then precip_fall.f90:97: wp_vel = rhofac * term_vel_qp
    term_vel = jnp.minimum(vt_max, v_coef * (rho_col * qp_safe) ** c_exp)
    vt = rhofac * term_vel
    vt = jnp.where(qp_safe > 1e-12, vt, 0.0)

    # irhoadz[k] = 1/(rho[k]*dz[k])  (gSAM: 1/(rho*adz), with dz factor in wp)
    irhoadz = 1.0 / (rho_col * dz_col)

    # Courant number (dimensionless, positive downward)
    # gSAM: wp = vt*dt/dz (scalar dz * adz → dz_col here)
    # For subcycling, we compute nprec and scale wp accordingly
    cfl_raw = vt * dt / dz_col  # unclipped CFL
    cfl_max = jnp.max(cfl_raw)

    # Adaptive subcycling: if CFL > 0.9, use multiple substeps (gSAM line 109-110)
    nprec = jnp.maximum(1, jnp.ceil(cfl_max / 0.9)).astype(jnp.int32)
    nprec_float = jnp.asarray(nprec, dtype=jnp.float32)

    # Scale velocity for the substeps
    wp = jnp.minimum(1.0, cfl_raw / nprec_float)  # each substep uses dt/nprec

    def _mpdata_step(qp_current):
        """Single MPDATA step: upwind + anti-diffusive correction with FCT."""
        # --- Pre-compute mx/mn from current field (gSAM nonos=.true. line 130-135) ---
        qp_above = jnp.concatenate([qp_current[1:], qp_current[-1:]])
        qp_below = jnp.concatenate([qp_current[:1], qp_current[:-1]])
        mx0 = jnp.maximum(jnp.maximum(qp_below, qp_above), qp_current)
        mn0 = jnp.minimum(jnp.minimum(qp_below, qp_above), qp_current)

        # --- Pass 1: first-order upwind (gSAM line 141-149) ---
        fz = qp_current * wp
        fz_top = jnp.concatenate([fz[1:], jnp.zeros((1,))])
        tmp_qp = qp_current - (fz_top - fz) * irhoadz
        tmp_qp = jnp.maximum(0.0, tmp_qp)

        # --- Anti-diffusive correction (gSAM line 151-162) ---
        tmp_flux = tmp_qp * wp
        tmp_flux_below = jnp.concatenate([tmp_flux[:1], tmp_flux[:-1]])
        www = 0.5 * (1.0 + wp * irhoadz) * (tmp_flux_below - tmp_flux)

        # --- FCT limiter (gSAM line 167-187) ---
        tmp_above = jnp.concatenate([tmp_qp[1:], tmp_qp[-1:]])
        tmp_below = jnp.concatenate([tmp_qp[:1], tmp_qp[:-1]])

        mx = jnp.maximum(jnp.maximum(jnp.maximum(tmp_below, tmp_above), tmp_qp), mx0)
        mn = jnp.minimum(jnp.minimum(jnp.minimum(tmp_below, tmp_above), tmp_qp), mn0)

        eps_fct = 1e-10
        www_top = jnp.concatenate([www[1:], jnp.zeros((1,))])

        ppos_www_top = jnp.maximum(0.0, www_top)
        pneg_www_top = jnp.maximum(0.0, -www_top)
        ppos_www     = jnp.maximum(0.0, www)
        pneg_www     = jnp.maximum(0.0, -www)

        # gSAM: mx(k) = rho(k)*adz(k)*(mx-tmp_qp) / (pneg(www(kc))+ppos(www(k))+eps)
        mx_lim = rho_col * dz_col * (mx - tmp_qp) / (pneg_www_top + ppos_www + eps_fct)
        mn_lim = rho_col * dz_col * (tmp_qp - mn) / (ppos_www_top + pneg_www + eps_fct)

        mx_below = jnp.concatenate([mx_lim[:1], mx_lim[:-1]])
        mn_below = jnp.concatenate([mn_lim[:1], mn_lim[:-1]])

        fz_corr = (fz
                   + ppos_www * jnp.minimum(1.0, jnp.minimum(mx_lim, mn_below))
                   - pneg_www * jnp.minimum(1.0, jnp.minimum(mx_below, mn_lim)))
        fz_corr_top = jnp.concatenate([fz_corr[1:], jnp.zeros((1,))])

        qp_out = jnp.maximum(0.0, qp_current - (fz_corr_top - fz_corr) * irhoadz)
        return qp_out

    # Run nprec substeps via while_loop (gSAM line 120-244)
    def _body_fun(carry):
        i, qp_step = carry
        qp_step = _mpdata_step(qp_step)
        return (i + 1, qp_step)

    def _cond_fun(carry):
        i, qp_step = carry
        return i < nprec

    _, qp_new = jax.lax.while_loop(_cond_fun, _body_fun, (0, qp_safe))

    return qp_new, cfl_max


def precip_fall(
    QR:   jax.Array,   # (nz, ny, nx)
    QS:   jax.Array,   # (nz, ny, nx)
    QG:   jax.Array,   # (nz, ny, nx)
    TABS: jax.Array,   # (nz, ny, nx)  for latent heat tendency
    metric: dict,
    params: MicroParams,
    dt: float,
) -> tuple:  # (QR_new, QS_new, QG_new, TABS_new)
    """
    Gravitational sedimentation of rain, snow, graupel.

    Full MPDATA with anti-diffusive correction + FCT limiter per column.
    Port of gSAM MICRO_SAM1MOM/precip_fall.f90.

    Terminal velocity: v_coef = a * gamma(4+b)/6 / (pi*rho_hyd*N0)^(b/4),
    matching gSAM microphysics.f90:484-486.  rhofac = sqrt(1.29/rho) applied
    in precip_fall (line 54,97).
    """
    rho_1d = np.array(metric["rho"])    # keep as numpy for vmap efficiency
    dz_1d  = np.array(metric["dz"])

    nz, ny, nx = QR.shape

    p = params
    # Pre-compute velocity coefficients: v_coef = a * gamma(4+b)/6 / (pi*rho_hyd*N0)^c
    # gSAM microphysics.f90:481-486
    crain = p.b_rain / 4.0
    csnow = p.b_snow / 4.0
    cgrau = p.b_grau / 4.0
    gamr3 = float(np.float32(_scipy_gamma(4.0 + p.b_rain)))
    gams3 = float(np.float32(_scipy_gamma(4.0 + p.b_snow)))
    gamg3 = float(np.float32(_scipy_gamma(4.0 + p.b_grau)))
    vrain = p.a_rain * gamr3 / 6.0 / (np.pi * p.rhor * p.nzeror) ** crain
    vsnow = p.a_snow * gams3 / 6.0 / (np.pi * p.rhos * p.nzeros) ** csnow
    vgrau = p.a_grau * gamg3 / 6.0 / (np.pi * p.rhog * p.nzerog) ** cgrau

    def _fall_species(qp_3d, v_coef, c_exp, vt_max):
        """Apply MPDATA sedimentation to a 3D species field."""
        def _col(qp_col):
            qp_new, _ = _fall_col_one(qp_col, rho_1d, dz_1d,
                                       v_coef, c_exp, vt_max, dt)
            return qp_new

        _vmap_i = jax.vmap(_col, in_axes=1, out_axes=1)
        _vmap_ji = jax.vmap(_vmap_i, in_axes=1, out_axes=1)
        return _vmap_ji(qp_3d)

    QR_new = _fall_species(QR, vrain, crain, 9.0)
    QS_new = _fall_species(QS, vsnow, csnow, 2.0)
    QG_new = _fall_species(QG, vgrau, cgrau, 10.0)

    # Latent heat tendency from sedimentation flux divergence
    # ΔT = -(lfac * d(rhow*vt*qp)/dz) / (rho*cp) * dt
    # Here we approximate: TABS changes by fac * (qp_old - qp_new)
    a_pr = 1.0 / (p.tprmax - p.tprmin)
    omp  = jnp.clip((TABS - p.tprmin) * a_pr, 0.0, 1.0)
    lfac = FAC_COND * omp + FAC_SUB * (1.0 - omp)
    dQR = QR_new - QR
    dQS = QS_new - QS
    dQG = QG_new - QG
    # Temperature cools where precip falls out, warms where it enters
    # (sign: if qp decreases in column → latent heat released above → warm above)
    TABS_new = TABS - lfac * (dQR + dQS + dQG)

    return QR_new, QS_new, QG_new, TABS_new


# ---------------------------------------------------------------------------
# Cloud-ice sedimentation  (port of MICRO_SAM1MOM/ice_fall.f90)
# ---------------------------------------------------------------------------

def _ice_fall_col(
    qi_col:   jax.Array,   # (nz,)   cloud-ice mixing ratio
    rho_col:  jax.Array,   # (nz,)   kg/m³
    dz_col:   jax.Array,   # (nz,)   m
    fudge:    float,
    gamma_rave: float,
    dt: float,
) -> tuple:  # (qi_new, dqi)   dqi used for latent-heat tendency
    qi_safe = jnp.maximum(qi_col, 0.0)

    # Heymsfield (JAS 2003, p.2607) ice terminal velocity, m/s
    # gSAM ice_fall.f90:83,95: qic = rho*qci; vt = 8.66*(qic)^0.24
    # argument is mass density (rho*q), NOT mixing ratio; no rhofac
    qic = rho_col * qi_safe
    vt = fudge * 8.66 * (jnp.maximum(0.0, qic) + 1e-10) ** 0.24 / gamma_rave
    vt = jnp.where(qi_safe > 0.0, vt, 0.0)

    # Work in mass-density units qrho = rho * q
    qrho = rho_col * qi_safe
    qu   = jnp.concatenate([qrho[1:], qrho[-1:]])   # kc = above
    qd   = jnp.concatenate([qrho[:1], qrho[:-1]])   # kb = below

    # MC flux limiter:  phi = max(0, min(0.5*(1+theta), 2, 2*theta))
    denom = qrho - qd
    safe  = jnp.abs(denom) >= 1e-6
    theta = jnp.where(safe, (qu - qrho) / jnp.where(safe, denom, 1.0), 0.0)
    phi   = jnp.where(
        safe,
        jnp.maximum(0.0, jnp.minimum(jnp.minimum(0.5 * (1.0 + theta), 2.0), 2.0 * theta)),
        0.0,
    )

    # Interface CFL coefficient (dz at bottom face of cell k)
    dz_below = jnp.concatenate([dz_col[:1], dz_col[:-1]])
    coef_if  = dt / (0.5 * (dz_below + dz_col))

    # Flux at bottom face of cell k (negative = downward)
    fz = -vt * (qrho - 0.5 * (1.0 - coef_if * vt) * phi * (qrho - qd))
    fz_top = jnp.concatenate([fz[1:], jnp.zeros((1,), dtype=fz.dtype)])  # fz[nz]=0

    # Flux-divergence increment
    dqi = (fz - fz_top) * dt / (rho_col * dz_col)

    qi_new = jnp.maximum(0.0, qi_safe + dqi)
    return qi_new, dqi


def ice_fall(
    QI:   jax.Array,   # (nz, ny, nx)
    TABS: jax.Array,   # (nz, ny, nx)
    metric: dict,
    params: MicroParams,
    dt: float,
) -> tuple:  # (QI_new, TABS_new)
    """
    Gravitational sedimentation of cloud ice with MC flux limiter.

    Port of gSAM MICRO_SAM1MOM/ice_fall.f90 (scalar icefall_fudge branch;
    Heymsfield 2003 terminal velocity).  Latent heat of sublimation is
    released into TABS following the same static-energy convention as
    precip_fall: ΔTABS = -fac_sub · ΔQI.
    """
    rho_1d = np.array(metric["rho"])
    dz_1d  = np.array(metric["dz"])

    def _col(qi_col):
        return _ice_fall_col(qi_col, rho_1d, dz_1d,
                             params.icefall_fudge, params.gamma_rave, dt)

    _vmap_i  = jax.vmap(_col, in_axes=1, out_axes=1)   # over nx
    _vmap_ji = jax.vmap(_vmap_i, in_axes=1, out_axes=1)  # over ny
    QI_new, dQI = _vmap_ji(QI)

    TABS_new = TABS - FAC_SUB * dQI
    return QI_new, TABS_new


# ---------------------------------------------------------------------------
# Top-level microphysics step
# ---------------------------------------------------------------------------

def micro_proc(
    state: ModelState,
    metric: dict,
    params: MicroParams,
    dt: float,
    tabs_phys: "jax.Array | None" = None,   # F11: physical TABS from previous satadj
    landmask: "jax.Array | None" = None,    # (ny, nx) bool/int for KK Ncc
) -> ModelState:
    """
    One microphysics step matching gSAM operator order.

    gSAM ordering (main.f90:299,324):
      advect_all_scalars calls precip_fall + ice_fall FIRST,
      then micro_proc calls cloud (satadj) + precip_proc.

    So: sedimentation → saturation adjustment → precipitation processes.

    Args:
        state     : current ModelState; state.TABS = t (static energy) in F11
                    mode, physical TABS otherwise
        metric    : dict from build_metric (must include 'pres' and 'gamaz')
        params    : MicroParams (use MicroParams() for gSAM defaults)
        dt        : physics timestep (seconds)
        tabs_phys : F11 mode — physical TABS from the previous satadj call
                    (= the TABS saved before advance_scalars in step.py).
                    When set, state.TABS is treated as the advected liquid-ice
                    static energy t.  tabs_phys is used for precip phase
                    fractions (omp) and as the Newton initial guess for satadj,
                    matching gSAM cloud.f90: ``tabs2=tabs; tabs1=tabs2``.

    Returns:
        new ModelState with updated TABS (physical temperature), QV, QC, QI,
        QR, QS, QG
    """
    # In F11 mode, TABS holds t (advected static energy); in standard mode it
    # holds physical temperature.  local variable name is kept as TABS for
    # brevity but the meaning differs between the two branches below.
    TABS = state.TABS
    QV, QC, QI = state.QV, state.QC, state.QI
    QR, QS, QG = state.QR, state.QS, state.QG

    # 1. Precipitation sedimentation (inside advect_all_scalars in gSAM)
    if tabs_phys is not None:
        # F11 mode: use physical TABS (tabs_phys) for omp phase fractions so
        # the rain/ice split is not distorted by the gamaz offset in t.
        # Accumulate the latent-heat delta into t (= TABS) rather than into
        # tabs_phys, matching gSAM where precip_fall updates t but not tabs.
        QR, QS, QG, _tabs_fall_out = precip_fall(QR, QS, QG, tabs_phys, metric, params, dt)
        TABS = TABS + (_tabs_fall_out - tabs_phys)   # t += latent-heat delta
    else:
        QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)

    # 2. Cloud-ice sedimentation (inside advect_all_scalars in gSAM)
    if params.do_ice_fall:
        if tabs_phys is not None:
            # Same pattern: use tabs_phys for the latent-heat bookkeeping;
            # ice_fall's ΔT = -FAC_SUB*ΔQI does not depend on the temperature
            # level, so passing tabs_phys only affects the addend's base value.
            QI, _tabs_ice_out = ice_fall(QI, tabs_phys, metric, params, dt)
            TABS = TABS + (_tabs_ice_out - tabs_phys)   # t += latent-heat delta
        else:
            QI, TABS = ice_fall(QI, TABS, metric, params, dt)

    # 3. Saturation adjustment (cloud() in gSAM micro_proc)
    if tabs_phys is not None:
        # F11 mode: step.py applies the F11 inverse (t → physical TABS) before
        # calling micro_proc, so TABS here is PHYSICAL temperature, not static
        # energy.  gSAM's tabs_dry = t - gamaz = TABS_phys - condensate_terms.
        # We compute it directly from the current physical TABS.
        # Newton initial guess = tabs_phys (physical TABS from previous satadj),
        # matching gSAM cloud.f90: ``tabs2=tabs; tabs1=tabs2``.
        a_pr_m = 1.0 / (params.tprmax - params.tprmin)
        _omp_m = jnp.clip((TABS - params.tprmin) * a_pr_m, 0.0, 1.0)
        _qp_m  = QR + QS + QG
        _tabs_dry = (TABS
                     - FAC_COND * (QC + _qp_m * _omp_m)
                     - FAC_SUB  * (QI + _qp_m * (1.0 - _omp_m)))
        TABS, QV, QC, QI = satadj(
            tabs_phys, QV, QC, QI, QR, QS, QG, metric, params,
            tabs_dry_override=_tabs_dry,
            tabs_guess=tabs_phys,
        )
    else:
        TABS, QV, QC, QI = satadj(
            TABS, QV, QC, QI, QR, QS, QG, metric, params,
        )

    # 4. Precipitation processes (precip_proc() in gSAM micro_proc)
    # Evaporation coefficients are recomputed every 10 steps, matching gSAM's
    # precip_init call frequency: mod(nstep,10).eq.0.and.icycle.eq.1
    nstep = int(state.nstep)
    if _evap_coef_cache["nstep"] < 0 or nstep % 10 == 0:
        _evap_coef_cache["coefs"] = _compute_evap_coefs(TABS, metric, params)
        _evap_coef_cache["nstep"] = nstep
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, QR, QS, QG, QV, metric, params, dt,
        evap_coefs=_evap_coef_cache["coefs"],
        landmask=landmask,
    )

    return ModelState(
        U   =state.U,
        V   =state.V,
        W   =state.W,
        TABS=TABS,
        QV  =QV,
        QC  =QC,
        QI  =QI,
        QR  =QR,
        QS  =QS,
        QG  =QG,
        TKE =state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep,
        time =state.time,
    )


def micro_proc_with_precip(
    state: ModelState,
    metric: dict,
    params: MicroParams,
    dt: float,
    tabs_phys: "jax.Array | None" = None,   # F11: physical TABS from previous satadj
    landmask: "jax.Array | None" = None,    # (ny, nx) bool/int for KK Ncc
) -> tuple[ModelState, jax.Array]:
    """Same as :func:`micro_proc` but also returns the surface precipitation
    flux at the reference (lowest) level in mm/s (= kg/m²/s), shape
    ``(ny, nx)``. Consumed by the SLM driver as gSAM's ``precip_ref``.

    The flux is computed as the column-integrated loss of rain+snow+graupel
    mass during sedimentation: for each column, ``sum_k rho[k] * dz[k] *
    ((QR+QS+QG)_before - (QR+QS+QG)_after) / dt``.

    Operator order matches gSAM: sedimentation first, then satadj + precip_proc.

    tabs_phys : F11 mode — physical TABS from previous satadj (see micro_proc).
    """
    rho = jnp.asarray(metric["rho"])
    dz  = jnp.asarray(metric["dz"])

    TABS = state.TABS
    QV, QC, QI = state.QV, state.QC, state.QI
    QR, QS, QG = state.QR, state.QS, state.QG

    # 1. Precipitation sedimentation (inside advect_all_scalars in gSAM)
    qp_pre_fall = QR + QS + QG
    if tabs_phys is not None:
        QR, QS, QG, _tabs_fall_out = precip_fall(QR, QS, QG, tabs_phys, metric, params, dt)
        TABS = TABS + (_tabs_fall_out - tabs_phys)   # accumulate latent-heat delta into t
    else:
        QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)
    qp_post_fall = QR + QS + QG

    # Mass lost to the surface during sedimentation (kg/m²):
    delta = (qp_pre_fall - qp_post_fall) * (rho * dz)[:, None, None]
    precip_ref = jnp.sum(delta, axis=0) / dt
    precip_ref = jnp.maximum(precip_ref, 0.0)

    # 2. Cloud-ice sedimentation
    if params.do_ice_fall:
        if tabs_phys is not None:
            QI, _tabs_ice_out = ice_fall(QI, tabs_phys, metric, params, dt)
            TABS = TABS + (_tabs_ice_out - tabs_phys)
        else:
            QI, TABS = ice_fall(QI, TABS, metric, params, dt)

    # 3. Saturation adjustment
    if tabs_phys is not None:
        _gamaz_3d = metric["gamaz"][:, None, None]
        TABS, QV, QC, QI = satadj(
            tabs_phys, QV, QC, QI, QR, QS, QG, metric, params,
            tabs_dry_override=TABS - _gamaz_3d,
            tabs_guess=tabs_phys,
        )
    else:
        TABS, QV, QC, QI = satadj(
            TABS, QV, QC, QI, QR, QS, QG, metric, params,
        )
    # 4. Precip processes (same cache logic as micro_proc)
    nstep = int(state.nstep)
    if _evap_coef_cache["nstep"] < 0 or nstep % 10 == 0:
        _evap_coef_cache["coefs"] = _compute_evap_coefs(TABS, metric, params)
        _evap_coef_cache["nstep"] = nstep
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, QR, QS, QG, QV, metric, params, dt,
        evap_coefs=_evap_coef_cache["coefs"],
        landmask=landmask,
    )

    new_state = ModelState(
        U   =state.U, V=state.V, W=state.W,
        TABS=TABS, QV=QV, QC=QC, QI=QI, QR=QR, QS=QS, QG=QG,
        TKE =state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, precip_ref
