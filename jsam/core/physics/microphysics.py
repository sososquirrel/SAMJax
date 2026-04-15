"""
SAM 1-moment bulk microphysics for jsam.

Port of gSAM MICRO_SAM1MOM (Khairoutdinov 2006) with simplifications for
differentiable JAX use:
  - Flat terrain (no k_terra)
  - Default namelist values (doKKauto=False, dowarmcloud=False, donograupel=False)
  - Saturation adjustment: 20-iteration Newton (jax.lax.scan, always runs)
  - Precipitation processes: implicit Kessler autoconversion + accretion + evaporation
  - Precipitation fall: upwind sedimentation (1 sub-step; add subcycling if CFL>1 needed)
  - Cloud-ice sedimentation: Heymsfield (2003) vt + MC flux limiter (ice_fall.f90)

Variable mapping (jsam ↔ gSAM SAM1MOM):
  jsam                   gSAM
  TABS                   tabs  (prognostic in jsam; diagnostic in gSAM)
  QV + QC + QI   =   q   (total non-precipitating water)
  QR + QS + QG   =   qp  (total precipitating water)
  t = TABS + gamaz - fac_cond*(QC+QR) - fac_sub*(QI+QS+QG)   (static energy, conserved)

Phase partitioning by temperature-dependent omega weights:
  omn = clip((T-tbgmin)/(tbgmax-tbgmin), 0, 1)   QC = qn*omn, QI = qn*(1-omn)
  omp = clip((T-tprmin)/(tprmax-tprmin), 0, 1)   QR = qp*omp
  omg = clip((T-tgrmin)/(tgrmax-tgrmin), 0, 1)   QG = qp*(1-omp)*omg, QS = qp*(1-omp)*(1-omg)

References
----------
  MICRO_SAM1MOM/cloud.f90      — saturation adjustment (Newton iteration)
  MICRO_SAM1MOM/precip_proc.f90 — autoconversion, accretion, evaporation
  MICRO_SAM1MOM/precip_fall.f90 — upwind sedimentation with anti-diffusion
  MICRO_SAM1MOM/precip_init.f90 — accretion/evaporation coefficient tables
  MICRO_SAM1MOM/micro_params.f90 — default constants
  gSAM SRC/consts.f90           — physical constants
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as _scipy_gamma

from jsam.core.state import ModelState


# ---------------------------------------------------------------------------
# Physical constants (matching gSAM SRC/consts.f90)
# ---------------------------------------------------------------------------

G_GRAV   = 9.79764      # m/s²      gravity
CP       = 1004.64      # J/(kg K)  specific heat of dry air
RV       = 461.5        # J/(kg K)  gas constant for water vapour
RGAS     = 287.04       # J/(kg K)  gas constant for dry air
EPS      = RGAS / RV    # ≈ 0.6220  Rd/Rv

LV       = 2.501e6      # J/kg      latent heat of vaporization (0°C)
LF       = 0.337e6      # J/kg      latent heat of fusion
LS       = LV + LF      # J/kg      latent heat of sublimation

FAC_COND = LV / CP      # K / (kg/kg)
FAC_FUS  = LF / CP
FAC_SUB  = LS / CP

THERCO   = 2.40e-2      # W/(m K)   thermal conductivity of air
DIFFELQ  = 2.21e-5      # m²/s      water-vapour diffusivity
MUELQ    = 1.717e-5     # kg/(m s)  dynamic viscosity of air


# ---------------------------------------------------------------------------
# Default SAM1MOM parameters  (micro_params.f90 defaults)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MicroParams:
    """Adjustable SAM1MOM parameters.  All values are gSAM defaults."""

    # Temperature limits for phase partitioning
    tbgmin: float = 253.16   # K  min T for cloud water (below → all ice)
    tbgmax: float = 273.16   # K  max T for cloud ice   (above → all liquid)
    tprmin: float = 268.16   # K  min T for rain        (below → all snow/graupel)
    tprmax: float = 283.16   # K  max T for snow+graupel (above → all rain)
    tgrmin: float = 223.16   # K  min T for graupel
    tgrmax: float = 283.16   # K  max T for graupel     (above → no graupel)

    # Autoconversion thresholds and rates
    qcw0:     float = 1.0e-3   # kg/kg  threshold for cloud water autoconversion
    qci0:     float = 1.0e-4   # kg/kg  threshold for cloud ice autoconversion
    alphaelq: float = 1.0e-3   # s⁻¹   Kessler cloud-water autoconversion rate
    betaelq:  float = 1.0e-3   # s⁻¹   cloud-ice autoconversion rate

    # Hydrometeor densities
    rhor: float = 1000.0   # kg/m³  liquid water
    rhos: float = 100.0    # kg/m³  snow
    rhog: float = 400.0    # kg/m³  graupel

    # Marshall-Palmer intercept parameters (m⁻⁴)
    nzeror: float = 8.0e6
    nzeros: float = 3.0e6
    nzerog: float = 4.0e6

    # Terminal-velocity coefficients  vt = a * (rho*q/(pi*rho_hyd*N0))^(b/4) * sqrt(1.29/rho)
    a_rain: float = 842.0    # m^(1-b)/s
    b_rain: float = 0.8
    a_snow: float = 4.84
    b_snow: float = 0.25
    a_grau: float = 94.5
    b_grau: float = 0.5

    # Collection efficiencies
    erccoef: float = 1.0   # rain / cloud water
    esccoef: float = 1.0   # snow / cloud water
    esicoef: float = 0.1   # snow / cloud ice
    egccoef: float = 1.0   # graupel / cloud water
    egicoef: float = 0.1   # graupel / cloud ice

    # Minimum precipitating water for evaporation
    qp_threshold: float = 1.0e-12   # kg/kg

    # Cloud-ice sedimentation (ice_fall.f90)
    icefall_fudge: float = 1.0   # vt tuning factor; ~0.2 for 25-km grids
    gamma_rave:    float = 1.0   # anvil sedimentation slowdown

    # Options
    donograupel: bool = False
    do_ice_fall: bool = True


def _gamma_coefs(p: MicroParams) -> dict:
    """Gamma function values derived from terminal-velocity exponents."""
    return {
        "gamr1": float(_scipy_gamma(3 + p.b_rain)),
        "gamr2": float(_scipy_gamma((5 + p.b_rain) / 2)),
        "gams1": float(_scipy_gamma(3 + p.b_snow)),
        "gams2": float(_scipy_gamma((5 + p.b_snow) / 2)),
        "gamg1": float(_scipy_gamma(3 + p.b_grau)),
        "gamg2": float(_scipy_gamma((5 + p.b_grau) / 2)),
    }


# ---------------------------------------------------------------------------
# Saturation vapour pressure functions (Buck 1981, matching gSAM atmosphere.f90)
# ---------------------------------------------------------------------------

def _esatw(tabs: jax.Array) -> jax.Array:
    """Saturation vapour pressure over liquid water (mb). Buck (1981)."""
    return 6.1121 * jnp.exp(17.502 * (tabs - 273.16) / (tabs - 32.18))


def _esati(tabs: jax.Array) -> jax.Array:
    """Saturation vapour pressure over ice (mb). Buck (1981)."""
    return 6.1121 * jnp.exp(22.587 * (tabs - 273.16) / (tabs + 0.7))


def qsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over liquid water (kg/kg)."""
    es = _esatw(tabs)
    # Clamp denominator to avoid negative qsat when es >= pres (T >> 360 K).
    # This only occurs during Newton-iteration overshoots; the result is never
    # used physically at such temperatures.
    return EPS * es / jnp.maximum(pres_mb - es, 1e-3)


def qsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over ice (kg/kg)."""
    es = _esati(tabs)
    return EPS * es / jnp.maximum(pres_mb - es, 1e-3)


def _dtqsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsatw)/d(tabs). Used inside Newton iteration."""
    es  = _esatw(tabs)
    des = es * 17.502 * (273.16 - 32.18) / (tabs - 32.18) ** 2
    denom = jnp.maximum(pres_mb - es, 1e-3)
    return EPS * des * pres_mb / denom ** 2


def _dtqsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsati)/d(tabs). Used inside Newton iteration."""
    es  = _esati(tabs)
    des = es * 22.587 * (273.16 + 0.7) / (tabs + 0.7) ** 2
    denom = jnp.maximum(pres_mb - es, 1e-3)
    return EPS * des * pres_mb / denom ** 2


# ---------------------------------------------------------------------------
# Saturation adjustment  (port of cloud.f90)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params", "n_iter"))
def satadj(
    TABS: jax.Array,   # (nz, ny, nx)  K
    QV:   jax.Array,   # (nz, ny, nx)  kg/kg
    QC:   jax.Array,   # (nz, ny, nx)
    QI:   jax.Array,   # (nz, ny, nx)
    QR:   jax.Array,   # (nz, ny, nx)
    QS:   jax.Array,   # (nz, ny, nx)
    QG:   jax.Array,   # (nz, ny, nx)
    metric: dict,
    params: MicroParams,
    n_iter: int = 20,
) -> tuple:  # (TABS_new, QV_new, QC_new, QI_new)
    """
    Moist saturation adjustment.

    Finds the temperature tabs1 and condensate qn that satisfy:
        tabs1 = tabs_dry + lstarn * (q - qsat(tabs1)) + lstarp * qp
    where tabs_dry = t - gamaz is the dry static-energy temperature.

    Precip fields (QR, QS, QG) are not changed here — only the phase
    partitioning of the non-precipitating water (QV, QC, QI) is adjusted.

    The Newton iteration always runs (n_iter fixed steps) so the function is
    JIT-compilable without data-dependent branching.
    """
    pres_mb  = metric["pres"][:, None, None] / 100.0   # Pa → mb, (nz,1,1)
    gamaz_3d = metric["gamaz"][:, None, None]            # (nz,1,1) K

    a_bg = 1.0 / (params.tbgmax - params.tbgmin)
    b_bg = params.tbgmin * a_bg
    a_pr = 1.0 / (params.tprmax - params.tprmin)
    b_pr = params.tprmin * a_pr
    a_gr = 0.0 if params.donograupel else 1.0 / (params.tgrmax - params.tgrmin)

    # Phase partition of existing precip (needed to compute t conserved variable)
    omp0 = jnp.clip((TABS - params.tprmin) * a_pr, 0.0, 1.0)
    qp_liq = (QR + QS + QG) * omp0       # liquid precip fraction
    qp_ice = (QR + QS + QG) * (1.0 - omp0)

    # Liquid-ice static energy temperature: tabs_dry = t - gamaz
    #   t = TABS + gamaz - fac_cond*(QC + qp_liq) - fac_sub*(QI + qp_ice)
    tabs_dry = TABS - FAC_COND * (QC + qp_liq) - FAC_SUB * (QI + qp_ice)

    q  = QV + QC + QI   # total non-precip water
    qp = QR + QS + QG   # total precip water

    # Newton iteration: find tabs1 such that residual fff = 0
    def _newton(tabs1):
        om = jnp.clip(a_bg * tabs1 - b_bg, 0.0, 1.0)    # cloud liquid fraction

        # Latent heat coefficient for non-precip condensate
        lstarn  = FAC_COND + (1.0 - om) * FAC_FUS
        dlstarn = jnp.where(
            (tabs1 > params.tbgmin) & (tabs1 < params.tbgmax),
            a_bg * FAC_FUS, 0.0,
        )

        # Latent heat coefficient for precipitating water
        omp_ = jnp.clip(a_pr * tabs1 - b_pr, 0.0, 1.0)
        lstarp  = FAC_COND + (1.0 - omp_) * FAC_FUS
        dlstarp = jnp.where(
            (tabs1 > params.tprmin) & (tabs1 < params.tprmax),
            a_pr * FAC_FUS, 0.0,
        )

        # Mixed-phase saturation mixing ratio
        qsat = jnp.where(
            tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, qsati(tabs1, pres_mb),
                om * qsatw(tabs1, pres_mb) + (1.0 - om) * qsati(tabs1, pres_mb),
            ),
        )
        dqsat = jnp.where(
            tabs1 >= params.tbgmax, _dtqsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, _dtqsati(tabs1, pres_mb),
                om * _dtqsatw(tabs1, pres_mb) + (1.0 - om) * _dtqsati(tabs1, pres_mb),
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
    tabs1 = TABS + 0.0  # copy; keep as jax array

    # Fixed Newton iterations (unrolled at trace time → no Python control-flow overhead)
    for _ in range(n_iter):
        tabs1 = _newton(tabs1)

    # Final condensate and phase partitioning
    om_f = jnp.clip(a_bg * tabs1 - b_bg, 0.0, 1.0)
    qsat_f = jnp.where(
        tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
        jnp.where(
            tabs1 <= params.tbgmin, qsati(tabs1, pres_mb),
            om_f * qsatw(tabs1, pres_mb) + (1.0 - om_f) * qsati(tabs1, pres_mb),
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
    gamr1, gamr2 = gcoefs["gamr1"], gcoefs["gamr2"]
    gams1, gams2 = gcoefs["gams1"], gcoefs["gams2"]
    gamg1, gamg2 = gcoefs["gamg1"], gcoefs["gamg2"]

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

    # ── Evaporation coefficients (precip_init.f90, computed on-the-fly) ──────
    rrr1 = 393.0 / (TABS + 120.0) * (TABS / 273.0) ** 1.5
    rrr2 = (TABS / 273.0) ** 1.94 * (1000.0 / pres_mb)
    estw = 100.0 * _esatw(TABS)   # Pa
    esti = 100.0 * _esati(TABS)   # Pa

    c1r = (LV / (TABS * RV) - 1.0) * LV / (THERCO * rrr1 * TABS)
    c2r = RV * TABS / (DIFFELQ * rrr2 * estw)
    evapr1 = (0.78  * 2.0 * np.pi * p.nzeror / jnp.sqrt(np.pi * p.rhor * p.nzeror * rho)
              / (c1r + c2r))
    evapr2 = (0.31  * 2.0 * np.pi * p.nzeror * gamr2 * 0.89
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

    qn = qcc + qii   # total cloud condensate

    # ── Branch 1: autoconversion + accretion (when qn > 0) ──────────────────
    # Cloud water autoconversion (Kessler)
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
    qcc_imp = (qcc + dt * autor * p.qcw0) / (1.0 + dt * (accrr + accrcs + accrcg + autor))
    qii_imp = (qii + dt * autos * p.qci0) / (1.0 + dt * (accris + accrig + autos))

    dq_conv = dt * (accrr * qcc_imp + autor * (qcc_imp - p.qcw0)
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

    # Apply net change: dq > 0 → cloud→precip;  dq < 0 → precip→vapour
    qp_new = jnp.maximum(0.0, qp + dq)
    q_new  = jnp.maximum(0.0, (QV + qn) - dq)   # q = QV + qn

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
    rhow_col: jax.Array,   # (nz+1,)
    dz_col:   jax.Array,   # (nz,)  m
    a_coef:   float,
    b_coef:   float,
    rho_hyd:  float,       # hydrometeor density (kg/m³)
    n_zero:   float,       # intercept parameter (m⁻⁴)
    dt:       float,
) -> tuple:  # (qp_new, wp_max) where wp_max is max Courant number for CFL check
    """
    Upwind sedimentation for one species in one column.

    Returns (qp_new, cfl_max).  cfl_max > 1 signals a CFL violation —
    add subcycling if this occurs.
    """
    qp_safe = jnp.maximum(qp_col, 0.0)
    rhofac  = jnp.sqrt(1.29 / rho_col)    # (nz,)

    # Terminal velocity (m/s, positive downward)
    vt = (a_coef * (rho_col * qp_safe / (np.pi * rho_hyd * n_zero + 1e-30))
          ** (b_coef / 4.0) * rhofac)
    vt = jnp.where(qp_safe > 1e-12, vt, 0.0)

    # Dimensionless upwind Courant factor:  wp[k] = -vt[k] * rhow[k] * dt / dz[k]
    # (negative = downward; rhow at bottom face of cell k = rhow_col[k])
    wp  = -vt * rhow_col[:len(qp_col)] * dt / dz_col   # (nz,)
    cfl = jnp.max(jnp.abs(wp))

    # Upwind flux: fz[k] = qp[k] * wp[k]  (at bottom face of cell k)
    fz      = qp_safe * wp                                  # (nz,)
    fz_top  = jnp.concatenate([fz[1:], jnp.zeros((1,))], axis=0)  # fz[k+1], 0 at top

    # Flux divergence:  qp[k] -= (fz_top[k] - fz[k]) / (rho[k]*dz[k])
    qp_new = qp_safe - (fz_top - fz) / (rho_col * dz_col)
    qp_new = jnp.maximum(0.0, qp_new)

    return qp_new, cfl


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

    Uses 1D upwind scheme per column (vmapped over ny, nx).
    Port of gSAM MICRO_SAM1MOM/precip_fall.f90 (upwind step only; no MPDATA
    anti-diffusive correction — add that for production runs).

    NOTE: no subcycling — assumes dt is short enough for CFL < 1.
    For rain at 8 m/s in a 250-m layer, max stable dt ≈ 30 s.
    """
    rho_1d  = np.array(metric["rho"])    # keep as numpy for vmap efficiency
    rhow_1d = np.array(metric["rhow"])
    dz_1d   = np.array(metric["dz"])

    nz, ny, nx = QR.shape

    def _fall_species(qp_3d, a, b, rho_hyd, n_zero):
        """Apply sedimentation to a 3D species field."""
        # vmap over (j, i): each column treated independently
        def _col(qp_col):
            qp_new, _ = _fall_col_one(qp_col, rho_1d, rhow_1d, dz_1d,
                                       a, b, rho_hyd, n_zero, dt)
            return qp_new

        # vmap over ny × nx
        _vmap_i = jax.vmap(_col, in_axes=1, out_axes=1)          # over nx
        _vmap_ji = jax.vmap(_vmap_i, in_axes=1, out_axes=1)     # over ny
        return _vmap_ji(qp_3d)

    p = params
    QR_new = _fall_species(QR, p.a_rain, p.b_rain, p.rhor, p.nzeror)
    QS_new = _fall_species(QS, p.a_snow, p.b_snow, p.rhos, p.nzeros)
    QG_new = _fall_species(QG, p.a_grau, p.b_grau, p.rhog, p.nzerog)

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
    rhofac  = jnp.sqrt(1.29 / rho_col)

    # Heymsfield (JAS 2003, p.2607) ice terminal velocity, m/s
    vt = fudge * 8.66 * (qi_safe + 1e-10) ** 0.24 * rhofac / gamma_rave
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
) -> ModelState:
    """
    One microphysics step: saturation adjustment → precip processes → sedimentation.

    Matches gSAM micro_proc() call sequence (cloud → precip_proc → precip_fall).
    U, V, W, TKE, nstep, time are passed through unchanged.

    Args:
        state  : current ModelState
        metric : dict from build_metric (must include 'pres' and 'gamaz')
        params : MicroParams (use MicroParams() for gSAM defaults)
        dt     : physics timestep (seconds)

    Returns:
        new ModelState with updated TABS, QV, QC, QI, QR, QS, QG
    """
    # 1. Saturation adjustment
    TABS, QV, QC, QI = satadj(
        state.TABS, state.QV, state.QC, state.QI,
        state.QR, state.QS, state.QG, metric, params,
    )

    # 2. Precipitation processes
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, state.QR, state.QS, state.QG, QV, metric, params, dt,
    )

    # 3. Precipitation sedimentation
    QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)

    # 4. Cloud-ice sedimentation
    if params.do_ice_fall:
        QI, TABS = ice_fall(QI, TABS, metric, params, dt)

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
) -> tuple[ModelState, jax.Array]:
    """Same as :func:`micro_proc` but also returns the surface precipitation
    flux at the reference (lowest) level in mm/s (= kg/m²/s), shape
    ``(ny, nx)``. Consumed by the SLM driver as gSAM's ``precip_ref``.

    The flux is computed as the column-integrated loss of rain+snow+graupel
    mass during sedimentation: for each column, ``sum_k rho[k] * dz[k] *
    ((QR+QS+QG)_before - (QR+QS+QG)_after) / dt``.
    """
    rho = jnp.asarray(metric["rho"])
    dz  = jnp.asarray(metric["dz"])

    qp_before = state.QR + state.QS + state.QG

    # Saturation adjustment
    TABS, QV, QC, QI = satadj(
        state.TABS, state.QV, state.QC, state.QI,
        state.QR, state.QS, state.QG, metric, params,
    )
    # Precip processes
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, state.QR, state.QS, state.QG, QV, metric, params, dt,
    )
    # Capture column precipitation mass *before* sedimentation drains it
    qp_pre_fall = QR + QS + QG
    # Sedimentation
    QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)
    qp_post_fall = QR + QS + QG

    # Mass lost to the surface during sedimentation (kg/m²):
    # ∑_k rho[k] * dz[k] * Δq[k]. Divide by dt → kg/m²/s = mm/s.
    delta = (qp_pre_fall - qp_post_fall) * (rho * dz)[:, None, None]
    precip_ref = jnp.sum(delta, axis=0) / dt
    precip_ref = jnp.maximum(precip_ref, 0.0)

    if params.do_ice_fall:
        QI, TABS = ice_fall(QI, TABS, metric, params, dt)

    new_state = ModelState(
        U   =state.U, V=state.V, W=state.W,
        TABS=TABS, QV=QV, QC=QC, QI=QI, QR=QR, QS=QS, QG=QG,
        TKE =state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, precip_ref
