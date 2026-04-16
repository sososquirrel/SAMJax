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
# Physical constants (matching gSAM SRC/consts.f90 — lsub is a literal
# 2.834e6 (NOT LV+LF))
# ---------------------------------------------------------------------------

G_GRAV   = 9.79764      # m/s²      gravity
CP       = 1004.64      # J/(kg K)  specific heat of dry air
CPV      = 1870.0       # J/(kg K)  specific heat of water vapour
CPW      = 3991.86795711963  # J/(kg K)  specific heat of seawater
RV       = 461.5        # J/(kg K)  gas constant for water vapour
RGAS     = 287.04       # J/(kg K)  gas constant for dry air
EPS      = 0.622        # gSAM sat.f90 uses literal 0.622, not Rd/Rv

LV       = 2.501e6      # J/kg      latent heat of vaporization (0°C)
LF       = 0.337e6      # J/kg      latent heat of fusion
LS       = 2.834e6      # J/kg      latent heat of sublimation (literal, NOT LV+LF)

FAC_COND = LV / CP      # K / (kg/kg)
FAC_FUS  = LF / CP
FAC_SUB  = LS / CP

THERCO   = 2.40e-2      # W/(m K)   thermal conductivity of air
DIFFELQ  = 2.21e-5      # m²/s      water-vapour diffusivity
MUELQ    = 1.717e-5     # kg/(m s)  dynamic viscosity of air

RAD_EARTH  = 6371229.0    # m         radius of Earth
SIGMA_SB   = 5.670373e-8  # W/(m² K⁴) Stefan-Boltzmann constant
EMIS_WATER = 0.98         # -         emissivity of water
PI         = float(np.pi)


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


# ---------------------------------------------------------------------------
# Saturation vapour pressure functions (Buck 1981, matching gSAM atmosphere.f90)
# ---------------------------------------------------------------------------

def _esatw(tabs: jax.Array) -> jax.Array:
    """Saturation vapour pressure over liquid water (mb). Buck (1981)."""
    return 6.1121 * jnp.exp(17.502 * (tabs - 273.16) / (tabs - 32.19))


def _esati(tabs: jax.Array) -> jax.Array:
    """Saturation vapour pressure over ice (mb). Buck (1981)."""
    return 6.1121 * jnp.exp(22.587 * (tabs - 273.16) / (tabs + 0.7))


def qsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over liquid water (kg/kg).

    Matches gSAM sat.f90: qsatw = 0.622 * es / max(es, p - es). The max(es, ...)
    flips behavior when es > p - es (low pressure / upper troposphere) so qsatw
    tends to 0.622 instead of diverging.
    """
    es = _esatw(tabs)
    return EPS * es / jnp.maximum(es, pres_mb - es)


def qsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """Saturation mixing ratio over ice (kg/kg). Matches gSAM sat.f90."""
    es = _esati(tabs)
    return EPS * es / jnp.maximum(es, pres_mb - es)


def _dtqsatw(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsatw)/d(tabs). Port of gSAM sat.f90:124-137."""
    es = _esatw(tabs)
    a1, T0, T1 = 17.502, 273.16, 32.19
    dtesatw = es * a1 * (T0 - T1) / (tabs - T1) ** 2
    return 0.622 * dtesatw / (pres_mb - es) * (1.0 + es / (pres_mb - es))


def _dtqsati(tabs: jax.Array, pres_mb: jax.Array) -> jax.Array:
    """d(qsati)/d(tabs). Port of gSAM sat.f90:149-162."""
    es = _esati(tabs)
    a1, T0, T1 = 22.587, 273.16, -0.7
    dtesati = es * a1 * (T0 - T1) / (tabs - T1) ** 2
    return 0.622 * dtesati / (pres_mb - es) * (1.0 + es / (pres_mb - es))


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
    # D21: gSAM uses full 3D pressure pp(i,j,k) for qsat; jsam uses 1D mean.
    # When 3D p' is available, add: pres_mb = pres_mb + pp_3d / 100.0
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

        # C12 fix: homogeneous freezing supersaturation (gSAM cloud.f90:48-52)
        # At T<235K with negligible existing ice, ice nucleation requires
        # supersaturation rh_homo = 2.583 - T/207.8 (IFS parameterization).
        rh_homo = jnp.where(
            (tabs1 < 235.0) & (QI < 1.0e-8),
            2.583 - tabs1 / 207.8,
            1.0,
        )

        # Mixed-phase saturation mixing ratio (with rh_homo on ice component)
        qsati_homo = qsati(tabs1, pres_mb) * rh_homo
        qsat = jnp.where(
            tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, qsati_homo,
                om * qsatw(tabs1, pres_mb) + (1.0 - om) * qsati_homo,
            ),
        )
        # Derivative: d(qsati*rh_homo)/dT = dqsati/dT * rh_homo + qsati * drh_homo/dT
        # drh_homo/dT = -1/207.8 when active, else 0
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
    tabs1 = TABS + 0.0  # copy; keep as jax array

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
    MPDATA sedimentation for one species in one column.

    Matches gSAM precip_fall.f90: upwind pass + anti-diffusive correction
    with non-oscillatory (FCT) flux limiting.

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
    wp = jnp.minimum(1.0, vt * dt / dz_col)
    cfl = jnp.max(wp)

    # --- Pre-compute mx/mn from original field (gSAM nonos=.true. line 130-135) ---
    qp_above = jnp.concatenate([qp_safe[1:], qp_safe[-1:]])
    qp_below = jnp.concatenate([qp_safe[:1], qp_safe[:-1]])
    mx0 = jnp.maximum(jnp.maximum(qp_below, qp_above), qp_safe)
    mn0 = jnp.minimum(jnp.minimum(qp_below, qp_above), qp_safe)

    # --- Pass 1: first-order upwind (gSAM line 141-149) ---
    fz = qp_safe * wp
    fz_top = jnp.concatenate([fz[1:], jnp.zeros((1,))])
    tmp_qp = qp_safe - (fz_top - fz) * irhoadz
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

    qp_new = jnp.maximum(0.0, qp_safe - (fz_corr_top - fz_corr) * irhoadz)

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
) -> ModelState:
    """
    One microphysics step matching gSAM operator order.

    gSAM ordering (main.f90:299,324):
      advect_all_scalars calls precip_fall + ice_fall FIRST,
      then micro_proc calls cloud (satadj) + precip_proc.

    So: sedimentation → saturation adjustment → precipitation processes.

    Args:
        state  : current ModelState
        metric : dict from build_metric (must include 'pres' and 'gamaz')
        params : MicroParams (use MicroParams() for gSAM defaults)
        dt     : physics timestep (seconds)

    Returns:
        new ModelState with updated TABS, QV, QC, QI, QR, QS, QG
    """
    TABS = state.TABS
    QV, QC, QI = state.QV, state.QC, state.QI
    QR, QS, QG = state.QR, state.QS, state.QG

    # 1. Precipitation sedimentation (inside advect_all_scalars in gSAM)
    QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)

    # 2. Cloud-ice sedimentation (inside advect_all_scalars in gSAM)
    if params.do_ice_fall:
        QI, TABS = ice_fall(QI, TABS, metric, params, dt)

    # 3. Saturation adjustment (cloud() in gSAM micro_proc)
    TABS, QV, QC, QI = satadj(
        TABS, QV, QC, QI, QR, QS, QG, metric, params,
    )

    # 4. Precipitation processes (precip_proc() in gSAM micro_proc)
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, QR, QS, QG, QV, metric, params, dt,
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
) -> tuple[ModelState, jax.Array]:
    """Same as :func:`micro_proc` but also returns the surface precipitation
    flux at the reference (lowest) level in mm/s (= kg/m²/s), shape
    ``(ny, nx)``. Consumed by the SLM driver as gSAM's ``precip_ref``.

    The flux is computed as the column-integrated loss of rain+snow+graupel
    mass during sedimentation: for each column, ``sum_k rho[k] * dz[k] *
    ((QR+QS+QG)_before - (QR+QS+QG)_after) / dt``.

    Operator order matches gSAM: sedimentation first, then satadj + precip_proc.
    """
    rho = jnp.asarray(metric["rho"])
    dz  = jnp.asarray(metric["dz"])

    TABS = state.TABS
    QV, QC, QI = state.QV, state.QC, state.QI
    QR, QS, QG = state.QR, state.QS, state.QG

    # 1. Precipitation sedimentation (inside advect_all_scalars in gSAM)
    qp_pre_fall = QR + QS + QG
    QR, QS, QG, TABS = precip_fall(QR, QS, QG, TABS, metric, params, dt)
    qp_post_fall = QR + QS + QG

    # Mass lost to the surface during sedimentation (kg/m²):
    delta = (qp_pre_fall - qp_post_fall) * (rho * dz)[:, None, None]
    precip_ref = jnp.sum(delta, axis=0) / dt
    precip_ref = jnp.maximum(precip_ref, 0.0)

    # 2. Cloud-ice sedimentation
    if params.do_ice_fall:
        QI, TABS = ice_fall(QI, TABS, metric, params, dt)

    # 3. Saturation adjustment
    TABS, QV, QC, QI = satadj(
        TABS, QV, QC, QI, QR, QS, QG, metric, params,
    )
    # 4. Precip processes
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, QR, QS, QG, QV, metric, params, dt,
    )

    new_state = ModelState(
        U   =state.U, V=state.V, W=state.W,
        TABS=TABS, QV=QV, QC=QC, QI=QI, QR=QR, QS=QS, QG=QG,
        TKE =state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, precip_ref
