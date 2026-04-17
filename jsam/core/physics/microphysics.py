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
    docloudfall: bool = False   # cloud liquid sedimentation (cloud_fall.f90); F for IRMA
    sigmag: float = 1.5         # lognormal dispersion for cloud drop size (cloud_fall.f90)
    # Khairoutdinov-Kogan (2000) autoconversion
    # Note: gSAM IRMA config (prm_debug500) uses doKKauto=.true. for better agreement
    # with observations; defaulting to True to match oracle behavior
    doKKauto: bool = True
    doKKaccr: bool = False
    auto_fudge: float = 1.0
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
           params: MicroParams, n_iter: int = 100,
           tabs_dry_override: "jax.Array | None" = None,
           tabs_guess: "jax.Array | None" = None,
           p_pert_pa: "jax.Array | None" = None) -> tuple:
    """Newton-iteration saturation adjustment; returns (TABS_new, QV_new, QC_new, QI_new).

    p_pert_pa : 3D perturbation pressure (Pa), shape (nz, ny, nx).  When given,
                the full 3D pressure pres_ref + p_pert is used for all qsat
                calls, matching gSAM cloud.f90 which uses pp(i,j,k) = pres + pp_pert.
    """
    pres_ref_mb = metric["pres"][:, None, None] / 100.0   # (nz,1,1) reference, mb
    if p_pert_pa is not None:
        pres_mb = pres_ref_mb + p_pert_pa / 100.0          # (nz,ny,nx) full 3D, mb
    else:
        pres_mb = pres_ref_mb
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

    def _newton_step(carry):
        """One Newton iteration with convergence masking (gSAM: exit when |dtabs|<0.001)."""
        tabs1, converged = carry

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

        # Fix 4.5: rh_homo only applied in pure-ice branch; NOT in mixed-phase
        # (cloud.f90 Newton loop lines 115-116 use plain qsati in mixed-phase)
        qsati_val = qsati(tabs1, pres_mb)
        qsati_homo = qsati_val * rh_homo
        qsat = jnp.where(
            tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, qsati_homo,
                om * qsatw(tabs1, pres_mb) + (1.0 - om) * qsati_val,
            ),
        )
        drh_homo = jnp.where(
            (tabs1 < 235.0) & (QI < 1.0e-8),
            -1.0 / 207.8,
            0.0,
        )
        dtqsati_val = _dtqsati(tabs1, pres_mb)
        dtqsati_homo = dtqsati_val * rh_homo + qsati_val * drh_homo
        dqsat = jnp.where(
            tabs1 >= params.tbgmax, _dtqsatw(tabs1, pres_mb),
            jnp.where(
                tabs1 <= params.tbgmin, dtqsati_homo,
                om * _dtqsatw(tabs1, pres_mb) + (1.0 - om) * dtqsati_val,
            ),
        )

        sat_excess = q - qsat
        fff  = tabs_dry - tabs1 + lstarn * sat_excess + lstarp * qp
        dfff = dlstarn * sat_excess + dlstarp * qp - lstarn * dqsat - 1.0
        dtabs = -fff / dfff

        # Fix 4.6: once |dtabs| < 0.001 K, stop updating that cell (convergence mask)
        newly_converged = jnp.abs(dtabs) < 0.001
        tabs1_new = jnp.where(converged | newly_converged, tabs1, tabs1 + dtabs)
        converged_new = converged | newly_converged
        return tabs1_new, converged_new

    # Fix 4.7: preliminary estimate before Newton loop (cloud.f90:58)
    # tabs1_prelim = (tabs_dry + fac1*qp) / (1 + fac2*qp)
    # where fac1 = fac_cond + (1+b_pr)*fac_fus, fac2 = fac_fus*a_pr
    # This is used only to select the saturation regime for the initial qsatt
    # test; the Newton loop itself starts from tabs_guess (= previous physical
    # TABS), matching gSAM cloud.f90 line 92: tabs1 = tabs2.
    _fac1_prelim = FAC_COND + (1.0 + b_pr) * FAC_FUS
    _fac2_prelim = FAC_FUS * a_pr
    # tabs_dry here corresponds to gSAM's tabs(i,j,k) = t - gamaz
    tabs1_prelim = (tabs_dry + _fac1_prelim * qp) / (1.0 + _fac2_prelim * qp)

    # F11: when tabs_guess is provided (= physical TABS from previous satadj),
    # use it as the Newton start, matching gSAM cloud.f90: tabs1 = tabs2.
    # When not provided, use tabs1_prelim as initial guess (matches gSAM warm/cold
    # path where tabs1 is computed from the preliminary estimate before the loop).
    tabs1 = (tabs_guess if tabs_guess is not None else tabs1_prelim) + 0.0

    # 100-iteration Newton loop with convergence mask (gSAM: while |dtabs|>0.001, niter<100)
    converged = jnp.zeros_like(tabs1, dtype=jnp.bool_)
    for _ in range(n_iter):
        tabs1, converged = _newton_step((tabs1, converged))

    # Final condensate and phase partitioning
    om_f = jnp.clip(a_bg * tabs1 - b_bg, 0.0, 1.0)
    # Fix 4.5: rh_homo applies only in pure-ice branch, not mixed-phase
    # (cloud.f90 line 107: qsati*rh_homo only when tabs1 <= tbgmin)
    rh_homo_f = jnp.where(
        (tabs1 < 235.0) & (QI < 1.0e-8),
        2.583 - tabs1 / 207.8,
        1.0,
    )
    qsati_val_f = qsati(tabs1, pres_mb)
    qsati_homo_f = qsati_val_f * rh_homo_f
    qsat_f = jnp.where(
        tabs1 >= params.tbgmax, qsatw(tabs1, pres_mb),
        jnp.where(
            tabs1 <= params.tbgmin, qsati_homo_f,
            om_f * qsatw(tabs1, pres_mb) + (1.0 - om_f) * qsati_val_f,
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
    p_pert_pa: "jax.Array | None" = None,  # 3D perturbation pressure (Pa), shape (nz, ny, nx)
) -> tuple:  # (TABS_new, QV_new, QC_new, QI_new, QR_new, QS_new, QG_new)
    """
    Microphysical source/sink terms: autoconversion, accretion, and evaporation.

    Does NOT include sedimentation (handled by precip_fall).
    All arithmetic is implicit to keep condensate non-negative.

    Port of gSAM MICRO_SAM1MOM/precip_proc.f90.

    p_pert_pa : 3D perturbation pressure (Pa), shape (nz, ny, nx).  When given,
                full 3D pressure is used for qsat (evaporation branch), matching
                gSAM which uses pp(i,j,k).
    """
    rho     = metric["rho"][:, None, None]     # (nz,1,1) kg/m³
    pres_ref_mb = metric["pres"][:, None, None] / 100.0   # (nz,1,1) reference, mb
    if p_pert_pa is not None:
        pres_mb = pres_ref_mb + p_pert_pa / 100.0          # (nz,ny,nx) full 3D, mb
    else:
        pres_mb = pres_ref_mb

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
        # Fix 4.4: Scale-dependence factor using full 2D cell area.
        # gSAM precip_proc.f90:105: autor *= min(1, 10000/(dx*mu(j)*dy*ady(j)))
        # cell_area = dx * mu[j] * dy * ady[j]  (m²)
        dx    = metric["dx_lon"]              # scalar, m
        dy    = metric["dy_lat_ref"]          # scalar reference dy, m
        mu_j  = metric["cos_lat"][None, :, None]     # (1, ny, 1)
        ady_j = metric["ady"][None, :, None]         # (1, ny, 1)
        cell_area = dx * mu_j * dy * ady_j + 1e-30  # (1, ny, 1) m²
        if p.do_scale_dependence_of_autoconv:
            scale_fac = jnp.minimum(1.0, 10000.0 / cell_area)
        else:
            scale_fac = 1.0
        autor = autor * scale_fac
    else:
        # Standard Kessler parameterization
        qcw0_eff = p.qcw0
        autor = jnp.where(qcc > p.qcw0, p.alphaelq, 0.0)

    # Fix 4.12: auto_fudge multiplier on autoconversion (micro_params.f90:50)
    autor = autor * p.auto_fudge

    # Cloud ice autoconversion
    autos = jnp.where(qii > p.qci0, p.betaelq * coefice, 0.0)

    # Accretion rates
    # Fix 4.11: KK (2000) accretion formula when doKKaccr=True (precip_proc.f90:131)
    # accrr = 67 * qrr^1.15 * qcc^0.15 (only when omp > 0.001)
    if p.doKKaccr:
        accrr = jnp.where(
            omp > 0.001,
            67.0 * jnp.maximum(qrr, 0.0) ** 1.15 * jnp.maximum(qcc, 0.0) ** 0.15,
            0.0,
        )
    else:
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
) -> tuple:  # (qp_new, cfl_max, fz_col)
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

    Returns (qp_new, cfl_max, fz_col) where fz_col[k] is the dimensionless
    downward mass flux (qp*wp) at the top face of cell k used in the final
    MPDATA step, matching gSAM's fz(k) (at bottom face of cell k+1 = top face
    of cell k in 0-indexed notation).  fz_col[nz-1] = 0 (top boundary).
    fz_col is accumulated over all substeps, scaled back to per-dt units.
    """
    qp_safe = jnp.maximum(qp_col, 0.0)
    rhofac = jnp.sqrt(1.29 / rho_col)

    # irhoadz[k] = 1/(rho[k]*dz[k])  (gSAM: 1/(rho*adz), with dz factor in wp)
    irhoadz = 1.0 / (rho_col * dz_col)

    def _compute_wp(qp_cur, nprec_f):
        """Compute per-level CFL from current qp (gSAM precip_fall.f90:97-99).

        Fix 4.9: gSAM recomputes wp from the updated qp after each substep
        (precip_fall.f90:225-237).  This helper is called at the start of
        each substep so wp reflects the current qp.
        """
        qp_c = jnp.maximum(qp_cur, 0.0)
        tv = jnp.minimum(vt_max, v_coef * (rho_col * qp_c) ** c_exp)
        vt_c = rhofac * tv
        vt_c = jnp.where(qp_c > 1e-12, vt_c, 0.0)
        return jnp.minimum(1.0, vt_c * dt / dz_col / nprec_f)

    # Compute initial terminal velocity from initial qp for CFL check
    term_vel_init = jnp.minimum(vt_max, v_coef * (rho_col * qp_safe) ** c_exp)
    vt_init = rhofac * term_vel_init
    vt_init = jnp.where(qp_safe > 1e-12, vt_init, 0.0)

    # Courant number (dimensionless, positive downward)
    # gSAM: wp = vt*dt/dz (scalar dz * adz → dz_col here)
    # For subcycling, we compute nprec and scale wp accordingly
    cfl_raw = vt_init * dt / dz_col  # unclipped CFL
    cfl_max = jnp.max(cfl_raw)

    # Adaptive subcycling: if CFL > 0.9, use multiple substeps (gSAM line 109-110)
    nprec = jnp.maximum(1, jnp.ceil(cfl_max / 0.9)).astype(jnp.int32)
    nprec_float = jnp.asarray(nprec, dtype=jnp.float32)

    nz = qp_col.shape[0]

    def _mpdata_step(carry):
        """Single MPDATA step: upwind + anti-diffusive correction with FCT.

        Fix 4.9: carry includes wp so it can be updated from qp after each
        substep, matching gSAM precip_fall.f90:225-237.
        Returns (qp_out, fz_corr, wp_next).
        """
        qp_current, wp = carry

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
        # Fix 4.9: recompute wp from updated qp for the next substep
        # (gSAM precip_fall.f90:225-237)
        wp_next = _compute_wp(qp_out, nprec_float)
        return qp_out, fz_corr, wp_next

    # Run nprec substeps via while_loop (gSAM line 120-244)
    # Carry: (step_index, qp, accumulated_fz, wp)
    # fz is accumulated over substeps (each substep applies its own flux divergence
    # to temperature in gSAM; we sum here and apply once at the end).
    # wp is carried and updated after each substep (Fix 4.9).
    wp_init = _compute_wp(qp_safe, nprec_float)

    def _body_fun(carry):
        i, qp_step, fz_acc, wp_step = carry
        qp_step, fz_step, wp_next = _mpdata_step((qp_step, wp_step))
        return (i + 1, qp_step, fz_acc + fz_step, wp_next)

    def _cond_fun(carry):
        i, qp_step, fz_acc, wp_step = carry
        return i < nprec

    _, qp_new, fz_acc, _ = jax.lax.while_loop(
        _cond_fun, _body_fun, (0, qp_safe, jnp.zeros(nz), wp_init)
    )

    return qp_new, cfl_max, fz_acc


def _fall_col_bulk(
    qp_col:    jax.Array,   # (nz,)  bulk qp mixing ratio (kg/kg)
    omp_col:   jax.Array,   # (nz,)  rain fraction (0=all ice, 1=all rain)
    omg_col:   jax.Array,   # (nz,)  graupel fraction of ice (0=all snow)
    rho_col:   jax.Array,   # (nz,)  kg/m³
    dz_col:    jax.Array,   # (nz,)  m
    vrain: float, crain: float,
    vsnow: float, csnow: float,
    vgrau: float, cgrau: float,
    dt: float,
) -> tuple:  # (qp_new, fz_acc)
    """
    MPDATA sedimentation for bulk qp = QR + QS + QG using composite terminal
    velocity, matching gSAM precip_fall.f90 + term_vel_qp (microphysics.f90:415-462).

    Fix 4.2: gSAM sediments a single bulk qp field with mass-weighted composite
    terminal velocity rather than QR/QS/QG independently.  After sedimentation
    the caller repartitions into species using the pre-sedimentation omega fractions.

    Terminal velocity (gSAM term_vel_qp):
      wp_rain  = vrain * (rho * qrr)^crain,  qrr = omp * qp
      wp_snow  = vsnow * (rho * qss)^csnow,  qss = (1-omp)*(1-omg) * qp
      wp_grau  = vgrau * (rho * qgg)^cgrau,  qgg = (1-omp)*omg * qp
      wp_bulk  = omp*wp_rain + (1-omp)*(omg*wp_grau + (1-omg)*wp_snow)
    velocity caps: rain 9 m/s, snow 2 m/s, graupel 10 m/s; rhofac = sqrt(1.29/rho).
    """
    qp_safe = jnp.maximum(qp_col, 0.0)
    rhofac  = jnp.sqrt(1.29 / rho_col)
    irhoadz = 1.0 / (rho_col * dz_col)
    eps_qp  = 1e-12

    def _composite_vt(qp_cur):
        """Composite terminal velocity from current bulk qp (gSAM term_vel_qp)."""
        qp_c   = jnp.maximum(qp_cur, 0.0)
        qrr    = qp_c * omp_col
        qss    = qp_c * (1.0 - omp_col) * (1.0 - omg_col)
        qgg    = qp_c * (1.0 - omp_col) * omg_col
        vt_r   = jnp.minimum(9.0,  vrain * (rho_col * jnp.maximum(qrr, 0.0)) ** crain)
        vt_s   = jnp.minimum(2.0,  vsnow * (rho_col * jnp.maximum(qss, 0.0)) ** csnow)
        vt_g   = jnp.minimum(10.0, vgrau * (rho_col * jnp.maximum(qgg, 0.0)) ** cgrau)
        wp_bulk = (omp_col * vt_r
                   + (1.0 - omp_col) * (omg_col * vt_g + (1.0 - omg_col) * vt_s))
        return jnp.where(qp_c > eps_qp, rhofac * wp_bulk, 0.0)

    def _compute_wp(qp_cur, nprec_f):
        vt_c = _composite_vt(qp_cur)
        return jnp.minimum(1.0, vt_c * dt / dz_col / nprec_f)

    vt_init = _composite_vt(qp_safe)
    cfl_raw = vt_init * dt / dz_col
    cfl_max = jnp.max(cfl_raw)

    nprec       = jnp.maximum(1, jnp.ceil(cfl_max / 0.9)).astype(jnp.int32)
    nprec_float = jnp.asarray(nprec, dtype=jnp.float32)
    nz = qp_col.shape[0]

    def _mpdata_step(carry):
        qp_current, wp = carry
        qp_above = jnp.concatenate([qp_current[1:], qp_current[-1:]])
        qp_below = jnp.concatenate([qp_current[:1], qp_current[:-1]])
        mx0 = jnp.maximum(jnp.maximum(qp_below, qp_above), qp_current)
        mn0 = jnp.minimum(jnp.minimum(qp_below, qp_above), qp_current)

        fz = qp_current * wp
        fz_top = jnp.concatenate([fz[1:], jnp.zeros((1,))])
        tmp_qp = jnp.maximum(0.0, qp_current - (fz_top - fz) * irhoadz)

        tmp_flux = tmp_qp * wp
        tmp_flux_below = jnp.concatenate([tmp_flux[:1], tmp_flux[:-1]])
        www = 0.5 * (1.0 + wp * irhoadz) * (tmp_flux_below - tmp_flux)

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

        mx_lim = rho_col * dz_col * (mx - tmp_qp) / (pneg_www_top + ppos_www + eps_fct)
        mn_lim = rho_col * dz_col * (tmp_qp - mn) / (ppos_www_top + pneg_www + eps_fct)
        mx_below = jnp.concatenate([mx_lim[:1], mx_lim[:-1]])
        mn_below = jnp.concatenate([mn_lim[:1], mn_lim[:-1]])

        fz_corr = (fz
                   + ppos_www * jnp.minimum(1.0, jnp.minimum(mx_lim, mn_below))
                   - pneg_www * jnp.minimum(1.0, jnp.minimum(mx_below, mn_lim)))
        fz_corr_top = jnp.concatenate([fz_corr[1:], jnp.zeros((1,))])
        qp_out  = jnp.maximum(0.0, qp_current - (fz_corr_top - fz_corr) * irhoadz)
        wp_next = _compute_wp(qp_out, nprec_float)
        return qp_out, fz_corr, wp_next

    wp_init = _compute_wp(qp_safe, nprec_float)

    def _body_fun(carry):
        i, qp_step, fz_acc, wp_step = carry
        qp_step, fz_step, wp_next = _mpdata_step((qp_step, wp_step))
        return (i + 1, qp_step, fz_acc + fz_step, wp_next)

    def _cond_fun(carry):
        i, qp_step, fz_acc, wp_step = carry
        return i < nprec

    _, qp_new, fz_acc, _ = jax.lax.while_loop(
        _cond_fun, _body_fun, (0, qp_safe, jnp.zeros(nz), wp_init)
    )
    return qp_new, fz_acc


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

    Fix 4.2: gSAM sediments bulk qp = QR+QS+QG with composite terminal velocity
    (term_vel_qp in microphysics.f90:415-462).  After sedimentation, species are
    repartitioned from the bulk using the pre-sedimentation omega fractions:
      QR_new = qp_new * omp
      QS_new = qp_new * (1-omp) * (1-omg)
      QG_new = qp_new * (1-omp) * omg

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

    # Fix 4.2: bulk sedimentation of qp = QR + QS + QG with composite terminal velocity.
    # Pre-sedimentation phase fractions (omp, omg) are frozen and used to:
    #  1. compute composite terminal velocity during sedimentation
    #  2. repartition qp into species after sedimentation
    a_pr_f = 1.0 / (p.tprmax - p.tprmin)
    a_gr_f = 0.0 if p.donograupel else 1.0 / (p.tgrmax - p.tgrmin)
    omp_3d = jnp.clip((TABS - p.tprmin) * a_pr_f, 0.0, 1.0)  # (nz,ny,nx)
    omg_3d = jnp.clip((TABS - p.tgrmin) * a_gr_f, 0.0, 1.0)  # (nz,ny,nx)

    QP = QR + QS + QG

    def _col_bulk(qp_col, omp_col, omg_col):
        """Sediment one column of bulk qp."""
        return _fall_col_bulk(
            qp_col, omp_col, omg_col, rho_1d, dz_1d,
            vrain, crain, vsnow, csnow, vgrau, cgrau, dt,
        )

    # vmap pattern mirrors _fall_species: outer over ny (axis=1 of (nz,ny,nx)),
    # inner over nx (axis=1 of the resulting (nz,nx) slices).
    _vmap_i  = jax.vmap(_col_bulk, in_axes=(1, 1, 1), out_axes=(1, 1))
    _vmap_ji = jax.vmap(_vmap_i,   in_axes=(1, 1, 1), out_axes=(1, 1))
    QP_new, fz_bulk = _vmap_ji(QP, omp_3d, omg_3d)

    # Repartition updated bulk qp into species using pre-sedimentation fractions
    QR_new = QP_new * omp_3d
    QS_new = QP_new * (1.0 - omp_3d) * (1.0 - omg_3d)
    QG_new = QP_new * (1.0 - omp_3d) * omg_3d

    # Fix 4.3: Interface-flux-divergence latent heat from sedimentation.
    # Matches gSAM precip_fall.f90 line 200:
    #   lat_heat = -(lfac(kc)*fz(kc) - lfac(k)*fz(k)) * irhoadz(k)
    #   t(i,j,k) = t(i,j,k) - lat_heat
    # where lfac(k) = fac_cond + (1-omega(k))*fac_fus at cell centre k,
    # fz(k) is the flux at the bottom face of cell k (top face of cell k-1),
    # and fz(nz)=0 (top boundary).
    #
    # In JAX convention (k=0 is bottom), fz_acc[k] is the accumulated flux
    # at the top face of cell k (= bottom face of cell k+1 in 1-indexed).
    # kc = k+1 in Fortran; fz(kc) = flux entering from above = fz_acc[k+1].
    # The divergence for cell k: fz(kc)*irhoadz(k) - fz(k)*irhoadz(k).
    # Combine lfac weighting: lfac(kc)*fz(kc) uses lfac from cell above.
    a_pr = 1.0 / (p.tprmax - p.tprmin)
    omp  = jnp.clip((TABS - p.tprmin) * a_pr, 0.0, 1.0)
    lfac = FAC_COND * omp + FAC_SUB * (1.0 - omp)  # (nz, ny, nx) cell-centre latent factor

    irhoadz = 1.0 / (jnp.array(rho_1d)[:, None, None] * jnp.array(dz_1d)[:, None, None])

    def _flux_div_dT(fz_acc, lfac_3d):
        """Temperature increment from interface-flux-weighted sedimentation divergence.

        Matches gSAM precip_fall.f90 lines 200-201:
          lat_heat = -(lfac(kc)*fz(kc) - lfac(k)*fz(k)) * irhoadz(k)
          t(k) = t(k) - lat_heat
        where in gSAM, fz(k) < 0 (downward, negative sign convention), kc = k+1 is
        the cell above k (Fortran 1-indexed, k=1 at bottom), and fz(nz)=0 at top.

        JSam convention (0-indexed, j=0 at bottom):
          fz_acc[j] > 0 = downward flux at bottom face of cell j (positive = downward).
          gSAM fz(k) = -fz_acc[j]  where j = k-1  (sign flip: gSAM is negative downward).
          gSAM fz(kc) = fz(k+1) = -fz_acc[j+1].

        Substituting into gSAM formula:
          lat_heat[j] = -(lfac[j+1]*(-fz_acc[j+1]) - lfac[j]*(-fz_acc[j])) * irhoadz[j]
                      = (lfac[j+1]*fz_acc[j+1] - lfac[j]*fz_acc[j]) * irhoadz[j]
          TABS_new[j] = TABS[j] - lat_heat[j]

        Boundaries:
          fz_acc[j+1] = 0 for top cell (j=nz-1): gSAM fz(nz)=0.
          lfac_above[nz-1] = 0: gSAM lfac(nz)=0.
        """
        # fz_acc[j+1]: flux from above (= 0 for top cell, gSAM fz(nz)=0)
        fz_above = jnp.concatenate([fz_acc[1:], jnp.zeros_like(fz_acc[:1])], axis=0)
        # lfac[j+1]: latent factor of cell above (= 0 for top cell, gSAM lfac(nz)=0)
        lfac_above = jnp.concatenate([lfac_3d[1:], jnp.zeros_like(lfac_3d[:1])], axis=0)
        # lat_heat[j] = (lfac[j+1]*fz_acc[j+1] - lfac[j]*fz_acc[j]) * irhoadz[j]
        lat_heat = (lfac_above * fz_above - lfac_3d * fz_acc) * irhoadz
        return lat_heat

    # Fix 4.2+4.3: apply latent heat flux divergence to the single bulk flux.
    # gSAM applies one call to precip_fall for all of qp with composite lfac.
    dT_bulk  = _flux_div_dT(fz_bulk, lfac)
    TABS_new = TABS - dT_bulk

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
# Cloud liquid sedimentation stub  (port of cloud_fall.f90)
# ---------------------------------------------------------------------------

def cloud_fall(
    QC:   jax.Array,   # (nz, ny, nx)  cloud liquid mixing ratio
    TABS: jax.Array,   # (nz, ny, nx)  temperature (for latent heat)
    metric: dict,
    params: MicroParams,
    dt: float,
    landmask: "jax.Array | None" = None,  # (ny, nx) 0=ocean 1=land
) -> tuple:  # (QC_new, TABS_new)
    """
    Fix 4.10: Stokes cloud liquid sedimentation (gSAM cloud_fall.f90).

    When params.docloudfall is False (IRMA default: docloudfall=F), this is a
    no-op that returns the inputs unchanged.  The field and stub exist so that
    MicroParams.docloudfall can be set to True without code changes.

    When True, the full Bretherton et al. (2007) lognormal Stokes velocity
    scheme would be implemented here.  For IRMA this branch is never reached.
    """
    if not params.docloudfall:
        return QC, TABS
    # docloudfall=True not yet implemented for IRMA (docloudfall=F in IRMA nml).
    # Raise at runtime if somehow enabled, to avoid silent no-op masking a bug.
    raise NotImplementedError(
        "cloud_fall: docloudfall=True is not yet implemented in JSAM. "
        "Set MicroParams(docloudfall=False) for IRMA."
    )


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

    # 2b. Cloud liquid sedimentation (Fix 4.10: gSAM cloud_fall.f90).
    # docloudfall=False for IRMA; stub is a no-op in that case.
    if params.docloudfall:
        QC, TABS = cloud_fall(QC, TABS, metric, params, dt, landmask=landmask)

    # 3. Saturation adjustment (cloud() in gSAM micro_proc)
    # Fix 4.1: use full 3D pressure (reference + perturbation) for qsat calls,
    # matching gSAM cloud.f90 which uses pp(i,j,k) = pres_ref + pp_pert.
    _p_pert = state.p_prev   # (nz, ny, nx) Pa perturbation pressure
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
            p_pert_pa=_p_pert,
        )
    else:
        TABS, QV, QC, QI = satadj(
            TABS, QV, QC, QI, QR, QS, QG, metric, params,
            p_pert_pa=_p_pert,
        )

    # 4. Precipitation processes (precip_proc() in gSAM micro_proc)
    # Evaporation coefficients are recomputed every 10 steps, matching gSAM's
    # precip_init call frequency: mod(nstep,10).eq.0.and.icycle.eq.1
    # micro_proc is not JIT-compiled, so int(state.nstep) works fine.
    nstep = int(state.nstep)
    if _evap_coef_cache["nstep"] < 0 or nstep % 10 == 0:
        _evap_coef_cache["coefs"] = _compute_evap_coefs(TABS, metric, params)
        _evap_coef_cache["nstep"] = nstep
    TABS, QV, QC, QI, QR, QS, QG = precip_proc(
        TABS, QC, QI, QR, QS, QG, QV, metric, params, dt,
        evap_coefs=_evap_coef_cache["coefs"],
        landmask=landmask,
        p_pert_pa=_p_pert,
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

    # 2b. Cloud liquid sedimentation (Fix 4.10: gSAM cloud_fall.f90).
    # docloudfall=False for IRMA; stub is a no-op in that case.
    if params.docloudfall:
        QC, TABS = cloud_fall(QC, TABS, metric, params, dt, landmask=landmask)

    # 3. Saturation adjustment
    # Fix 4.1: use full 3D pressure for qsat calls (gSAM uses pp(i,j,k) = pres + pp_pert).
    _p_pert = state.p_prev   # (nz, ny, nx) Pa perturbation pressure
    if tabs_phys is not None:
        _gamaz_3d = metric["gamaz"][:, None, None]
        TABS, QV, QC, QI = satadj(
            tabs_phys, QV, QC, QI, QR, QS, QG, metric, params,
            tabs_dry_override=TABS - _gamaz_3d,
            tabs_guess=tabs_phys,
            p_pert_pa=_p_pert,
        )
    else:
        TABS, QV, QC, QI = satadj(
            TABS, QV, QC, QI, QR, QS, QG, metric, params,
            p_pert_pa=_p_pert,
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
        p_pert_pa=_p_pert,
    )

    new_state = ModelState(
        U   =state.U, V=state.V, W=state.W,
        TABS=TABS, QV=QV, QC=QC, QI=QI, QR=QR, QS=QS, QG=QG,
        TKE =state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, precip_ref
