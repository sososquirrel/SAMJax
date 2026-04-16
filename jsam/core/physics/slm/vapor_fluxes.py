"""
Surface vapour (latent heat) fluxes for the Simple Land Model.

Direct port of gSAM ``SRC/SLM/vapor_fluxes_canopy.f90`` and
``SRC/SLM/vapor_fluxes_soil.f90``, merged into a single pure JAX function
operating elementwise over the ``(ny, nx)`` land grid. The ground-level
saturation humidity ``q_gr`` — computed immediately before the vapour
flux call in ``run_slm.f90`` (lines 185-206) — is also produced here so
the block is self-contained and so ``fh_calc`` (a small helper from
``slm_vars.f90:1659-1666``) has exactly one call site in the port.

Semantics
---------
* **Canopy** (vegetated cells only):
    ``evapo_wet``  direct evaporation from the intercepted water ``mw``,
                   weighted by the wet fraction ``mw/mw_mx``
    ``evapo_dry``  transpiration through the ``2 r_b + r_c`` pathway,
                   zeroed on layers whose ``soilw < 0.05`` via ``rootF``
    ``evp_canop``  = ``evapo_wet + evapo_dry``              [kg/m²/s]
    ``lhf_canop``  = ``L * evp_canop``                       [W/m²]
    ``mw_new``     = ``mw - dtn_iter * evapo_wet``  (pre wet-fraction
                   cap; this is the exact Fortran update, kg/m²)

* **Soil** (below canopy for vegetated cells, at surface for bare soil):
    ``q_gr``       ground-level saturation humidity (with ``fh_calc``
                   soil-moisture-potential reduction over unsaturated,
                   warm, snow-free soil; ``qsati`` over snow / ice / cold
                   soil; raw ``qsatw`` over saturated warm soil)
    ``evp_soil``   flux of water vapour from the soil pore space through
                   the ``r_soil``-plus-aerodynamic resistance stack
    ``lhf_soil``   = ``L * evp_soil``                        [W/m²]

* **Totals to the atmosphere**:
    ``evp_air = evp_canop + evp_soil`` (for vegetated cells). Over bare
    soil the canopy contribution is zero and ``evp_air`` is recomputed
    from the ``qsfc-qr`` gradient exactly as in
    ``vapor_fluxes_soil.f90`` lines 98-102.

Latent heat of vaporisation vs sublimation is selected per-cell by the
same ``ice_flag`` branches used in the Fortran (``lcond`` over warm,
non-icy surfaces; ``lsub`` otherwise).

The subcycling loop ``niter = max(1, nint(dt/1))`` that wraps the canopy
call in ``run_slm.f90`` is intentionally **not** implemented here — this
function returns the instantaneous tendency for a single sub-step and
the caller is responsible for iterating ``mw`` forward.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import jax
import jax.numpy as jnp

from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.sat import qsati, qsatw
from jsam.core.physics.slm.state import SLMState, SLMStatic


# ---------------------------------------------------------------------------
# Physical constants (match gSAM ``SRC/params.f90`` / ``SRC/SLM/slm_vars.f90``)
# ---------------------------------------------------------------------------
_GGR: float = 9.81       # gravity, m/s²
_RV:  float = 461.5      # gas constant for water vapour, J/kg/K


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class VaporFluxOutput:
    """Per-cell vapour fluxes returned by :func:`vapor_fluxes`.

    All fields are shape ``(ny, nx)``. Fluxes are in kg/m²/s for the
    ``evp_*`` entries and W/m² for the ``lhf_*`` entries. ``mw_new`` is
    the canopy-intercepted water content (mm, equivalently kg/m²) after
    the Fortran update ``mw += -dtn_iter * evapo_wet`` — this is the raw
    post-evaporation store, before the ``min(1, mw/mw_mx)`` cap is
    applied to scale ``evapo_wet`` into the fraction-weighted flux.
    """
    evapo_wet: jax.Array
    evapo_dry: jax.Array
    evp_canop: jax.Array
    lhf_canop: jax.Array
    evp_soil:  jax.Array
    lhf_soil:  jax.Array
    evp_air:   jax.Array
    lhf_air:   jax.Array
    q_gr:      jax.Array
    mw_new:    jax.Array
    r_soil:    jax.Array   # (ny, nx) soil resistance (s/m) — exposed for q_cas diagnostic

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "evapo_wet", "evapo_dry",
        "evp_canop", "lhf_canop",
        "evp_soil",  "lhf_soil",
        "evp_air",   "lhf_air",
        "q_gr",      "mw_new",
        "r_soil",
    )

    def tree_flatten(self):
        names = tuple(f.name for f in fields(self))
        children = [getattr(self, n) for n in names]
        return children, names

    @classmethod
    def tree_unflatten(cls, names, children):
        return cls(**dict(zip(names, children)))


jax.tree_util.register_pytree_node(
    VaporFluxOutput, VaporFluxOutput.tree_flatten, VaporFluxOutput.tree_unflatten
)


# ---------------------------------------------------------------------------
# Helper — fractional humidity reduction from soil-moisture potential
# ---------------------------------------------------------------------------
def fh_calc(t: jax.Array,
            mps: jax.Array,
            sw: jax.Array,
            B: jax.Array) -> jax.Array:
    """Fractional humidity at the soil surface from the Clapp & Hornberger
    moisture potential, matching ``slm_vars.f90:fh_calc`` (lines 1659-1666).

    ``moist_pot1 = mps / max(1e-10, sw**B) / 1000``  (m of water head)
    ``moist_pot1 = max(-1e8, moist_pot1)``
    ``fh        = min(1, exp(moist_pot1 * g / (Rv * T)))``
    """
    denom = jnp.maximum(1.0e-10, sw ** B)
    moist_pot1 = mps / denom / 1000.0
    moist_pot1 = jnp.maximum(-1.0e8, moist_pot1)
    return jnp.minimum(1.0, jnp.exp(moist_pot1 * _GGR / (_RV * t)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def vapor_fluxes(
    state: SLMState,
    static: SLMStatic,
    params: SLMParams,
    r_a:      jax.Array,
    r_b:      jax.Array,
    r_c:      jax.Array,
    r_d:      jax.Array,
    r_litter: jax.Array,
    qr:       jax.Array,
    pressf:   jax.Array,
    rhosf:    jax.Array,
    dtn_iter: float | jax.Array,
    lcond:    float,
    lsub:     float,
) -> VaporFluxOutput:
    """Combined canopy + soil vapour flux computation.

    Parameters
    ----------
    state, static, params
        Standard SLM containers.
    r_a, r_b, r_c, r_d, r_litter
        Aerodynamic / boundary-layer / canopy / under-canopy / litter
        resistances (s/m), each shape ``(ny, nx)``. ``r_litter`` is zero
        in the current gSAM configuration but is carried for parity.
    qr
        Reference-level specific humidity (kg/kg), shape ``(ny, nx)``.
    pressf
        Surface-level pressure passed to ``qsatw/qsati`` — same units as
        the Fortran (millibar / hPa). Shape ``(ny, nx)``.
    rhosf
        Surface-level air density (kg/m³), shape ``(ny, nx)``.
    dtn_iter
        Sub-step length (s) used for the Fortran cap
        ``evapo_wet <= mw/dtn_iter`` that prevents draining more canopy
        water than exists in a single step.
    lcond, lsub
        Latent heats of condensation and sublimation (J/kg).

    Returns
    -------
    :class:`VaporFluxOutput`
    """
    soilt0 = state.soilt[0]
    soilw0 = state.soilw[0]

    Bconst0    = static.Bconst[0]
    m_pot_sat0 = static.m_pot_sat[0]
    w_s_FC0    = static.w_s_FC[0]

    icemask   = static.icemask
    vegetated = static.vegetated
    vege_YES  = static.vege_YES
    IMPERV    = static.IMPERV
    rootF     = static.rootF          # (nsoil, ny, nx)
    mw_mx     = static.mw_mx

    snow_mass = state.snow_mass
    snowt     = state.snowt
    mw        = state.mw
    mws       = state.mws
    t_canop   = state.t_canop

    tfriz = params.tfriz
    pii   = params.pii

    # ------------------------------------------------------------------
    # q_gr  — ground-level saturation humidity  (run_slm.f90 lines 185-206)
    # ------------------------------------------------------------------
    qsatw_soil = qsatw(soilt0, pressf)
    qsati_soil = qsati(soilt0, pressf)
    qsati_snow = qsati(snowt,  pressf)

    fh = fh_calc(soilt0, m_pot_sat0, soilw0, Bconst0)

    warm_soil = (icemask == 0) & (soilt0 > tfriz)           # bool
    puddle    = mws > 0.0

    # Branch A: warm, ice-free soil with standing water → raw qsatw, sdew=1
    # Branch B: warm, ice-free soil, no puddle → fh*qsatw, sdew=(fh>0.99)
    # Branch C: icy / cold, with snow → qsati(snowt)
    # Branch D: icy / cold, no snow → qsati(soilt0)
    q_gr_warm_puddle = qsatw_soil
    q_gr_warm_dry    = fh * qsatw_soil
    q_gr_cold        = jnp.where(snow_mass > 0.0, qsati_snow, qsati_soil)

    q_gr = jnp.where(
        warm_soil,
        jnp.where(puddle, q_gr_warm_puddle, q_gr_warm_dry),
        q_gr_cold,
    )

    sdew_warm_dry = jnp.where(fh > 0.99, 1.0, 0.0)
    sdew = jnp.where(
        warm_soil,
        jnp.where(puddle, 1.0, sdew_warm_dry),
        1.0,   # cold branch: sdew = 1 (see run_slm.f90 line 205)
    )

    # qsfc — surface-level humidity seen by the turbulent transfer block.
    # For vegetated cells this is the canopy-air-space humidity q_cas;
    # for bare soil it is q_gr.  Additional override: when sdew<0.1 and
    # qsfc<qr, qsfc is pulled up to qr (run_slm.f90:224).
    q_sfc = jnp.where(vegetated, state.q_cas, q_gr)
    q_sfc = jnp.where((sdew < 0.1) & (q_sfc < qr), qr, q_sfc)

    # ==================================================================
    # CANOPY   (vapor_fluxes_canopy.f90)
    # ==================================================================
    canop_warm = (icemask == 0) & (t_canop >= tfriz)
    qsat_canop = jnp.where(
        canop_warm,
        qsatw(t_canop, pressf),
        qsati(t_canop, pressf),
    )
    flag_ice = jnp.where(canop_warm, 1.0, 0.0)

    # --- direct evaporation from intercepted canopy water -------------
    # evapo_wet = min(mw/dtn_iter, flag_ice*(qsat_canop-q_sfc)*rhosf/(2 r_b)) * vege_YES
    gradient_wet = flag_ice * (qsat_canop - q_sfc) * rhosf / (2.0 * r_b)
    evapo_wet = jnp.minimum(mw / dtn_iter, gradient_wet) * vege_YES

    # --- canopy water store update ------------------------------------
    mw_inc = -dtn_iter * evapo_wet
    mw_new = mw + mw_inc
    wet_canop = jnp.minimum(1.0, mw_new / jnp.maximum(mw_mx, 1.0e-20))
    evapo_wet = wet_canop * evapo_wet

    # --- transpiration -------------------------------------------------
    # evapo_dry = max(0, flag_ice*(qsat_canop-q_sfc)*rhosf*(1-wet_canop)/(2 r_b + r_c)) * vege_YES
    evapo_dry0 = jnp.maximum(
        0.0,
        flag_ice * (qsat_canop - q_sfc) * rhosf * (1.0 - wet_canop)
        / (2.0 * r_b + r_c),
    ) * vege_YES

    # Root-layer water availability: remove evapo_dry0*rootF[k] from any
    # layer whose soilw < 0.05.  The Fortran only applies the cut when
    # evapo_dry0 > 0; with evapo_dry0 initialised via max(0,...) above,
    # the subtraction is a no-op when the flux is already zero.
    soilw_all = state.soilw                                  # (nsoil, ny, nx)
    dry_layers = (soilw_all < 0.05).astype(soilw_all.dtype)  # 1 where dry
    dry_root_frac = jnp.sum(dry_layers * rootF, axis=0)      # (ny, nx)
    evapo_dry = evapo_dry0 - evapo_dry0 * dry_root_frac

    # --- canopy totals -------------------------------------------------
    evp_canop = evapo_wet + evapo_dry
    L_canop   = flag_ice * lcond + (1.0 - flag_ice) * lsub
    lhf_canop = L_canop * evp_canop

    # ==================================================================
    # SOIL   (vapor_fluxes_soil.f90)
    # ==================================================================
    # Reference-level q for the soil branch:
    #    vegetated  → qref_tmp = q_sfc  (i.e. canopy-air humidity)
    #    baresoil   → qref_tmp = qr
    qref_tmp = jnp.where(vegetated, q_sfc, qr)

    # ice_flag for the soil branch (note: independent of the canopy one)
    soil_icy = (icemask == 1) | (snow_mass > 0.0) | (soilt0 < tfriz)
    ice_flag = jnp.where(soil_icy, 0.0, 1.0)

    # --- soil diffusion factor (Sakaguchi & Zeng 2009; no snow factor)
    saturated_or_dew = (soilw0 >= w_s_FC0) | (qref_tmp > q_gr)
    soilw_clip = jnp.maximum(0.01, soilw0)
    soil_diff_unsat = 0.25 * (1.0 - jnp.cos(pii * soilw_clip / w_s_FC0)) ** 2
    soil_diff = jnp.where(saturated_or_dew, 1.0, soil_diff_unsat)

    inv_diff_minus_1 = 1.0 / soil_diff - 1.0

    # --- r_soil ---------------------------------------------------------
    # baresoil icy/snow    → r_soil = 0
    # baresoil warm        → clamp(r_a * (1/diff - 1), 100, 10000)
    # vegetated            → clamp(r_d * (1/diff - 1),  50, 10000)
    r_soil_bare_warm = jnp.minimum(10000.0,
                                   jnp.maximum(100.0, r_a * inv_diff_minus_1))
    bare_icy = (icemask == 1) | (snow_mass > 0.0)
    r_soil_bare = jnp.where(bare_icy, 0.0, r_soil_bare_warm)

    r_soil_vege = jnp.minimum(10000.0,
                              jnp.maximum(50.0, r_d * inv_diff_minus_1))

    r_soil = jnp.where(vegetated, r_soil_vege, r_soil_bare)

    totalR_soil = jnp.where(
        vegetated,
        r_soil + r_d + r_litter,
        r_soil + r_a,
    )

    # --- evapo_s --------------------------------------------------------
    # evapo_s = vege_YES * (q_gr - q_sfc) * rhosf / totalR_soil
    # dew (negative) only allowed when sdew==1 (saturated surface)
    evapo_s = vege_YES * (q_gr - q_sfc) * rhosf / totalR_soil
    evapo_s = jnp.where(evapo_s < 0.0, evapo_s * sdew, evapo_s)

    # --- partitioning into evp_soil / evp_air --------------------------
    L_soil = ice_flag * lcond + (1.0 - ice_flag) * lsub

    # Vegetated branch
    evp_soil_vege = evapo_s * (1.0 - IMPERV)
    evp_air_vege  = evp_canop + evp_soil_vege
    lhf_air_vege  = lhf_canop + L_soil * evp_soil_vege

    # Bare-soil branch (vapor_fluxes_soil.f90 lines 96-102)
    # evp_air = (q_sfc - qr) * rhosf / (r_soil + r_a) * (1-IMPERV)
    # sign-gate by sdew, then evapo_s = evp_air, evp_soil = evp_air
    evp_air_bare_raw = (q_sfc - qr) * rhosf / (r_soil + r_a) * (1.0 - IMPERV)
    evp_air_bare = jnp.where(evp_air_bare_raw < 0.0,
                             evp_air_bare_raw * sdew, evp_air_bare_raw)
    evp_soil_bare = evp_air_bare
    lhf_air_bare  = L_soil * evp_air_bare

    # Merge
    evp_soil = jnp.where(vegetated, evp_soil_vege, evp_soil_bare)
    evp_air  = jnp.where(vegetated, evp_air_vege,  evp_air_bare)
    lhf_air  = jnp.where(vegetated, lhf_air_vege,  lhf_air_bare)
    lhf_soil = L_soil * evp_soil

    # ==================================================================
    # Cast all outputs to float32 and pack
    # ==================================================================
    f32 = jnp.float32
    return VaporFluxOutput(
        evapo_wet=evapo_wet.astype(f32),
        evapo_dry=evapo_dry.astype(f32),
        evp_canop=evp_canop.astype(f32),
        lhf_canop=lhf_canop.astype(f32),
        evp_soil=evp_soil.astype(f32),
        lhf_soil=lhf_soil.astype(f32),
        evp_air=evp_air.astype(f32),
        lhf_air=lhf_air.astype(f32),
        q_gr=q_gr.astype(f32),
        mw_new=mw_new.astype(f32),
        r_soil=r_soil.astype(f32),
    )
