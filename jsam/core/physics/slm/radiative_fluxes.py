"""
Radiative flux transfer over the land surface.

Direct port of gSAM ``SRC/SLM/radiative_fluxes.f90``. Computes net absorbed
shortwave and longwave radiation by the canopy and the soil/snow/ice surface,
along with the upwelling surface radiation and a provisional skin temperature
derived from the upwelling LW at canopy top.

The port is fully vectorised over ``(ny, nx)``. Every Fortran branch is
rewritten as a ``jnp.where`` on boolean masks so the function is trace-safe
under ``jax.jit``. The goal is bit-close reproduction of gSAM's single
precision output so downstream IRMA debug tensors diff cleanly against the
oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import jax
import jax.numpy as jnp

from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.state import SLMState, SLMStatic


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class RadiativeOutput:
    """Per-cell radiative fluxes returned by :func:`radiative_fluxes`.

    All fields are shape ``(ny, nx)``. The ``_canop`` suffix refers to the
    canopy layer (Fortran index 1), the ``_soil`` suffix refers to the ground
    surface (Fortran index 2, which covers bare soil, snow or land ice as
    appropriate), and ``_up`` is the atmosphere-facing upwelling flux at the
    reference level (Fortran ``net_*(1)`` for upwelling fluxes after the
    canopy transmittance step).
    """

    net_sw_canop:  jax.Array
    net_sw_soil:   jax.Array
    net_sw_up:     jax.Array   # upwelling SW from canopy top to atmosphere

    net_lw_canop:  jax.Array
    net_lw_soil:   jax.Array
    net_lw_up:     jax.Array   # upwelling LW from canopy top to atmosphere
    net_lw_dn:     jax.Array   # downwelling LW on canopy top (pass-through)

    net_rad_canop: jax.Array   # total net radiation absorbed by canopy
    net_rad_soil:  jax.Array   # total net radiation absorbed by ground

    t_skin:        jax.Array   # provisional skin temperature (K)

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "net_sw_canop", "net_sw_soil", "net_sw_up",
        "net_lw_canop", "net_lw_soil", "net_lw_up", "net_lw_dn",
        "net_rad_canop", "net_rad_soil",
        "t_skin",
    )

    def tree_flatten(self):
        children = [getattr(self, f) for f in self._dynamic_fields]
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._dynamic_fields, children)))


jax.tree_util.register_pytree_node(
    RadiativeOutput, RadiativeOutput.tree_flatten, RadiativeOutput.tree_unflatten
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def radiative_fluxes(
    state: SLMState,
    static: SLMStatic,
    params: SLMParams,
    sw_dir_vis: jax.Array,
    sw_dif_vis: jax.Array,
    sw_dir_nir: jax.Array,
    sw_dif_nir: jax.Array,
    lwds: jax.Array,
    coszrs: jax.Array,
    *,
    dolongwave: bool = True,
    doshortwave: bool = True,
) -> RadiativeOutput:
    """Compute net shortwave and longwave radiation partitioning over land.

    Mirrors the per-cell Fortran subroutine ``radiative_fluxes(i, j)`` in
    ``gSAM/SRC/SLM/radiative_fluxes.f90``. Inputs with suffix ``_vis``/``_nir``
    are the RRTMG (or oracle) visible and near-IR down-welling beams split
    into direct (``_dir``) and diffuse (``_dif``). ``lwds`` is down-welling
    longwave and ``coszrs`` is the cosine of the solar zenith angle.
    """
    sigma = jnp.float32(params.sigma)
    IR_emis_snow = jnp.float32(params.IR_emis_snow)
    IR_emis_ice = jnp.float32(params.IR_emis_ice)

    snow_mass = state.snow_mass
    snowt = state.snowt
    soilt1 = state.soilt[0]
    soilw1 = state.soilw[0]
    t_canop = state.t_canop

    LAI = static.LAI
    phi_1 = static.phi_1
    phi_2 = static.phi_2
    alb_vis_v = static.alb_vis_v
    alb_nir_v = static.alb_nir_v
    alb_vis_s = static.alb_vis_s
    alb_nir_s = static.alb_nir_s
    IR_emis_vege = static.IR_emis_vege
    IR_emis_grnd = static.IR_emis_grnd
    icemask = static.icemask

    zero = jnp.zeros_like(snow_mass)

    has_snow = snow_mass > 0.0
    is_ice = icemask == 1.0

    # ------------------------------------------------------------------
    # Provisional t_skin (pre-LW update): snow surface if snow, else
    # top-layer soil temperature.
    # ------------------------------------------------------------------
    t_skin_prov = jnp.where(has_snow, snowt, soilt1)

    # ------------------------------------------------------------------
    # SHORTWAVE
    # Fortran gates the SW block on ``doshortwave .and. coszrsxy > 0``.
    # We compute the body unconditionally using a floored cosine so the
    # arithmetic is safe, then mask out cells where coszrs<=0.
    # ------------------------------------------------------------------
    coszrs_safe = jnp.maximum(coszrs, jnp.float32(0.01))
    ka_dir = phi_1 / coszrs_safe + phi_2
    explai = jnp.exp(-ka_dir * LAI)                      # direct beam
    ka_dif = phi_1 + phi_2
    explai0 = jnp.exp(-ka_dif * LAI)                     # diffuse beam

    # Net absorbed SW by canopy (Fortran lines 80..83)
    net_rad_canop_sw = (
          sw_dir_vis * (1.0 - alb_vis_v * (1.0 - explai)  - explai)
        + sw_dif_vis * (1.0 - alb_vis_v * (1.0 - explai0) - explai)
        + sw_dir_nir * (1.0 - alb_nir_v * (1.0 - explai)  - explai)
        + sw_dif_nir * (1.0 - alb_nir_v * (1.0 - explai0) - explai)
    )

    net_swdn_canop = sw_dir_vis + sw_dif_vis + sw_dir_nir + sw_dif_nir
    net_swup_canop = net_rad_canop_sw - net_swdn_canop * (1.0 - explai)

    # Net absorbed SW by soil (snow-free branch, Fortran 96..100)
    wetfactor = 1.0 - 0.5 * soilw1
    soil_sw_nosnow = (
          sw_dir_vis * (1.0 - alb_vis_s * wetfactor) * explai
        + sw_dif_vis * (1.0 - alb_vis_s * wetfactor) * explai0
        + sw_dir_nir * (1.0 - alb_nir_s * wetfactor) * explai
        + sw_dif_nir * (1.0 - alb_nir_s * wetfactor) * explai0
    )

    # Snow branch (Fortran 102..105): hard-coded 0.75/0.45 albedo factors.
    soil_sw_snow = (
          sw_dir_vis * (1.0 - 0.75) * explai
        + sw_dif_vis * (1.0 - 0.45) * explai0
        + sw_dir_nir * (1.0 - 0.75) * explai
        + sw_dif_nir * (1.0 - 0.45) * explai0
    )

    net_rad_soil_sw = jnp.where(has_snow, soil_sw_snow, soil_sw_nosnow)

    net_swdn_soil = net_swdn_canop * explai
    net_swup_soil = net_rad_soil_sw - net_swdn_soil

    # Apply the ``doshortwave .and. coszrs>0`` gate after the fact.
    sw_active = coszrs > 0.0
    if not doshortwave:
        sw_active = jnp.zeros_like(sw_active)

    net_sw_canop = jnp.where(sw_active, net_rad_canop_sw, zero)
    net_sw_soil  = jnp.where(sw_active, net_rad_soil_sw,  zero)
    net_swup_canop = jnp.where(sw_active, net_swup_canop, zero)

    # Running totals (Fortran `net_rad` accumulator)
    net_rad_canop = net_sw_canop
    net_rad_soil  = net_sw_soil

    # ------------------------------------------------------------------
    # LONGWAVE
    # ------------------------------------------------------------------
    # Emitted TIR from canopy and ground surfaces (Fortran 135..144).
    tir_canop = IR_emis_vege * sigma * t_canop ** 4
    tir_snow  = IR_emis_snow * sigma * snowt ** 4
    tir_ice   = IR_emis_ice  * sigma * soilt1 ** 4
    tir_bare  = IR_emis_grnd * sigma * soilt1 ** 4
    tir_ground = jnp.where(
        has_snow,
        tir_snow,
        jnp.where(is_ice, tir_ice, tir_bare),
    )

    # The Fortran ground emissivity used in the "reflected back" term
    # (line 191) is always IR_emis_grnd regardless of snow/ice — we keep
    # that exactly for bit-close parity.
    emis_grnd_reflect = IR_emis_grnd

    # --- Step 1: transmit LW through canopy (Fortran 149..164) --------
    fdn1_canop = lwds
    net_lwdn_canop = fdn1_canop
    fdn2_canop = (1.0 - IR_emis_vege) * fdn1_canop + tir_canop
    net_lw_canop_pre = fdn1_canop - fdn2_canop

    # Canopy SW + downward LW absorption contribution.
    net_rad_canop_lw1 = net_rad_canop + (fdn1_canop - fdn2_canop)

    # --- Step 2: ground surface LW budget (Fortran 179..203) ----------
    fdn1_soil = fdn2_canop
    net_lwdn_soil = fdn1_soil
    fdn2_soil = jnp.zeros_like(fdn1_soil)
    fup2_soil = jnp.zeros_like(fdn1_soil)

    fup1_soil = tir_ground + (1.0 - emis_grnd_reflect) * fdn1_soil
    net_lwup_soil = fup1_soil

    net_lw_soil_contrib = fdn1_soil - fdn2_soil - fup1_soil + fup2_soil
    net_rad_soil_lw = net_rad_soil + net_lw_soil_contrib
    net_lw_soil = net_lw_soil_contrib

    # --- Step 3: upwelling back through canopy (Fortran 210..224) -----
    fup2_canop = fup1_soil
    fup1_canop = (1.0 - IR_emis_vege) * fup2_canop + tir_canop

    net_lwup_canop = fup1_canop
    t_skin_lw = (fup1_canop / sigma) ** 0.25

    net_lw_canop = net_lw_canop_pre + (fup2_canop - fup1_canop)
    net_rad_canop_lw = net_rad_canop_lw1 + (fup2_canop - fup1_canop)

    # ------------------------------------------------------------------
    # Apply the dolongwave gate.
    # ------------------------------------------------------------------
    if dolongwave:
        net_lw_canop_out  = net_lw_canop
        net_lw_soil_out   = net_lw_soil
        net_lw_up_out     = net_lwup_canop
        net_lw_dn_out     = net_lwdn_canop
        net_rad_canop_out = net_rad_canop_lw
        net_rad_soil_out  = net_rad_soil_lw
        t_skin_out        = t_skin_lw
    else:
        net_lw_canop_out  = zero
        net_lw_soil_out   = zero
        net_lw_up_out     = zero
        net_lw_dn_out     = zero
        net_rad_canop_out = net_rad_canop
        net_rad_soil_out  = net_rad_soil
        t_skin_out        = t_skin_prov

    return RadiativeOutput(
        net_sw_canop=net_sw_canop,
        net_sw_soil=net_sw_soil,
        net_sw_up=net_swup_canop,
        net_lw_canop=net_lw_canop_out,
        net_lw_soil=net_lw_soil_out,
        net_lw_up=net_lw_up_out,
        net_lw_dn=net_lw_dn_out,
        net_rad_canop=net_rad_canop_out,
        net_rad_soil=net_rad_soil_out,
        t_skin=t_skin_out,
    )
