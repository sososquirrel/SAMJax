"""
SLM state containers.

Two dataclasses, both registered as JAX pytrees:

* :class:`SLMStatic` — frozen land-surface parameters. Static for the
  duration of a run: masks, IGBP-derived fields (z0, ztop, roots, albedos,
  IR emissivities, Rc_min, …), Cosby soil properties (ks, Bconst,
  poro_soil, theta_FC, theta_WP, …) and the (nsoil) layer geometry. Held
  on ``StepConfig`` and treated as JAX leaves so it is traced once by jit
  but never mutated.

* :class:`SLMState` — prognostic variables advanced each step:
  ``soilt, soilw`` (nsoil, ny, nx), plus canopy / snow scalars.

nsoil is hard-coded to 9 throughout gSAM — we do the same here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import jax
import jax.numpy as jnp


NSOIL: int = 9


# ---------------------------------------------------------------------------
# Static fields (frozen parameters)
# ---------------------------------------------------------------------------
@dataclass
class SLMStatic:
    # Masks (ny, nx) int8
    landmask:     jax.Array
    seaicemask:   jax.Array
    landicemask:  jax.Array
    icemask:      jax.Array
    landtype:     jax.Array   # IGBP class 1..16
    vegetated:    jax.Array   # bool (ny, nx)
    vege_YES:     jax.Array   # float (ny, nx), 0 or 1

    # Surface / canopy parameters (ny, nx)
    z0_sfc:       jax.Array
    ztop:         jax.Array
    disp_hgt:     jax.Array
    BAI:          jax.Array
    IMPERV:       jax.Array

    # Albedos (ny, nx)
    alb_vis_v:    jax.Array
    alb_nir_v:    jax.Array
    alb_vis_s:    jax.Array
    alb_nir_s:    jax.Array

    # IR emissivities (ny, nx)
    IR_emis_vege: jax.Array
    IR_emis_grnd: jax.Array

    # Canopy-absorption geometry (ny, nx)
    khai_L:       jax.Array
    phi_1:        jax.Array
    phi_2:        jax.Array
    precip_extinc: jax.Array

    # Stomatal resistance
    Rc_min:       jax.Array
    Rgl:          jax.Array
    hs_rc:        jax.Array

    # Roots
    rootL:        jax.Array   # (ny, nx)
    root_a:       jax.Array
    root_b:       jax.Array
    rootF:        jax.Array   # (nsoil, ny, nx)

    # Soil texture + geometry (nsoil, ny, nx) unless noted
    SAND:         jax.Array
    CLAY:         jax.Array
    s_depth:      jax.Array
    node_z:       jax.Array
    interface_z:  jax.Array

    # Cosby 1984 derived soil hydraulics (nsoil, ny, nx)
    Bconst:       jax.Array
    m_pot_sat:    jax.Array
    ks:           jax.Array
    poro_soil:    jax.Array
    theta_FC:     jax.Array
    theta_WP:     jax.Array
    w_s_FC:       jax.Array
    w_s_WP:       jax.Array
    sst_cond:     jax.Array   # soil solid thermal conductivity
    sst_capa:     jax.Array   # soil heat capacity

    # Canopy water storage cap (ny, nx) — mw_mx (depends on LAI so refreshed
    # whenever LAI is interpolated by month; kept on static because
    # prm_debug500 uses readseaice=false / no slm_forcing path, so the LAI
    # is frozen from the initial month lookup)
    mw_mx:        jax.Array
    mws_mx:       jax.Array   # scalar or (ny, nx)

    # LAI (ny, nx) — resolved from monthly climatology at init time
    LAI:          jax.Array

    # ── pytree registration ───────────────────────────────────────────────
    _dynamic_fields: ClassVar[tuple[str, ...]] = ()

    def tree_flatten(self):
        names = tuple(f.name for f in fields(self))
        children = [getattr(self, n) for n in names]
        return children, names

    @classmethod
    def tree_unflatten(cls, names, children):
        return cls(**dict(zip(names, children)))


jax.tree_util.register_pytree_node(
    SLMStatic, SLMStatic.tree_flatten, SLMStatic.tree_unflatten
)


# ---------------------------------------------------------------------------
# Prognostic state
# ---------------------------------------------------------------------------
@dataclass
class SLMState:
    soilt:     jax.Array   # (nsoil, ny, nx)  K
    soilw:     jax.Array   # (nsoil, ny, nx)  dimensionless wetness (0..1)

    t_canop:   jax.Array   # (ny, nx)  K — leaf temperature
    t_cas:     jax.Array   # (ny, nx)  K — canopy-air-space temperature
    q_cas:     jax.Array   # (ny, nx)  kg/kg — canopy-air-space humidity

    mw:        jax.Array   # (ny, nx)  mm — canopy intercepted water
    mws:       jax.Array   # (ny, nx)  mm — puddle water

    snow_mass: jax.Array   # (ny, nx)  kg/m² — snow water equivalent
    snowt:     jax.Array   # (ny, nx)  K — snow surface temperature
    t_skin:    jax.Array   # (ny, nx)  K — ground skin temperature

    # Optional two-iter M-O state carried step-to-step so the next step
    # can warm-start from the previous converged values (mirrors gSAM
    # which keeps ustar/tstar in the module-level state).
    ustar:     jax.Array   # (ny, nx)  m/s
    tstar:     jax.Array   # (ny, nx)  K

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "soilt", "soilw",
        "t_canop", "t_cas", "q_cas",
        "mw", "mws",
        "snow_mass", "snowt", "t_skin",
        "ustar", "tstar",
    )

    def tree_flatten(self):
        children = [getattr(self, f) for f in self._dynamic_fields]
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._dynamic_fields, children)))

    @classmethod
    def zeros(cls, ny: int, nx: int, nsoil: int = NSOIL) -> "SLMState":
        z2 = jnp.zeros((ny, nx))
        z3 = jnp.zeros((nsoil, ny, nx))
        return cls(
            soilt=z3, soilw=z3,
            t_canop=z2, t_cas=z2, q_cas=z2,
            mw=z2, mws=z2,
            snow_mass=z2, snowt=z2, t_skin=z2,
            ustar=jnp.full((ny, nx), 0.1), tstar=z2,
        )


jax.tree_util.register_pytree_node(
    SLMState, SLMState.tree_flatten, SLMState.tree_unflatten
)
