"""
SLM canopy / surface resistances.

Direct port of ``gSAM1.8.7/SRC/SLM/resistances.f90``. Computes:

* ``r_b``      : leaf boundary layer resistance (set to 0.5 * r_a, per IFS-
                 style simplification used by gSAM since Jan 2023).
* ``r_c``      : stomatal (canopy) resistance with Jarvis-style multiplicative
                 stress factors (light, VPD, temperature, soil moisture),
                 and a two-leaf sunlit/shaded LAI partition for the light
                 factor.
* ``r_d``      : under-canopy aerodynamic resistance (canopy-air-space ↔
                 ground) with stability correction and snow over-write.
* ``r_litter`` : litter resistance — currently set to 0 in gSAM, kept here
                 for interface parity.

``r_a`` is NOT computed in this routine in gSAM; it is produced by
``transfer_coef`` earlier in the step and is an INPUT here (needed to
form ``r_b``). ``r_soil`` is also not set here — gSAM computes it later
in ``vapor_fluxes_soil.f90`` — so it is NOT returned by this function.
The caller (run_slm) is responsible for ``r_soil``.

All arrays are (ny, nx); fully vectorised over the grid with ``jnp.where``
branches instead of Python ``if``s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.sat import esatw
from jsam.core.physics.slm.state import SLMState, SLMStatic


# Physical constants (from gSAM params.f90 / slm_params.f90) ---------------
_GGR: float = 9.79764       # gravitational acceleration (m/s^2) — gSAM consts.f90 ggr
_Z0_SOIL: float = 0.005     # baresoil roughness length (m)
_RHO_SNOW: float = 100.0    # snow density (kg/m^3)


# ---------------------------------------------------------------------------
# Output dataclass (pytree)
# ---------------------------------------------------------------------------
@dataclass
class ResistancesOutput:
    r_b:      jax.Array   # (ny, nx)  leaf boundary-layer resistance (s/m)
    r_c:      jax.Array   # (ny, nx)  stomatal (canopy) resistance   (s/m)
    r_d:      jax.Array   # (ny, nx)  under-canopy aerodynamic       (s/m)
    r_litter: jax.Array   # (ny, nx)  litter resistance (always 0)   (s/m)

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "r_b", "r_c", "r_d", "r_litter",
    )

    def tree_flatten(self):
        children = [getattr(self, f) for f in self._dynamic_fields]
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls._dynamic_fields, children)))


jax.tree_util.register_pytree_node(
    ResistancesOutput,
    ResistancesOutput.tree_flatten,
    ResistancesOutput.tree_unflatten,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def resistances(
    state: SLMState,
    static: SLMStatic,
    params: SLMParams,
    radsfc_par: jax.Array,   # (ny, nx)  downwelling SW at canopy top (W/m^2)
    pressf: jax.Array,       # (ny, nx)  surface pressure (hPa/mbar)
    r_a: jax.Array,          # (ny, nx)  aerodynamic resistance from transfer_coef
) -> ResistancesOutput:
    """Port of ``SUBROUTINE resistances`` (resistances.f90).

    Parameters
    ----------
    state       : prognostic SLM state (uses soilt, soilw, t_cas, q_cas,
                  snow_mass, ustar).
    static      : frozen land-surface parameters.
    params      : scalar SLM parameters (Rc_max, T_opt).
    radsfc_par  : downwelling shortwave radiation at canopy top (W/m^2),
                  i.e. gSAM's ``swdsxy``.
    pressf      : surface pressure in hPa (gSAM's ``pressf``).
    r_a         : aerodynamic resistance produced by ``transfer_coef``.
                  Needed because in gSAM ``r_b = 0.5 * r_a``.

    Returns
    -------
    ResistancesOutput with r_b, r_c, r_d, r_litter — each (ny, nx).
    """
    # Pull state / static fields -------------------------------------------
    soilt      = state.soilt          # (nsoil, ny, nx)
    soilw      = state.soilw          # (nsoil, ny, nx)
    t_cas      = state.t_cas          # (ny, nx)
    q_cas      = state.q_cas          # (ny, nx)
    snow_mass  = state.snow_mass      # (ny, nx)
    ustar      = state.ustar          # (ny, nx)

    LAI        = static.LAI           # (ny, nx)
    ztop       = static.ztop          # (ny, nx)
    Rc_min     = static.Rc_min        # (ny, nx)
    Rgl        = static.Rgl           # (ny, nx)
    hs_rc      = static.hs_rc         # (ny, nx)
    phi_1      = static.phi_1         # (ny, nx)
    phi_2      = static.phi_2         # (ny, nx)
    rootF      = static.rootF         # (nsoil, ny, nx)
    s_depth    = static.s_depth       # (nsoil, ny, nx)
    w_s_FC     = static.w_s_FC        # (nsoil, ny, nx)
    w_s_WP     = static.w_s_WP        # (nsoil, ny, nx)
    theta_FC   = static.theta_FC      # (nsoil, ny, nx)
    theta_WP   = static.theta_WP      # (nsoil, ny, nx)
    poro_soil  = static.poro_soil     # (nsoil, ny, nx)
    vegetated  = static.vegetated     # (ny, nx)  bool

    Rc_max     = params.Rc_max
    T_opt      = params.T_opt

    # Guard against ustar = 0 for non-vegetated cells feeding into the
    # vegetated branch (we still evaluate both branches under vmap/where).
    ustar_safe = jnp.maximum(ustar, 1.0e-6)

    # ------------------------------------------------------------------
    # r_d : under-canopy aerodynamic resistance
    # ------------------------------------------------------------------
    soilt1 = soilt[0]                        # top soil layer (ny, nx)
    temp_diff = t_cas - soilt1               # >0: stable under-canopy

    # Stability correction factor:
    #   unstable: factor = 1
    #   stable  : factor = 1 / (1 + 0.5 * min(10, rd_correc_fac))
    #     rd_correc_fac = g * ztop * max(0, temp_diff) / soilt1 / ustar^2
    soilt1_safe = jnp.maximum(soilt1, 1.0e-6)
    rd_correc_fac = (
        _GGR * ztop * jnp.maximum(0.0, temp_diff)
        / soilt1_safe / (ustar_safe ** 2)
    )
    factor_stab = 1.0 / (1.0 + 0.5 * jnp.minimum(10.0, rd_correc_fac))
    factor = jnp.where(temp_diff < 0.0, 1.0, factor_stab)

    # Turbulent transfer coefficient under dense canopy (with stab corr)
    Cs_dense = 0.004 * factor

    # Bare-soil transfer coefficient (Zeng/Oleson form):
    #   Cs_bare = 0.4/0.13 * (z0_soil * ustar / 1.5e-5)^(-0.45)
    Cs_bare = (0.4 / 0.13) * (
        (_Z0_SOIL * ustar_safe / 1.5e-5) ** (-0.45)
    )

    # LAI-weighted combination: exp(-LAI) weights the bare fraction.
    expmlai = jnp.exp(-LAI)
    Cs = Cs_bare * expmlai + Cs_dense * (1.0 - expmlai)

    # r_d = 1/(ustar*Cs), capped at 400 s/m
    r_d_veg = jnp.minimum(400.0, 1.0 / (ustar_safe * jnp.maximum(Cs, 1.0e-12)))

    # ------------------------------------------------------------------
    # r_b : leaf boundary layer resistance (simplified, IFS-style)
    # ------------------------------------------------------------------
    r_b_veg = 0.5 * r_a

    # ------------------------------------------------------------------
    # r_c : stomatal resistance — Jarvis-style multiplicative factors
    # ------------------------------------------------------------------
    # Two-leaf sun/shade partition of LAI
    k_beer = phi_1 + phi_2                         # extinction coef (ny, nx)
    # Protect k_beer from zero (degenerate canopy geometry)
    k_beer_safe = jnp.where(jnp.abs(k_beer) < 1.0e-6,
                            1.0e-6, k_beer)
    lai_sun_raw   = (1.0 - jnp.exp(-k_beer_safe * LAI)) / k_beer_safe
    lai_shade_raw = LAI - lai_sun_raw

    lai_sun   = jnp.maximum(1.0e-6, lai_sun_raw)
    lai_shade = jnp.maximum(0.0,     lai_shade_raw)

    # Radiation reaching each layer. f_shade = 0.1 (empirical).
    f_shade = 0.1
    sw_sun   = radsfc_par
    sw_shade = f_shade * radsfc_par

    Rgl_safe = jnp.maximum(Rgl, 1.0e-6)
    tmp_sun   = 0.55 * sw_sun   * 2.0 / Rgl_safe / lai_sun
    tmp_shade = (
        0.55 * sw_shade * 2.0 / Rgl_safe
        / jnp.maximum(1.0e-6, lai_shade)
    )

    rc_fac_sun   = (Rc_min / Rc_max + tmp_sun)   / (1.0 + tmp_sun)
    rc_fac_shade = (Rc_min / Rc_max + tmp_shade) / (1.0 + tmp_shade)

    LAI_safe = jnp.maximum(LAI, 1.0e-6)
    rc_fac_rad = (
        lai_sun * rc_fac_sun + lai_shade * rc_fac_shade
    ) / LAI_safe

    # Vapor pressure deficit factor
    #   e_cas = q_cas * pressf / (0.622 + 0.388 * q_cas)  [hPa]
    #   rc_fac_vpd = exp(-hs_rc * (esatw(t_cas) - e_cas))
    e_cas = q_cas * pressf / (0.622 + 0.388 * q_cas)
    rc_fac_vpd = jnp.exp(-hs_rc * (esatw(t_cas) - e_cas))

    # Temperature factor: max(0, 1 - 0.0016*(T_opt - t_cas)^2)
    rc_fac_t = jnp.maximum(0.0, 1.0 - 0.0016 * (T_opt - t_cas) ** 2)

    # Rootzone soil moisture factor — vectorised over the vertical layers.
    # Per-layer contribution:
    #   if rootF[k] <= 0           : 0           (layer not rooted)
    #   elif soilw[k] >= w_s_FC[k] : s_depth * rootF          (no stress)
    #   elif soilw[k] <= w_s_WP[k] : 0                        (wilted)
    #   else                       : s_depth * rootF *
    #                                 (soilw*poro - theta_WP)/(theta_FC - theta_WP)
    # And d_root accumulates s_depth*rootF except for the wilted case
    # (matches Fortran: d_root is only incremented in the "above-FC" and
    # "between-WP-and-FC" branches, NOT when below WP).
    rooted     = rootF > 0.0                          # (nsoil, ny, nx)
    above_FC   = soilw >= w_s_FC
    below_WP   = soilw <= w_s_WP
    in_between = (~above_FC) & (~below_WP)

    # numerator term per layer
    denom_moist = jnp.maximum(theta_FC - theta_WP, 1.0e-12)
    temp_between = (
        s_depth
        * (soilw * poro_soil - theta_WP)
        / denom_moist
        * rootF
    )
    temp_above = s_depth * rootF

    temp_k = jnp.where(
        rooted & above_FC, temp_above,
        jnp.where(rooted & in_between, temp_between, 0.0),
    )
    # d_root increment per layer (not incremented for below_WP)
    droot_k = jnp.where(
        rooted & (above_FC | in_between),
        s_depth * rootF,
        0.0,
    )

    rc_fac_sw_num = jnp.sum(temp_k,  axis=0)         # (ny, nx)
    d_root        = jnp.sum(droot_k, axis=0)         # (ny, nx)
    rc_fac_sw = rc_fac_sw_num / jnp.maximum(1.0e-6, d_root)

    # Final r_c for vegetated cells.
    # tmp2 = max(1e-6, f_rad * f_vpd * f_t * f_sw)
    # r_c  = min(Rc_max, Rc_min / LAI / tmp2)
    tmp2 = jnp.maximum(
        1.0e-6, rc_fac_rad * rc_fac_vpd * rc_fac_t * rc_fac_sw
    )
    r_c_veg = jnp.minimum(Rc_max, Rc_min / LAI_safe / tmp2)

    # ------------------------------------------------------------------
    # Non-vegetated branch : Fortran sets all four to 0, but JAX evaluates
    # the vegetated expressions for all cells before masking via jnp.where.
    # Use sentinel=1.0 so dead-branch divisions produce finite (discarded)
    # values — matching the Fortran if(vegetated) guard semantics.
    # ------------------------------------------------------------------
    veg = vegetated.astype(bool)
    zero = jnp.zeros_like(r_c_veg)
    one  = jnp.ones_like(r_c_veg)

    r_b = jnp.where(veg, r_b_veg, one)
    r_c = jnp.where(veg, r_c_veg, one)
    r_d = jnp.where(veg, r_d_veg, one)
    r_litter = zero    # gSAM hard-codes r_litter = 0

    # ------------------------------------------------------------------
    # Snow over-write on r_d (applies everywhere, even non-veg)
    #   if snow_mass > 0: r_d = max(r_d, 10000 * snow_mass / rho_snow)
    # ------------------------------------------------------------------
    r_d_snow = 10000.0 * snow_mass / _RHO_SNOW
    r_d = jnp.where(
        snow_mass > 0.0,
        jnp.maximum(r_d, r_d_snow),
        r_d,
    )

    return ResistancesOutput(r_b=r_b, r_c=r_c, r_d=r_d, r_litter=r_litter)
