"""Bulk aerodynamic surface fluxes (ocean only): sensible, latent, momentum.
Uses Monin-Obukhov stability correction and Large & Pond (1981) neutral coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsam.core.state import ModelState
from jsam.core.physics.sgs import SurfaceFluxes
from jsam.core.physics.microphysics import qsatw as _qsatw_micro, qsati as _qsati_micro


@dataclass(frozen=True)
class BulkParams:
    """Bulk-flux parameters (gSAM defaults)."""
    umin:        float = 1.0
    karman:      float = 0.4
    epsv:        float = 0.61
    salt_factor: float = 0.981  # gSAM surface.f90:55
    ug:          float = 0.0
    vg:          float = 0.0
    p00:         float = 1.0e5
    Rd:          float = 287.04
    Rv:          float = 461.5
    cp:          float = 1004.64
    g:           float = 9.79764
    g_mo:        float = 9.81    # gSAM surface.f90:360 — local g used inside MO stability loop


# ---------------------------------------------------------------------------
# Stability functions (matches oceflx.f90 psimhu / psixhu)
# ---------------------------------------------------------------------------

def _psimhu(xd: jax.Array) -> jax.Array:
    """Unstable momentum stability function (Paulson 1970)."""
    return (jnp.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
            - 2.0 * jnp.arctan(xd) + 1.571)

def _psixhu(xd: jax.Array) -> jax.Array:
    """Unstable heat/moisture stability function."""
    return 2.0 * jnp.log((1.0 + xd * xd) / 2.0)


# ---------------------------------------------------------------------------
# Neutral drag / heat / moisture coefficients (Large & Pond 1981)
# ---------------------------------------------------------------------------

def _cdn(u10: jax.Array) -> jax.Array:
    """Neutral drag coefficient at 10 m."""
    return 0.0027 / u10 + 0.000142 + 0.0000764 * u10


def _neutral_coeffs(vmag: jax.Array, delt: jax.Array) -> tuple:
    """
    Initial neutral exchange coefficients (sqrt form).

    Returns (rdn, rhn, ren) — square roots of the exchange coefficients.
    """
    stable = 0.5 + jnp.sign(delt) * 0.5   # 1 if stable (delt > 0), 0 if not
    rdn = jnp.sqrt(_cdn(vmag))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = jnp.full_like(vmag, 0.0346)
    return rdn, rhn, ren


# ---------------------------------------------------------------------------
# Core bulk-flux computation — one column
# (fully vectorised; operates on (ny, nx) arrays)
# ---------------------------------------------------------------------------

def _one_iteration(vmag: jax.Array, delt: jax.Array, delq: jax.Array,
                   thbot: jax.Array, qbot: jax.Array, zbot: jax.Array,
                   rdn: jax.Array, rhn: jax.Array, ren: jax.Array,
                   params: BulkParams) -> tuple:
    """One Monin-Obukhov iteration; returns (rdn_new, rhn_new, ren_new, ustar, tstar, qstar)."""
    karman = params.karman
    epsv   = params.epsv
    zref   = 10.0   # reference height (m)

    alz = jnp.log(zbot / zref)   # ln(z_bot / z_ref)  (scalar)

    # --- stability parameter z/L ---
    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq

    hol = (karman * params.g_mo * zbot   # gSAM oceflx.f90:147
           * (tstar / thbot + qstar / (1.0 / epsv + qbot))
           / ustar ** 2)
    hol    = jnp.clip(hol, -10.0, 10.0)
    stable = 0.5 + jnp.sign(hol) * 0.5

    # Monin-Obukhov stability functions (matches gSAM oceflx.f90 lines 151-154)
    # am=5.0, bm=16.0 are the gSAM oceflx.f90 constants (NOT Businger 1973 values)
    am, bm = 5.0, 16.0
    xsq    = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - bm * hol)), 1.0)
    xqq    = jnp.sqrt(xsq)   # net: max((abs(1-16*hol))^0.25, 1.0)
    psimh  = -am * hol * stable + (1.0 - stable) * _psimhu(xqq)
    psixh  = -am * hol * stable + (1.0 - stable) * _psixhu(xqq)

    # --- shift rd to measurement height ---
    rd   = rdn / (1.0 + rdn / karman * (alz - psimh))
    u10n = vmag * rd / rdn   # neutral 10-m wind speed

    # --- updated neutral coefficients ---
    rdn_new = jnp.sqrt(_cdn(u10n))
    rhn_new = (1.0 - stable) * 0.0327 + stable * 0.018
    ren_new = jnp.full_like(vmag, 0.0346)

    # --- shift all coeffs to zbot and stability ---
    rd_new = rdn_new / (1.0 + rdn_new / karman * (alz - psimh))
    rh_new = rhn_new / (1.0 + rhn_new / karman * (alz - psixh))
    re_new = ren_new / (1.0 + ren_new / karman * (alz - psixh))

    ustar_new = rd_new * vmag
    tstar_new = rh_new * delt
    qstar_new = re_new * delq

    return rdn_new, rhn_new, ren_new, ustar_new, tstar_new, qstar_new


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

import functools

@functools.partial(jax.jit, static_argnames=("params",))
def bulk_surface_fluxes(state: ModelState, metric: dict, sst: jax.Array,
                        params: BulkParams = BulkParams()) -> SurfaceFluxes:
    """Compute bulk aerodynamic fluxes (shf, lhf, tau_x, tau_y) from lowest-level state."""
    pres0 = metric["pres"][0]
    z0    = metric["z"][0]

    # Use the bottom interface pressure (presi[0]) for the surface Exner function,
    # matching gSAM's use of prespoti(k) = (1000/presi(k))^(Rd/cp).
    pres_sfc = metric["presi"][0]   # Pa, bottom interface level

    exner0   = (pres0    / params.p00) ** (params.Rd / params.cp)
    exner_s  = (pres_sfc / params.p00) ** (params.Rd / params.cp)

    TABS0 = state.TABS[0, :, :]
    QV0   = state.QV[0, :, :]

    u_cc = 0.5 * (state.U[0, :, :-1] + state.U[0, :, 1:])
    v_cc = 0.5 * (state.V[0, :-1, :] + state.V[0, 1:, :])
    u_cc = u_cc + params.ug
    v_cc = v_cc + params.vg

    vmag  = jnp.maximum(params.umin, jnp.sqrt(u_cc ** 2 + v_cc ** 2))

    sst_safe = jnp.where(jnp.isnan(sst), TABS0, sst)

    thbot = TABS0 / exner0
    ts    = sst_safe / exner_s

    delt  = thbot - ts

    pres_sfc_mb = pres_sfc / 100.0   # Pa → hPa (mb) for saturation functions
    qs_water = params.salt_factor * _qsatw_micro(sst_safe, pres_sfc_mb)
    qs_ice   = _qsati_micro(sst_safe, pres_sfc_mb)
    # gSAM surface.f90 line 70-76: use ice saturation when SST < 271 K (sea ice),
    # liquid saturation with salt factor otherwise.
    qs_sfc = jnp.where(sst_safe < 271.0, qs_ice, qs_water)
    delq   = QV0 - qs_sfc

    rdn, rhn, ren = _neutral_coeffs(vmag, delt)

    for _ in range(2):  # gSAM oceflx.f90: exactly 2 passes (first estimate + one iteration)
        rdn, rhn, ren, ustar, tstar, qstar = _one_iteration(
            vmag, delt, delq, thbot, QV0, z0, rdn, rhn, ren, params,
        )

    shf   = -ustar * tstar
    lhf   = -ustar * qstar

    tau_x = -(ustar ** 2) * u_cc / vmag
    tau_y = -(ustar ** 2) * v_cc / vmag

    is_ocean = ~jnp.isnan(sst)
    shf   = jnp.where(is_ocean, shf,   0.0)
    lhf   = jnp.where(is_ocean, lhf,   0.0)
    tau_x = jnp.where(is_ocean, tau_x, 0.0)
    tau_y = jnp.where(is_ocean, tau_y, 0.0)

    return SurfaceFluxes(shf=shf, lhf=lhf, tau_x=tau_x, tau_y=tau_y)
