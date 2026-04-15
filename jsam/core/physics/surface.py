"""
Bulk aerodynamic surface flux computation for jsam.

Port of gSAM ``oceflx.f90`` (CESM1 origin, Marat Khairoutdinov 2018).
Computes ocean surface fluxes — sensible heat, latent heat, and momentum
stress — from the model's lowest layer and a prescribed SST field.

Algorithm
---------
  1. Compute cell-centre wind speed at k=0 from staggered U/V faces.
  2. Neutral drag/heat/moisture coefficients (Large & Pond 1981):
       cdn(U10) = 0.0027/U10 + 0.000142 + 0.0000764*U10
       ctn (unstable) = 0.0327*sqrt(cdn),  ctn (stable) = 0.018*sqrt(cdn)
       cen            = 0.0346*sqrt(cdn)
  3. Two-iteration Monin-Obukhov stability correction for z/L.
  4. Fluxes:
       shf  = −u* · t*          (K·m/s,      positive upward)
       lhf  = −u* · q*          (kg/kg·m/s,  positive upward)
       tau_x = −u*² · u / V     (m²/s²,      negative when u > 0)
       tau_y = −u*² · v / V     (m²/s²)

All outputs are on the cell-centre (ny, nx) grid and are compatible
with ``SurfaceFluxes`` (used by ``sgs_proc``).

Differences from full gSAM ``oceflx``:
  - No TC mode (dotc=False)
  - No gustiness parameter (wd=0)
  - No land / SLM (ocean only)
  - No diagnostic 2-m/10-m fields returned

References
----------
  gSAM SRC/oceflx.f90
  gSAM SRC/surface.f90
  Large & Pond (1981) JPO 11, 324–336
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsam.core.state import ModelState
from jsam.core.physics.sgs import SurfaceFluxes
from jsam.core.physics.microphysics import qsatw as _qsatw_micro


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkParams:
    """Bulk-flux tuning parameters (matches gSAM consts.f90 / params.f90)."""
    umin:        float = 1.0       # minimum wind speed (m/s)
    karman:      float = 0.4       # von Kármán constant
    epsv:        float = 0.61      # (Rv/Rd − 1); virtual temp correction
    salt_factor: float = 0.98      # ocean salinity reduction of qs
    p00:         float = 1.0e5     # reference pressure (Pa)
    Rd:          float = 287.04    # dry air gas constant (J/kg/K)
    Rv:          float = 461.5     # water vapour gas constant (J/kg/K)
    cp:          float = 1004.64   # specific heat of dry air (J/kg/K)
    g:           float = 9.79764   # gravitational acceleration (m/s²)


# ---------------------------------------------------------------------------
# Stability functions (matches oceflx.f90 psimhu / psixhu)
# ---------------------------------------------------------------------------

def _psimhu(xd: jax.Array) -> jax.Array:
    """Unstable momentum stability function (Paulson 1970)."""
    return (
        jnp.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
        - 2.0 * jnp.arctan(xd) + 1.5707963   # π/2 − 3·ln2 ≈ 1.571
    )


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

def _one_iteration(
    vmag:   jax.Array,   # (ny, nx) wind speed
    delt:   jax.Array,   # (ny, nx) θ_atm − θ_sfc
    delq:   jax.Array,   # (ny, nx) q_atm − q_sfc
    thbot:  jax.Array,   # (ny, nx) atmospheric potential temperature
    qbot:   jax.Array,   # (ny, nx) atmospheric specific humidity
    zbot:   jax.Array,    # scalar: height of lowest cell centre (m)
    rdn:    jax.Array,   # (ny, nx) current sqrt(Cd) at neutral
    rhn:    jax.Array,   # (ny, nx) current sqrt(Ch) at neutral
    ren:    jax.Array,   # (ny, nx) current sqrt(Ce) at neutral
    params: BulkParams,
) -> tuple:
    """
    One M-O iteration.  Returns (rdn_new, rhn_new, ren_new, ustar, tstar, qstar).
    """
    karman = params.karman
    epsv   = params.epsv
    zref   = 10.0   # reference height (m)

    alz = jnp.log(zbot / zref)   # ln(z_bot / z_ref)  (scalar)

    # --- stability parameter z/L ---
    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq

    hol = (karman * params.g * zbot
           * (tstar / thbot + qstar / (1.0 / epsv + qbot))
           / (ustar ** 2 + 1.0e-20))
    hol    = jnp.clip(hol, -10.0, 10.0)
    stable = 0.5 + jnp.sign(hol) * 0.5

    xsq    = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
    xqq    = jnp.sqrt(xsq)
    psimh  = -5.0 * hol * stable + (1.0 - stable) * _psimhu(xqq)
    psixh  = -5.0 * hol * stable + (1.0 - stable) * _psixhu(xqq)

    # --- shift rd to measurement height ---
    rd   = rdn / (1.0 + rdn / karman * (alz - psimh))
    u10n = vmag * rd / rdn   # neutral 10-m wind speed

    # --- updated neutral coefficients ---
    delt_stable = 0.5 + jnp.sign(delt) * 0.5
    rdn_new = jnp.sqrt(_cdn(u10n))
    rhn_new = (1.0 - delt_stable) * 0.0327 + delt_stable * 0.018
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
def bulk_surface_fluxes(
    state:  ModelState,
    metric: dict,
    sst:    jax.Array,      # (ny, nx) sea-surface temperature [K]
    params: BulkParams = BulkParams(),
) -> SurfaceFluxes:
    """
    Compute bulk aerodynamic surface fluxes over ocean.

    Parameters
    ----------
    state   : ModelState — uses lowest layer TABS, QV, U, V
    metric  : grid metric dict — uses rho[0], pres[0], dz[0], z[0]
    sst     : (ny, nx) sea-surface temperature [K]
    params  : BulkParams (optional, defaults to gSAM/CESM1 values)

    Returns
    -------
    SurfaceFluxes(shf, lhf, tau_x, tau_y) all (ny, nx)
      shf   : sensible heat flux (K·m/s,      positive upward)
      lhf   : latent heat flux   (kg/kg·m/s,  positive upward)
      tau_x : zonal stress       (m²/s²,      negative when u > 0)
      tau_y : meridional stress  (m²/s²)
    """
    # --- grid constants at the lowest level ---
    rho0  = metric["rho"][0]    # kg/m³
    pres0 = metric["pres"][0]   # Pa (cell-centre pressure)
    dz0   = metric["dz"][0]     # m
    z0    = metric["z"][0]      # height of cell centre (= dz0/2 for flat ground)

    # Surface pressure: hydrostatic half-step below cell centre
    pres_sfc = pres0 + rho0 * params.g * dz0 * 0.5   # Pa

    # Exner functions  π = (p/p00)^(Rd/cp)
    exner0   = (pres0    / params.p00) ** (params.Rd / params.cp)
    exner_s  = (pres_sfc / params.p00) ** (params.Rd / params.cp)

    # --- atmospheric state at k=0, interpolated to cell centres ---
    TABS0 = state.TABS[0, :, :]   # (ny, nx)
    QV0   = state.QV[0, :, :]     # (ny, nx)

    # Cell-centre winds from staggered faces
    u_cc = 0.5 * (state.U[0, :, :-1] + state.U[0, :, 1:])   # (ny, nx)
    v_cc = 0.5 * (state.V[0, :-1, :] + state.V[0, 1:, :])   # (ny, nx)

    vmag  = jnp.maximum(params.umin, jnp.sqrt(u_cc ** 2 + v_cc ** 2))

    # Replace NaN SST (land points) with a neutral dummy so the M-O iteration
    # stays finite.  The resulting fluxes will be zeroed out below.
    sst_safe = jnp.where(jnp.isnan(sst), TABS0, sst)

    # Potential temperatures
    thbot = TABS0 * exner0            # atmospheric θ at k=0
    ts    = sst_safe * exner_s        # surface θ (referenced to pres_sfc)

    delt  = thbot - ts                # (ny, nx)

    # Surface saturation humidity (with salt factor).
    # qsatw matches gSAM sat.f90 (Buck 1981 / IFS) exactly; pressure in mb.
    qs_sfc = params.salt_factor * _qsatw_micro(sst_safe, pres_sfc / 100.0)
    delq   = QV0 - qs_sfc             # (ny, nx)

    # --- neutral coefficients (first guess) ---
    rdn, rhn, ren = _neutral_coeffs(vmag, delt)

    # --- two M-O iterations (matches gSAM oceflx) ---
    for _ in range(2):
        rdn, rhn, ren, ustar, tstar, qstar = _one_iteration(
            vmag, delt, delq, thbot, QV0, z0, rdn, rhn, ren, params,
        )

    # --- fluxes (positive upward) ---
    shf   = -ustar * tstar                          # K·m/s
    lhf   = -ustar * qstar                          # kg/kg·m/s

    # Momentum stress: τ = ρ·u*² in N/m²; divide by ρ to get m²/s²
    # tau_x/tau_y at cell centres; diffuse_momentum will interpolate to faces
    tau_x = -(ustar ** 2) * u_cc / vmag            # m²/s²
    tau_y = -(ustar ** 2) * v_cc / vmag            # m²/s²

    # Land mask: ERA5 SST is NaN over land. Zero out all fluxes there so
    # NaN does not propagate into the SGS diffusion bottom BC.
    is_ocean = ~jnp.isnan(sst)
    shf   = jnp.where(is_ocean, shf,   0.0)
    lhf   = jnp.where(is_ocean, lhf,   0.0)
    tau_x = jnp.where(is_ocean, tau_x, 0.0)
    tau_y = jnp.where(is_ocean, tau_y, 0.0)

    return SurfaceFluxes(shf=shf, lhf=lhf, tau_x=tau_x, tau_y=tau_y)
