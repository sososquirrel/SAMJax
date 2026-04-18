"""Smagorinsky SGS: tk=Cs²·smix²·√def2, tkh=Pr·tk. Explicit diffusion, anelastic vertical.
Zero-flux BCs at walls. No prognostic TKE, no DNS, no terrain."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsam.core.state import ModelState


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SGSParams:
    """Smagorinsky params: Cs (const), Pr (Prandtl), delta_max (m)."""
    Cs:        float = 0.19
    Ck:        float = 0.1
    Pr:        float = 1.0
    delta_max: float = 1000.0

@jax.tree_util.register_pytree_node_class
@dataclass
class SurfaceFluxes:
    """Surface fluxes (ny,nx): shf, lhf, tau_x, tau_y (positive upward)."""
    shf:   jax.Array
    lhf:   jax.Array
    tau_x: jax.Array
    tau_y: jax.Array

    @classmethod
    def zeros(cls, ny: int, nx: int) -> "SurfaceFluxes":
        z = jnp.zeros((ny, nx))
        return cls(shf=z, lhf=z, tau_x=z, tau_y=z)

    def tree_flatten(self):
        return (self.shf, self.lhf, self.tau_x, self.tau_y), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dzw(dz: jax.Array) -> jax.Array:
    """Distance between cell centres at w-faces; shape (nz+1,)."""
    return jnp.concatenate([dz[:1] * 0.5, 0.5 * (dz[:-1] + dz[1:]), dz[-1:] * 0.5])


# ---------------------------------------------------------------------------
# Strain-rate invariant  def2 = 2 S_ij S_ij
# ---------------------------------------------------------------------------

@jax.jit
def shear_prod(U: jax.Array, V: jax.Array, W: jax.Array, metric: dict) -> jax.Array:
    """Compute 2·S_ij·S_ij at cell centres; Cartesian approx for y."""
    dx  = metric["dx_lon"]
    dy  = metric["dy_lat"]
    dz  = metric["dz"]
    rdx = 1.0 / dx
    rdy  = (1.0 / dy)[None, :, None]
    dzw_ = _dzw(dz)

    S11 = (U[:, :, 1:] - U[:, :, :-1]) * rdx
    S22 = (V[:, 1:, :] - V[:, :-1, :]) * rdy
    S33 = (W[1:, :, :] - W[:-1, :, :]) / dz[:, None, None]

    diag = 2.0 * (S11**2 + S22**2 + S33**2)


    # ---- U-V cross terms: (du/dy + dv/dx)² averaged over 4 corners ----
    Up = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')
    Vx = jnp.concatenate([V[:, :, -1:], V, V[:, :, :1]], axis=2)

    dudy_NE = (Up[:, 2:,  1:] - Up[:, 1:-1,  1:]) * rdy
    dvdx_NE = (Vx[:, 1:, 2:] - Vx[:, 1:, 1:-1]) * rdx
    dudy_NW = (Up[:, 2:, :-1] - Up[:, 1:-1, :-1]) * rdy
    dvdx_NW = (Vx[:, 1:, 1:-1] - Vx[:, 1:, :-2]) * rdx
    dudy_SE = (Up[:, 1:-1,  1:] - Up[:, :-2,  1:]) * rdy
    dvdx_SE = (Vx[:, :-1, 2:] - Vx[:, :-1, 1:-1]) * rdx
    dudy_SW = (Up[:, 1:-1, :-1] - Up[:, :-2, :-1]) * rdy
    dvdx_SW = (Vx[:, :-1, 1:-1] - Vx[:, :-1, :-2]) * rdx

    cross_uv = 0.25 * (
        (dudy_NE + dvdx_NE)**2 + (dudy_NW + dvdx_NW)**2 +
        (dudy_SE + dvdx_SE)**2 + (dudy_SW + dvdx_SW)**2
    )

    Uz = jnp.pad(U, ((1, 1), (0, 0), (0, 0)), mode='edge')
    Wx = jnp.concatenate([W[:, :, -1:], W, W[:, :, :1]], axis=2)

    dzw_above = dzw_[1:][:, None, None]
    dzw_below = dzw_[:-1][:, None, None]

    dudz_ab_ip1 = (Uz[2:, :, 1:] - Uz[1:-1, :, 1:]) / dzw_above
    dudz_ab_i   = (Uz[2:, :, :-1] - Uz[1:-1, :, :-1]) / dzw_above
    dudz_bel_ip1 = (Uz[1:-1, :, 1:] - Uz[:-2, :, 1:]) / dzw_below
    dudz_bel_i   = (Uz[1:-1, :, :-1] - Uz[:-2, :, :-1]) / dzw_below
    dwdx_above = (Wx[1:, :, 2:] - Wx[1:, :, 1:-1]) * rdx
    dwdx_below = (Wx[:-1, :, 2:] - Wx[:-1, :, 1:-1]) * rdx

    cross_uw = 0.25 * (
        (dudz_ab_ip1  + dwdx_above)**2 + (dudz_ab_i   + dwdx_above)**2 +
        (dudz_bel_ip1 + dwdx_below)**2 + (dudz_bel_i  + dwdx_below)**2
    )

    Vz = jnp.pad(V, ((1, 1), (0, 0), (0, 0)), mode='edge')
    Wy = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')

    dvdz_ab_jp1 = (Vz[2:, 1:, :] - Vz[1:-1, 1:, :]) / dzw_above
    dvdz_ab_j   = (Vz[2:, :-1, :] - Vz[1:-1, :-1, :]) / dzw_above
    dvdz_bel_jp1 = (Vz[1:-1, 1:, :] - Vz[:-2, 1:, :]) / dzw_below
    dvdz_bel_j   = (Vz[1:-1, :-1, :] - Vz[:-2, :-1, :]) / dzw_below
    dwdy_above = (Wy[1:, 2:, :] - Wy[1:, 1:-1, :]) * rdy
    dwdy_below = (Wy[:-1, 2:, :] - Wy[:-1, 1:-1, :]) * rdy

    cross_vw = 0.25 * (
        (dvdz_ab_jp1  + dwdy_above)**2 + (dvdz_ab_j   + dwdy_above)**2 +
        (dvdz_bel_jp1 + dwdy_below)**2 + (dvdz_bel_j  + dwdy_below)**2
    )

    # Fix 2.8: surface boundary correction for uw/vw cross-terms.
    # gSAM shear_prod3D.f90:90-95: at k=0 (surface level), only the upward
    # (above) interface exists; the downward (below) terms are absent.
    # Interior: 0.25*(above_ip1² + above_i² + below_ip1² + below_i²)
    # Surface:  0.5 *(above_ip1² + above_i²)   [coefficient 0.5, not 0.25]
    cross_uw_sfc = 0.5 * (
        (dudz_ab_ip1[0]  + dwdx_above[0])**2 + (dudz_ab_i[0]   + dwdx_above[0])**2
    )
    cross_vw_sfc = 0.5 * (
        (dvdz_ab_jp1[0]  + dwdy_above[0])**2 + (dvdz_ab_j[0]   + dwdy_above[0])**2
    )
    cross_uw = cross_uw.at[0].set(cross_uw_sfc)
    cross_vw = cross_vw.at[0].set(cross_vw_sfc)

    return diag + cross_uv + cross_uw + cross_vw


# ---------------------------------------------------------------------------
# Smagorinsky eddy viscosity / diffusivity
# ---------------------------------------------------------------------------

def _compute_buoy_sgs(
    TABS:   jax.Array,       # (nz, ny, nx) K
    tabs0:  jax.Array,       # (nz,) K reference profile
    metric: dict,
    QV:     jax.Array | None = None,   # moisture fields (all nz,ny,nx)
    QC:     jax.Array | None = None,
    QI:     jax.Array | None = None,
    QR:     jax.Array | None = None,
    QS:     jax.Array | None = None,
    QG:     jax.Array | None = None,
    fluxbt: jax.Array | None = None,   # D14: (ny,nx) surface heat flux (K·m/s)
    fluxbq: jax.Array | None = None,   # D14: (ny,nx) surface moisture flux (kg/kg·m/s)
) -> jax.Array:
    """
    Full moist SGS buoyancy at cell centres, matching gSAM tke_full.f90:96-242.

    Includes virtual temperature, condensate loading, precipitation drag,
    and moist-adiabatic saturation correction when the interface mixture
    is supersaturated.

    Falls back to the dry-only formulation when moisture fields are None.
    """
    from jsam.core.physics.microphysics import (
        qsatw, qsati, _dtqsatw, _dtqsati,
        FAC_COND, FAC_FUS, FAC_SUB, G_GRAV, CP,
    )

    z      = metric["z"]
    gamaz  = metric["gamaz"]
    adzw   = metric["adzw"]
    pres   = metric["pres"]
    dz_ref = metric["dz_ref"]
    g      = G_GRAV
    EPSV   = 0.61

    bet = g / tabs0   # (nz,)

    # --- Dry fallback ---
    if QV is None:
        g_cp = g / CP
        t_liq = TABS + g_cp * z[:, None, None]
        dz_c = (z[1:] - z[:-1])[:, None, None]
        tabs0_face = 0.5 * (tabs0[1:] + tabs0[:-1])[:, None, None]
        buoy_face = (g / tabs0_face) * (t_liq[1:] - t_liq[:-1]) / dz_c
        return jnp.concatenate(
            [buoy_face[:1],
             0.5 * (buoy_face[:-1] + buoy_face[1:]),
             buoy_face[-1:]], axis=0)

    # --- Full moist buoyancy (gSAM tke_full.f90:122-242) ---
    t_se = (TABS + gamaz[:, None, None]
            - FAC_COND * (QC + QR)
            - FAC_SUB  * (QI + QS + QG))

    qpl  = QR
    qpi  = QS + QG
    qtot = QV + QC + QI

    pres_mb  = pres / 100.0
    presi_mb = jnp.concatenate([pres_mb[:1],
                                0.5 * (pres_mb[:-1] + pres_mb[1:]),
                                pres_mb[-1:]])

    kb = slice(None, -1)
    kc = slice(1, None)

    bet_face = 0.5 * (bet[1:] + bet[:-1])
    betdz = (bet_face / (dz_ref * adzw[1:-1]))[:, None, None]

    # Unsaturated (tke_full.f90:143-159)
    tabs_if = 0.5 * (TABS[kc] + FAC_COND * QC[kc] + FAC_SUB * QI[kc]
                    + TABS[kb] + FAC_COND * QC[kb] + FAC_SUB * QI[kb])
    qtot_if = 0.5 * (qtot[kc] + qtot[kb])
    qp_if   = 0.5 * (qpl[kc] + qpi[kc] + qpl[kb] + qpi[kb])

    bbb = 1.0 + EPSV * qtot_if - qp_if
    buoy_unsat = betdz * (
        bbb * (t_se[kc] - t_se[kb])
        + EPSV * tabs_if * (qtot[kc] - qtot[kb])
        + (bbb * FAC_COND - tabs_if) * (qpl[kc] - qpl[kb])
        + (bbb * FAC_SUB  - tabs_if) * (qpi[kc] - qpi[kb]))

    # Saturated check (tke_full.f90:170-237)
    qctot = QC[kc] + QI[kc] + QC[kb] + QI[kb]
    has_cloud = qctot > 0.0
    omn = (QC[kc] + QC[kb]) / (qctot + 1e-20)
    presi_if = presi_mb[1:-1, None, None]
    qsat_chk = omn * qsatw(tabs_if, presi_if) + (1.0 - omn) * qsati(tabs_if, presi_if)
    is_sat = has_cloud & (qtot_if > qsat_chk)

    lstarn = FAC_COND + (1.0 - omn) * FAC_FUS
    tabs_if_sat = 0.5 * (TABS[kc] + TABS[kb])
    dqsat = (omn * _dtqsatw(tabs_if_sat, presi_if)
             + (1.0 - omn) * _dtqsati(tabs_if_sat, presi_if))
    qsatt = (omn * qsatw(tabs_if_sat, presi_if)
             + (1.0 - omn) * qsati(tabs_if_sat, presi_if))
    bbb_s = (1.0 + EPSV * qsatt + qsatt - qtot_if - qp_if
             + 1.61 * tabs_if_sat * dqsat) / (1.0 + lstarn * dqsat)
    # Fix 2.7: gSAM tke_full.f90:225-226 uses tabs(i,j,k) (cell-centre of lower
    # cell kb) for precipitation drag terms — not the face-averaged tabs_if_sat.
    tabs_k = TABS[kb]
    buoy_sat = betdz * (
        bbb_s * (t_se[kc] - t_se[kb])
        + (bbb_s * lstarn - (1.0 + lstarn * dqsat) * tabs_k)
            * (qtot[kc] - qtot[kb])
        + (bbb_s * FAC_COND - (1.0 + FAC_COND * dqsat) * tabs_k)
            * (qpl[kc] - qpl[kb])
        + (bbb_s * FAC_SUB  - (1.0 + FAC_SUB * dqsat) * tabs_k)
            * (qpi[kc] - qpi[kb]))

    buoy_face = jnp.where(is_sat, buoy_sat, buoy_unsat)

    # Fix 2.1 + 2.6: gSAM tke_full.f90:96-118 — at the surface, buoy_sgs is
    # derived from surface fluxes using the nonlinear Smagorinsky inversion.
    # gSAM uses SST (sstxy) not the reference temperature tabs0[0].
    #   a_prod_bu = bet(k) * fluxbt + bet(k) * epsv * (t00+sst) * fluxbq
    # (bbb=1+epsv*qv_sfc multiplied in gSAM but dropped here; consistent with
    #  gSAM comment that only works for positive flux.)
    #   if a_prod_bu > 0: buoy_below = -(a_prod_bu^2 * Ce / (Ck^3 * Pr * grd^4))^(1/3)
    #   else:             buoy_below = 0
    if fluxbt is not None:
        _Ck_sfc = 0.1
        _Cs_sfc = 0.19
        _Pr_sfc = 1.0
        _Ce_sfc = _Ck_sfc**3 / _Cs_sfc**4
        _bet0 = g / tabs0[0]
        _fbq = fluxbq if fluxbq is not None else jnp.zeros_like(fluxbt)
        # Fix 2.6: use local SST if available; falls back to tabs0[0]
        _sst = metric.get("sst", None)
        _sst_val = _sst if _sst is not None else tabs0[0]
        # gSAM tke_full.f90:101: bbb = 1 + epsv*qv(k=1) multiplies sensible heat term
        _bbb = 1.0 + EPSV * QV[0]   # (ny, nx) — density correction at lowest level
        _a_prod_bu = _bet0 * (_bbb * fluxbt + EPSV * _sst_val * _fbq)   # (ny,nx)
        # grd at surface level k=0: coef(j) = min(delta_max, dx*mu) * min(delta_max, dy*ady)
        # dz_ref*adz(k=0) ≈ dz[0] (adz[0] = 1 in flat terrain with uniform vertical grid)
        _dz_sfc = dz_ref   # dz_ref = metric["dz_ref"] already in scope (JAX array)
        _dx_sfc = metric["dx_lon"]
        _cos_lat_sfc = metric.get("cos_lat", None)
        _dy_sfc = metric.get("dy_lat", None)
        if _cos_lat_sfc is not None:
            _dx_eff_sfc = jnp.minimum(1000.0, _dx_sfc * _cos_lat_sfc)   # (ny,)
        else:
            _dx_eff_sfc = jnp.minimum(1000.0, _dx_sfc) * jnp.ones((1,))
        if _dy_sfc is not None:
            _dy_eff_sfc = jnp.minimum(1000.0, _dy_sfc)   # (ny,)
        else:
            _dy_eff_sfc = jnp.minimum(1000.0, metric.get("dy_lat_ref", _dx_sfc)) * jnp.ones((1,))
        _coef_sfc = _dx_eff_sfc * _dy_eff_sfc   # (ny,)
        _grd_sfc = (_dz_sfc * _coef_sfc) ** 0.33333   # (ny,)
        # nonlinear inversion (tke_full.f90:114):
        #   buoy_sgs_below = -(a_prod_bu^2 * Ce / (Ck^3 * Pr * grd^4))^(1/3)
        _grd4 = (_grd_sfc[:, None]) ** 4   # (ny, 1) — broadcast over nx
        _buoy_unstable = -(
            (_a_prod_bu**2 * _Ce_sfc / (_Ck_sfc**3 * _Pr_sfc * _grd4))**0.3333
        )
        _buoy_sfc = jnp.where(_a_prod_bu > 0.0, _buoy_unstable, 0.0)
        buoy_face = buoy_face.at[0].set(_buoy_sfc)

    return jnp.concatenate(
        [buoy_face[:1],
         0.5 * (buoy_face[:-1] + buoy_face[1:]),
         buoy_face[-1:]], axis=0)


@functools.partial(jax.jit, static_argnames=("params",))
def smag_viscosity(
    def2:   jax.Array,   # (nz, ny, nx)  s⁻²
    metric: dict,
    params: SGSParams,
    TABS:   jax.Array | None = None,   # (nz, ny, nx) K
    tabs0:  jax.Array | None = None,   # (nz,) K reference profile
    QV:     jax.Array | None = None,   # moisture fields for moist buoyancy
    QC:     jax.Array | None = None,
    QI:     jax.Array | None = None,
    QR:     jax.Array | None = None,
    QS:     jax.Array | None = None,
    QG:     jax.Array | None = None,
    tk_prev: jax.Array | None = None,  # D13: previous-step tk for smix limiter
    fluxbt: jax.Array | None = None,   # D14: (ny,nx) surface heat flux (K·m/s)
    fluxbq: jax.Array | None = None,   # D14: (ny,nx) surface moisture flux (kg/kg·m/s)
) -> tuple[jax.Array, jax.Array]:
    """
    Returns (tk, tkh), both (nz, ny, nx) in m²/s.

    Pure Smagorinsky (TABS/tabs0 None):
        tk = (Cs·smix)² · sqrt(max(0, def2))

    gSAM dosmagor=True with buoyancy suppression (TABS and tabs0 given):
        buoy = N²  (>0 stable)
        if buoy > 0:  smix = min(grd, max(0.1·grd, √(0.76·tk_prev/Ck/√buoy)))
        Cee  = (Ce/0.7) · (0.19 + 0.51·smix/grd),   Ce = Ck³/Cs⁴
        tk   = √(Ck³/Cee · max(0, def2 − Pr·buoy)) · smix²

    D13: tk_prev is the previous-step tk used in the mixing length limiter
    (gSAM tke_full.f90:288 uses stored tk array). Falls back to fixed-point
    iteration from grd when tk_prev is None.

    When moisture fields (QV..QG) are provided, uses the full moist buoyancy
    from gSAM tke_full.f90 including virtual temp, condensate loading,
    and saturation correction.  Falls back to dry buoyancy when None.
    """
    dx        = metric["dx_lon"]
    dy        = metric["dy_lat"]   # (ny,) per-row
    dz        = metric["dz"]   # (nz,)
    delta_max = params.delta_max
    Cs        = params.Cs
    Ck        = params.Ck
    Pr        = params.Pr

    # F8/F10 fix: gSAM tke_full.f90:42 computes
    #   coef(j) = min(delta_max, dx*mu(j)) * min(delta_max, dy*ady(j))
    # where dx is the EQUATORIAL spacing (= dx_lon), mu(j)=cos(lat), and
    # dy*ady(j) = actual per-row meridional spacing (= dy_lat per-row).
    # jsam metric["dy_lat"] already equals dy_ref*ady(j) so dy_eff is correct.
    # But dx must be multiplied by cos(lat) to get the actual zonal spacing.
    _cos_lat_sgs = metric.get("cos_lat", None)
    if _cos_lat_sgs is not None:
        # dx * cos_lat(j) = actual zonal grid spacing (m) per latitude row
        dx_eff = jnp.minimum(delta_max, dx * _cos_lat_sgs)[None, :, None]   # (1, ny, 1)
    else:
        dx_eff = jnp.minimum(delta_max, dx)
    dy_eff = jnp.minimum(delta_max, dy)[None, :, None]

    grd = (dz[:, None, None] * dx_eff * dy_eff) ** 0.33333

    if TABS is None or tabs0 is None:
        tk  = Cs**2 * grd**2 * jnp.sqrt(jnp.maximum(0.0, def2))
        tkh = params.Pr * tk
        return tk, tkh

    # --- gSAM dosmagor=True branch with buoyancy suppression ---
    buoy_sgs  = _compute_buoy_sgs(TABS, tabs0, metric,
                                   QV=QV, QC=QC, QI=QI, QR=QR, QS=QS, QG=QG,
                                   fluxbt=fluxbt, fluxbq=fluxbq)

    # Constants (gSAM tke_full.f90:28-31)
    Ce  = Ck**3 / Cs**4                 # ≈ 0.7673
    Ce1 = Ce / 0.7 * 0.19
    Ce2 = Ce / 0.7 * 0.51

    def2_adj = jnp.maximum(0.0, def2 - Pr * buoy_sgs)

    # D13 fix: gSAM uses previous-step tk in the mixing length limiter
    # (tke_full.f90:288). When tk_prev is available, use it directly;
    # otherwise fall back to the single fixed-point iteration.
    if tk_prev is not None:
        tk_for_smix = tk_prev
    else:
        # Fixed-point iter 0: smix = grd, Cee = Ce1 + Ce2 = Ce
        tk_for_smix = jnp.sqrt(Ck**3 / Ce * def2_adj) * grd**2

    # gSAM smix limiter for stable layers (buoy > 0)
    stable = buoy_sgs > 0.0
    smix_stable = jnp.minimum(
        grd,
        jnp.maximum(
            0.1 * grd,
            jnp.sqrt(0.76 * tk_for_smix / Ck / jnp.sqrt(buoy_sgs + 1e-10)),
        ),
    )
    smix = jnp.where(stable, smix_stable, grd)

    ratio = smix / grd
    Cee   = Ce1 + Ce2 * ratio
    tk    = jnp.sqrt(Ck**3 / Cee * def2_adj) * smix**2
    tkh   = Pr * tk
    return tk, tkh


# ---------------------------------------------------------------------------
# SGS diffusion of a scalar field
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_scalar(
    field:  jax.Array,   # (nz, ny, nx)
    tkh:    jax.Array,   # (nz, ny, nx)  eddy diffusivity
    metric: dict,
    fluxb:  jax.Array | None = None,   # (ny, nx) surface flux (kg/m² s or K·m/s)
    fluxt:  jax.Array | None = None,   # (ny, nx) top flux
    tk_max: jax.Array | None = None,   # Fix 2.5: (nz,ny,1) per-face stability cap
) -> jax.Array:
    """
    Explicit SGS diffusion tendency d(field)/dt (same units as field / s).
    Horizontal: second-order centred with imu² metric in x.
    Vertical:   anelastic flux form with rho/rhow weighting.
    BCs: zero-flux at y-walls; fluxb/fluxt at z-walls (default 0).

    Fix 2.5: clamp tkh locally at each face before flux computation
    (matches gSAM per-face tkmax clamping in diffuse_scalar3D.f90:51-52, 65-66, 106).
    """
    dx  = metric["dx_lon"]
    dy  = metric["dy_lat"]   # (ny,) per-row
    dz  = metric["dz"]       # (nz,)
    rho  = metric["rho"]     # (nz,)
    rhow = metric["rhow"]    # (nz+1,)
    dzw_ = _dzw(dz)           # (nz+1,)
    nz, ny, nx = field.shape

    # ---- Horizontal x-direction — with imu(j) = 1/cos(lat) metric (C8 fix) ----
    field_xp = jnp.concatenate([field[:, :, -1:], field, field[:, :, :1]], axis=2)  # (nz,ny,nx+2)
    tkh_xp   = jnp.concatenate([tkh[:, :, -1:],  tkh,  tkh[:, :, :1]],  axis=2)
    rdx2     = (1.0 / dx) ** 2                                             # scalar
    # C8 fix: gSAM diffuse_scalar3D.f90:48 — rdx5 = rdx2 * imu(j)^2
    cos_lat  = metric.get("cos_lat", None)
    if cos_lat is not None:
        imu2 = (1.0 / cos_lat[None, :, None]) ** 2   # (1, ny, 1)
    else:
        imu2 = 1.0
    rdx2_j   = rdx2 * imu2                                                # (1, ny, 1) or scalar
    # Fix 2.5: interpolate tkh to x-face, then cap locally (gSAM lines 51-52)
    tkh_fx   = 0.5 * (tkh_xp[:, :, :-1] + tkh_xp[:, :, 1:])              # (nz,ny,nx+1) x-faces
    if tk_max is not None:
        # Cap at x-faces (shape: nz, ny, nx+1) - tk_max is (nz, ny, 1)
        tkh_fx = jnp.minimum(tkh_fx, tk_max)
    flx_x    = -rdx2_j * tkh_fx * (field_xp[:, :, 1:] - field_xp[:, :, :-1])  # (nz,ny,nx+1)
    dfdt     = -(flx_x[:, :, 1:] - flx_x[:, :, :-1])                     # (nz,ny,nx)

    # ---- Horizontal y-direction (non-uniform, with spherical metrics) ----
    # Fix 2.2: gSAM diffuse_scalar3D.f90:63-64 applies spherical corrections:
    #   flux: rdy5 = rdy2 / adyv(jc) * muv(jc)  → flux *= muv[face] / adyv[face]
    #   div:  rdy5 = 1/(ady(j)*mu(j))            → div  /= mu[cell]
    # Since dy_v_full = adyv * dy_ref and dy = ady * dy_ref, the combined formula is:
    #   flux = -muv[face] / (dy_v_full * dy_ref) * tkh * df
    #   dfdt -= (flux_n - flux_s) / (dy * mu)
    # Pad field and tkh in y with edge (Neumann → zero flux at walls).
    field_yp = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode='edge')      # (nz,ny+2,nx)
    tkh_yp   = jnp.pad(tkh,   ((0, 0), (1, 1), (0, 0)), mode='edge')
    dy_v_int = 0.5 * (dy[:-1] + dy[1:])                                   # (ny-1,)
    # v-face spacings at every face (ny+1):  at boundaries duplicate edge.
    dy_v_full = jnp.concatenate([dy_v_int[:1], dy_v_int, dy_v_int[-1:]])  # (ny+1,)
    # Fix 2.2: muv factor at each y-face (cos_v shape ny+1)
    cos_lat_s = metric.get("cos_lat", None)
    cos_v_s   = metric.get("cos_v", None)
    if cos_v_s is not None:
        muv_face = cos_v_s[None, :, None]                                  # (1, ny+1, 1)
    else:
        muv_face = 1.0
    if cos_lat_s is not None:
        mu_cell = cos_lat_s[None, :, None]                                 # (1, ny, 1)
    else:
        mu_cell = 1.0
    # Fix 2.5: interpolate tkh to y-face, then cap locally (gSAM lines 65-66)
    tkh_fy   = 0.5 * (tkh_yp[:, :-1, :] + tkh_yp[:, 1:, :])              # (nz,ny+1,nx) y-faces
    if tk_max is not None:
        # Interpolate tk_max from mass levels (ny) to y-faces (ny+1)
        # Bottom face: use tk_max at j=0; interior: interpolate; top face: use tk_max at j=ny-1
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])  # (nz, ny-1, 1)
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)  # (nz, ny+1, 1)
        tkh_fy = jnp.minimum(tkh_fy, tk_max_yface)
    # rdy_ref: gSAM rdy2 = 1/dy_ref^2; combined with adyv/muv factor the flux is:
    #   flux = -muv[face] / (dy_v_full[face] * dy_ref) * tkh * df
    dy_ref_s  = metric["dy_lat_ref"]                                       # scalar
    flx_y    = (-muv_face / (dy_v_full[None, :, None] * dy_ref_s)
                * tkh_fy * (field_yp[:, 1:, :] - field_yp[:, :-1, :]))    # (nz,ny+1,nx)
    dfdt     = dfdt - (flx_y[:, 1:, :] - flx_y[:, :-1, :]) / (dy[None, :, None] * mu_cell)

    # ---- Vertical — anelastic: d(rhow*tkh*dfield/dz)/(rho*dz) ----
    # Interior fluxes at w-faces 1..nz-1
    # Fix 2.5: interpolate tkh to z-face, then cap locally (gSAM line 106)
    tkh_fz    = 0.5 * (tkh[:-1, :, :] + tkh[1:, :, :])                    # (nz-1,ny,nx)
    if tk_max is not None:
        tkh_fz = jnp.minimum(tkh_fz, tk_max[:-1, :, :])                   # cap interior z-faces only
    dz_face   = dzw_[1:-1][:, None, None]                                  # (nz-1,1,1)
    rhow_int  = rhow[1:-1][:, None, None]                                  # (nz-1,1,1)
    flx_z_int = (-rhow_int * tkh_fz
                 * (field[1:, :, :] - field[:-1, :, :]) / dz_face)        # (nz-1,ny,nx)

    # Boundary fluxes
    if fluxb is None:
        fluxb = jnp.zeros((ny, nx))
    if fluxt is None:
        fluxt = jnp.zeros((ny, nx))

    flx_z_bot = rhow[0] * fluxb[None, :, :]    # (1,ny,nx)  downward positive
    flx_z_top = rhow[-1] * fluxt[None, :, :]   # (1,ny,nx)

    # Full flux array at all nz+1 faces (k=0 bottom, k=nz top)
    flx_z = jnp.concatenate([flx_z_bot, flx_z_int, flx_z_top], axis=0)   # (nz+1,ny,nx)

    rho_dz = (rho * dz)[:, None, None]   # (nz,1,1)
    dfdt   = dfdt - (flx_z[1:, :, :] - flx_z[:-1, :, :]) / rho_dz

    return dfdt


# ---------------------------------------------------------------------------
# Horizontal-only SGS diffusion of a scalar field
# (gSAM diffuse_scalar3D.f90 with doimplicitdiff=.true. — skips vertical)
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_scalar_horiz(
    field:  jax.Array,   # (nz, ny, nx)
    tkh:    jax.Array,   # (nz, ny, nx)  eddy diffusivity
    metric: dict,
    tk_max: jax.Array | None = None,   # (nz,ny,1) per-face stability cap
) -> jax.Array:
    """
    Horizontal-only explicit SGS diffusion tendency d(field)/dt.
    Matches gSAM diffuse_scalar3D.f90 with doimplicitdiff=.true.: applies
    horizontal (x and y) diffusion only and returns early; the vertical is
    handled separately by diffuse_scalar_z_implicit (the implicit solver).

    No fluxb/fluxt arguments — boundary fluxes are applied only by the
    implicit vertical solver.
    """
    dx  = metric["dx_lon"]
    dy  = metric["dy_lat"]   # (ny,) per-row

    # ---- Horizontal x-direction — with imu(j) = 1/cos(lat) metric (C8 fix) ----
    field_xp = jnp.concatenate([field[:, :, -1:], field, field[:, :, :1]], axis=2)
    tkh_xp   = jnp.concatenate([tkh[:, :, -1:],  tkh,  tkh[:, :, :1]],  axis=2)
    rdx2     = (1.0 / dx) ** 2
    cos_lat  = metric.get("cos_lat", None)
    if cos_lat is not None:
        imu2 = (1.0 / cos_lat[None, :, None]) ** 2
    else:
        imu2 = 1.0
    rdx2_j   = rdx2 * imu2
    tkh_fx   = 0.5 * (tkh_xp[:, :, :-1] + tkh_xp[:, :, 1:])
    if tk_max is not None:
        tkh_fx = jnp.minimum(tkh_fx, tk_max)
    flx_x    = -rdx2_j * tkh_fx * (field_xp[:, :, 1:] - field_xp[:, :, :-1])
    dfdt     = -(flx_x[:, :, 1:] - flx_x[:, :, :-1])

    # ---- Horizontal y-direction (non-uniform, with spherical metrics) ----
    field_yp = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode='edge')
    tkh_yp   = jnp.pad(tkh,   ((0, 0), (1, 1), (0, 0)), mode='edge')
    dy_v_int = 0.5 * (dy[:-1] + dy[1:])
    dy_v_full = jnp.concatenate([dy_v_int[:1], dy_v_int, dy_v_int[-1:]])
    cos_v_s   = metric.get("cos_v", None)
    if cos_v_s is not None:
        muv_face = cos_v_s[None, :, None]
    else:
        muv_face = 1.0
    if cos_lat is not None:
        mu_cell = cos_lat[None, :, None]
    else:
        mu_cell = 1.0
    tkh_fy   = 0.5 * (tkh_yp[:, :-1, :] + tkh_yp[:, 1:, :])
    if tk_max is not None:
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)
        tkh_fy = jnp.minimum(tkh_fy, tk_max_yface)
    dy_ref_s  = metric["dy_lat_ref"]
    flx_y    = (-muv_face / (dy_v_full[None, :, None] * dy_ref_s)
                * tkh_fy * (field_yp[:, 1:, :] - field_yp[:, :-1, :]))
    dfdt     = dfdt - (flx_y[:, 1:, :] - flx_y[:, :-1, :]) / (dy[None, :, None] * mu_cell)

    return dfdt


# ---------------------------------------------------------------------------
# SGS diffusion of momentum
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_momentum(
    U: jax.Array,   # (nz, ny, nx+1)
    V: jax.Array,   # (nz, ny+1, nx)
    W: jax.Array,   # (nz+1, ny, nx)
    tk:     jax.Array,   # (nz, ny, nx) eddy viscosity at cell centres
    metric: dict,
    tau_x: jax.Array | None = None,  # (ny, nx) surface x-stress (m²/s²)
    tau_y: jax.Array | None = None,  # (ny, nx) surface y-stress (m²/s²)
    tk_max: jax.Array | None = None, # Fix 2.5: (nz,ny,1) per-face stability cap
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns (dU/dt, dV/dt, dW/dt) tendencies from SGS diffusion (s⁻²).
    Follows diffuse_mom3D.f90 structure.  Simplified: uses nearest cell-centre
    tk without the 4-point corner interpolation in gSAM (adequate for CRM).
    Anelastic vertical; Cartesian horizontal.

    tau_x, tau_y: surface (bottom) momentum flux (m²/s²), positive upward.
    They are interpolated from cell-centre positions to U/V staggered faces.
    Default (None) = zero-flux bottom BC.

    Fix 2.5: clamp tk locally at each face before flux computation
    (matches gSAM per-face tkmax clamping in diffuse_mom3D.f90:42, 45, 48, 71, 74, 77, 119, 122, 125).
    """
    dx  = metric["dx_lon"]
    # Gap 8 (2026-04-12): dy_lat is now per-row.  SGS momentum diffusion is
    # a closure parameter (not an advective/pressure operator) and uses a
    # reference scalar dy via `dy_lat_ref` — keeping the conservative split
    # through the full (ny+1) staggered stencil (U/V/W) complicates the
    # code without any dynamical benefit.  The correctness-critical sites
    # (advection, pressure, divergence) use the per-row dy_lat array.
    dy_ref = metric["dy_lat_ref"]   # scalar (Python float or 0-d traced)
    dz  = metric["dz"]    # (nz,)
    rho  = metric["rho"]  # (nz,)
    rhow = metric["rhow"] # (nz+1,)
    dzw_ = _dzw(dz)        # (nz+1,)

    # C8 fix: apply imu(j)^2 = 1/cos(lat)^2 metric to x-direction
    # Match gSAM diffuse_mom3D.f90: use different metrics for each staggered grid
    cos_lat = metric.get("cos_lat", None)
    cos_v_metric = metric.get("cos_v", None)
    rdx2_base = (1.0 / dx)**2
    if cos_lat is not None:
        imu2 = (1.0 / cos_lat[None, :, None]) ** 2   # (1, ny, 1)
    else:
        imu2 = 1.0

    # For V (ny+1 rows): use cos_v from metric (computed in build_metric),
    # matching Fortran's use of imuv(j) for V grid (differ from imu for U/W)
    if cos_v_metric is not None:
        imuv2 = (1.0 / cos_v_metric[None, :, None]) ** 2     # (1, ny+1, 1)
    else:
        imuv2 = imu2  # fallback if cos_v not available
    rdx2u = rdx2_base * imu2                         # for U grid (ny rows)
    rdx2v = rdx2_base * imuv2                        # for V grid (ny+1 rows) — match Fortran imuv(j)
    rdx2w = rdx2_base * imu2                         # for W grid (ny rows)

    # Fix 2.3: spherical y-metrics for momentum diffusion.
    # gSAM diffuse_mom3D.f90:65-67:
    #   rdy2u = rdy2/adyv(jc)*muv(jc)  (at face between cells j and j+1)
    #   rdy2v = rdy2/ady(j)*mu(j)       (at mass cell j, for V-flux between V-rows j and j+1)
    #   rdy2w = rdy2/adyv(jc)*muv(jc)
    # gSAM divergence (lines 91-93):
    #   rdy2u = 1/(ady(j)*mu(j))        (at mass cell j)
    #   rdy2v = 1/(adyv(j)*muv(j))      (at V-face j, between cells j-1 and j)
    #   rdy2w = 1/(ady(j)*mu(j))
    ady_m  = metric.get("ady", None)   # (ny,) mass-cell spacing ratios
    if ady_m is not None and cos_lat is not None:
        # adyv (ny+1,): V-face spacing ratios; interior = 0.5*(ady[j]+ady[j+1])
        adyv_int_m = 0.5 * (ady_m[:-1] + ady_m[1:])                      # (ny-1,)
        adyv_m = jnp.concatenate([ady_m[:1], adyv_int_m, ady_m[-1:]])    # (ny+1,)
        # muv at V-faces (ny+1,) — use cos_v if available, else interpolate
        if cos_v_metric is not None:
            muv_m = cos_v_metric                                           # (ny+1,)
        else:
            cos_v_int = 0.5 * (cos_lat[:-1] + cos_lat[1:])
            muv_m = jnp.concatenate([cos_lat[:1], cos_v_int, cos_lat[-1:]])
        # dy_v_m (ny+1,): V-face physical spacings in metres = adyv * dy_ref
        dy_v_m = adyv_m * dy_ref                                           # (ny+1,)
        # dy_row_m (ny,): mass-cell physical spacings in metres = ady * dy_ref
        dy_row_m = ady_m * dy_ref                                          # (ny,)
        # U/W flux factor at V-face: muv[face] / (dy_v[face] * dy_ref)
        rdy_uw_flux = (muv_m / (dy_v_m * dy_ref))[None, :, None]         # (1, ny+1, 1)
        # U/W divergence factor at mass cell: 1 / (dy_row[j] * mu[j])
        rdy_uw_div  = (1.0 / (dy_row_m * cos_lat))[None, :, None]        # (1, ny, 1)
        # V flux factor at mass cell j (between V-rows j and j+1):
        #   rdy2v = rdy2/ady(j)*mu(j) = mu(j) / (dy_row(j) * dy_ref)
        rdy_v_flux  = (cos_lat / (dy_row_m * dy_ref))[None, :, None]     # (1, ny, 1)
        # V divergence factor at V-face j (between cells j-1 and j):
        #   1 / (adyv(j) * muv(j))
        rdy_v_div   = (1.0 / (adyv_m * muv_m))[None, :, None]            # (1, ny+1, 1)
    else:
        # Fallback: Cartesian (no spherical corrections)
        rdy2_scalar = 1.0 / dy_ref**2
        rdy_uw_flux = rdy2_scalar
        rdy_uw_div  = 1.0 / dy_ref
        rdy_v_flux  = rdy2_scalar
        rdy_v_div   = 1.0 / dy_ref

    # Helper: interpolate cell-centre tk to U x-faces (periodic in x)
    # tk_Ux[k,j,i] ≈ mean of cell tk to the left and right of U-face i
    tk_Ux = 0.5 * (jnp.roll(tk, 1, axis=2) + tk)  # (nz,ny,nx), valid at face i using tk[i-1] and tk[i]
    # Extend to nx+1 with periodic wrap: face nx same as face 0
    tk_at_U = jnp.concatenate([tk_Ux, tk_Ux[:, :, :1]], axis=2)          # (nz,ny,nx+1)
    # D12 fix: 4-point corner interpolation for staggered tk (gSAM diffuse_mom3D.f90:44-48)
    # At V y-face (j+1/2, i): average tk from 4 surrounding cells (j,i-1), (j,i), (j+1,i-1), (j+1,i)
    tk_xm = jnp.roll(tk, 1, axis=2)                                       # tk at i-1
    tk_yp   = jnp.pad(tk, ((0, 0), (1, 0), (0, 0)), mode='edge')          # (nz,ny+1,nx)
    tk_xm_yp = jnp.pad(tk_xm, ((0, 0), (1, 0), (0, 0)), mode='edge')    # (nz,ny+1,nx)
    tk_at_V = 0.25 * (tk_yp[:, :-1, :] + tk_yp[:, 1:, :]
                     + tk_xm_yp[:, :-1, :] + tk_xm_yp[:, 1:, :])         # (nz,ny,nx)
    tk_at_V = jnp.pad(tk_at_V, ((0, 0), (0, 1), (0, 0)), mode='edge')    # (nz,ny+1,nx) add top row
    # Interpolate to W z-faces: 4-point average over (k-1,i), (k,i), (k-1,i-1), (k,i-1)
    tk_zp = jnp.pad(tk, ((1, 0), (0, 0), (0, 0)), mode='edge')            # (nz+1,ny,nx)
    tk_xm_zp = jnp.pad(tk_xm, ((1, 0), (0, 0), (0, 0)), mode='edge')
    tk_at_W_int = 0.25 * (tk_zp[:-1, :, :] + tk_zp[1:, :, :]
                        + tk_xm_zp[:-1, :, :] + tk_xm_zp[1:, :, :])      # (nz,ny,nx)
    tk_at_W = jnp.pad(
        0.5 * (tk[:-1, :, :] + tk[1:, :, :]),                             # (nz-1,ny,nx) interior
        ((1, 1), (0, 0), (0, 0)), mode='edge',
    )                                                                      # (nz+1,ny,nx)

    rho_dz  = (rho * dz)[:, None, None]                                   # (nz,1,1)
    rhow_full = rhow[:, None, None]                                        # (nz+1,1,1)

    # ======== dU/dt: shape (nz, ny, nx+1) ========

    # x: flux between adjacent U-faces at cell centres (periodic)
    # Fix 2.5: cap tk locally at x-flux computation (gSAM line 42)
    tk_x = tk
    if tk_max is not None:
        # Cap at x-faces (shape: nz, ny, nx+1) - tk_max is (nz, ny, 1)
        tk_x = jnp.minimum(tk_x, tk_max)
    F_ux = -rdx2u * tk_x * (U[:, :, 1:] - U[:, :, :-1])                    # (nz,ny,nx)  F[i] at cell i
    dUdt = jnp.roll(F_ux, 1, axis=2) - F_ux                               # (nz,ny,nx)  = F[i-1]-F[i]
    dUdt = jnp.concatenate([dUdt, dUdt[:, :, :1]], axis=2)                # (nz,ny,nx+1) periodic

    # y: flux at y-interfaces between U-face rows (Neumann at poles)
    # Fix 2.3: flux uses rdy2u=muv[face]/(dy_v[face]*dy_ref); div uses 1/(dy_row*mu)
    U_yp = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')              # (nz,ny+2,nx+1)
    tkU_yp = jnp.pad(tk_at_U, ((0, 0), (1, 1), (0, 0)), mode='edge')     # (nz,ny+2,nx+1)
    # Fix 2.5: cap tk_at_U at y-face before computing flux (gSAM line 71)
    tkU_y_face = 0.5 * (tkU_yp[:, :-1, :] + tkU_yp[:, 1:, :])             # (nz,ny+1,nx+1)
    if tk_max is not None:
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)
        tkU_y_face = jnp.minimum(tkU_y_face, tk_max_yface)
    fy_U = (-rdy_uw_flux * tkU_y_face
            * (U_yp[:, 1:, :] - U_yp[:, :-1, :]))                        # (nz,ny+1,nx+1)
    dUdt = dUdt - rdy_uw_div * (fy_U[:, 1:, :] - fy_U[:, :-1, :])

    # z: anelastic flux between U levels (zero-flux at top/bottom)
    dzw_int  = dzw_[1:-1][:, None, None]                                  # (nz-1,1,1)
    rhow_int = rhow[1:-1][:, None, None]                                   # (nz-1,1,1)
    tkUz = 0.5 * (tk_at_U[:-1, :, :] + tk_at_U[1:, :, :])                # (nz-1,ny,nx+1) at interior w-faces
    # Fix 2.5: cap tkUz at z-face before computing flux (gSAM line 119)
    if tk_max is not None:
        tkUz = jnp.minimum(tkUz, tk_max[:-1, :, :])
    fz_Uint = -rhow_int * tkUz * (U[1:, :, :] - U[:-1, :, :]) / dzw_int  # (nz-1,ny,nx+1)
    # Surface (bottom) flux for U: interpolate tau_x from cell centres to U x-faces
    if tau_x is None:
        fz_U_bot = jnp.zeros((1, U.shape[1], U.shape[2]))
    else:
        tau_x_ux = 0.5 * (jnp.roll(tau_x, 1, axis=-1) + tau_x)           # (ny, nx) at interior x-faces
        tau_x_u  = jnp.concatenate(
            [tau_x_ux, tau_x_ux[:, :1]], axis=-1,
        )                                                                  # (ny, nx+1) periodic
        fz_U_bot = rhow[0] * tau_x_u[None, :, :]                          # (1, ny, nx+1)
    fz_U = jnp.concatenate([
        fz_U_bot,
        fz_Uint,
        jnp.zeros((1, U.shape[1], U.shape[2])),
    ], axis=0)                                                             # (nz+1,ny,nx+1)
    dUdt = dUdt - (fz_U[1:, :, :] - fz_U[:-1, :, :]) / rho_dz

    # ======== dV/dt: shape (nz, ny+1, nx) ========

    # x: flux at x-interfaces between V-face columns (periodic)
    V_xp = jnp.concatenate([V[:, :, -1:], V, V[:, :, :1]], axis=2)       # (nz,ny+1,nx+2)
    tkV_xp = jnp.concatenate(
        [tk_at_V[:, :, -1:], tk_at_V, tk_at_V[:, :, :1]], axis=2
    )                                                                      # (nz,ny+1,nx+2)
    # Fix 2.5: cap tk_at_V at x-face before computing flux (gSAM line 45)
    tkV_x_face = 0.5 * (tkV_xp[:, :, :-1] + tkV_xp[:, :, 1:])             # (nz,ny+1,nx+1)
    if tk_max is not None:
        # Cap at x-faces (shape: nz, ny+1, nx+1) - tk_max is (nz, ny, 1)
        # Interpolate tk_max to y-faces for proper broadcasting (nz, ny+1, 1)
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)
        tkV_x_face = jnp.minimum(tkV_x_face, tk_max_yface)
    fx_V = (-rdx2v * tkV_x_face
            * (V_xp[:, :, 1:] - V_xp[:, :, :-1]))                        # (nz,ny+1,nx+1)
    dVdt = -(fx_V[:, :, 1:] - fx_V[:, :, :-1])                           # (nz,ny+1,nx)

    # y: second-difference of V in y.  gSAM boundaries.f90:101-129 applies an
    # antisymmetric wall mirror to V at the polar walls:
    #     v(:,0,:) = -v(:,2,:)  (low halo)     v(:,ny+2,:) = -v(:,ny,:)  (high halo)
    # In Python V is (nz, ny+1, nx) with V[:,0,:] and V[:,ny,:] the pole walls,
    # so the halo-low row = -V[:,1,:] and halo-high row = -V[:,ny-1,:].
    # Scalars (tk_at_V) use the symmetric mirror, which for a 1-row halo is
    # identical to mode='edge'.
    V_yp = jnp.concatenate(
        [-V[:, 1:2, :], V, -V[:, -2:-1, :]], axis=1,
    )                                                                      # (nz,ny+3,nx)
    tkV_yp = jnp.pad(tk_at_V, ((0, 0), (1, 1), (0, 0)), mode='edge')     # (nz,ny+3,nx)
    # Fix 2.3: V y-flux uses rdy2v=mu[cell]/(dy_row[cell]*dy_ref); div uses 1/(adyv[face]*muv[face])
    # V_yp has ny+3 rows; fy_V flux at ny+2 faces, face j connects V_yp[j] and V_yp[j+1].
    # Face j=1..ny connects V[j-1] and V[j], so the flux factor is at mass cell j-1.
    # rdy_v_flux shape (1,ny,1): pad to (1,ny+2,1) with edge so padded[j] = rdy_v_flux[j-1 clipped].
    if hasattr(rdy_v_flux, 'shape'):
        rdy_v_flux_padded = jnp.pad(rdy_v_flux, ((0, 0), (1, 1), (0, 0)), mode='edge')  # (1,ny+2,1)
    else:
        rdy_v_flux_padded = rdy_v_flux                                     # scalar fallback
    # Fix 2.5: cap tk_at_V at y-face before computing flux (gSAM line 74)
    tkV_y_face = 0.5 * (tkV_yp[:, :-1, :] + tkV_yp[:, 1:, :])             # (nz,ny+2,nx)
    if tk_max is not None:
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)
        tkV_y_face = jnp.minimum(tkV_y_face, tk_max_yface)
    fy_V = (-rdy_v_flux_padded * tkV_y_face
            * (V_yp[:, 1:, :] - V_yp[:, :-1, :]))                        # (nz,ny+2,nx)
    dVdt = dVdt - rdy_v_div * (fy_V[:, 1:, :] - fy_V[:, :-1, :])         # (nz,ny+1,nx)

    # z: anelastic
    tkVz = 0.5 * (tk_at_V[:-1, :, :] + tk_at_V[1:, :, :])                # (nz-1,ny+1,nx)
    # Fix 2.5: cap tkVz at z-face before computing flux (gSAM line 122)
    if tk_max is not None:
        tkVz = jnp.minimum(tkVz, tk_max[:-1, :, :])
    fz_Vint = -rhow_int * tkVz * (V[1:, :, :] - V[:-1, :, :]) / dzw_int  # (nz-1,ny+1,nx)
    # Surface (bottom) flux for V: interpolate tau_y from cell centres to V y-faces
    if tau_y is None:
        fz_V_bot = jnp.zeros((1, V.shape[1], V.shape[2]))
    else:
        tau_y_yp  = jnp.pad(tau_y, ((1, 0), (0, 0)), mode='edge')         # (ny+1, nx) left-pad
        tau_y_v   = 0.5 * (tau_y_yp[:-1, :] + tau_y_yp[1:, :])           # (ny, nx) interior faces
        tau_y_v   = jnp.pad(tau_y_v, ((0, 1), (0, 0)), mode='edge')       # (ny+1, nx) add top row
        fz_V_bot  = rhow[0] * tau_y_v[None, :, :]                         # (1, ny+1, nx)
    fz_V = jnp.concatenate([
        fz_V_bot,
        fz_Vint,
        jnp.zeros((1, V.shape[1], V.shape[2])),
    ], axis=0)
    dVdt = dVdt - (fz_V[1:, :, :] - fz_V[:-1, :, :]) / rho_dz
    # Polar walls: V=0 is a specified BC — enforce zero tendency after all diffusion
    dVdt = dVdt.at[:, 0, :].set(0.0)
    dVdt = dVdt.at[:, -1, :].set(0.0)

    # ======== dW/dt: shape (nz+1, ny, nx) ========

    # x: flux at x-interfaces between W columns (periodic)
    W_xp = jnp.concatenate([W[:, :, -1:], W, W[:, :, :1]], axis=2)       # (nz+1,ny,nx+2)
    tkW_xp = jnp.concatenate(
        [tk_at_W[:, :, -1:], tk_at_W, tk_at_W[:, :, :1]], axis=2
    )                                                                      # (nz+1,ny,nx+2)
    # Fix 2.5: cap tk_at_W at x-face before computing flux (gSAM line 48)
    tkW_x_face = 0.5 * (tkW_xp[:, :, :-1] + tkW_xp[:, :, 1:])             # (nz+1,ny,nx+1)
    if tk_max is not None:
        # Cap at x-faces (shape: nz+1, ny, nx+1) - tk_max is (nz, ny, 1)
        # Need to extend tk_max to nz+1 dimension first
        tk_max_extended = jnp.concatenate([tk_max, tk_max[-1:, :, :]], axis=0)
        tkW_x_face = jnp.minimum(tkW_x_face, tk_max_extended)
    fx_W = (-rdx2w * tkW_x_face
            * (W_xp[:, :, 1:] - W_xp[:, :, :-1]))                        # (nz+1,ny,nx+1)
    dWdt = -(fx_W[:, :, 1:] - fx_W[:, :, :-1])                           # (nz+1,ny,nx)

    # y: Neumann pad
    # Fix 2.3: W y-flux uses rdy2w=muv[face]/(dy_v[face]*dy_ref); div uses 1/(dy_row*mu)
    W_yp  = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')             # (nz+1,ny+2,nx)
    tkW_yp = jnp.pad(tk_at_W, ((0, 0), (1, 1), (0, 0)), mode='edge')     # (nz+1,ny+2,nx)
    # Fix 2.5: cap tk_at_W at y-face before computing flux (gSAM line 77)
    tkW_y_face = 0.5 * (tkW_yp[:, :-1, :] + tkW_yp[:, 1:, :])             # (nz+1,ny+1,nx)
    if tk_max is not None:
        tk_max_interior = 0.5 * (tk_max[:, :-1, :] + tk_max[:, 1:, :])
        tk_max_yface = jnp.concatenate([tk_max[:, :1, :], tk_max_interior, tk_max[:, -1:, :]], axis=1)
        tkW_y_face = jnp.minimum(tkW_y_face, tk_max_yface)
    fy_W = (-rdy_uw_flux * tkW_y_face
            * (W_yp[:, 1:, :] - W_yp[:, :-1, :]))                        # (nz+1,ny+1,nx)
    dWdt = dWdt - rdy_uw_div * (fy_W[:, 1:, :] - fy_W[:, :-1, :])

    # z: second-difference of W at w-face positions (W is already at w-faces)
    # gSAM diffuse_mom3D.f90:124 uses tkz=tk(i,j,k) directly (cell-centre),
    # not a face average.  Flux at face between W[k] and W[k+1] uses tk[k].
    dW_dz = W[1:, :, :] - W[:-1, :, :]                                    # (nz,ny,nx)
    # Fix 2.5: cap tk at z-flux computation (gSAM line 125)
    tk_z = tk
    if tk_max is not None:
        tk_z = jnp.minimum(tk_z, tk_max)
    fz_Wint = -tk_z * dW_dz / dz[:, None, None]                              # (nz,ny,nx)
    # D24 fix: gSAM diffuse_mom3D.f90:137 zeros fwz(i,j,2) — the flux at the
    # lowest interior W face (between W[1] and W[2], 0-indexed = fz_Wint[1]).
    fz_Wint = fz_Wint.at[1].set(0.0)
    fz_W = jnp.concatenate([
        jnp.zeros((1, W.shape[1], W.shape[2])),
        fz_Wint,
        jnp.zeros((1, W.shape[1], W.shape[2])),
    ], axis=0)                                                             # (nz+2,ny,nx)
    # dzw_ has shape (nz+1,) — distance between cell centres at each w-face
    rhow_dzw = (rhow * dzw_)[:, None, None]                               # (nz+1,1,1)
    dWdt = dWdt - (fz_W[1:, :, :] - fz_W[:-1, :, :]) / rhow_dzw

    # Rigid-lid BCs
    dWdt = dWdt.at[0, :, :].set(0.0)
    dWdt = dWdt.at[-1, :, :].set(0.0)

    return dUdt, dVdt, dWdt


# ---------------------------------------------------------------------------
# Horizontal-only SGS momentum diffusion (for use with implicit vertical)
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_momentum_horiz(
    U: jax.Array,   # (nz, ny, nx+1)
    V: jax.Array,   # (nz, ny+1, nx)
    W: jax.Array,   # (nz+1, ny, nx)
    tk:     jax.Array,   # (nz, ny, nx) eddy viscosity at cell centres
    metric: dict,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Horizontal-only SGS momentum diffusion tendencies.
    Vertical diffusion is handled by the implicit solver (diffuse_damping_mom_z).
    Port of gSAM diffuse_mom3D.f90 with doimplicitdiff=.true. (goto 333).
    """
    dx  = metric["dx_lon"]
    dy_ref = metric["dy_lat_ref"]          # SGS closure uses reference dy (Gap 8)
    dz  = metric["dz"]

    # C8 fix: apply imu(j)^2 = 1/cos(lat)^2 metric to x-direction
    cos_lat = metric.get("cos_lat", None)
    cos_v_h = metric.get("cos_v", None)
    rdx2_base = (1.0 / dx)**2
    if cos_lat is not None:
        imu2 = (1.0 / cos_lat[None, :, None]) ** 2   # (1, ny, 1)
    else:
        imu2 = 1.0
    if cos_v_h is not None:
        imuv2_h = (1.0 / cos_v_h[None, :, None]) ** 2   # (1, ny+1, 1)
    else:
        imuv2_h = imu2
    rdx2   = rdx2_base * imu2
    rdx2_v = rdx2_base * imuv2_h   # for V x-diffusion

    # Fix 2.3: spherical y-metrics (same as diffuse_momentum)
    ady_h = metric.get("ady", None)
    if ady_h is not None and cos_lat is not None:
        adyv_int_h = 0.5 * (ady_h[:-1] + ady_h[1:])
        adyv_h = jnp.concatenate([ady_h[:1], adyv_int_h, ady_h[-1:]])
        muv_h  = cos_v_h if cos_v_h is not None else jnp.concatenate(
            [cos_lat[:1], 0.5 * (cos_lat[:-1] + cos_lat[1:]), cos_lat[-1:]]
        )
        dy_v_h    = adyv_h * dy_ref
        dy_row_h  = ady_h * dy_ref
        rdy_uw_flux_h = (muv_h / (dy_v_h * dy_ref))[None, :, None]
        rdy_uw_div_h  = (1.0 / (dy_row_h * cos_lat))[None, :, None]
        rdy_v_flux_h  = (cos_lat / (dy_row_h * dy_ref))[None, :, None]
        rdy_v_div_h   = (1.0 / (adyv_h * muv_h))[None, :, None]
    else:
        rdy2_s = 1.0 / dy_ref**2
        rdy_uw_flux_h = rdy2_s
        rdy_uw_div_h  = 1.0 / dy_ref
        rdy_v_flux_h  = rdy2_s
        rdy_v_div_h   = 1.0 / dy_ref

    # Interpolate tk to staggered positions
    tk_Ux = 0.5 * (jnp.roll(tk, 1, axis=2) + tk)
    tk_at_U = jnp.concatenate([tk_Ux, tk_Ux[:, :, :1]], axis=2)
    tk_yp   = jnp.pad(tk, ((0, 0), (1, 0), (0, 0)), mode='edge')
    tk_at_V = 0.5 * (tk_yp[:, :-1, :] + tk_yp[:, 1:, :])
    tk_at_V = jnp.pad(tk_at_V, ((0, 0), (0, 1), (0, 0)), mode='edge')
    tk_at_W = jnp.pad(
        0.5 * (tk[:-1, :, :] + tk[1:, :, :]),
        ((1, 1), (0, 0), (0, 0)), mode='edge',
    )

    # ======== dU/dt (horiz only) ========
    F_ux = -rdx2 * tk * (U[:, :, 1:] - U[:, :, :-1])
    dUdt = jnp.roll(F_ux, 1, axis=2) - F_ux
    dUdt = jnp.concatenate([dUdt, dUdt[:, :, :1]], axis=2)

    # Fix 2.3: U y-flux uses muv[face]/(dy_v[face]*dy_ref); div uses 1/(dy_row*mu)
    U_yp = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')
    tkU_yp = jnp.pad(tk_at_U, ((0, 0), (1, 1), (0, 0)), mode='edge')
    fy_U = (-rdy_uw_flux_h * 0.5 * (tkU_yp[:, :-1, :] + tkU_yp[:, 1:, :])
            * (U_yp[:, 1:, :] - U_yp[:, :-1, :]))
    dUdt = dUdt - rdy_uw_div_h * (fy_U[:, 1:, :] - fy_U[:, :-1, :])

    # ======== dV/dt (horiz only) ========
    V_xp = jnp.concatenate([V[:, :, -1:], V, V[:, :, :1]], axis=2)
    tkV_xp = jnp.concatenate(
        [tk_at_V[:, :, -1:], tk_at_V, tk_at_V[:, :, :1]], axis=2)
    fx_V = (-rdx2_v * 0.5 * (tkV_xp[:, :, :-1] + tkV_xp[:, :, 1:])
            * (V_xp[:, :, 1:] - V_xp[:, :, :-1]))
    dVdt = -(fx_V[:, :, 1:] - fx_V[:, :, :-1])

    # Antisymmetric wall mirror for V (gSAM boundaries.f90:101-129).
    # Fix 2.3: V y-flux uses mu[cell]/(dy_row[cell]*dy_ref); div uses 1/(adyv[face]*muv[face])
    V_yp = jnp.concatenate(
        [-V[:, 1:2, :], V, -V[:, -2:-1, :]], axis=1,
    )
    tkV_yp = jnp.pad(tk_at_V, ((0, 0), (1, 1), (0, 0)), mode='edge')
    if hasattr(rdy_v_flux_h, 'shape'):
        rdy_v_flux_h_padded = jnp.pad(rdy_v_flux_h, ((0, 0), (1, 1), (0, 0)), mode='edge')
    else:
        rdy_v_flux_h_padded = rdy_v_flux_h
    fy_V = (-rdy_v_flux_h_padded * 0.5 * (tkV_yp[:, :-1, :] + tkV_yp[:, 1:, :])
            * (V_yp[:, 1:, :] - V_yp[:, :-1, :]))
    dVdt = dVdt - rdy_v_div_h * (fy_V[:, 1:, :] - fy_V[:, :-1, :])
    dVdt = dVdt.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)

    # ======== dW/dt (horiz only) ========
    W_xp = jnp.concatenate([W[:, :, -1:], W, W[:, :, :1]], axis=2)
    tkW_xp = jnp.concatenate(
        [tk_at_W[:, :, -1:], tk_at_W, tk_at_W[:, :, :1]], axis=2)
    fx_W = (-rdx2 * 0.5 * (tkW_xp[:, :, :-1] + tkW_xp[:, :, 1:])
            * (W_xp[:, :, 1:] - W_xp[:, :, :-1]))
    dWdt = -(fx_W[:, :, 1:] - fx_W[:, :, :-1])

    # Fix 2.3: W y-flux uses muv[face]/(dy_v[face]*dy_ref); div uses 1/(dy_row*mu)
    W_yp  = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')
    tkW_yp = jnp.pad(tk_at_W, ((0, 0), (1, 1), (0, 0)), mode='edge')
    fy_W = (-rdy_uw_flux_h * 0.5 * (tkW_yp[:, :-1, :] + tkW_yp[:, 1:, :])
            * (W_yp[:, 1:, :] - W_yp[:, :-1, :]))
    dWdt = dWdt - rdy_uw_div_h * (fy_W[:, 1:, :] - fy_W[:, :-1, :])

    dWdt = dWdt.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)

    return dUdt, dVdt, dWdt


# ---------------------------------------------------------------------------
# Thomas tridiagonal solver (vectorised over spatial dims)
# ---------------------------------------------------------------------------

def _thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system along axis 0 using the Thomas algorithm.
    Vectorised over all other dimensions via jax.lax.scan.

    Parameters
    ----------
    a, b, c, d : (nz, *spatial)
        Sub-diagonal (a[0] unused), diagonal, super-diagonal (c[-1] unused),
        and right-hand side arrays.

    Returns
    -------
    x : (nz, *spatial)  solution
    """
    def fwd_step(carry, inp):
        al, be = carry
        ak, bk, ck, dk = inp
        e = bk + ak * al
        al_new = -ck / e
        be_new = (dk - ak * be) / e
        return (al_new, be_new), (al_new, be_new)

    al0 = -c[0] / b[0]
    be0 = d[0] / b[0]

    _, (als, bes) = jax.lax.scan(
        fwd_step, (al0, be0), (a[1:], b[1:], c[1:], d[1:]),
    )
    alpha = jnp.concatenate([al0[None], als], axis=0)
    beta  = jnp.concatenate([be0[None], bes], axis=0)

    def bwd_step(x_next, inp):
        al_k, be_k = inp
        x_k = al_k * x_next + be_k
        return x_k, x_k

    x_last = beta[-1]
    _, xs = jax.lax.scan(bwd_step, x_last, (alpha[:-1], beta[:-1]),
                          reverse=True)
    return jnp.concatenate([xs, x_last[None]], axis=0)


# ---------------------------------------------------------------------------
# Implicit vertical SGS diffusion + damping for momentum
# (gSAM diffuse_damping_mom_z.f90)
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_damping_mom_z(
    U:  jax.Array,       # (nz, ny, nx+1)
    V:  jax.Array,       # (nz, ny+1, nx)
    W:  jax.Array,       # (nz+1, ny, nx)
    tk: jax.Array,       # (nz, ny, nx) eddy viscosity
    metric: dict,
    dt: float,
    damping_u_cu: float = 0.25,
    damping_w_cu: float = 0.3,
    fluxbu: jax.Array | None = None,   # (ny, nx+1) surface U flux (m/s * m/s)
    fluxbv: jax.Array | None = None,   # (ny+1, nx) surface V flux
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Implicit vertical SGS diffusion + all damping for U, V, W.
    Replaces explicit damping() + explicit vertical SGS diffusion.

    Combines:
      - SGS vertical diffusion (tridiagonal implicit solve)
      - Top sponge on W (dodamping, nub=0.6)
      - W CFL limiter (dodamping_w)
      - Polar velocity limiter for U, V (dodamping_poles)
      - Upper-level U, V damping (pres < 70 hPa)
      - Surface momentum flux BCs

    Port of gSAM SGS_TKE/diffuse_damping_mom_z.f90.
    """
    nz = tk.shape[0]
    ny = tk.shape[1]
    nx = tk.shape[2]
    dz     = metric["dz"]        # (nz,)  adz * dz_ref — actual cell thickness (used for sponge zi)
    rho    = metric["rho"]       # (nz,)
    rhow   = metric["rhow"]      # (nz+1,)
    adz    = metric["adz"]       # (nz,)  (zi[k+1]-zi[k]) / dz_ref
    adzw   = metric["adzw"]      # (nz+1,) (z[k]-z[k-1]) / dz_ref; adzw[0]=1
    dz_ref = metric["dz_ref"]    # scalar, reference dz = zi[1]-zi[0]
    z      = metric["z"]         # (nz,)  cell-centre heights (used for sponge nu)
    cos_lat = metric["cos_lat"]  # (ny,)
    dx   = metric["dx_lon"]      # scalar
    pres = metric["pres"]        # (nz,) Pa

    tau_max = 1.0 / dt

    # Interior w-face arrays (faces 1..nz-1 in 0-indexed = gSAM faces 2..nzm).
    # adzw_int[f] = adzw[f+1] = (z[f+1]-z[f])/dz_ref  (center-to-center spacing ratio)
    # adz_lo[f]   = adz[f]   = thickness ratio of cell below face f+1
    # adz_hi[f]   = adz[f+1] = thickness ratio of cell above face f+1
    adzw_int = adzw[1:-1]    # (nz-1,) face-spacing ratios at interior w-faces
    adz_lo   = adz[:-1]      # (nz-1,) adz of cell below each interior face
    adz_hi   = adz[1:]       # (nz-1,) adz of cell above each interior face
    rhow_int = rhow[1:-1]    # (nz-1,) density at interior w-faces

    # ================================================================
    # U implicit solve  (nz, ny, nx+1)
    # ================================================================

    # SGS viscosity interpolated to U x-faces, then averaged to w-faces
    tk_ux = 0.5 * (jnp.roll(tk, 1, axis=2) + tk)   # (nz, ny, nx)
    tk_uf = jnp.concatenate([tk_ux, tk_ux[:, :, :1]], axis=2)  # (nz, ny, nx+1)
    # tkz at nz-1 interior w-faces: average of levels above and below
    tkz_u = 0.5 * (tk_uf[:-1] + tk_uf[1:])  # (nz-1, ny, nx+1)

    # Polar + upper-level damping coefficients for U
    mu = cos_lat                                            # (ny,)
    tauy = tau_max * (1.0 - mu ** 2) ** 200                # (ny,)
    umax_polar = damping_u_cu * dx * mu / dt               # (ny,)
    umax_3d = umax_polar[None, :, None]                    # (1, ny, 1)

    # pres < 70 hPa → use tau_max; else tauy
    pres_hpa = pres / 100.0
    tau_base_u = jnp.where(
        pres_hpa[:, None] < 70.0, tau_max, tauy[None, :],
    )  # (nz, ny)

    U_exceeds = jnp.abs(U) > umax_3d
    tau_vel_u = jnp.where(U_exceeds, tau_base_u[:, :, None], 0.0)
    vel0_u = jnp.where(U > umax_3d, umax_3d,
                        jnp.where(U < -umax_3d, -umax_3d, 0.0))

    # Tridiagonal coefficients — exact gSAM diffuse_damping_mom_z.f90 notation.
    # rdz2 = dt/dz_ref^2; for row k (0-indexed), c coefficient uses face above (adzw[k+1]):
    #   c[k] = -tkz * rdz2 * rhow[k+1] / (adzw[k+1] * adz[k] * rho[k])
    # and a coefficient at row k uses face below (adzw[k]):
    #   a[k] = -tkz * rdz2 * rhow[k]   / (adzw[k]   * adz[k] * rho[k])
    # Here adzw_int[f] = adzw[f+1], adz_lo[f] = adz[f] (cell below face f+1).
    rdz2 = dt / dz_ref ** 2
    c_vals = (-rdz2 * rhow_int[:, None, None] * tkz_u
              / (adzw_int[:, None, None] * adz_lo[:, None, None] * rho[:-1, None, None]))
    c_u = jnp.concatenate([c_vals, jnp.zeros((1, ny, nx + 1))], axis=0)

    # a[k+1] uses face adzw[k+1] (= adzw_int[k]) and adz[k+1] = adz_hi[k].
    a_vals = (-rdz2 * rhow_int[:, None, None] * tkz_u
              / (adzw_int[:, None, None] * adz_hi[:, None, None] * rho[1:, None, None]))
    a_u = jnp.concatenate([jnp.zeros((1, ny, nx + 1)), a_vals], axis=0)

    b_u = 1.0 + dt * tau_vel_u - a_u - c_u
    d_u = U + dt * tau_vel_u * vel0_u

    # Surface flux BC at k = k_terrau(i,j).  jsam is flat-only so
    # k_terrau ≡ 0 everywhere, matching gSAM diffuse_damping_mom_z.f90:121-138:
    #     d(i,j,k) += dtn*rhow(k)/(dz_ref*adz(k)*rho(k))*fluxbu(i,j)
    # (gSAM uses dz=dz_ref and adz(k_terrau)=1 for flat terrain, but we use
    # adz[0] for generality to match the Fortran coefficient exactly.)
    if fluxbu is not None:
        d_u = d_u.at[0].add(dt * rhow[0] / (dz_ref * adz[0] * rho[0]) * fluxbu)

    U_new = _thomas_solve(a_u, b_u, c_u, d_u)

    # ================================================================
    # V implicit solve  (nz, ny+1, nx)
    # ================================================================

    # SGS viscosity interpolated to V y-faces, then averaged to w-faces
    tk_yp = jnp.pad(tk, ((0, 0), (1, 0), (0, 0)), mode='edge')
    tk_vf = 0.5 * (tk_yp[:, :-1, :] + tk_yp[:, 1:, :])
    tk_vf = jnp.pad(tk_vf, ((0, 0), (0, 1), (0, 0)), mode='edge')  # (nz, ny+1, nx)
    tkz_v = 0.5 * (tk_vf[:-1] + tk_vf[1:])  # (nz-1, ny+1, nx)

    # Polar + upper-level damping for V
    # D7 fix: gSAM uses mass-cell mu(j) and umax(j) for BOTH U and V.
    # Map interior v-faces to the average of the two adjacent mass-row values.
    cos_v_half = jnp.pad(
        0.5 * (cos_lat[:-1] + cos_lat[1:]),
        (1, 1), mode='edge',
    )  # (ny+1,)
    # Use mass-cell-averaged tau and umax (not v-face cos)
    tauy_mass = tau_max * (1.0 - cos_lat ** 2) ** 200    # (ny,)
    umax_mass = damping_u_cu * dx * cos_lat / dt         # (ny,)
    tauy_v = jnp.pad(
        0.5 * (tauy_mass[:-1] + tauy_mass[1:]),
        (1, 1), constant_values=tau_max,
    )  # (ny+1,)
    umax_v = jnp.pad(
        0.5 * (umax_mass[:-1] + umax_mass[1:]),
        (1, 1), constant_values=0.0,
    )  # (ny+1,)
    umax_v3d = umax_v[None, :, None]

    tau_base_v = jnp.where(
        pres_hpa[:, None] < 70.0, tau_max, tauy_v[None, :],
    )
    V_exceeds = jnp.abs(V) > umax_v3d
    tau_vel_v = jnp.where(V_exceeds, tau_base_v[:, :, None], 0.0)
    vel0_v = jnp.where(V > umax_v3d, umax_v3d,
                        jnp.where(V < -umax_v3d, -umax_v3d, 0.0))

    c_vals_v = (-rdz2 * rhow_int[:, None, None] * tkz_v
                / (adzw_int[:, None, None] * adz_lo[:, None, None] * rho[:-1, None, None]))
    c_v = jnp.concatenate([c_vals_v, jnp.zeros((1, ny + 1, nx))], axis=0)

    a_vals_v = (-rdz2 * rhow_int[:, None, None] * tkz_v
                / (adzw_int[:, None, None] * adz_hi[:, None, None] * rho[1:, None, None]))
    a_v = jnp.concatenate([jnp.zeros((1, ny + 1, nx)), a_vals_v], axis=0)

    b_v = 1.0 + dt * tau_vel_v - a_v - c_v
    d_v = V + dt * tau_vel_v * vel0_v

    # Surface flux BC at k = k_terrav(i,j) ≡ 0 for flat-only jsam.
    # Matches gSAM diffuse_damping_mom_z.f90:243-260: dtn*rhow(k)/(dz*adz(k)*rho(k)).
    if fluxbv is not None:
        d_v = d_v.at[0].add(dt * rhow[0] / (dz_ref * adz[0] * rho[0]) * fluxbv)

    V_new = _thomas_solve(a_v, b_v, c_v, d_v)
    # Enforce polar wall BC: V=0 at j=0 and j=ny
    V_new = V_new.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)

    # ================================================================
    # W implicit solve  (interior w-faces 1..nz-1, BCs W[0]=W[nz]=0)
    # ================================================================

    # Top sponge: tauz at cell centres (gSAM dodamping, nub=0.6)
    zi_0   = z[0] - 0.5 * dz[0]
    zi_top = z[-1] + 0.5 * dz[-1]
    nub = 0.6
    nu_c = (z - zi_0) / (zi_top - zi_0)
    zzz  = jnp.where(nu_c > nub, 100.0 * ((nu_c - nub) / (1.0 - nub)) ** 2, 0.0)
    tauz_c = tau_max * zzz / (1.0 + zzz)  # (nz,) at cell centres
    # Interpolate tauz to interior w-faces (nz-1 values at faces 1..nz-1)
    tauz_w = 0.5 * (tauz_c[:-1] + tauz_c[1:])  # (nz-1,)

    # W CFL limiter (dodamping_w): only below sponge (tauz_w == 0)
    # gSAM: wmax = damping_w_cu * dz * adzw(k) / dtn = damping_w_cu * dz_ref * adzw_int / dt
    wmax_w = damping_w_cu * dz_ref * adzw_int / dt  # (nz-1,)
    W_int = W[1:-1]  # (nz-1, ny, nx) interior w-faces

    below_sponge = (tauz_w == 0.0)[:, None, None]
    W_exceeds_w = jnp.abs(W_int) > wmax_w[:, None, None]
    tau_vel_w = jnp.where(below_sponge & W_exceeds_w, tau_max, 0.0)
    vel0_w = jnp.where(
        W_int > wmax_w[:, None, None], wmax_w[:, None, None],
        jnp.where(W_int < -wmax_w[:, None, None], -wmax_w[:, None, None], 0.0),
    )

    # For W at interior face f (0-indexed in the nz-1 system, = gSAM w-face k=f+2):
    # cell below = f (gSAM kb=f+1), cell above = f+1 (gSAM k=f+2)
    # gSAM diffuse_damping_mom_z.f90 lines 368-369 (1-indexed k, kc=k+1, kb=k-1):
    #   iadz  = rdz2*rho(kb)/(adzw(k)*adz(kb)*rhow(k))  -> a = -tkz_below * iadz
    #   iadzw = rdz2*rho(k) /(adzw(k)*adz(k) *rhow(k))  -> c = -tkz_above * iadzw
    # Both use adzw(k) (the same face), but adz(kb) vs adz(k) for a vs c.
    # 0-indexed: adzw(k=f+2) = adzw[f+1] = adzw_int[f]
    #            adz(kb=f+1) = adz[f]    = adz_lo[f]
    #            adz(k=f+2)  = adz[f+1]  = adz_hi[f]

    tk_below = tk[:-1]   # (nz-1, ny, nx) = tk at cells 0..nz-2  (gSAM tk(kb))
    tk_above = tk[1:]    # (nz-1, ny, nx) = tk at cells 1..nz-1  (gSAM tk(k))
    rhow_w   = rhow[1:-1]  # (nz-1,) = density at interior w-faces (gSAM rhow(k))

    a_w_vals = (-rdz2 * tk_below * rho[:-1, None, None]
                / (adzw_int[:, None, None] * adz_lo[:, None, None] * rhow_w[:, None, None]))
    c_w_vals = (-rdz2 * tk_above * rho[1:, None, None]
                / (adzw_int[:, None, None] * adz_hi[:, None, None] * rhow_w[:, None, None]))

    # First face (f=0): a connects to W[0]=0, so a is effectively unused
    # but we set alpha[0]=0, beta[0] properly by keeping a[0] in the system.
    # Since W[0]=0, the Thomas algorithm handles this naturally:
    # at f=0, a*W[-1] doesn't exist. We zero out a[0].
    a_w = a_w_vals.at[0].set(0.0)

    # Last face (f=nz-2): c connects to W[nz]=0. Zero out c[-1].
    c_w = c_w_vals.at[-1].set(0.0)

    b_w = (1.0 + dt * (tauz_w[:, None, None] + tau_vel_w)
           - a_w - c_w)
    d_w = W_int + dt * tau_vel_w * vel0_w

    W_int_new = _thomas_solve(a_w, b_w, c_w, d_w)

    # Reassemble full W with boundary conditions
    W_new = jnp.concatenate([
        jnp.zeros((1, ny, nx)),
        W_int_new,
        jnp.zeros((1, ny, nx)),
    ], axis=0)

    return U_new, V_new, W_new


# ---------------------------------------------------------------------------
# Implicit vertical SGS diffusion for scalars
# (gSAM SGS_TKE/diffuse_scalar_z.f90)
# ---------------------------------------------------------------------------

@jax.jit
def diffuse_scalar_z_implicit(
    field: jax.Array,     # (nz, ny, nx)
    tkh:   jax.Array,     # (nz, ny, nx) eddy diffusivity
    metric: dict,
    dt:    float,
    fluxb: jax.Array | None = None,  # (ny, nx) surface flux
    fluxt: jax.Array | None = None,  # (ny, nx) top flux
) -> jax.Array:
    """
    Implicit vertical SGS diffusion for a scalar field.
    Returns the updated field (not a tendency).

    Port of gSAM SGS_TKE/diffuse_scalar_z.f90.

    Tridiagonal coefficients match gSAM diffuse_scalar_z.f90 exactly:
      rdz2   = dt / dz_ref^2
      c[k]   = -tkz * rdz2 * rhow[k+1] / (adzw[k+1] * adz[k] * rho[k])   (super-diagonal)
      a[k]   = -tkz * rdz2 * rhow[k]   / (adzw[k]   * adz[k] * rho[k])   (sub-diagonal)
    where tkz is the face-average of tkh at the relevant w-face.
    """
    nz, ny, nx = field.shape
    rho    = metric["rho"]      # (nz,)
    rhow   = metric["rhow"]     # (nz+1,)
    adz    = metric["adz"]      # (nz,)   (zi[k+1]-zi[k]) / dz_ref
    adzw   = metric["adzw"]     # (nz+1,) (z[k]-z[k-1]) / dz_ref; adzw[0]=1
    dz_ref = metric["dz_ref"]   # scalar

    # Interior face arrays (gSAM faces 2..nzm → Python indices 1..nz-1).
    # adzw_int[f] = adzw[f+1]  (face-spacing ratio at interior face f+1)
    # adz_lo[f]   = adz[f]     (cell-thickness ratio of cell below face f+1)
    # adz_hi[f]   = adz[f+1]   (cell-thickness ratio of cell above face f+1)
    adzw_int = adzw[1:-1]   # (nz-1,)
    adz_lo   = adz[:-1]     # (nz-1,)
    adz_hi   = adz[1:]      # (nz-1,)
    rhow_int = rhow[1:-1]   # (nz-1,) at interior w-faces

    # tkh at interior w-faces: average of adjacent cell-centre values.
    # Matches gSAM: tkz = 0.5*(tkh(k-1)+tkh(k)) for sub, 0.5*(tkh(k)+tkh(k+1)) for super.
    tkh_fz = 0.5 * (tkh[:-1] + tkh[1:])  # (nz-1, ny, nx)

    # Tridiagonal coefficients — exact gSAM diffuse_scalar_z.f90 notation.
    # rdz2 = dt/dz_ref^2 (= gSAM's dtn/dz^2 where dz=dz_ref).
    rdz2 = dt / dz_ref ** 2
    # c[k] (super-diagonal, row k → row k+1):
    #   iadzw = rdz2 * rhow(kc) / (adzw(kc) * adz(k) * rho(k))  [gSAM line 70]
    c_vals = (-rdz2 * rhow_int[:, None, None] * tkh_fz
              / (adzw_int[:, None, None] * adz_lo[:, None, None] * rho[:-1, None, None]))
    c_f = jnp.concatenate([c_vals, jnp.zeros((1, ny, nx))], axis=0)

    # a[k] (sub-diagonal, row k → row k-1):
    #   iadz = rdz2 * rhow(k) / (adzw(k) * adz(k) * rho(k))  [gSAM line 69]
    # Applied at row k+1 (= f+1) using face f+1 spacing (adzw_int[f]=adzw[f+1])
    # and cell f+1 thickness (adz_hi[f]=adz[f+1]).
    a_vals = (-rdz2 * rhow_int[:, None, None] * tkh_fz
              / (adzw_int[:, None, None] * adz_hi[:, None, None] * rho[1:, None, None]))
    a_f = jnp.concatenate([jnp.zeros((1, ny, nx)), a_vals], axis=0)

    b_f = 1.0 - a_f - c_f
    d_f = field.copy()

    # Surface flux BC at k=0 (gSAM diffuse_scalar_z.f90 line 61):
    #   d += dtn * rhow(k_terra) / (rho(k) * dz * adz(k)) * fluxb
    if fluxb is not None:
        d_f = d_f.at[0].add(dt * rhow[0] / (dz_ref * adz[0] * rho[0]) * fluxb)
    # Top flux BC at k=nz-1 (gSAM line 97, kc=nzm+1=nz → rhow[nz]):
    #   d -= dtn * rhow(kc) / (rho(k) * dz * adz(k)) * fluxt
    if fluxt is not None:
        d_f = d_f.at[-1].add(-dt * rhow[-1] / (dz_ref * adz[-1] * rho[-1]) * fluxt)

    return _thomas_solve(a_f, b_f, c_f, d_f)


# ---------------------------------------------------------------------------
# Top-level SGS step
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("params",))
def sgs_proc(
    state:   ModelState,
    metric:  dict,
    params:  SGSParams,
    dt:      float,
    surface: SurfaceFluxes | None = None,
) -> ModelState:
    """
    Apply one explicit SGS diffusion step.
    Returns updated ModelState with new U,V,W,TABS,QV,QC,QI,QR,QS,QG.

    surface: optional SurfaceFluxes with shf (K·m/s), lhf (kg/kg·m/s),
             tau_x and tau_y (m²/s²) surface flux fields of shape (ny, nx).
             Pass None (default) for zero-flux bottom BCs.
    """
    U, V, W = state.U, state.V, state.W

    # Compute strain-rate invariant and eddy coefficients
    def2       = shear_prod(U, V, W, metric)
    tk, tkh    = smag_viscosity(def2, metric, params)

    # C9 fix: per-(j,k) 3D CFL stability cap; Fix 2.5: passed into diffusion
    # functions for local clamping at each face rather than global pre-cap.
    tk_max = _tkmax_3d(metric, dt)

    # Unpack surface fluxes (None → zero BC handled inside diffuse_* functions)
    shf   = None if surface is None else surface.shf
    lhf   = None if surface is None else surface.lhf
    tau_x = None if surface is None else surface.tau_x
    tau_y = None if surface is None else surface.tau_y

    # Scalar diffusion tendencies
    dTABS = diffuse_scalar(state.TABS, tkh, metric, fluxb=shf,  tk_max=tk_max)
    dQV   = diffuse_scalar(state.QV,   tkh, metric, fluxb=lhf,  tk_max=tk_max)
    dQC   = diffuse_scalar(state.QC,   tkh, metric, tk_max=tk_max)
    dQI   = diffuse_scalar(state.QI,   tkh, metric, tk_max=tk_max)
    dQR   = diffuse_scalar(state.QR,   tkh, metric, tk_max=tk_max)
    dQS   = diffuse_scalar(state.QS,   tkh, metric, tk_max=tk_max)
    dQG   = diffuse_scalar(state.QG,   tkh, metric, tk_max=tk_max)

    # Momentum diffusion tendencies
    dU, dV, dW = diffuse_momentum(U, V, W, tk, metric, tau_x=tau_x, tau_y=tau_y,
                                   tk_max=tk_max)

    return ModelState(
        U    = U     + dt * dU,
        V    = V     + dt * dV,
        W    = W     + dt * dW,
        TABS = state.TABS + dt * dTABS,
        QV   = state.QV + dt * dQV,
        QC   = state.QC + dt * dQC,
        QI   = state.QI + dt * dQI,
        QR   = state.QR + dt * dQR,
        QS   = state.QS + dt * dQS,
        QG   = state.QG + dt * dQG,
        TKE  = state.TKE,   # Smagorinsky: no prognostic TKE equation
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )


# ---------------------------------------------------------------------------
# Split SGS: momentum only (before advance_momentum) and scalars only (after
# advance_scalars).  Matches gSAM operator order:
#   sgs_mom_proc  → advance_momentum → advance_scalars → sgs_scalars_proc
# ---------------------------------------------------------------------------

def _tkmax_3d(metric: dict, dt: float) -> jax.Array:
    """C9 fix: per-(j,k) CFL stability cap for explicit SGS diffusion.

    Matches gSAM SGS_TKE.SAM/tke_full.f90:
        cx = dx^2/dt;  cy = dy^2/dt;  cz = (dz*min(adzw_lo,adzw_hi))^2/dt
        tkmax = 0.46 / (1/cx + 1/cy + 1/cz)

    With the C8 imu metric, effective dx in x is dx*cos(lat).
    Returns (nz, ny, 1) array.
    """
    dx  = metric["dx_lon"]
    dy  = metric["dy_lat"]           # (ny,)
    dz  = metric["dz"]              # (nz,)
    cos_lat = metric.get("cos_lat", None)

    if cos_lat is not None:
        cx = ((dx * cos_lat) ** 2 / dt)[None, :, None]   # (1, ny, 1)
    else:
        cx = dx ** 2 / dt

    cy = (dy ** 2 / dt)[None, :, None]                    # (1, ny, 1)

    dzw_lo = jnp.concatenate([dz[:1], 0.5 * (dz[:-1] + dz[1:])])
    dzw_hi = jnp.concatenate([0.5 * (dz[:-1] + dz[1:]), dz[-1:]])
    dz_eff = jnp.minimum(dzw_lo, dzw_hi)
    cz = (dz_eff ** 2 / dt)[:, None, None]                # (nz, 1, 1)

    return 0.46 / (1.0 / cx + 1.0 / cy + 1.0 / cz)       # (nz, ny, 1)


def _sgs_coefs(
    state:   ModelState,
    metric:  dict,
    params:  SGSParams,
    dt:      float,
    tabs0:   jax.Array | None = None,
    fluxbt:  jax.Array | None = None,   # D14: (ny,nx) surface heat flux (K·m/s)
    fluxbq:  jax.Array | None = None,   # D14: (ny,nx) surface moisture flux (kg/kg·m/s)
):
    """Shared helper: compute tk, tkh with stability cap."""
    U, V, W = state.U, state.V, state.W
    def2      = shear_prod(U, V, W, metric)
    TABS_arg  = state.TABS if tabs0 is not None else None
    # D13 fix: pass previous-step tk (stored in state.TKE) for smix limiter
    # D14 fix: pass surface fluxes to _compute_buoy_sgs via smag_viscosity
    tk, tkh   = smag_viscosity(def2, metric, params, TABS=TABS_arg, tabs0=tabs0,
                               QV=state.QV, QC=state.QC, QI=state.QI,
                               QR=state.QR, QS=state.QS, QG=state.QG,
                               tk_prev=state.TKE,
                               fluxbt=fluxbt, fluxbq=fluxbq)
    # C9 fix: per-(j,k) stability cap — computed here and applied locally inside
    # diffusion functions (Fix 2.5) rather than globally pre-capping tk/tkh.
    tk_max = _tkmax_3d(metric, dt)

    # Fix 2.10: gSAM tke_full.f90:51-89 — at nstep==1, floor tk at surface level
    # with the equilibrium value implied by surface buoyancy fluxes so the smix
    # limiter does not start from zero when the surface buoyancy flux is positive.
    #   tke_eq  = (grd / Cee * max(1e-20, 0.5*a_prod_bu)) ** (2/3)
    #   tk_eq   = Ck * grd * sqrt(tke_eq)
    # where a_prod_bu is the surface buoyancy production (same formula as Fix 2.1).
    if fluxbt is not None and tabs0 is not None:
        from jsam.core.physics.microphysics import G_GRAV
        _Ck    = params.Ck
        _Cs    = params.Cs
        _Ce    = _Ck**3 / _Cs**4          # Ces in gSAM (= Ce for tke_full)
        _Ce1   = _Ce / 0.7 * 0.19
        _Ce2   = _Ce / 0.7 * 0.51
        _Cee   = _Ce1 + _Ce2              # smix = grd at equilibrium → ratio = 1
        EPSV   = 0.61
        _bet0  = G_GRAV / tabs0[0]        # bet at surface level
        _fbq   = fluxbq if fluxbq is not None else jnp.zeros_like(fluxbt)
        _sst   = metric.get("sst", None)
        _sst_val = _sst if _sst is not None else tabs0[0]
        # Surface buoyancy production: a_prod_bu = bet * fluxbt + bet * epsv * sst * fluxbq
        _a_prod_bu = _bet0 * (fluxbt + EPSV * _sst_val * _fbq)   # (ny, nx)
        # grd at surface: (dz[0] * coef(j))^(1/3), coef = min(delta_max,dx*mu)*min(delta_max,dy)
        _dz_ref    = metric["dz_ref"]
        _dx_sfc    = metric["dx_lon"]
        _cos_lat10 = metric.get("cos_lat", None)
        _dy_sfc    = metric.get("dy_lat", None)
        if _cos_lat10 is not None:
            _dx_eff10 = jnp.minimum(params.delta_max, _dx_sfc * _cos_lat10)  # (ny,)
        else:
            _dx_eff10 = jnp.minimum(params.delta_max, _dx_sfc) * jnp.ones((1,))
        if _dy_sfc is not None:
            _dy_eff10 = jnp.minimum(params.delta_max, _dy_sfc)               # (ny,)
        else:
            _dy_eff10 = jnp.minimum(params.delta_max,
                                    metric.get("dy_lat_ref", _dx_sfc)) * jnp.ones((1,))
        _coef10   = _dx_eff10 * _dy_eff10                                    # (ny,)
        _grd10    = (_dz_ref * _coef10) ** 0.33333                    # (ny,)
        # tke_eq = (grd / Cee * max(1e-20, 0.5 * a_prod_bu)) ^ (2/3)  — per (ny, nx)
        _half_apb = jnp.maximum(1e-20, 0.5 * _a_prod_bu)                    # (ny, nx)
        _tke_eq   = (_grd10[:, None] / _Cee * _half_apb) ** (2.0 / 3.0)    # (ny, nx)
        # tk_eq at surface (k=0): Ck * grd * sqrt(tke_eq)
        _tk_eq    = _Ck * _grd10[:, None] * jnp.sqrt(_tke_eq)               # (ny, nx)
        # Apply only when a_prod_bu > 0 (equilibrium only for unstable surface)
        _tk_eq    = jnp.where(_a_prod_bu > 0.0, _tk_eq, 0.0)
        # Floor tk[0] with equilibrium when nstep == 1 (JIT-compatible via jnp.where)
        tk_eq_3d  = jnp.maximum(tk[0], _tk_eq)                              # (ny, nx)
        tk_sfc    = jnp.where(state.nstep == 1, tk_eq_3d, tk[0])
        tk        = tk.at[0].set(tk_sfc)
        # tkh follows Pr * tk
        tkh       = tkh.at[0].set(params.Pr * tk[0])

    return tk, tkh, tk_max


@functools.partial(jax.jit, static_argnames=("params",))
def sgs_mom_proc(
    state:   ModelState,
    metric:  dict,
    params:  SGSParams,
    dt:      float,
    surface: SurfaceFluxes | None = None,
) -> ModelState:
    """
    Apply SGS momentum diffusion only (U, V, W).

    Call BEFORE advance_momentum so SGS-adjusted velocities are used for
    advection (matches gSAM sgs.f90 + diffuse_mom3D.f90 → adamsA order).
    Scalars are left unchanged.
    """
    U, V, W = state.U, state.V, state.W
    tk, _, tk_max = _sgs_coefs(state, metric, params, dt)

    tau_x = None if surface is None else surface.tau_x
    tau_y = None if surface is None else surface.tau_y
    dU, dV, dW = diffuse_momentum(U, V, W, tk, metric, tau_x=tau_x, tau_y=tau_y,
                                   tk_max=tk_max)

    return ModelState(
        U    = U + dt * dU,
        V    = V + dt * dV,
        W    = W + dt * dW,
        TABS = state.TABS,
        QV   = state.QV,
        QC   = state.QC,
        QI   = state.QI,
        QR   = state.QR,
        QS   = state.QS,
        QG   = state.QG,
        TKE  = state.TKE,
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )


@functools.partial(jax.jit, static_argnames=("params",))
def sgs_scalars_proc(
    state:   ModelState,
    metric:  dict,
    params:  SGSParams,
    dt:      float,
    surface: SurfaceFluxes | None = None,
    tabs0:   jax.Array | None = None,
) -> ModelState:
    """
    Apply SGS scalar diffusion only (TABS, QV, QC, QI, QR, QS, QG).

    Call AFTER advance_scalars so SGS is applied to post-advection fields,
    matching gSAM operator order: advect_all_scalars → sgs_scalars.
    Momentum fields are left unchanged.
    """
    _, tkh, tk_max = _sgs_coefs(state, metric, params, dt, tabs0=tabs0)

    shf   = None if surface is None else surface.shf
    lhf   = None if surface is None else surface.lhf

    dTABS = diffuse_scalar(state.TABS, tkh, metric, fluxb=shf,  tk_max=tk_max)
    dQV   = diffuse_scalar(state.QV,   tkh, metric, fluxb=lhf,  tk_max=tk_max)
    dQC   = diffuse_scalar(state.QC,   tkh, metric, tk_max=tk_max)
    dQI   = diffuse_scalar(state.QI,   tkh, metric, tk_max=tk_max)
    dQR   = diffuse_scalar(state.QR,   tkh, metric, tk_max=tk_max)
    dQS   = diffuse_scalar(state.QS,   tkh, metric, tk_max=tk_max)
    dQG   = diffuse_scalar(state.QG,   tkh, metric, tk_max=tk_max)

    return ModelState(
        U    = state.U,
        V    = state.V,
        W    = state.W,
        TABS = state.TABS + dt * dTABS,
        QV   = state.QV + dt * dQV,
        QC   = state.QC + dt * dQC,
        QI   = state.QI + dt * dQI,
        QR   = state.QR + dt * dQR,
        QS   = state.QS + dt * dQS,
        QG   = state.QG + dt * dQG,
        TKE  = state.TKE,
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )
