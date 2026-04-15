"""
Smagorinsky SGS turbulence for jsam.

Port of gSAM SGS_TKE (Khairoutdinov 2012), Smagorinsky branch only
(dosmagor=True).  No prognostic TKE equation, no DNS, no terrain.

Eddy viscosity:
    tk  = Cs² · smix² · sqrt(max(0, def2))       (m²/s)
    tkh = Pr · tk                                  (m²/s)

where smix = (dz · dx_eff · dy_eff)^(1/3) is the filter length scale
and def2 = 2 S_ij S_ij is the strain-rate invariant from shear_prod().

Diffusion is explicit in time (forward Euler applied to the tendency).
Anelastic vertical diffusion uses rho/rhow weighting.
Horizontal diffusion uses imu = 1/cos(lat) for x only (Cartesian y approx).
Zero-flux BCs at all walls (bottom, top, y-walls).

References
----------
  SGS_TKE/tke_full.f90       — Smagorinsky viscosity computation
  SGS_TKE/shear_prod3D.f90   — strain-rate tensor
  SGS_TKE/diffuse_scalar3D.f90
  SGS_TKE/diffuse_mom3D.f90
"""
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
    """Smagorinsky SGS closure parameters (gSAM defaults)."""
    Cs:        float = 0.19     # Smagorinsky constant
    Ck:        float = 0.1      # (for reference; not used in pure Smagorinsky)
    Pr:        float = 1.0      # turbulent Prandtl number (tkh = Pr * tk)
    delta_max: float = 1000.0   # max horizontal length scale (m)


@jax.tree_util.register_pytree_node_class
@dataclass
class SurfaceFluxes:
    """
    Bottom-boundary (surface) flux fields, all shape (ny, nx).

    Sign convention: positive = upward flux from surface into atmosphere.

    Fields
    ------
    shf   : sensible heat flux  (K·m/s)
    lhf   : latent heat flux    (kg/kg·m/s)
    tau_x : x-momentum stress   (m²/s²) on cell-centre grid
    tau_y : y-momentum stress   (m²/s²) on cell-centre grid
    """
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
    """
    Distance between cell centres at each w-face, shape (nz+1,).
    dzw[0]  = dz[0]/2      (surface half-cell)
    dzw[k]  = (dz[k-1]+dz[k])/2   for k=1..nz-1
    dzw[nz] = dz[nz-1]/2  (top half-cell)
    """
    return jnp.concatenate([
        dz[:1] * 0.5,
        0.5 * (dz[:-1] + dz[1:]),
        dz[-1:] * 0.5,
    ])


# ---------------------------------------------------------------------------
# Strain-rate invariant  def2 = 2 S_ij S_ij
# ---------------------------------------------------------------------------

@jax.jit
def shear_prod(
    U: jax.Array,   # (nz, ny, nx+1)  u at x-faces
    V: jax.Array,   # (nz, ny+1, nx)  v at y-faces
    W: jax.Array,   # (nz+1, ny, nx)  w at z-faces
    metric: dict,
) -> jax.Array:     # def2 (nz, ny, nx)  s⁻²
    """
    Compute 2·S_ij·S_ij at cell centres.  Follows shear_prod3D.f90.
    Cartesian approximation for y (ady=adyv=1).
    """
    dx  = metric["dx_lon"]   # scalar (m)
    dy  = metric["dy_lat"]   # (ny,) per-row
    dz  = metric["dz"]       # (nz,)
    rdx = 1.0 / dx    # scalar — Cartesian (no cos(lat))
    # Per-row 1/dy for y-differences.  Strain-rate corner terms at j+½
    # faces should in principle use dy_v[j+½] = 0.5*(dy[j]+dy[j+1]); we
    # broadcast the mass-row dy for all (nz, ny, ·) tensors (matches gSAM
    # ady(j) — one factor per mass row).  On strongly non-uniform rows
    # the residual error is absorbed into the Cs calibration.
    rdy  = (1.0 / dy)[None, :, None]               # (1, ny, 1)
    dzw_ = _dzw(dz)                   # (nz+1,)

    # ---- Diagonal terms: 2*(S11² + S22² + S33²) ----
    # S11 = du/dx  at cell (k,j,i) using U[k,j,i] (left face) and U[k,j,i+1] (right face)
    S11 = (U[:, :, 1:] - U[:, :, :-1]) * rdx          # (nz,ny,nx)
    # S22 = dv/dy
    S22 = (V[:, 1:, :] - V[:, :-1, :]) * rdy           # (nz,ny,nx)
    # S33 = dw/dz
    S33 = (W[1:, :, :] - W[:-1, :, :]) / dz[:, None, None]   # (nz,ny,nx)

    diag = 2.0 * (S11**2 + S22**2 + S33**2)

    # ---- U-V cross terms: (du/dy + dv/dx)² averaged over 4 corners ----
    # Pad U in y (Neumann), V in x (periodic)
    Up = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')   # (nz,ny+2,nx+1)
    Vx = jnp.concatenate([V[:, :, -1:], V, V[:, :, :1]], axis=2)  # (nz,ny+1,nx+2)

    # Corner NE (i+½, j+½):
    dudy_NE = (Up[:, 2:,  1:] - Up[:, 1:-1,  1:]) * rdy
    dvdx_NE = (Vx[:, 1:, 2:] - Vx[:, 1:, 1:-1]) * rdx
    # Corner NW (i-½, j+½):
    dudy_NW = (Up[:, 2:, :-1] - Up[:, 1:-1, :-1]) * rdy
    dvdx_NW = (Vx[:, 1:, 1:-1] - Vx[:, 1:, :-2]) * rdx
    # Corner SE (i+½, j-½):
    dudy_SE = (Up[:, 1:-1,  1:] - Up[:, :-2,  1:]) * rdy
    dvdx_SE = (Vx[:, :-1, 2:] - Vx[:, :-1, 1:-1]) * rdx
    # Corner SW (i-½, j-½):
    dudy_SW = (Up[:, 1:-1, :-1] - Up[:, :-2, :-1]) * rdy
    dvdx_SW = (Vx[:, :-1, 1:-1] - Vx[:, :-1, :-2]) * rdx

    cross_uv = 0.25 * (
        (dudy_NE + dvdx_NE)**2 + (dudy_NW + dvdx_NW)**2 +
        (dudy_SE + dvdx_SE)**2 + (dudy_SW + dvdx_SW)**2
    )

    # ---- U-W cross terms: (du/dz + dw/dx)² averaged over 4 corners ----
    Uz = jnp.pad(U, ((1, 1), (0, 0), (0, 0)), mode='edge')   # (nz+2,ny,nx+1)
    Wx = jnp.concatenate([W[:, :, -1:], W, W[:, :, :1]], axis=2)  # (nz+1,ny,nx+2)

    dzw_above = dzw_[1:][:, None, None]   # (nz,1,1) w-face k+1
    dzw_below = dzw_[:-1][:, None, None]  # (nz,1,1) w-face k

    # du/dz at w-face above (k+1), at U-face i+1 and i:
    dudz_ab_ip1 = (Uz[2:, :, 1:] - Uz[1:-1, :, 1:]) / dzw_above
    dudz_ab_i   = (Uz[2:, :, :-1] - Uz[1:-1, :, :-1]) / dzw_above
    # du/dz at w-face below (k), at U-face i+1 and i:
    dudz_bel_ip1 = (Uz[1:-1, :, 1:] - Uz[:-2, :, 1:]) / dzw_below
    dudz_bel_i   = (Uz[1:-1, :, :-1] - Uz[:-2, :, :-1]) / dzw_below
    # dw/dx at w-face above (k+1) and below (k):
    dwdx_above = (Wx[1:, :, 2:] - Wx[1:, :, 1:-1]) * rdx    # (nz,ny,nx)
    dwdx_below = (Wx[:-1, :, 2:] - Wx[:-1, :, 1:-1]) * rdx

    cross_uw = 0.25 * (
        (dudz_ab_ip1  + dwdx_above)**2 + (dudz_ab_i   + dwdx_above)**2 +
        (dudz_bel_ip1 + dwdx_below)**2 + (dudz_bel_i  + dwdx_below)**2
    )

    # ---- V-W cross terms: (dv/dz + dw/dy)² averaged over 4 corners ----
    Vz = jnp.pad(V, ((1, 1), (0, 0), (0, 0)), mode='edge')   # (nz+2,ny+1,nx)
    Wy = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')   # (nz+1,ny+2,nx)

    # dv/dz at w-face above (k+1), at V-face j+1 and j:
    dvdz_ab_jp1 = (Vz[2:, 1:, :] - Vz[1:-1, 1:, :]) / dzw_above
    dvdz_ab_j   = (Vz[2:, :-1, :] - Vz[1:-1, :-1, :]) / dzw_above
    # dv/dz at w-face below (k):
    dvdz_bel_jp1 = (Vz[1:-1, 1:, :] - Vz[:-2, 1:, :]) / dzw_below
    dvdz_bel_j   = (Vz[1:-1, :-1, :] - Vz[:-2, :-1, :]) / dzw_below
    # dw/dy:
    dwdy_above = (Wy[1:, 2:, :] - Wy[1:, 1:-1, :]) * rdy
    dwdy_below = (Wy[:-1, 2:, :] - Wy[:-1, 1:-1, :]) * rdy

    cross_vw = 0.25 * (
        (dvdz_ab_jp1  + dwdy_above)**2 + (dvdz_ab_j   + dwdy_above)**2 +
        (dvdz_bel_jp1 + dwdy_below)**2 + (dvdz_bel_j  + dwdy_below)**2
    )

    return diag + cross_uv + cross_uw + cross_vw


# ---------------------------------------------------------------------------
# Smagorinsky eddy viscosity / diffusivity
# ---------------------------------------------------------------------------

def _compute_buoy_sgs(
    TABS:   jax.Array,     # (nz, ny, nx) K
    tabs0:  jax.Array,     # (nz,) K reference profile
    z:      jax.Array,     # (nz,) m cell-centre heights
    g_cp:   float = 9.79764 / 1004.64,
    g:      float = 9.79764,
) -> jax.Array:
    """
    gSAM-style SGS buoyancy frequency squared at cell centres, dry limit.

    buoy_sgs = (g / tabs0) * d(t_liq)/dz   where t_liq ≈ TABS + (g/cp)·z
             ≈ Brunt–Väisälä frequency squared (>0 stable, <0 unstable).

    Ignores moisture and cloud latent contributions (sufficient for
    stratosphere/upper troposphere where W runaway happens).  Returns
    a (nz, ny, nx) array, averaged from face values to cell centres.
    """
    t_liq = TABS + g_cp * z[:, None, None]                          # (nz,ny,nx)

    dz_c = (z[1:] - z[:-1])[:, None, None]                          # (nz-1,1,1)
    tabs0_face = 0.5 * (tabs0[1:] + tabs0[:-1])[:, None, None]      # (nz-1,1,1)
    buoy_face = (g / tabs0_face) * (t_liq[1:] - t_liq[:-1]) / dz_c  # (nz-1,ny,nx)

    # Face → cell-centre average (k=0 and k=nz-1 use nearest face)
    buoy_centre = jnp.concatenate(
        [buoy_face[:1],
         0.5 * (buoy_face[:-1] + buoy_face[1:]),
         buoy_face[-1:]],
        axis=0,
    )
    return buoy_centre


@functools.partial(jax.jit, static_argnames=("params",))
def smag_viscosity(
    def2:   jax.Array,   # (nz, ny, nx)  s⁻²
    metric: dict,
    params: SGSParams,
    TABS:   jax.Array | None = None,   # (nz, ny, nx) K
    tabs0:  jax.Array | None = None,   # (nz,) K reference profile
) -> tuple[jax.Array, jax.Array]:
    """
    Returns (tk, tkh), both (nz, ny, nx) in m²/s.

    Pure Smagorinsky (TABS/tabs0 None):
        tk = (Cs·smix)² · sqrt(max(0, def2))

    gSAM dosmagor=True with buoyancy suppression (TABS and tabs0 given):
        buoy = N²  (>0 stable)
        if buoy > 0:  smix = min(grd, max(0.1·grd, √(0.76·tk_iter/Ck/√buoy)))
        Cee  = (Ce/0.7) · (0.19 + 0.51·smix/grd),   Ce = Ck³/Cs⁴
        tk   = √(Ck³/Cee · max(0, def2 − Pr·buoy)) · smix²
    tk_iter is one fixed-point iteration with smix₀ = grd, needed because
    gSAM uses the previous-step tk in the smix limiter.
    """
    dx        = metric["dx_lon"]
    dy        = metric["dy_lat"]   # (ny,) per-row
    dz        = metric["dz"]   # (nz,)
    delta_max = params.delta_max
    Cs        = params.Cs
    Ck        = params.Ck
    Pr        = params.Pr

    # Cartesian: uniform equatorial dx everywhere (no cos(lat) shrinkage).
    # Per-row dy → per-row dy_eff → per-row mixing length.
    dx_eff = jnp.minimum(delta_max, dx)                         # scalar
    dy_eff = jnp.minimum(delta_max, dy)[None, :, None]          # (1, ny, 1)

    grd = (dz[:, None, None] * dx_eff * dy_eff) ** (1.0 / 3.0)

    if TABS is None or tabs0 is None:
        # Legacy pure-Smagorinsky path (no buoyancy suppression)
        tk  = Cs**2 * grd**2 * jnp.sqrt(jnp.maximum(0.0, def2))
        tkh = params.Pr * tk
        return tk, tkh

    # --- gSAM dosmagor=True branch with buoyancy suppression ---
    z         = metric["z"]
    buoy_sgs  = _compute_buoy_sgs(TABS, tabs0, z)

    # Constants (gSAM tke_full.f90:28-31)
    Ce  = Ck**3 / Cs**4                 # ≈ 0.7673
    Ce1 = Ce / 0.7 * 0.19
    Ce2 = Ce / 0.7 * 0.51

    def2_adj = jnp.maximum(0.0, def2 - Pr * buoy_sgs)

    # Fixed-point iter 0: smix = grd, Cee = Ce1 + Ce2 = Ce
    tk_iter = jnp.sqrt(Ck**3 / Ce * def2_adj) * grd**2

    # gSAM smix limiter for stable layers (buoy > 0)
    stable = buoy_sgs > 0.0
    smix_stable = jnp.minimum(
        grd,
        jnp.maximum(
            0.1 * grd,
            jnp.sqrt(0.76 * tk_iter / Ck / jnp.sqrt(buoy_sgs + 1e-10)),
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
) -> jax.Array:
    """
    Explicit SGS diffusion tendency d(field)/dt (same units as field / s).
    Horizontal: second-order centred with imu² metric in x.
    Vertical:   anelastic flux form with rho/rhow weighting.
    BCs: zero-flux at y-walls; fluxb/fluxt at z-walls (default 0).
    """
    dx  = metric["dx_lon"]
    dy  = metric["dy_lat"]   # (ny,) per-row
    dz  = metric["dz"]       # (nz,)
    rho  = metric["rho"]     # (nz,)
    rhow = metric["rhow"]    # (nz+1,)
    dzw_ = _dzw(dz)           # (nz+1,)
    nz, ny, nx = field.shape

    # ---- Horizontal x-direction — Cartesian (no imu) ----
    field_xp = jnp.concatenate([field[:, :, -1:], field, field[:, :, :1]], axis=2)  # (nz,ny,nx+2)
    tkh_xp   = jnp.concatenate([tkh[:, :, -1:],  tkh,  tkh[:, :, :1]],  axis=2)
    rdx2     = (1.0 / dx) ** 2                                             # scalar
    tkh_fx   = 0.5 * (tkh_xp[:, :, :-1] + tkh_xp[:, :, 1:])              # (nz,ny,nx+1) x-faces
    flx_x    = -rdx2 * tkh_fx * (field_xp[:, :, 1:] - field_xp[:, :, :-1])  # (nz,ny,nx+1)
    dfdt     = -(flx_x[:, :, 1:] - flx_x[:, :, :-1])                     # (nz,ny,nx)

    # ---- Horizontal y-direction (non-uniform) ----
    # Pad field and tkh in y with edge (Neumann → zero flux at walls).
    # Non-uniform form: flux at v-face uses dy_v = 0.5*(dy[j-1]+dy[j]);
    # divergence at mass cell uses dy_row[j].  On a uniform grid both
    # equal dy so this collapses to rdy² = 1/dy².
    field_yp = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode='edge')      # (nz,ny+2,nx)
    tkh_yp   = jnp.pad(tkh,   ((0, 0), (1, 1), (0, 0)), mode='edge')
    dy_v_int = 0.5 * (dy[:-1] + dy[1:])                                   # (ny-1,)
    # v-face spacings at every face (ny+1):  at boundaries duplicate edge.
    dy_v_full = jnp.concatenate([dy_v_int[:1], dy_v_int, dy_v_int[-1:]])  # (ny+1,)
    inv_dy_v  = (1.0 / dy_v_full)[None, :, None]                          # (1, ny+1, 1)
    inv_dy_r  = (1.0 / dy)[None, :, None]                                 # (1, ny, 1)
    tkh_fy   = 0.5 * (tkh_yp[:, :-1, :] + tkh_yp[:, 1:, :])              # (nz,ny+1,nx) y-faces
    flx_y    = -inv_dy_v * tkh_fy * (field_yp[:, 1:, :] - field_yp[:, :-1, :])   # (nz,ny+1,nx)
    dfdt     = dfdt - inv_dy_r * (flx_y[:, 1:, :] - flx_y[:, :-1, :])

    # ---- Vertical — anelastic: d(rhow*tkh*dfield/dz)/(rho*dz) ----
    # Interior fluxes at w-faces 1..nz-1
    tkh_fz    = 0.5 * (tkh[:-1, :, :] + tkh[1:, :, :])                    # (nz-1,ny,nx)
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns (dU/dt, dV/dt, dW/dt) tendencies from SGS diffusion (s⁻²).
    Follows diffuse_mom3D.f90 structure.  Simplified: uses nearest cell-centre
    tk without the 4-point corner interpolation in gSAM (adequate for CRM).
    Anelastic vertical; Cartesian horizontal.

    tau_x, tau_y: surface (bottom) momentum flux (m²/s²), positive upward.
    They are interpolated from cell-centre positions to U/V staggered faces.
    Default (None) = zero-flux bottom BC.
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

    # Cartesian: uniform 1/dx² everywhere (no imu)
    rdx2u = (1.0 / dx)**2   # scalar — same for U, V, W grids
    rdx2v = rdx2u
    rdx2w = rdx2u
    rdy2  = 1.0 / dy_ref**2

    # Helper: interpolate cell-centre tk to U x-faces (periodic in x)
    # tk_Ux[k,j,i] ≈ mean of cell tk to the left and right of U-face i
    tk_Ux = 0.5 * (jnp.roll(tk, 1, axis=2) + tk)  # (nz,ny,nx), valid at face i using tk[i-1] and tk[i]
    # Extend to nx+1 with periodic wrap: face nx same as face 0
    tk_at_U = jnp.concatenate([tk_Ux, tk_Ux[:, :, :1]], axis=2)          # (nz,ny,nx+1)
    # Interpolate to V y-faces (Neumann pad): tk_at_V[k,j,i] = mean of cell rows j-1 and j
    tk_yp   = jnp.pad(tk, ((0, 0), (1, 0), (0, 0)), mode='edge')          # (nz,ny+1,nx)
    tk_at_V = 0.5 * (tk_yp[:, :-1, :] + tk_yp[:, 1:, :])                 # (nz,ny,nx) at V interior rows 1..ny-1
    tk_at_V = jnp.pad(tk_at_V, ((0, 0), (0, 1), (0, 0)), mode='edge')    # (nz,ny+1,nx) add top row
    # Interpolate to W z-faces (edge pad): tk_at_W[k,j,i] = mean of cells k-1 and k
    tk_at_W = jnp.pad(
        0.5 * (tk[:-1, :, :] + tk[1:, :, :]),                             # (nz-1,ny,nx) interior
        ((1, 1), (0, 0), (0, 0)), mode='edge',
    )                                                                      # (nz+1,ny,nx)

    rho_dz  = (rho * dz)[:, None, None]                                   # (nz,1,1)
    rhow_full = rhow[:, None, None]                                        # (nz+1,1,1)

    # ======== dU/dt: shape (nz, ny, nx+1) ========

    # x: flux between adjacent U-faces at cell centres (periodic)
    F_ux = -rdx2u * tk * (U[:, :, 1:] - U[:, :, :-1])                    # (nz,ny,nx)  F[i] at cell i
    dUdt = jnp.roll(F_ux, 1, axis=2) - F_ux                               # (nz,ny,nx)  = F[i-1]-F[i]
    dUdt = jnp.concatenate([dUdt, dUdt[:, :, :1]], axis=2)                # (nz,ny,nx+1) periodic

    # y: flux at y-interfaces between U-face rows (Neumann at poles)
    U_yp = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')              # (nz,ny+2,nx+1)
    tkU_yp = jnp.pad(tk_at_U, ((0, 0), (1, 1), (0, 0)), mode='edge')     # (nz,ny+2,nx+1)
    fy_U = (-rdy2 * 0.5 * (tkU_yp[:, :-1, :] + tkU_yp[:, 1:, :])
            * (U_yp[:, 1:, :] - U_yp[:, :-1, :]))                        # (nz,ny+1,nx+1)
    dUdt = dUdt - (fy_U[:, 1:, :] - fy_U[:, :-1, :])

    # z: anelastic flux between U levels (zero-flux at top/bottom)
    dzw_int  = dzw_[1:-1][:, None, None]                                  # (nz-1,1,1)
    rhow_int = rhow[1:-1][:, None, None]                                   # (nz-1,1,1)
    tkUz = 0.5 * (tk_at_U[:-1, :, :] + tk_at_U[1:, :, :])                # (nz-1,ny,nx+1) at interior w-faces
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
    fx_V = (-rdx2v * 0.5 * (tkV_xp[:, :, :-1] + tkV_xp[:, :, 1:])
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
    fy_V = (-rdy2 * 0.5 * (tkV_yp[:, :-1, :] + tkV_yp[:, 1:, :])
            * (V_yp[:, 1:, :] - V_yp[:, :-1, :]))                        # (nz,ny+2,nx)
    dVdt = dVdt - (fy_V[:, 1:, :] - fy_V[:, :-1, :])                     # (nz,ny+1,nx)

    # z: anelastic
    tkVz = 0.5 * (tk_at_V[:-1, :, :] + tk_at_V[1:, :, :])                # (nz-1,ny+1,nx)
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
    fx_W = (-rdx2w * 0.5 * (tkW_xp[:, :, :-1] + tkW_xp[:, :, 1:])
            * (W_xp[:, :, 1:] - W_xp[:, :, :-1]))                        # (nz+1,ny,nx+1)
    dWdt = -(fx_W[:, :, 1:] - fx_W[:, :, :-1])                           # (nz+1,ny,nx)

    # y: Neumann pad
    W_yp  = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')             # (nz+1,ny+2,nx)
    tkW_yp = jnp.pad(tk_at_W, ((0, 0), (1, 1), (0, 0)), mode='edge')     # (nz+1,ny+2,nx)
    fy_W = (-rdy2 * 0.5 * (tkW_yp[:, :-1, :] + tkW_yp[:, 1:, :])
            * (W_yp[:, 1:, :] - W_yp[:, :-1, :]))                        # (nz+1,ny+1,nx)
    dWdt = dWdt - (fy_W[:, 1:, :] - fy_W[:, :-1, :])

    # z: second-difference of W at w-face positions (W is already at w-faces)
    # Flux between W[k] and W[k+1]: use tk_at_W averaged across those two faces
    dW_dz = W[1:, :, :] - W[:-1, :, :]                                    # (nz,ny,nx)
    tk_Wz2 = 0.5 * (tk_at_W[:-1, :, :] + tk_at_W[1:, :, :])              # (nz,ny,nx)
    fz_Wint = -tk_Wz2 * dW_dz / dz[:, None, None]                         # (nz,ny,nx)
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

    rdx2 = (1.0 / dx)**2
    rdy2 = 1.0 / dy_ref**2

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

    U_yp = jnp.pad(U, ((0, 0), (1, 1), (0, 0)), mode='edge')
    tkU_yp = jnp.pad(tk_at_U, ((0, 0), (1, 1), (0, 0)), mode='edge')
    fy_U = (-rdy2 * 0.5 * (tkU_yp[:, :-1, :] + tkU_yp[:, 1:, :])
            * (U_yp[:, 1:, :] - U_yp[:, :-1, :]))
    dUdt = dUdt - (fy_U[:, 1:, :] - fy_U[:, :-1, :])

    # ======== dV/dt (horiz only) ========
    V_xp = jnp.concatenate([V[:, :, -1:], V, V[:, :, :1]], axis=2)
    tkV_xp = jnp.concatenate(
        [tk_at_V[:, :, -1:], tk_at_V, tk_at_V[:, :, :1]], axis=2)
    fx_V = (-rdx2 * 0.5 * (tkV_xp[:, :, :-1] + tkV_xp[:, :, 1:])
            * (V_xp[:, :, 1:] - V_xp[:, :, :-1]))
    dVdt = -(fx_V[:, :, 1:] - fx_V[:, :, :-1])

    # Antisymmetric wall mirror for V (gSAM boundaries.f90:101-129).
    V_yp = jnp.concatenate(
        [-V[:, 1:2, :], V, -V[:, -2:-1, :]], axis=1,
    )
    tkV_yp = jnp.pad(tk_at_V, ((0, 0), (1, 1), (0, 0)), mode='edge')
    fy_V = (-rdy2 * 0.5 * (tkV_yp[:, :-1, :] + tkV_yp[:, 1:, :])
            * (V_yp[:, 1:, :] - V_yp[:, :-1, :]))
    dVdt = dVdt - (fy_V[:, 1:, :] - fy_V[:, :-1, :])
    dVdt = dVdt.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)

    # ======== dW/dt (horiz only) ========
    W_xp = jnp.concatenate([W[:, :, -1:], W, W[:, :, :1]], axis=2)
    tkW_xp = jnp.concatenate(
        [tk_at_W[:, :, -1:], tk_at_W, tk_at_W[:, :, :1]], axis=2)
    fx_W = (-rdx2 * 0.5 * (tkW_xp[:, :, :-1] + tkW_xp[:, :, 1:])
            * (W_xp[:, :, 1:] - W_xp[:, :, :-1]))
    dWdt = -(fx_W[:, :, 1:] - fx_W[:, :, :-1])

    W_yp  = jnp.pad(W, ((0, 0), (1, 1), (0, 0)), mode='edge')
    tkW_yp = jnp.pad(tk_at_W, ((0, 0), (1, 1), (0, 0)), mode='edge')
    fy_W = (-rdy2 * 0.5 * (tkW_yp[:, :-1, :] + tkW_yp[:, 1:, :])
            * (W_yp[:, 1:, :] - W_yp[:, :-1, :]))
    dWdt = dWdt - (fy_W[:, 1:, :] - fy_W[:, :-1, :])

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
    dz   = metric["dz"]        # (nz,)
    rho  = metric["rho"]       # (nz,)
    rhow = metric["rhow"]      # (nz+1,)
    z    = metric["z"]          # (nz,) cell centres
    cos_lat = metric["cos_lat"] # (ny,)
    dx   = metric["dx_lon"]     # scalar
    pres = metric["pres"]       # (nz,) Pa

    tau_max = 1.0 / dt

    # dzw[f] = z[f] - z[f-1] for interior w-faces f=0..nz-2
    # (distance between cell centres f and f+1)
    dzw = z[1:] - z[:-1]     # (nz-1,)
    rhow_int = rhow[1:-1]    # (nz-1,) at interior w-faces

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

    # Tridiagonal coefficients
    # c[k]: coefficient for U[k+1], at face between k and k+1
    #   = -dt * rhow[k+1] * tkz_u[k] / (dzw[k] * dz[k] * rho[k])
    c_vals = (-dt * rhow_int[:, None, None] * tkz_u
              / (dzw[:, None, None] * dz[:-1, None, None] * rho[:-1, None, None]))
    c_u = jnp.concatenate([c_vals, jnp.zeros((1, ny, nx + 1))], axis=0)

    # a[k]: coefficient for U[k-1], at face between k-1 and k
    #   = -dt * rhow[k] * tkz_u[k-1] / (dzw[k-1] * dz[k] * rho[k])
    a_vals = (-dt * rhow_int[:, None, None] * tkz_u
              / (dzw[:, None, None] * dz[1:, None, None] * rho[1:, None, None]))
    a_u = jnp.concatenate([jnp.zeros((1, ny, nx + 1)), a_vals], axis=0)

    b_u = 1.0 + dt * tau_vel_u - a_u - c_u
    d_u = U + dt * tau_vel_u * vel0_u

    # Surface flux BC at k = k_terrau(i,j).  jsam is flat-only so
    # k_terrau ≡ 0 everywhere, matching gSAM diffuse_damping_mom_z.f90:121-138
    # for the flat-terrain limit:
    #     d(i,j,k_terrau) += dtn*rhow(k)/(dz*adz(k)*rho(k))*fluxbu(i,j)
    # In jsam the metric dz[0] already equals dz_ref*adz(0), so the flux
    # coefficient is rhow[0] / (rho[0] * dz[0]).
    if fluxbu is not None:
        d_u = d_u.at[0].add(dt * rhow[0] / (rho[0] * dz[0]) * fluxbu)

    U_new = _thomas_solve(a_u, b_u, c_u, d_u)

    # ================================================================
    # V implicit solve  (nz, ny+1, nx)
    # ================================================================

    # SGS viscosity interpolated to V y-faces, then averaged to w-faces
    tk_yp = jnp.pad(tk, ((0, 0), (1, 0), (0, 0)), mode='edge')
    tk_vf = 0.5 * (tk_yp[:, :-1, :] + tk_yp[:, 1:, :])
    tk_vf = jnp.pad(tk_vf, ((0, 0), (0, 1), (0, 0)), mode='edge')  # (nz, ny+1, nx)
    tkz_v = 0.5 * (tk_vf[:-1] + tk_vf[1:])  # (nz-1, ny+1, nx)

    # Polar + upper-level damping for V (use cos at v-face latitudes)
    # V y-faces are between mass latitudes: cos_v ≈ cos at half-lat
    cos_v_half = jnp.pad(
        0.5 * (cos_lat[:-1] + cos_lat[1:]),
        (1, 1), mode='edge',
    )  # (ny+1,)
    tauy_v = tau_max * (1.0 - cos_v_half ** 2) ** 200
    umax_v = damping_u_cu * dx * cos_v_half / dt
    umax_v3d = umax_v[None, :, None]

    tau_base_v = jnp.where(
        pres_hpa[:, None] < 70.0, tau_max, tauy_v[None, :],
    )
    V_exceeds = jnp.abs(V) > umax_v3d
    tau_vel_v = jnp.where(V_exceeds, tau_base_v[:, :, None], 0.0)
    vel0_v = jnp.where(V > umax_v3d, umax_v3d,
                        jnp.where(V < -umax_v3d, -umax_v3d, 0.0))

    c_vals_v = (-dt * rhow_int[:, None, None] * tkz_v
                / (dzw[:, None, None] * dz[:-1, None, None] * rho[:-1, None, None]))
    c_v = jnp.concatenate([c_vals_v, jnp.zeros((1, ny + 1, nx))], axis=0)

    a_vals_v = (-dt * rhow_int[:, None, None] * tkz_v
                / (dzw[:, None, None] * dz[1:, None, None] * rho[1:, None, None]))
    a_v = jnp.concatenate([jnp.zeros((1, ny + 1, nx)), a_vals_v], axis=0)

    b_v = 1.0 + dt * tau_vel_v - a_v - c_v
    d_v = V + dt * tau_vel_v * vel0_v

    # Surface flux BC at k = k_terrav(i,j) ≡ 0 for flat-only jsam.
    # Matches gSAM diffuse_damping_mom_z.f90:243-260 in the flat limit.
    if fluxbv is not None:
        d_v = d_v.at[0].add(dt * rhow[0] / (rho[0] * dz[0]) * fluxbv)

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
    # wmax = damping_w_cu * dzw / dt at each interior w-face
    wmax_w = damping_w_cu * dzw / dt  # (nz-1,)
    W_int = W[1:-1]  # (nz-1, ny, nx) interior w-faces

    below_sponge = (tauz_w == 0.0)[:, None, None]
    W_exceeds_w = jnp.abs(W_int) > wmax_w[:, None, None]
    tau_vel_w = jnp.where(below_sponge & W_exceeds_w, tau_max, 0.0)
    vel0_w = jnp.where(
        W_int > wmax_w[:, None, None], wmax_w[:, None, None],
        jnp.where(W_int < -wmax_w[:, None, None], -wmax_w[:, None, None], 0.0),
    )

    # For W at interior face f (0-indexed in the nz-1 system, = w-face f+1):
    # cell below = f, cell above = f+1
    # tkz_below = tk[f], tkz_above = tk[f+1]
    # dzw_f = z[f+1] - z[f]
    #
    # a[f] = -dt * tk[f]   * rho[f]   / (dzw_f * dz[f]   * rhow[f+1])
    # c[f] = -dt * tk[f+1] * rho[f+1] / (dzw_f * dz[f+1] * rhow[f+1])
    #
    # But f goes from 0..nz-2 in this system.
    # Cell below of system-f = cell f (0-indexed)
    # Cell above of system-f = cell f+1

    tk_below = tk[:-1]   # (nz-1, ny, nx) = tk at cells 0..nz-2
    tk_above = tk[1:]    # (nz-1, ny, nx) = tk at cells 1..nz-1
    rhow_w   = rhow[1:-1]  # (nz-1,) density at interior w-faces

    a_w_vals = (-dt * tk_below * rho[:-1, None, None]
                / (dzw[:, None, None] * dz[:-1, None, None] * rhow_w[:, None, None]))
    c_w_vals = (-dt * tk_above * rho[1:, None, None]
                / (dzw[:, None, None] * dz[1:, None, None] * rhow_w[:, None, None]))

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
    """
    nz, ny, nx = field.shape
    dz   = metric["dz"]
    rho  = metric["rho"]
    rhow = metric["rhow"]
    z    = metric["z"]

    dzw = z[1:] - z[:-1]      # (nz-1,)
    rhow_int = rhow[1:-1]     # (nz-1,)

    # tkh at interior w-faces: average of adjacent cell-centre values
    tkh_fz = 0.5 * (tkh[:-1] + tkh[1:])  # (nz-1, ny, nx)

    # Tridiagonal coefficients (same structure as momentum but no damping)
    c_vals = (-dt * rhow_int[:, None, None] * tkh_fz
              / (dzw[:, None, None] * dz[:-1, None, None] * rho[:-1, None, None]))
    c_f = jnp.concatenate([c_vals, jnp.zeros((1, ny, nx))], axis=0)

    a_vals = (-dt * rhow_int[:, None, None] * tkh_fz
              / (dzw[:, None, None] * dz[1:, None, None] * rho[1:, None, None]))
    a_f = jnp.concatenate([jnp.zeros((1, ny, nx)), a_vals], axis=0)

    b_f = 1.0 - a_f - c_f
    d_f = field.copy()

    # Surface flux BC at k=0
    if fluxb is not None:
        d_f = d_f.at[0].add(dt * rhow[0] / (rho[0] * dz[0]) * fluxb)
    # Top flux BC at k=nz-1
    if fluxt is not None:
        d_f = d_f.at[-1].add(-dt * rhow[-1] / (rho[-1] * dz[-1]) * fluxt)

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

    # Stability cap for explicit forward-Euler diffusion.
    # Vertical is the most restrictive dimension at GCM resolution (dz << dx,dy).
    # Constraint: K * dt / dz² ≤ 0.5  →  K_max = 0.5 * dz² / dt
    # Without this, Smagorinsky tk ≈ O(1e5) m²/s at 2° resolution causes
    # the diffusion tendency to grow faster than dt can damp it.
    dz_3d  = metric["dz"][:, None, None]           # (nz,1,1)
    tk_max = 0.5 * dz_3d ** 2 / dt                  # (nz,1,1)  m²/s
    tk     = jnp.minimum(tk,  tk_max)
    tkh    = jnp.minimum(tkh, tk_max)

    # Unpack surface fluxes (None → zero BC handled inside diffuse_* functions)
    shf   = None if surface is None else surface.shf
    lhf   = None if surface is None else surface.lhf
    tau_x = None if surface is None else surface.tau_x
    tau_y = None if surface is None else surface.tau_y

    # Scalar diffusion tendencies
    dTABS = diffuse_scalar(state.TABS, tkh, metric, fluxb=shf)
    dQV   = diffuse_scalar(state.QV,   tkh, metric, fluxb=lhf)
    dQC   = diffuse_scalar(state.QC,   tkh, metric)
    dQI   = diffuse_scalar(state.QI,   tkh, metric)
    dQR   = diffuse_scalar(state.QR,   tkh, metric)
    dQS   = diffuse_scalar(state.QS,   tkh, metric)
    dQG   = diffuse_scalar(state.QG,   tkh, metric)

    # Momentum diffusion tendencies
    dU, dV, dW = diffuse_momentum(U, V, W, tk, metric, tau_x=tau_x, tau_y=tau_y)

    return ModelState(
        U    = U     + dt * dU,
        V    = V     + dt * dV,
        W    = W     + dt * dW,
        TABS = state.TABS + dt * dTABS,
        QV   = jnp.maximum(0.0, state.QV + dt * dQV),
        QC   = jnp.maximum(0.0, state.QC + dt * dQC),
        QI   = jnp.maximum(0.0, state.QI + dt * dQI),
        QR   = jnp.maximum(0.0, state.QR + dt * dQR),
        QS   = jnp.maximum(0.0, state.QS + dt * dQS),
        QG   = jnp.maximum(0.0, state.QG + dt * dQG),
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

def _sgs_coefs(
    state:   ModelState,
    metric:  dict,
    params:  SGSParams,
    dt:      float,
    tabs0:   jax.Array | None = None,
):
    """Shared helper: compute tk, tkh with stability cap."""
    U, V, W = state.U, state.V, state.W
    def2      = shear_prod(U, V, W, metric)
    TABS_arg  = state.TABS if tabs0 is not None else None
    tk, tkh   = smag_viscosity(def2, metric, params, TABS=TABS_arg, tabs0=tabs0)
    dz_3d  = metric["dz"][:, None, None]
    tk_max = 0.5 * dz_3d ** 2 / dt
    tk     = jnp.minimum(tk,  tk_max)
    tkh    = jnp.minimum(tkh, tk_max)
    return tk, tkh


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
    tk, _   = _sgs_coefs(state, metric, params, dt)

    tau_x = None if surface is None else surface.tau_x
    tau_y = None if surface is None else surface.tau_y
    dU, dV, dW = diffuse_momentum(U, V, W, tk, metric, tau_x=tau_x, tau_y=tau_y)

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
    _, tkh  = _sgs_coefs(state, metric, params, dt, tabs0=tabs0)

    shf   = None if surface is None else surface.shf
    lhf   = None if surface is None else surface.lhf

    dTABS = diffuse_scalar(state.TABS, tkh, metric, fluxb=shf)
    dQV   = diffuse_scalar(state.QV,   tkh, metric, fluxb=lhf)
    dQC   = diffuse_scalar(state.QC,   tkh, metric)
    dQI   = diffuse_scalar(state.QI,   tkh, metric)
    dQR   = diffuse_scalar(state.QR,   tkh, metric)
    dQS   = diffuse_scalar(state.QS,   tkh, metric)
    dQG   = diffuse_scalar(state.QG,   tkh, metric)

    return ModelState(
        U    = state.U,
        V    = state.V,
        W    = state.W,
        TABS = state.TABS + dt * dTABS,
        QV   = jnp.maximum(0.0, state.QV + dt * dQV),
        QC   = jnp.maximum(0.0, state.QC + dt * dQC),
        QI   = jnp.maximum(0.0, state.QI + dt * dQI),
        QR   = jnp.maximum(0.0, state.QR + dt * dQR),
        QS   = jnp.maximum(0.0, state.QS + dt * dQS),
        QG   = jnp.maximum(0.0, state.QG + dt * dQG),
        TKE  = state.TKE,
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )
