"""
Scalar advection — 5th-order ULTIMATE-MACHO scheme (anelastic lat-lon).

Port of gSAM ADV_UM5/advect_um_lib.f90 + advect_scalar3D.f90.

Algorithm (MACHO case 2: x → y → z)
-------------------------------------
  fadv = phi.copy()
  1. face_x(fadv) → fx;  fadv += adv_cn_x * (fx_west − fx_east)   [x predictor]
  2. face_y(fadv) → fy;  fadv += adv_cn_y * (fy_south − fy_north)  [y predictor]
  3. face_z(fadv) → fz                                              [z faces, no update]
  4. phi_new = phi + flux_div_x(fx) + flux_div_y(fy) + flux_div_z(fz)

The two advective predictor steps improve face-value accuracy before the
single final flux-form update.  Unlike a plain sequential flux-form split,
no explicit divergence correction is needed.

adv_cn(cn_left, cn_right):
  = cn_left   if both > 0  (rightward/upward flow)
  = cn_right  if both < 0  (leftward/downward flow)
  = 0         otherwise     (convergent/divergent zone)

Face values use the 5th-order ULTIMATE formula with monotone limiter
(clip to [min(f_im1,f_i), max(f_im1,f_i)]).

Coordinate / velocity conventions
----------------------------------
  phi : (nz, ny, nx)      scalar at cell centres
  U   : (nz, ny, nx+1)    zonal velocity at east faces; U[...,0]=U[...,nx] periodic
  V   : (nz, ny+1, nx)    meridional velocity at north faces; V[:,0,:]=V[:,ny,:]=0
  W   : (nz+1, ny, nx)    vertical velocity at top faces; W[0,...]=W[nz,...]=0

References
----------
  gSAM ADV_UM5/advect_um_lib.f90   — face_5th(), adv_form_update_*(), advective_cn()
  gSAM ADV_UM5/advect_scalar3D.f90 — 3-D MACHO ordering and final flux-form
  Yamaguchi, Randall & Khairoutdinov (2011) MWR 139, 3248–3264
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jsam.core.grid.latlon import LatLonGrid


# ---------------------------------------------------------------------------
# 1.  Core face-value formula (matches gSAM face_5th exactly)
# ---------------------------------------------------------------------------

def _face5(
    f_im3: jax.Array,
    f_im2: jax.Array,
    f_im1: jax.Array,
    f_i:   jax.Array,
    f_ip1: jax.Array,
    f_ip2: jax.Array,
    cn:    jax.Array,
) -> jax.Array:
    """
    5th-order ULTIMATE face value at the im1/i interface.

    Matches gSAM face_5th() in ADV_UM5/advect_um_lib.f90.
    All args are JAX arrays of identical shape (vectorised over all faces).

    Args:
        f_im3 .. f_ip2 : cell values at positions im3 .. ip2
        cn             : Courant number at the face  (positive = flow im1 → i)

    Returns:
        face value at the im1/i interface (same shape as inputs)
    """
    d2 = f_ip1 - f_i - f_im1 + f_im2
    d3 = f_ip1 - 3*f_i + 3*f_im1 - f_im2
    d4 = f_ip2 - 3*f_ip1 + 2*f_i + 2*f_im1 - 3*f_im2 + f_im3
    d5 = f_ip2 - 5*f_ip1 + 10*f_i - 10*f_im1 + 5*f_im2 - f_im3
    face = 0.5 * (
        f_i + f_im1
        - cn * (f_i - f_im1)
        + (1/6)   * (cn**2 - 1) * (d2 - 0.5*cn*d3)
        + (1/120) * (cn**2 - 1) * (cn**2 - 4) * (d4 - jnp.sign(cn)*d5)
    )
    # ULTIMATE monotone limiter (gSAM advect_um_lib.f90):
    #   clip face value to [min(f_im1,f_i), max(f_im1,f_i)]
    # This prevents overshoots from the 5th-order stencil at the poles and
    # wherever the flow creates strong local convergence.
    lo = jnp.minimum(f_im1, f_i)
    hi = jnp.maximum(f_im1, f_i)
    return jnp.clip(face, lo, hi)


# ---------------------------------------------------------------------------
# 2.  Advective Courant number (gSAM advective_cn)
# ---------------------------------------------------------------------------

def _adv_cn(cn_left: jax.Array, cn_right: jax.Array) -> jax.Array:
    """
    Upwind advective Courant number for the MACHO predictor.

    Matches gSAM advect_um_lib.f90 advective_cn():
      if cn_right > 0 and cn_left >= 0 → cn_left   (rightward/upward flow)
      if cn_right <= 0 and cn_left < 0 → cn_right  (leftward/downward flow)
      else                              → 0          (convergent zone)
    """
    return jnp.where(
        (cn_right > 0.0) & (cn_left >= 0.0), cn_left,
        jnp.where((cn_right <= 0.0) & (cn_left < 0.0), cn_right, 0.0),
    )


# ---------------------------------------------------------------------------
# 3.  Public API — MACHO scalar advection
# ---------------------------------------------------------------------------

@jax.jit
def advect_scalar(
    phi:    jax.Array,   # (nz, ny, nx)  scalar field
    U:      jax.Array,   # (nz, ny, nx+1)  zonal velocity at east faces (m/s)
    V:      jax.Array,   # (nz, ny+1, nx)  meridional velocity at north faces (m/s)
    W:      jax.Array,   # (nz+1, ny, nx)  vertical velocity at top faces (m/s)
    metric: dict,
    dt:     float,
) -> jax.Array:          # (nz, ny, nx)  updated phi
    """
    One timestep of 5th-order ULTIMATE-MACHO scalar advection (anelastic lat-lon)
    with Zalesak FCT to enforce monotonicity and positivity.

    Implements gSAM advect_scalar3D case 2 (x→y→z) + fct3D.

    Algorithm:
      1. MACHO predictor: compute high-order face values (fx, fy, fz)
      2. FCT (Zalesak 1979):
         a. Compute 1st-order upwind fluxes → upwind update (monotone, diffusive)
         b. Antidiffusive flux = high-order flux − upwind flux
         c. Limit antidiffusive flux so result stays within local min/max
         d. Final update with positivity floor: max(0, f_upwind + div(limited_aflx))
    """
    nz, ny, _ = phi.shape

    dx    = metric["dx_lon"]
    dy    = metric["dy_lat"]   # (ny,) per-row
    rho   = metric["rho"]      # (nz,)
    rhow  = metric["rhow"]     # (nz+1,)
    dz    = metric["dz"]       # (nz,)

    irho = 1.0 / rho[:, None, None]           # (nz, 1, 1)
    iadz = 1.0 / dz[:, None, None]            # (nz, 1, 1)
    dy3   = dy[None, :, None]                  # (1, ny, 1)

    # Mass-weighted Courant numbers (= velocity * dt / dx, matching gSAM cu/cv/cw)
    cu_w = U[:, :, :-1]   * dt / dx            # west face of cell i  (nz, ny, nx)
    cu_e = jnp.roll(cu_w, -1, axis=2)          # east face of cell i
    cv_s = V[:, 0:ny,   :] * dt / dy3          # south face of cell j
    cv_n = V[:, 1:ny+1, :] * dt / dy3          # north face of cell j
    cw_t = W[1:nz+1, :, :] * dt * iadz         # top face of cell k   (nz, ny, nx)

    # ------------------------------------------------------------------ #
    # MACHO predictor — compute high-order face values                    #
    # ------------------------------------------------------------------ #
    fadv = phi

    # --- x: face values → advective update ---
    fx = _face5(
        jnp.roll(fadv,  3, axis=2),
        jnp.roll(fadv,  2, axis=2),
        jnp.roll(fadv,  1, axis=2),
        fadv,
        jnp.roll(fadv, -1, axis=2),
        jnp.roll(fadv, -2, axis=2),
        cu_w,
    )   # fx[:,j,i] = face value at west face of cell i
    fx_e = jnp.roll(fx, -1, axis=2)
    fadv = fadv + _adv_cn(cu_w, cu_e) * (fx - fx_e)

    # --- y: face values → advective update ---
    fp = jnp.pad(fadv, ((0, 0), (3, 3), (0, 0)), mode='edge')
    fy_s = _face5(
        fp[:, 0:ny, :], fp[:, 1:ny+1, :], fp[:, 2:ny+2, :],
        fp[:, 3:ny+3, :], fp[:, 4:ny+4, :], fp[:, 5:ny+5, :],
        cv_s,
    )   # fy_s[:,j,:] = face value at south face of cell j
    fy_n = jnp.concatenate([fy_s[:, 1:, :], fy_s[:, -1:, :]], axis=1)
    fadv = fadv + _adv_cn(cv_s, cv_n) * (fy_s - fy_n)

    # --- z: face values only (last direction — no advective update) ---
    fp = jnp.pad(fadv, ((3, 3), (0, 0), (0, 0)), mode='edge')
    fz_t = _face5(
        fp[1:nz+1], fp[2:nz+2], fp[3:nz+3],
        fp[4:nz+4], fp[5:nz+5], fp[6:nz+6],
        cw_t,
    )   # fz_t[k] = face value at top face of cell k
    fz_b = jnp.concatenate([jnp.zeros_like(fz_t[:1]), fz_t[:-1]], axis=0)

    # ------------------------------------------------------------------ #
    # FCT (Zalesak 1979) — matches gSAM fct3D in advect_um_lib.f90       #
    #                                                                      #
    # Flux conventions match the original flux-form update above:          #
    #   x,y: velocity * face_value * dt / dx  (no rho — uniform in x,y)  #
    #   z:   rhow * W * face_value / (rho * dz) * dt                     #
    # ------------------------------------------------------------------ #

    rw_t = rhow[1:nz+1, None, None]
    rw_b = rhow[0:nz,   None, None]
    rho3 = rho[:, None, None]

    # High-order fluxes (same as original flux-form update)
    ho_x_w = U[:, :, :-1] * fx * dt / dx
    ho_x_e = U[:, :, 1:]  * fx_e * dt / dx
    ho_y_s = V[:, :ny, :]  * fy_s * dt / dy3
    ho_y_n = V[:, 1:,  :]  * fy_n * dt / dy3
    ho_z_t = rw_t * W[1:nz+1] * fz_t / (rho3 * dz[:, None, None]) * dt
    ho_z_b = jnp.concatenate([jnp.zeros_like(ho_z_t[:1]),
                               rw_b[1:] * W[1:nz] * fz_b[1:] / (rho3[1:] * dz[1:, None, None]) * dt],
                              axis=0)

    # --- Step 1: local min/max of original field (7-point stencil) ---
    f_xm = jnp.roll(phi,  1, axis=2)
    f_xp = jnp.roll(phi, -1, axis=2)
    f_yp = jnp.pad(phi, ((0, 0), (0, 1), (0, 0)), mode='edge')[:, 1:, :]
    f_ym = jnp.pad(phi, ((0, 0), (1, 0), (0, 0)), mode='edge')[:, :-1, :]
    f_zp = jnp.pad(phi, ((0, 1), (0, 0), (0, 0)), mode='edge')[1:, :, :]
    f_zm = jnp.pad(phi, ((1, 0), (0, 0), (0, 0)), mode='edge')[:-1, :, :]

    mn0 = jnp.minimum(jnp.minimum(jnp.minimum(phi, f_xm), jnp.minimum(f_xp, f_ym)),
                       jnp.minimum(f_yp, jnp.minimum(f_zm, f_zp)))
    mx0 = jnp.maximum(jnp.maximum(jnp.maximum(phi, f_xm), jnp.maximum(f_xp, f_ym)),
                       jnp.maximum(f_yp, jnp.maximum(f_zm, f_zp)))

    # --- Step 2: 1st-order upwind fluxes (same conventions as high-order) ---
    # x: upwind at west face of cell i
    U_w = U[:, :, :-1]
    U_e = U[:, :, 1:]
    up_x_w = (jnp.roll(phi, 1, axis=2) * jnp.maximum(0.0, U_w)
            + phi * jnp.minimum(0.0, U_w)) * dt / dx
    up_x_e = (phi * jnp.maximum(0.0, U_e)
            + jnp.roll(phi, -1, axis=2) * jnp.minimum(0.0, U_e)) * dt / dx

    # y: upwind at south face of cell j
    V_s = V[:, :ny, :]
    V_n = V[:, 1:, :]
    up_y_s = (f_ym * jnp.maximum(0.0, V_s)
            + phi * jnp.minimum(0.0, V_s)) * dt / dy3
    up_y_n = (phi * jnp.maximum(0.0, V_n)
            + f_yp * jnp.minimum(0.0, V_n)) * dt / dy3

    # z: upwind at top face of cell k (with rhow/rho weighting)
    W_t = W[1:nz+1]
    up_z_t = (phi * jnp.maximum(0.0, W_t)
            + f_zp * jnp.minimum(0.0, W_t)) * rw_t / (rho3 * dz[:, None, None]) * dt
    up_z_b = jnp.concatenate([jnp.zeros_like(up_z_t[:1]),
        (f_zm[1:] * jnp.maximum(0.0, W[1:nz])
       + phi[1:] * jnp.minimum(0.0, W[1:nz])) * rw_b[1:] / (rho3[1:] * dz[1:, None, None]) * dt],
        axis=0)

    # --- Step 3: upwind update ---
    f_up = phi + (up_x_w - up_x_e) - (up_y_n - up_y_s) - (up_z_t - up_z_b)

    # --- Step 4: antidiffusive flux = high-order - upwind ---
    afx_w = ho_x_w - up_x_w
    afx_e = ho_x_e - up_x_e
    afy_s = ho_y_s - up_y_s
    afy_n = ho_y_n - up_y_n
    afz_t = ho_z_t - up_z_t
    afz_b = ho_z_b - up_z_b

    # --- Step 5: update min/max bounds with upwind-updated field ---
    fu_xm = jnp.roll(f_up,  1, axis=2)
    fu_xp = jnp.roll(f_up, -1, axis=2)
    fu_yp = jnp.pad(f_up, ((0, 0), (0, 1), (0, 0)), mode='edge')[:, 1:, :]
    fu_ym = jnp.pad(f_up, ((0, 0), (1, 0), (0, 0)), mode='edge')[:, :-1, :]
    fu_zp = jnp.pad(f_up, ((0, 1), (0, 0), (0, 0)), mode='edge')[1:, :, :]
    fu_zm = jnp.pad(f_up, ((1, 0), (0, 0), (0, 0)), mode='edge')[:-1, :, :]

    mn = jnp.minimum(mn0, jnp.minimum(
        jnp.minimum(jnp.minimum(f_up, fu_xm), jnp.minimum(fu_xp, fu_ym)),
        jnp.minimum(fu_yp, jnp.minimum(fu_zm, fu_zp))))
    mx = jnp.maximum(mx0, jnp.maximum(
        jnp.maximum(jnp.maximum(f_up, fu_xm), jnp.maximum(fu_xp, fu_ym)),
        jnp.maximum(fu_yp, jnp.maximum(fu_zm, fu_zp))))

    # --- Step 6: FCT scale factors ---
    eps = 1e-10

    # Total antidiffusive outflow (positive flux leaving cell)
    out_flux = (jnp.maximum(0.0, afx_e) - jnp.minimum(0.0, afx_w)
              + jnp.maximum(0.0, afy_n) - jnp.minimum(0.0, afy_s)
              + jnp.maximum(0.0, afz_t) - jnp.minimum(0.0, afz_b))
    scale_out = (f_up - mn) / (out_flux + eps)

    # Total antidiffusive inflow (positive flux entering cell)
    in_flux = (jnp.maximum(0.0, afx_w) - jnp.minimum(0.0, afx_e)
             + jnp.maximum(0.0, afy_s) - jnp.minimum(0.0, afy_n)
             + jnp.maximum(0.0, afz_b) - jnp.minimum(0.0, afz_t))
    scale_in = (mx - f_up) / (in_flux + eps)

    # --- Step 7: limit antidiffusive fluxes ---
    scale_out_xm = jnp.roll(scale_out, 1, axis=2)
    scale_in_xm  = jnp.roll(scale_in,  1, axis=2)
    afx_w = (jnp.maximum(0.0, afx_w) * jnp.minimum(1.0, jnp.minimum(scale_out_xm, scale_in))
           + jnp.minimum(0.0, afx_w) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_xm)))
    afx_e = (jnp.maximum(0.0, afx_e) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_xm))
           + jnp.minimum(0.0, afx_e) * jnp.minimum(1.0, jnp.minimum(scale_out_xm, scale_in)))

    scale_out_ym = jnp.pad(scale_out, ((0,0),(1,0),(0,0)), mode='edge')[:, :-1, :]
    scale_in_ym  = jnp.pad(scale_in,  ((0,0),(1,0),(0,0)), mode='edge')[:, :-1, :]
    afy_s = (jnp.maximum(0.0, afy_s) * jnp.minimum(1.0, jnp.minimum(scale_out_ym, scale_in))
           + jnp.minimum(0.0, afy_s) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_ym)))
    scale_out_yp = jnp.pad(scale_out, ((0,0),(0,1),(0,0)), mode='edge')[:, 1:, :]
    scale_in_yp  = jnp.pad(scale_in,  ((0,0),(0,1),(0,0)), mode='edge')[:, 1:, :]
    afy_n = (jnp.maximum(0.0, afy_n) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_yp))
           + jnp.minimum(0.0, afy_n) * jnp.minimum(1.0, jnp.minimum(scale_out_yp, scale_in)))

    scale_out_zm = jnp.pad(scale_out, ((1,0),(0,0),(0,0)), mode='edge')[:-1, :, :]
    scale_in_zm  = jnp.pad(scale_in,  ((1,0),(0,0),(0,0)), mode='edge')[:-1, :, :]
    afz_b = (jnp.maximum(0.0, afz_b) * jnp.minimum(1.0, jnp.minimum(scale_out_zm, scale_in))
           + jnp.minimum(0.0, afz_b) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_zm)))
    scale_out_zp = jnp.pad(scale_out, ((0,1),(0,0),(0,0)), mode='edge')[1:, :, :]
    scale_in_zp  = jnp.pad(scale_in,  ((0,1),(0,0),(0,0)), mode='edge')[1:, :, :]
    afz_t = (jnp.maximum(0.0, afz_t) * jnp.minimum(1.0, jnp.minimum(scale_out, scale_in_zp))
           + jnp.minimum(0.0, afz_t) * jnp.minimum(1.0, jnp.minimum(scale_out_zp, scale_in)))

    # --- Step 8: final update = upwind + limited antidiffusive, with positivity ---
    return jnp.maximum(0.0, f_up
        + (afx_w - afx_e) - (afy_n - afy_s) - (afz_t - afz_b))


# ---------------------------------------------------------------------------
# 4.  3rd-order upwind momentum advection (C-grid, gSAM nadv_mom=3)
# ---------------------------------------------------------------------------

def _flux3(
    phi_m1: jax.Array,
    phi_0:  jax.Array,
    phi_p1: jax.Array,
    phi_p2: jax.Array,
    u_adv:  jax.Array,
) -> jax.Array:
    """
    3rd-order upwind-biased face flux: u_adv * phi_interp.

    Matches gSAM advect23_mom_xy/z with alpha_hybrid=0 (flat terrain, wg=0).
    Face is between phi_0 and phi_p1.
      u_adv >= 0: (2*phi_p1 + 5*phi_0 - phi_m1) / 6
      u_adv <  0: (2*phi_0  + 5*phi_p1 - phi_p2) / 6
    """
    f_pos = (2 * phi_p1 + 5 * phi_0 - phi_m1) / 6.0
    f_neg = (2 * phi_0  + 5 * phi_p1 - phi_p2) / 6.0
    return u_adv * jnp.where(u_adv >= 0, f_pos, f_neg)


def _mom_adv_tend(
    U: jax.Array, V: jax.Array, W: jax.Array, metric: dict, dt: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Spherical mass-weighted momentum advective tendencies * dt.

    Transposition of gSAM advect23_mom_xy.f90 + advect23_mom_z.f90
    (nadv_mom=3, alpha_hybrid=0, gamma_RAVE=1, flat terrain), using the
    mass-flux form with Jacobians gu/gv/gw:

        u1(j,k)    = U * rho(k)  * (dt/dx)     * adz(k) * ady(j)      [U-face]
        v1(j_v,k)  = V * rho(k)  * (dt/dy_ref) * adz(k) * muv(j_v)    [V-face]
        w1(j,k_w)  = W * rhow(k_w)*(dt/dz_ref) * ady(j) * mu(j)       [W-face]

        gu(j,k)    = mu(j)   * rho(k)  * ady(j)  * adz(k)
        gv(j_v,k)  = muv(j_v)* rho(k)  * adyv(j_v)*adz(k)
        gw(j,k_w)  = mu(j)   * rhow(k_w)*ady(j) * adzw(k_w)

    Tendency form (gSAM advect23_mom_xy.f90 lines 38, 59, 80, 103, 127, 150,
    advect23_mom_z.f90 lines 102, 103, 130):

        dudt -= (1/gu) * (f_east - f_west + f_north - f_south + f_top - f_bot)

    with face fluxes f = 0.5*(u1_a+u1_b) * _flux3_poly (the 0.5 absorbs the
    factor-of-2 that gSAM leaves in uuu and compensates with d12=1/12
    instead of 1/6).

    Returns (dU, dV, dW):
      dU shape (nz, ny, nx)       — U update at west-face index i=0..nx-1
      dV shape (nz, ny-1, nx)     — interior V rows j_v=1..ny-1
      dW shape (nz-1, ny, nx)     — interior W levels k_w=1..nz-1
    The prognostic variables are updated once by the caller.
    """
    mu     = metric["cos_lat"]    # (ny,)
    muv    = metric["cos_v"]      # (ny+1,)  ady-weighted (gSAM muv)
    ady    = metric["ady"]        # (ny,)    dy_row / dy_ref
    rho    = metric["rho"]        # (nz,)
    rhow   = metric["rhow"]       # (nz+1,)
    dz     = metric["dz"]         # (nz,)
    dx     = metric["dx_lon"]     # scalar (equatorial dx)
    dy_ref = metric["dy_lat_ref"] # scalar

    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1

    # ------------------------------------------------------------------
    # Non-dimensional vertical thickness ratios.  dz_ref is arbitrary;
    # adz/adzw always appear alongside their reciprocals in g*, so the
    # normalisation cancels analytically.
    # ------------------------------------------------------------------
    dz_ref   = dz[0]
    adz      = dz / dz_ref                            # (nz,)
    adzw_int = 0.5 * (adz[:-1] + adz[1:])             # (nz-1,)
    adzw     = jnp.concatenate([adz[:1], adzw_int, adz[-1:]])   # (nz+1,)

    # adyv[j_v]: interior = 0.5*(ady[j-1]+ady[j]); poles get the adjacent
    # mass-row value (unused because V=0 there).
    adyv_int = 0.5 * (ady[:-1] + ady[1:])             # (ny-1,)
    adyv     = jnp.concatenate([ady[:1], adyv_int, ady[-1:]])   # (ny+1,)

    dtdx = dt / dx
    dtdy = dt / dy_ref
    dtdz = dt / dz_ref

    # ------------------------------------------------------------------
    # Mass fluxes (gSAM u1/v1/w1; dt absorbed).
    # ------------------------------------------------------------------
    u1 = U * (rho[:, None, None] * dtdx
              * adz[:, None, None] * ady[None, :, None])            # (nz, ny, nx+1)
    v1 = V * (rho[:, None, None] * dtdy
              * adz[:, None, None] * muv[None, :, None])            # (nz, ny+1, nx)
    w1 = W * (rhow[:, None, None] * dtdz
              * ady[None, :, None] * mu[None, :, None])             # (nz+1, ny, nx)

    # ------------------------------------------------------------------
    # Jacobians (gSAM gu/gv/gw).
    # ------------------------------------------------------------------
    gu3 = (mu[None, :, None] * rho[:, None, None]
           * ady[None, :, None] * adz[:, None, None])               # (nz, ny, 1)
    gv3_int = (muv[None, 1:ny, None] * rho[:, None, None]
               * adyv[None, 1:ny, None] * adz[:, None, None])       # (nz, ny-1, 1)
    gw3_int = (mu[None, :, None] * rhow[1:nz, None, None]
               * ady[None, :, None] * adzw[1:nz, None, None])       # (nz-1, ny, 1)

    # ==================================================================
    # U tendency (U-CV centred on west face of mass cell i, i=0..nx-1)
    # ==================================================================
    U_c   = U[:, :, :nx]             # (nz, ny, nx)
    u1_c  = u1[:, :, :nx]
    u1_cp = jnp.roll(u1_c, -1, axis=2)

    # --- x: east x-face of U-CV i is at mass cell i centre;
    #        advecting mass flux = 0.5*(u1(i) + u1(i+1)).
    U_m1 = jnp.roll(U_c, +1, axis=2)
    U_p1 = jnp.roll(U_c, -1, axis=2)
    U_p2 = jnp.roll(U_c, -2, axis=2)
    flux_x_U = _flux3(U_m1, U_c, U_p1, U_p2, 0.5 * (u1_c + u1_cp))
    dU_x = -(flux_x_U - jnp.roll(flux_x_U, +1, axis=2)) / gu3

    # --- y: north y-face of U-CV at mass row j lies at v-face j_v=j+1;
    #        advecting v1 = 0.5*(v1(i) + v1(i-1)) at v-face j+1.
    v1_face_all = 0.5 * (v1 + jnp.roll(v1, +1, axis=-1))   # (nz, ny+1, nx)
    v_adv_U_n   = v1_face_all[:, 1:ny+1, :]                # (nz, ny, nx)
    U_py = jnp.pad(U_c, ((0, 0), (1, 2), (0, 0)), mode='edge')   # (nz, ny+3, nx)
    fy_n_U = _flux3(U_py[:, 0:ny, :],   U_py[:, 1:ny+1, :],
                    U_py[:, 2:ny+2, :], U_py[:, 3:ny+3, :], v_adv_U_n)
    fy_s_U = jnp.concatenate([jnp.zeros_like(fy_n_U[:, :1, :]),
                              fy_n_U[:, :-1, :]], axis=1)
    dU_y = -(fy_n_U - fy_s_U) / gu3

    # --- z: fuz at W-faces 1..nz-1 (interior), f=0 at k_w=0, nz.
    #        Stencil at W-face k_w uses U at mass levels k_w-2..k_w+1.
    #        Advecting w1 = 0.5*(w1(i) + w1(i-1)) at the interior W-faces.
    w1_face_all = 0.5 * (w1 + jnp.roll(w1, +1, axis=-1))   # (nz+1, ny, nx)
    w_adv_U_int = w1_face_all[1:nz, :, :]                  # (nz-1, ny, nx)
    U_pz = jnp.pad(U_c, ((2, 2), (0, 0), (0, 0)), mode='edge')   # (nz+4, ny, nx)
    fuz_int = _flux3(U_pz[1:nz],   U_pz[2:nz+1],
                     U_pz[3:nz+2], U_pz[4:nz+3], w_adv_U_int)   # (nz-1, ny, nx)
    fuz_full = jnp.concatenate([jnp.zeros_like(fuz_int[:1]),
                                fuz_int,
                                jnp.zeros_like(fuz_int[:1])], axis=0)  # (nz+1,…)
    dU_z = -(fuz_full[1:nz+1] - fuz_full[0:nz]) / gu3

    dU = dU_x + dU_y + dU_z

    # ==================================================================
    # V tendency (V-CV centred on v-face j_v=1..ny-1)
    # ==================================================================
    V_prog = V[:, 1:ny, :]        # (nz, ny-1, nx)

    # --- x: east x-face of V-CV at cell i;
    #        advecting u1 = 0.5*(u1(i+1,j)+u1(i+1,j-1)) — east face of cell i,
    #        averaged across the two mass rows straddling the v-face.
    u1_east = u1[:, :, 1:nx+1]                              # (nz, ny, nx)
    u1_V_e  = 0.5 * (u1_east[:, 1:ny, :] + u1_east[:, 0:ny-1, :])   # (nz, ny-1, nx)
    V_m1 = jnp.roll(V_prog, +1, axis=2)
    V_p1 = jnp.roll(V_prog, -1, axis=2)
    V_p2 = jnp.roll(V_prog, -2, axis=2)
    flux_x_V = _flux3(V_m1, V_prog, V_p1, V_p2, u1_V_e)
    dV_x = -(flux_x_V - jnp.roll(flux_x_V, +1, axis=2)) / gv3_int

    # --- y: V self-advects.  At mass row j (between V-faces j and j+1) the
    #        flux is fvy(j); gSAM stencil uses V at v-faces (j-1,j,j+1,j+2).
    #        Advecting v1 = 0.5*(v1(j+1)+v1(j)) at mass row j.
    #        For prognostic V indices jp=0..ny-2 (v-face j_v=jp+1) the north
    #        face of V-CV is at mass row j_v, i.e., uses v1(j_v+1)+v1(j_v).
    V_ext = jnp.concatenate([V, jnp.zeros_like(V[:, :1, :])], axis=1)  # (nz, ny+2, nx)
    v1_V_n_a = v1[:, 1:ny,   :]       # v1(j_v)   for j_v=1..ny-1
    v1_V_n_b = v1[:, 2:ny+1, :]       # v1(j_v+1) for j_v=1..ny-1
    v_adv_V_n = 0.5 * (v1_V_n_a + v1_V_n_b)                  # (nz, ny-1, nx)
    fy_n_V = _flux3(V_ext[:, 0:ny-1, :],  V_ext[:, 1:ny,   :],
                    V_ext[:, 2:ny+1, :],  V_ext[:, 3:ny+2, :], v_adv_V_n)
    fy_s_V = jnp.concatenate([jnp.zeros_like(fy_n_V[:, :1, :]),
                              fy_n_V[:, :-1, :]], axis=1)
    dV_y = -(fy_n_V - fy_s_V) / gv3_int

    # --- z: fvz at W-faces 1..nz-1; advecting w1 = 0.5*(w1(j)+w1(j-1)).
    w1_V_all = 0.5 * (w1[:, 1:ny, :] + w1[:, 0:ny-1, :])     # (nz+1, ny-1, nx)
    w_adv_V_int = w1_V_all[1:nz, :, :]                       # (nz-1, ny-1, nx)
    V_pz = jnp.pad(V_prog, ((2, 2), (0, 0), (0, 0)), mode='edge')   # (nz+4, ny-1, nx)
    fvz_int = _flux3(V_pz[1:nz],   V_pz[2:nz+1],
                     V_pz[3:nz+2], V_pz[4:nz+3], w_adv_V_int)
    fvz_full = jnp.concatenate([jnp.zeros_like(fvz_int[:1]),
                                fvz_int,
                                jnp.zeros_like(fvz_int[:1])], axis=0)  # (nz+1,…)
    dV_z = -(fvz_full[1:nz+1] - fvz_full[0:nz]) / gv3_int

    dV = dV_x + dV_y + dV_z

    # ==================================================================
    # W tendency (W-CV centred on W-face k_w=1..nz-1)
    # ==================================================================
    W_core = W[1:nz, :, :]    # (nz-1, ny, nx)

    # --- x: east x-face of W-CV at cell i;
    #        advecting u1 = 0.5*(u1(i+1,k)+u1(i+1,k-1)) — east face of cell i,
    #        averaged across mass levels k_w-1 and k_w.
    u1_W_e = 0.5 * (u1_east[0:nz-1, :, :] + u1_east[1:nz, :, :])   # (nz-1, ny, nx)
    W_m1 = jnp.roll(W_core, +1, axis=2)
    W_p1 = jnp.roll(W_core, -1, axis=2)
    W_p2 = jnp.roll(W_core, -2, axis=2)
    flux_x_W = _flux3(W_m1, W_core, W_p1, W_p2, u1_W_e)
    dW_x = -(flux_x_W - jnp.roll(flux_x_W, +1, axis=2)) / gw3_int

    # --- y: north face at v-face j+1, advecting v1 averaged across
    #        mass levels k_w-1 and k_w.
    v1_Wy_a = v1[0:nz-1, :, :]
    v1_Wy_b = v1[1:nz,   :, :]
    v1_W    = 0.5 * (v1_Wy_a + v1_Wy_b)                     # (nz-1, ny+1, nx)
    v_adv_W_n = v1_W[:, 1:ny+1, :]                           # (nz-1, ny, nx)
    W_py = jnp.pad(W_core, ((0, 0), (1, 2), (0, 0)), mode='edge')  # (nz-1, ny+3, nx)
    fy_n_W = _flux3(W_py[:, 0:ny, :],   W_py[:, 1:ny+1, :],
                    W_py[:, 2:ny+2, :], W_py[:, 3:ny+3, :], v_adv_W_n)
    fy_s_W = jnp.concatenate([jnp.zeros_like(fy_n_W[:, :1, :]),
                              fy_n_W[:, :-1, :]], axis=1)
    dW_y = -(fy_n_W - fy_s_W) / gw3_int

    # --- z: W self-advects; fwz at mass levels k_m=0..nz-1, stencil in W
    #        uses indices k_m-1..k_m+2.  Advecting w1 at mass level k_m is
    #        0.5*(w1(k_m)+w1(k_m+1)) (gSAM line 114: w1(kc)+w1(k), kc=k+1).
    w_adv_W = 0.5 * (w1[0:nz, :, :] + w1[1:nz+1, :, :])      # (nz, ny, nx)
    W_pz = jnp.pad(W, ((1, 2), (0, 0), (0, 0)), mode='edge')   # (nz+4, ny, nx)
    fwz = _flux3(W_pz[0:nz],   W_pz[1:nz+1],
                 W_pz[2:nz+2], W_pz[3:nz+3], w_adv_W)          # (nz, ny, nx)
    dW_z = -(fwz[1:nz, :, :] - fwz[0:nz-1, :, :]) / gw3_int

    dW = dW_x + dW_y + dW_z

    return dU, dV, dW


@jax.jit
def advect_momentum(
    U:      jax.Array,   # (nz, ny, nx+1)
    V:      jax.Array,   # (nz, ny+1, nx)
    W:      jax.Array,   # (nz+1, ny, nx)
    metric: dict,
    dt:     float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    One timestep of 3rd-order upwind momentum advection on an anelastic
    lat-lon C-grid.

    Port of gSAM advect23_mom_xy + advect23_mom_z (nadv_mom=3, alpha_hybrid=0,
    gamma_RAVE=1, flat terrain).

    Returns (U_new, V_new, W_new) after one advection step of length dt.
    Boundary conditions:
      U — periodic in x (U[...,nx] = U[...,0])
      V — polar rows V[:,0,:] and V[:,ny,:] held at 0
      W — rigid-lid rows W[0,...] and W[nz,...] held at 0
    """
    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1
    nz_w = W.shape[0]    # nz+1
    ny_v = V.shape[1]    # ny+1

    dU, dV, dW = _mom_adv_tend(U, V, W, metric, dt)

    U_new = U.at[:, :, :nx].add(dU)
    U_new = U_new.at[:, :, nx].set(U_new[:, :, 0])   # periodic

    V_new = V.at[:, 1:ny_v-1, :].add(dV)             # interior rows only

    W_new = W.at[1:nz_w-1, :, :].add(dW)             # interior levels only

    return U_new, V_new, W_new
