"""
Scalar advection — 5th-order ULTIMATE-MACHO scheme (anelastic lat-lon).
Port of gSAM ADV_UM5 with MACHO direction cycling and Zalesak FCT.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jsam.core.grid.latlon import LatLonGrid


def _face5(
    f_im3: jax.Array,
    f_im2: jax.Array,
    f_im1: jax.Array,
    f_i:   jax.Array,
    f_ip1: jax.Array,
    f_ip2: jax.Array,
    cn:    jax.Array,
) -> jax.Array:
    """5th-order ULTIMATE face value at the im1/i interface."""
    d2 = f_ip1 - f_i - f_im1 + f_im2
    d3 = f_ip1 - 3*f_i + 3*f_im1 - f_im2
    d4 = f_ip2 - 3*f_ip1 + 2*f_i + 2*f_im1 - 3*f_im2 + f_im3
    d5 = f_ip2 - 5*f_ip1 + 10*f_i - 10*f_im1 + 5*f_im2 - f_im3
    return 0.5 * (
        f_i + f_im1
        - cn * (f_i - f_im1)
        + (1/6)   * (cn**2 - 1) * (d2 - 0.5*cn*d3)
        + (1/120) * (cn**2 - 1) * (cn**2 - 4) * (d4 - jnp.sign(cn)*d5)
    )


def _adv_cn(cn_left: jax.Array, cn_right: jax.Array) -> jax.Array:
    """Upwind advective Courant number for the MACHO predictor."""
    return jnp.where(
        (cn_right > 0.0) & (cn_left >= 0.0), cn_left,
        jnp.where((cn_right <= 0.0) & (cn_left < 0.0), cn_right, 0.0),
    )


@jax.jit(static_argnums=(6,))
def _advect_scalar_jit(
    phi:    jax.Array,
    U:      jax.Array,
    V:      jax.Array,
    W:      jax.Array,
    metric: dict,
    dt:     float,
    macho_order: int = 0,
) -> jax.Array:
    """Core scalar advection logic with static macho_order parameter.

    The macho_order parameter (index 6) is declared as static via jax.jit,
    so Python conditionals work correctly despite being in a traced function.
    """
    nz, ny, _ = phi.shape

    dx    = metric["dx_lon"]
    dy    = metric["dy_lat"]
    rho   = metric["rho"]
    rhow  = metric["rhow"]
    dz    = metric["dz"]

    irho = 1.0 / rho[:, None, None]
    iadz = 1.0 / dz[:, None, None]
    dy3   = dy[None, :, None]

    ady   = metric["ady"][:, None]
    ady3  = ady[None, :, :]
    adz   = (dz / dz[0])[:, None, None]

    cu_w = U[:, :, :-1]   * dt / dx * adz * ady3
    cu_e = jnp.roll(cu_w, -1, axis=2)
    cv_s = V[:, 0:ny,   :] * dt / dy3 * adz * ady3
    cv_n = V[:, 1:ny+1, :] * dt / dy3 * adz * ady3
    cw_t = W[1:nz+1, :, :] * dt * iadz

    # ------------------------------------------------------------------ #
    # MACHO predictor — compute high-order face values                    #
    # C5 fix: cycle through 6 direction orderings via mod(nstep-1, 6)     #
    #   0: z→x→y  1: y→z→x  2: x→y→z  3: z→y→x  4: x→z→y  5: y→x→z    #
    # The first two directions get advective-form predictor updates;      #
    # the third direction computes face values only (no predictor update).#
    # ------------------------------------------------------------------ #
    cw_b = jnp.concatenate([jnp.zeros_like(cw_t[:1]), cw_t[:-1]], axis=0)

    def _face_x(fadv_in):
        return _face5(
            jnp.roll(fadv_in,  3, axis=2), jnp.roll(fadv_in,  2, axis=2),
            jnp.roll(fadv_in,  1, axis=2), fadv_in,
            jnp.roll(fadv_in, -1, axis=2), jnp.roll(fadv_in, -2, axis=2),
            cu_w,
        )

    def _adv_update_x(fadv_in, fx_in):
        fx_e_in = jnp.roll(fx_in, -1, axis=2)
        return fadv_in + _adv_cn(cu_w, cu_e) * (fx_in - fx_e_in), fx_e_in

    def _face_y(fadv_in):
        fp = jnp.pad(fadv_in, ((0, 0), (3, 3), (0, 0)), mode='edge')
        return _face5(
            fp[:, 0:ny, :], fp[:, 1:ny+1, :], fp[:, 2:ny+2, :],
            fp[:, 3:ny+3, :], fp[:, 4:ny+4, :], fp[:, 5:ny+5, :],
            cv_s,
        )

    def _adv_update_y(fadv_in, fy_s_in):
        fy_n_in = jnp.concatenate([fy_s_in[:, 1:, :], fy_s_in[:, -1:, :]], axis=1)
        return fadv_in + _adv_cn(cv_s, cv_n) * (fy_s_in - fy_n_in), fy_n_in

    def _face_z(fadv_in):
        fp = jnp.pad(fadv_in, ((3, 3), (0, 0), (0, 0)), mode='edge')
        fz_t_in = _face5(
            fp[1:nz+1], fp[2:nz+2], fp[3:nz+3],
            fp[4:nz+4], fp[5:nz+5], fp[6:nz+6],
            cw_t,
        )
        fz_t_in = fz_t_in.at[-1].set(0.0)  # rigid-lid BC
        return fz_t_in

    def _adv_update_z(fadv_in, fz_t_in):
        fz_b_in = jnp.concatenate([jnp.zeros_like(fz_t_in[:1]), fz_t_in[:-1]], axis=0)
        return fadv_in + _adv_cn(cw_b, cw_t) * (fz_b_in - fz_t_in), fz_b_in

    fadv = phi

    # Execute the 6 MACHO orderings.
    # Each ordering: dir1 (face+update), dir2 (face+update), dir3 (face only).
    # Use Python conditionals since macho_order is now a static int (not traced)
    if macho_order == 0:    # z → x → y
        fz_t = _face_z(fadv);         fadv, fz_b = _adv_update_z(fadv, fz_t)
        fx   = _face_x(fadv);         fadv, fx_e = _adv_update_x(fadv, fx)
        fy_s = _face_y(fadv);         fy_n = jnp.concatenate([fy_s[:, 1:, :], fy_s[:, -1:, :]], axis=1)
    elif macho_order == 1:  # y → z → x
        fy_s = _face_y(fadv);         fadv, fy_n = _adv_update_y(fadv, fy_s)
        fz_t = _face_z(fadv);         fadv, fz_b = _adv_update_z(fadv, fz_t)
        fx   = _face_x(fadv);         fx_e = jnp.roll(fx, -1, axis=2)
    elif macho_order == 2:  # x → y → z  (original case 2)
        fx   = _face_x(fadv);         fadv, fx_e = _adv_update_x(fadv, fx)
        fy_s = _face_y(fadv);         fadv, fy_n = _adv_update_y(fadv, fy_s)
        fz_t = _face_z(fadv);         fz_b = jnp.concatenate([jnp.zeros_like(fz_t[:1]), fz_t[:-1]], axis=0)
    elif macho_order == 3:  # z → y → x
        fz_t = _face_z(fadv);         fadv, fz_b = _adv_update_z(fadv, fz_t)
        fy_s = _face_y(fadv);         fadv, fy_n = _adv_update_y(fadv, fy_s)
        fx   = _face_x(fadv);         fx_e = jnp.roll(fx, -1, axis=2)
    elif macho_order == 4:  # x → z → y
        fx   = _face_x(fadv);         fadv, fx_e = _adv_update_x(fadv, fx)
        fz_t = _face_z(fadv);         fadv, fz_b = _adv_update_z(fadv, fz_t)
        fy_s = _face_y(fadv);         fy_n = jnp.concatenate([fy_s[:, 1:, :], fy_s[:, -1:, :]], axis=1)
    else:                   # 5: y → x → z
        fy_s = _face_y(fadv);         fadv, fy_n = _adv_update_y(fadv, fy_s)
        fx   = _face_x(fadv);         fadv, fx_e = _adv_update_x(fadv, fx)
        fz_t = _face_z(fadv);         fz_b = jnp.concatenate([jnp.zeros_like(fz_t[:1]), fz_t[:-1]], axis=0)

    rw_t = rhow[1:nz+1, None, None]
    rw_b = rhow[0:nz,   None, None]
    rho3 = rho[:, None, None]

    dz3 = dz[:, None, None]
    ho_x_w = U[:, :, :-1] * fx * dt / dx * irho * iadz
    ho_x_e = U[:, :, 1:]  * fx_e * dt / dx * irho * iadz
    ho_y_s = V[:, :ny, :]  * fy_s * dt / dy3 * irho * iadz
    ho_y_n = V[:, 1:,  :]  * fy_n * dt / dy3 * irho * iadz
    ho_z_t = rw_t * W[1:nz+1] * fz_t / (rho3 * dz3) * dt * iadz
    ho_z_b = jnp.concatenate([jnp.zeros_like(ho_z_t[:1]),
                               rw_b[1:] * W[1:nz] * fz_b[1:] / (rho3[1:] * dz[1:, None, None]) * dt * iadz[1:]],
                              axis=0)

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

    U_w = U[:, :, :-1]
    U_e = U[:, :, 1:]
    up_x_w = (jnp.roll(phi, 1, axis=2) * jnp.maximum(0.0, U_w)
            + phi * jnp.minimum(0.0, U_w)) * dt / dx * irho * iadz
    up_x_e = (phi * jnp.maximum(0.0, U_e)
            + jnp.roll(phi, -1, axis=2) * jnp.minimum(0.0, U_e)) * dt / dx * irho * iadz

    V_s = V[:, :ny, :]
    V_n = V[:, 1:, :]
    up_y_s = (f_ym * jnp.maximum(0.0, V_s)
            + phi * jnp.minimum(0.0, V_s)) * dt / dy3 * irho * iadz
    up_y_n = (phi * jnp.maximum(0.0, V_n)
            + f_yp * jnp.minimum(0.0, V_n)) * dt / dy3 * irho * iadz

    W_t = W[1:nz+1]
    up_z_t = (phi * jnp.maximum(0.0, W_t)
            + f_zp * jnp.minimum(0.0, W_t)) * rw_t / (rho3 * dz3) * dt * iadz
    up_z_b = jnp.concatenate([jnp.zeros_like(up_z_t[:1]),
        (f_zm[1:] * jnp.maximum(0.0, W[1:nz])
       + phi[1:] * jnp.minimum(0.0, W[1:nz])) * rw_b[1:] / (rho3[1:] * dz[1:, None, None]) * dt * iadz[1:]],
        axis=0)

    f_up = phi + (up_x_w - up_x_e) - (up_y_n - up_y_s) - (up_z_t - up_z_b)

    afx_w = ho_x_w - up_x_w
    afx_e = ho_x_e - up_x_e
    afy_s = ho_y_s - up_y_s
    afy_n = ho_y_n - up_y_n
    afz_t = ho_z_t - up_z_t
    afz_b = ho_z_b - up_z_b
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

    eps = 1e-10

    out_flux = (jnp.maximum(0.0, afx_e) - jnp.minimum(0.0, afx_w)
              + jnp.maximum(0.0, afy_n) - jnp.minimum(0.0, afy_s)
              + jnp.maximum(0.0, afz_t) - jnp.minimum(0.0, afz_b))
    scale_out = (f_up - mn) / (out_flux + eps)

    in_flux = (jnp.maximum(0.0, afx_w) - jnp.minimum(0.0, afx_e)
             + jnp.maximum(0.0, afy_s) - jnp.minimum(0.0, afy_n)
             + jnp.maximum(0.0, afz_b) - jnp.minimum(0.0, afz_t))
    scale_in = (mx - f_up) / (in_flux + eps)
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

    return jnp.maximum(0.0, f_up
        + (afx_w - afx_e) - (afy_n - afy_s) - (afz_t - afz_b))


def advect_scalar(
    phi:    jax.Array,
    U:      jax.Array,
    V:      jax.Array,
    W:      jax.Array,
    metric: dict,
    dt:     float,
    nstep:  int = 0,
) -> jax.Array:
    """One timestep of 5th-order MACHO scalar advection with Zalesak FCT.

    Wrapper that computes macho_order from nstep (statically) and dispatches
    to the JIT-compiled kernel.
    """
    # Convert nstep to Python int to ensure macho_order is a static argument
    macho_order = int((int(nstep) - 1) % 6)
    return _advect_scalar_jit(phi, U, V, W, metric, dt, macho_order)


def _flux3(
    phi_m1: jax.Array,
    phi_0:  jax.Array,
    phi_p1: jax.Array,
    phi_p2: jax.Array,
    u_adv:  jax.Array,
) -> jax.Array:
    """3rd-order upwind-biased face flux."""
    f_pos = (2 * phi_p1 + 5 * phi_0 - phi_m1) / 6.0
    f_neg = (2 * phi_0  + 5 * phi_p1 - phi_p2) / 6.0
    return u_adv * jnp.where(u_adv >= 0, f_pos, f_neg)


def _mom_adv_tend(
    U: jax.Array, V: jax.Array, W: jax.Array, metric: dict, dt: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Spherical mass-weighted momentum advective tendencies."""
    mu     = metric["cos_lat"]
    muv    = metric["cos_v"]
    ady    = metric["ady"]
    rho    = metric["rho"]
    rhow   = metric["rhow"]
    dz     = metric["dz"]
    dx     = metric["dx_lon"]
    dy_ref = metric["dy_lat_ref"]

    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1

    dz_ref   = dz[0]
    adz      = dz / dz_ref
    adzw_int = 0.5 * (adz[:-1] + adz[1:])
    adzw     = jnp.concatenate([adz[:1], adzw_int, adz[-1:]])

    adyv_int = 0.5 * (ady[:-1] + ady[1:])
    adyv     = jnp.concatenate([ady[:1], adyv_int, ady[-1:]])

    dtdx = dt / dx
    dtdy = dt / dy_ref
    dtdz = dt / dz_ref

    u1 = U * (rho[:, None, None] * dtdx
              * adz[:, None, None] * ady[None, :, None])
    v1 = V * (rho[:, None, None] * dtdy
              * adz[:, None, None] * muv[None, :, None])
    w1 = W * (rhow[:, None, None] * dtdz
              * ady[None, :, None] * mu[None, :, None])

    gu3 = (mu[None, :, None] * rho[:, None, None]
           * ady[None, :, None] * adz[:, None, None])
    gv3_int = (muv[None, 1:ny, None] * rho[:, None, None]
               * adyv[None, 1:ny, None] * adz[:, None, None])
    gw3_int = (mu[None, :, None] * rhow[1:nz, None, None]
               * ady[None, :, None] * adzw[1:nz, None, None])

    U_c   = U[:, :, :nx]
    u1_c  = u1[:, :, :nx]
    u1_cp = jnp.roll(u1_c, -1, axis=2)

    U_m1 = jnp.roll(U_c, +1, axis=2)
    U_p1 = jnp.roll(U_c, -1, axis=2)
    U_p2 = jnp.roll(U_c, -2, axis=2)
    flux_x_U = _flux3(U_m1, U_c, U_p1, U_p2, 0.5 * (u1_c + u1_cp))
    dU_x = -(flux_x_U - jnp.roll(flux_x_U, +1, axis=2)) / gu3

    v1_face_all = 0.5 * (v1 + jnp.roll(v1, +1, axis=-1))
    v_adv_U_n   = v1_face_all[:, 1:ny+1, :]
    U_py = jnp.concatenate(
        [U_c[:, :1, :], U_c, U_c[:, -1:, :], U_c[:, -2:-1, :]], axis=1,
    )
    fy_n_U = _flux3(U_py[:, 0:ny, :],   U_py[:, 1:ny+1, :],
                    U_py[:, 2:ny+2, :], U_py[:, 3:ny+3, :], v_adv_U_n)
    fy_s_U = jnp.concatenate([jnp.zeros_like(fy_n_U[:, :1, :]),
                              fy_n_U[:, :-1, :]], axis=1)
    dU_y = -(fy_n_U - fy_s_U) / gu3

    w1_face_all = 0.5 * (w1 + jnp.roll(w1, +1, axis=-1))
    w_adv_U_int = w1_face_all[1:nz, :, :]
    U_pz = jnp.pad(U_c, ((2, 2), (0, 0), (0, 0)), mode='edge')
    fuz_int = _flux3(U_pz[1:nz],   U_pz[2:nz+1],
                     U_pz[3:nz+2], U_pz[4:nz+3], w_adv_U_int)
    fuz_full = jnp.concatenate([jnp.zeros_like(fuz_int[:1]),
                                fuz_int,
                                jnp.zeros_like(fuz_int[:1])], axis=0)
    dU_z = -(fuz_full[1:nz+1] - fuz_full[0:nz]) / gu3

    dU = dU_x + dU_y + dU_z

    V_prog = V[:, 1:ny, :]

    u1_east = u1[:, :, 1:nx+1]
    u1_V_e  = 0.5 * (u1_east[:, 1:ny, :] + u1_east[:, 0:ny-1, :])
    V_m1 = jnp.roll(V_prog, +1, axis=2)
    V_p1 = jnp.roll(V_prog, -1, axis=2)
    V_p2 = jnp.roll(V_prog, -2, axis=2)
    flux_x_V = _flux3(V_m1, V_prog, V_p1, V_p2, u1_V_e)
    dV_x = -(flux_x_V - jnp.roll(flux_x_V, +1, axis=2)) / gv3_int

    V_ext = jnp.concatenate([V, -V[:, -2:-1, :]], axis=1)
    v1_V_n_a = v1[:, 1:ny,   :]
    v1_V_n_b = v1[:, 2:ny+1, :]
    v_adv_V_n = 0.5 * (v1_V_n_a + v1_V_n_b)
    fy_n_V = _flux3(V_ext[:, 0:ny-1, :],  V_ext[:, 1:ny,   :],
                    V_ext[:, 2:ny+1, :],  V_ext[:, 3:ny+2, :], v_adv_V_n)
    fy_s_V = jnp.concatenate([jnp.zeros_like(fy_n_V[:, :1, :]),
                              fy_n_V[:, :-1, :]], axis=1)
    dV_y = -(fy_n_V - fy_s_V) / gv3_int

    w1_V_all = 0.5 * (w1[:, 1:ny, :] + w1[:, 0:ny-1, :])
    w_adv_V_int = w1_V_all[1:nz, :, :]
    V_pz = jnp.pad(V_prog, ((2, 2), (0, 0), (0, 0)), mode='edge')
    fvz_int = _flux3(V_pz[1:nz],   V_pz[2:nz+1],
                     V_pz[3:nz+2], V_pz[4:nz+3], w_adv_V_int)
    fvz_full = jnp.concatenate([jnp.zeros_like(fvz_int[:1]),
                                fvz_int,
                                jnp.zeros_like(fvz_int[:1])], axis=0)
    dV_z = -(fvz_full[1:nz+1] - fvz_full[0:nz]) / gv3_int

    dV = dV_x + dV_y + dV_z

    W_core = W[1:nz, :, :]

    u1_W_e = 0.5 * (u1_east[0:nz-1, :, :] + u1_east[1:nz, :, :])
    W_m1 = jnp.roll(W_core, +1, axis=2)
    W_p1 = jnp.roll(W_core, -1, axis=2)
    W_p2 = jnp.roll(W_core, -2, axis=2)
    flux_x_W = _flux3(W_m1, W_core, W_p1, W_p2, u1_W_e)
    dW_x = -(flux_x_W - jnp.roll(flux_x_W, +1, axis=2)) / gw3_int

    v1_Wy_a = v1[0:nz-1, :, :]
    v1_Wy_b = v1[1:nz,   :, :]
    v1_W    = 0.5 * (v1_Wy_a + v1_Wy_b)
    v_adv_W_n = v1_W[:, 1:ny+1, :]
    W_py = jnp.concatenate(
        [W_core[:, :1, :], W_core, W_core[:, -1:, :], W_core[:, -2:-1, :]], axis=1,
    )
    fy_n_W = _flux3(W_py[:, 0:ny, :],   W_py[:, 1:ny+1, :],
                    W_py[:, 2:ny+2, :], W_py[:, 3:ny+3, :], v_adv_W_n)
    fy_s_W = jnp.concatenate([jnp.zeros_like(fy_n_W[:, :1, :]),
                              fy_n_W[:, :-1, :]], axis=1)
    dW_y = -(fy_n_W - fy_s_W) / gw3_int

    w_adv_W = 0.5 * (w1[0:nz, :, :] + w1[1:nz+1, :, :])
    W_pz = jnp.pad(W, ((1, 2), (0, 0), (0, 0)), mode='edge')
    fwz = _flux3(W_pz[0:nz],   W_pz[1:nz+1],
                 W_pz[2:nz+2], W_pz[3:nz+3], w_adv_W)
    dW_z = -(fwz[1:nz, :, :] - fwz[0:nz-1, :, :]) / gw3_int

    dW = dW_x + dW_y + dW_z

    return dU, dV, dW


@jax.jit
def advect_momentum(
    U:      jax.Array,
    V:      jax.Array,
    W:      jax.Array,
    metric: dict,
    dt:     float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One timestep of 3rd-order upwind momentum advection on a lat-lon C-grid."""
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
