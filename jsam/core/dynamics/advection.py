"""
Scalar advection — 5th-order ULTIMATE-MACHO scheme (anelastic lat-lon).
Port of gSAM ADV_UM5 with MACHO direction cycling and Zalesak FCT.
"""
from __future__ import annotations

import functools

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
    mu    = metric["cos_lat"]          # shape (ny,)
    muv   = metric["cos_v"]            # shape (ny+1,)

    cu_w = U[:, :, :-1]   * dt / dx * adz * ady3
    cu_e = jnp.roll(cu_w, -1, axis=2)
    # Fix 1.5: cv uses muv (cos-lat at V-points), not ady
    cv_s = V[:, 0:ny,   :] * dt / dy3 * adz * muv[None, :ny,    None]
    cv_n = V[:, 1:ny+1, :] * dt / dy3 * adz * muv[None, 1:ny+1, None]
    # Fix 1.3/1.5: cw includes ady*mu factors (rhow cancels: w1=w*rhow*dt/dz*ady*mu,
    # then cw=w1/(rhow*adz_actual) = w*dt/dz_actual*ady*mu, matching gSAM face_z)
    ady1d = metric["ady"]                                     # shape (ny,), raw 1-D
    cw_t = W[1:nz+1, :, :] * dt * iadz * ady1d[None, :, None] * mu[None, :, None]

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

    # Fix 1.2: non-uniform vertical grid stencil matching gSAM face_5th_z.
    # Build 1-D adz and adzw arrays (nzm = nz levels).
    # adz[k] = dz[k]/dz[0]; adzw[0]=adz[0], adzw[k]=0.5*(adz[k-1]+adz[k]) for k=1..nzm-1,
    # adzw[nzm]=adz[nzm-1].  (Fortran adz(k_f) = adz1d[k_f-1], same for adzw.)
    adz1d  = dz / dz[0]                                            # shape (nz,)
    adzw1d = jnp.concatenate([
        adz1d[:1],
        0.5 * (adz1d[:-1] + adz1d[1:]),
        adz1d[-1:],
    ])                                                              # shape (nz+1,)

    def _face_z(fadv_in):
        # Non-uniform vertical grid stencil matching gSAM face_5th_z / face_3rd_z / face_2nd_z.
        #
        # Index mapping (corrected):
        #   jsam fz_t[k] = Fortran fz(k+2)   [derived from adv_form_update_z equivalence]
        #   Fortran face index i = k_idx + 2  (k_idx is 0-based Python index into fz_t)
        #
        # Field stencil for face i = k_idx+2:
        #   fadv(i-3..i+2) = fadv[k_idx-1..k_idx+3]  (0-based)
        # Pad 2 cells at bottom and 4 at top so fp[2+k] = fadv_in[k], giving:
        #   f_im3 = fp[0:nz]    = fadv[k_idx-2]  (Fortran fadv(i-3))
        #   f_im2 = fp[1:nz+1]  = fadv[k_idx-1]  (Fortran fadv(i-2))
        #   f_im1 = fp[2:nz+2]  = fadv[k_idx]    (Fortran fadv(i-1))
        #   f_i   = fp[3:nz+3]  = fadv[k_idx+1]  (Fortran fadv(i))
        #   f_ip1 = fp[4:nz+4]  = fadv[k_idx+2]  (Fortran fadv(i+1))
        #   f_ip2 = fp[5:nz+5]  = fadv[k_idx+3]  (Fortran fadv(i+2))
        #
        # Metric arrays for face i = k_idx+2:
        #   adz(i)   = adz1d[k_idx+1];  adz(i-1) = adz1d[k_idx];  adz(i-2) = adz1d[k_idx-1]
        #   adz(i+1) = adz1d[k_idx+2]
        #   adzw(i)   = adzw1d[k_idx+1];  adzw(i-1) = adzw1d[k_idx];  etc.
        # (adzw1d has nz+1 entries 0..nz; clip indices to [0, nz].)
        #
        # Courant number: cw_t[k_idx] = W[k_idx+1] = Fortran cw(k_idx+2) = cw(i)  ✓
        fp = jnp.pad(fadv_in, ((2, 4), (0, 0), (0, 0)), mode='edge')

        k_idx = jnp.arange(nz)

        az_i   = adz1d[jnp.clip(k_idx + 1, 0, nz - 1)]   # adz(i)
        az_im1 = adz1d[jnp.clip(k_idx,     0, nz - 1)]   # adz(i-1)
        az_im2 = adz1d[jnp.clip(k_idx - 1, 0, nz - 1)]   # adz(i-2)
        az_ip1 = adz1d[jnp.clip(k_idx + 2, 0, nz - 1)]   # adz(i+1)

        aw_im2 = adzw1d[jnp.clip(k_idx - 1, 0, nz)]      # adzw(i-2)
        aw_im1 = adzw1d[jnp.clip(k_idx,     0, nz)]       # adzw(i-1)
        aw_i   = adzw1d[jnp.clip(k_idx + 1, 0, nz)]       # adzw(i)
        aw_ip1 = adzw1d[jnp.clip(k_idx + 2, 0, nz)]       # adzw(i+1)
        aw_ip2 = adzw1d[jnp.clip(k_idx + 3, 0, nz)]       # adzw(i+2)

        def _b(x): return x[:, None, None]   # broadcast (nz,) -> (nz,1,1)

        cn = cw_t   # shape (nz, ny, nx)

        # Stencil slices (same for all order variants)
        f_im3 = fp[0:nz]; f_im2 = fp[1:nz+1]; f_im1 = fp[2:nz+2]
        f_i   = fp[3:nz+3]; f_ip1 = fp[4:nz+4]; f_ip2 = fp[5:nz+5]

        # Precompute inverse denominators
        iaw_i   = 1.0 / _b(aw_i)
        iaz_i   = 1.0 / _b(az_i)
        iaz_im1 = 1.0 / _b(az_im1)
        iaz_ip1 = 1.0 / _b(az_ip1)
        iaz_im2 = 1.0 / _b(az_im2)

        # --- Shared linear term (also face_2nd_z entire result) ---
        # face_2nd_z = 0.5*(f_i + f_im1 - cn*adz(i)/adzw(i)*(f_i-f_im1))
        lin = f_i + f_im1 - cn * _b(az_i) * iaw_i * (f_i - f_im1)  # inner (without 0.5)

        # --- face_3rd_z non-uniform (Fortran: face_3rd_z, used at k=2 and k=nz-2) ---
        # Fortran returns 0.5*(lin_inner + positive_3rd + negative_3rd + sign*(pos-neg))
        # positive_3rd = 1/6*(cn^2*adz(i)^2/adzw(i)/adzw(i-1)-1)
        #               *(adzw(i-1)/adz(i-1)*(f_i-f_im1) - adzw(i)/adz(i-1)*(f_im1-f_im2))
        # negative_3rd = 1/6*(cn^2*adz(i)^2*adzw(i)/adzw(i+1)-1)
        #               *(adzw(i)/adz(i)*(f_ip1-f_i) - adzw(i+1)/adz(i)*(f_i-f_im1))
        p3 = ((1.0/6.0) * (cn*cn * _b(az_i*az_i) * iaw_i / _b(aw_im1) - 1.0)
              * (_b(aw_im1) * iaz_im1 * (f_i - f_im1) - _b(aw_i) * iaz_im1 * (f_im1 - f_im2)))
        n3 = ((1.0/6.0) * (cn*cn * _b(az_i*az_i) * _b(aw_i) / _b(aw_ip1) - 1.0)
              * (_b(aw_i) * iaz_i * (f_ip1 - f_i) - _b(aw_ip1) * iaz_i * (f_i - f_im1)))
        fz3 = 0.5 * (lin + p3 + n3 + jnp.sign(cn) * (p3 - n3))

        # --- face_5th_z non-uniform (interior levels k=3..nz-3) ---
        # Fortran returns 0.5*(lin_inner + c3 + c_asym + p5 + n5 + sign*(p5-n5))
        # (a) 3rd-order correction (Fortran line 170-172):
        # +1/3*(cn^2*adz(i)^2/adzw(i+1)/adzw(i-1)-1)
        #      *(adzw(i-1)/(adz(i)+adz(i-1))*(f_ip1-f_i) - adzw(i+1)/(adz(i)+adz(i-1))*(f_im1-f_im2))
        denom3 = _b(az_i + az_im1)
        c3_coeff = (1.0/3.0) * (cn*cn * _b(az_i*az_i) / (_b(aw_ip1) * _b(aw_im1)) - 1.0)
        c3 = c3_coeff * (_b(aw_im1) / denom3 * (f_ip1 - f_i)
                        - _b(aw_ip1) / denom3 * (f_im1 - f_im2))

        # (b) Asymmetric 5th correction (Fortran line 173-176):
        # -1/12*(cn^2*adz(i)^2*adzw(i-1)/adzw(i+1)-1)*cn*adz(i)/adzw(i)
        #      *(adzw(i-1)/adz(i)*(f_ip1-f_i)
        #        - adzw(i+1)*adzw(i-1)*(adz(i-1)+adz(i))/adzw(i)/adz(i)/adz(i-1)*(f_i-f_im1)
        #        + adzw(i+1)/adz(i-1)*(f_im1-f_im2))
        # Note: outer coefficient uses adzw(i-1) NOT inverted (Fortran line 173)
        c_asym_coeff = ((-1.0/12.0)
                        * (cn*cn * _b(az_i*az_i) * _b(aw_im1) / _b(aw_ip1) - 1.0)
                        * cn * _b(az_i) * iaw_i)
        c_asym = c_asym_coeff * (
              _b(aw_im1) * iaz_i * (f_ip1 - f_i)
            - _b(aw_ip1 * aw_im1) * _b(az_im1 + az_i) * iaw_i * iaz_i * iaz_im1 * (f_i - f_im1)
            + _b(aw_ip1) * iaz_im1 * (f_im1 - f_im2)
        )

        # (c) positive_5th (Fortran line 150-157):
        # 1/120*(cn^2*adz^2/adzw(i)/adzw(i-1)-1)*(cn^2*adz^2/adzw(i+1)/adzw(i-2)-4)
        # * (  adzw(i-1)*adzw(i-2)/adz(i)/adz(i-1)*(f_ip1-f_i)
        #    - adzw(i+1)*adzw(i-2)*(adzw(i-1)*adz(i-1)+adzw(i-1)*adz(i)+adzw(i)*adz(i))
        #       /adzw(i)/adz(i)/adz(i-1)^2*(f_i-f_im1)
        #    + adzw(i+1)*adzw(i-2)*(adzw(i-1)*adz(i-2)+adzw(i)*adz(i-2)+adzw(i)*adz(i-1))
        #       /adzw(i-1)/adz(i-1)^2/adz(i-2)*(f_im1-f_im2)
        #    - adzw(i+1)*adzw(i)/adz(i-1)/adz(i-2)*(f_im2-f_im3) )
        p5_a = ((1.0/120.0)
                * (cn*cn * _b(az_i*az_i) / (_b(aw_i) * _b(aw_im1)) - 1.0)
                * (cn*cn * _b(az_i*az_i) / (_b(aw_ip1) * _b(aw_im2)) - 4.0))
        p5 = p5_a * (
              _b(aw_im1 * aw_im2) * iaz_i * iaz_im1 * (f_ip1 - f_i)
            - _b(aw_ip1 * aw_im2) * _b(aw_im1*az_im1 + aw_im1*az_i + aw_i*az_i)
              * iaw_i * iaz_i * iaz_im1 * iaz_im1 * (f_i - f_im1)
            + _b(aw_ip1 * aw_im2) * _b(aw_im1*az_im2 + aw_i*az_im2 + aw_i*az_im1)
              / _b(aw_im1) * iaz_im1 * iaz_im1 * iaz_im2 * (f_im1 - f_im2)
            - _b(aw_ip1 * aw_i) * iaz_im1 * iaz_im2 * (f_im2 - f_im3)
        )

        # (d) negative_5th (Fortran line 159-166):
        # 1/120*(cn^2*adz^2/adzw(i+1)/adzw(i)-1)*(cn^2*adz^2/adzw(i+2)/adzw(i-1)-4)
        # * (  adzw(i)*adzw(i-1)/adz(i+1)/adz(i)*(f_ip2-f_ip1)
        #    - adzw(i+2)*adzw(i-1)*(adzw(i)*adz(i)+adzw(i)*adz(i+1)+adzw(i+1)*adz(i+1))
        #       /adzw(i+1)/adz(i+1)/adz(i)^2*(f_ip1-f_i)
        #    + adzw(i+2)*adzw(i-1)*(adzw(i)*adz(i-1)+adzw(i+1)*adz(i-1)+adzw(i+1)*adz(i))
        #       /adzw(i)/adz(i)^2/adz(i-1)*(f_i-f_im1)
        #    - adzw(i+2)*adzw(i+1)/adz(i)/adz(i-1)*(f_im1-f_im2) )
        n5_a = ((1.0/120.0)
                * (cn*cn * _b(az_i*az_i) / (_b(aw_ip1) * _b(aw_i)) - 1.0)
                * (cn*cn * _b(az_i*az_i) / (_b(aw_ip2) * _b(aw_im1)) - 4.0))
        n5 = n5_a * (
              _b(aw_i * aw_im1) * iaz_ip1 * iaz_i * (f_ip2 - f_ip1)
            - _b(aw_ip2 * aw_im1) * _b(aw_i*az_i + aw_i*az_ip1 + aw_ip1*az_ip1)
              / _b(aw_ip1) * iaz_ip1 * iaz_i * iaz_i * (f_ip1 - f_i)
            + _b(aw_ip2 * aw_im1) * _b(aw_i*az_im1 + aw_ip1*az_im1 + aw_ip1*az_i)
              * iaw_i * iaz_i * iaz_i * iaz_im1 * (f_i - f_im1)
            - _b(aw_ip2 * aw_ip1) * iaz_i * iaz_im1 * (f_im1 - f_im2)
        )

        fz5 = 0.5 * (lin + c3 + c_asym + p5 + n5 + jnp.sign(cn) * (p5 - n5))

        # Select order by level — matches gSAM face_z_5th (advect_um_lib.f90:250-279).
        # jsam fz_t[k] = Fortran fz(k+2), so the mapping is:
        # k=0  (fz(2)):      face_2nd_z with i=2
        # k=1  (fz(3)):      face_3rd_z with i=3
        # k=2..nz-4 (fz(4..nzm-2)): face_5th_z
        # k=nz-3 (fz(nzm-1)): face_3rd_z with i=nzm-1
        # k=nz-2 (fz(nzm)):  face_2nd_z with i=nzm
        # k=nz-1 (fz(nzm+1)=fz(nz)): top rigid-lid BC = 0
        fz2 = 0.5 * lin   # face_2nd_z = 0.5 * lin_inner
        k_arr = jnp.arange(nz)[:, None, None]
        fz_t_in = jnp.where(k_arr == 0,          fz2,   # face_2nd_z  (Fortran fz(2))
                   jnp.where(k_arr == 1,          fz3,   # face_3rd_z  (Fortran fz(3))
                   jnp.where(k_arr == nz - 3,     fz3,   # face_3rd_z  (Fortran fz(nzm-1))
                   jnp.where(k_arr == nz - 2,     fz2,   # face_2nd_z  (Fortran fz(nzm))
                   jnp.where(k_arr == nz - 1,     0.0,   # top rigid-lid BC (Fortran fz(nz))
                              fz5)))))                     # face_5th_z  (interior)
        return fz_t_in

    def _adv_update_z(fadv_in, fz_t_in):
        fz_b_in = jnp.concatenate([jnp.zeros_like(fz_t_in[:1]), fz_t_in[:-1]], axis=0)
        update = _adv_cn(cw_b, cw_t) * (fz_b_in - fz_t_in)
        # Fix 1.7: gSAM adv_form_update_z loops k=2..nzm-1 (Fortran 1-based),
        # skipping k=1 (bottom) and k=nzm (top).  In 0-based terms: skip index 0
        # and index nz-1.
        mask = jnp.ones(nz).at[0].set(0.0).at[-1].set(0.0)
        return fadv_in + update * mask[:, None, None], fz_b_in

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

    # Fix 1.6: gSAM fct3D (advect_um_lib.f90:478-488) computes both scale
    # factors as (f - mn) / (out_flux * irho(k) + eps) where irho(k) = 1/rho(k).
    # The entire sum (horizontal + vertical/iadz terms) is multiplied by irho
    # before adding eps.  Match this density weighting in the denominator.
    out_flux = (jnp.maximum(0.0, afx_e) - jnp.minimum(0.0, afx_w)
              + jnp.maximum(0.0, afy_n) - jnp.minimum(0.0, afy_s)
              + (jnp.maximum(0.0, afz_t) - jnp.minimum(0.0, afz_b)) * iadz)
    scale_out = (f_up - mn) / (out_flux * irho + eps)

    in_flux = (jnp.maximum(0.0, afx_w) - jnp.minimum(0.0, afx_e)
             + jnp.maximum(0.0, afy_s) - jnp.minimum(0.0, afy_n)
             + (jnp.maximum(0.0, afz_b) - jnp.minimum(0.0, afz_t)) * iadz)
    scale_in = (mx - f_up) / (in_flux * irho + eps)
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


@functools.partial(jax.jit, static_argnums=(6,))
def _advect_scalars_batch_jit(
    fields:      jax.Array,   # (N, nz, ny, nx) — N scalars stacked
    U:           jax.Array,
    V:           jax.Array,
    W:           jax.Array,
    metric:      dict,
    dt:          float,
    macho_order: int,         # static — same for all N scalars this step
) -> jax.Array:               # (N, nz, ny, nx)
    """Batch-advect N scalar fields in one vmapped JIT call."""
    return jax.vmap(
        lambda phi: _advect_scalar_jit(phi, U, V, W, metric, dt, macho_order),
    )(fields)


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
    """Spherical mass-weighted momentum advective tendencies.

    Uses 2nd-order centered differences (gSAM nadv_mom=2, advect2_mom_xy +
    advect2_mom_z).  The centered flux formula at face i+1/2 is:
        F = 0.25 * (u1_L + u1_R) * (phi_L + phi_R)
    where u1 are mass-flux-weighted advective velocities and phi is the
    transported momentum component.
    """
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

    # Mass-flux-weighted advective velocities (same as before — verified
    # against gSAM advect_mom.f90: gu = mu*rho*ady*adz, etc.)
    u1 = U * (rho[:, None, None] * dtdx
              * adz[:, None, None] * ady[None, :, None])
    v1 = V * (rho[:, None, None] * dtdy
              * adz[:, None, None] * muv[None, :, None])
    w1 = W * (rhow[:, None, None] * dtdz
              * ady[None, :, None] * mu[None, :, None])

    # Grid Jacobians (1/g) for tendency normalisation.
    # Clamp mu/muv to 1e-5 (matches gSAM advect_mom.f90: gv=max(1.e-5,muv)*...).
    mu_safe  = jnp.maximum(mu,  1e-5)
    muv_safe = jnp.maximum(muv, 1e-5)
    gu3 = (mu_safe[None, :, None] * rho[:, None, None]
           * ady[None, :, None] * adz[:, None, None])
    gv3_int = (muv_safe[None, 1:ny, None] * rho[:, None, None]
               * adyv[None, 1:ny, None] * adz[:, None, None])
    gw3_int = (mu_safe[None, :, None] * rhow[1:nz, None, None]
               * ady[None, :, None] * adzw[1:nz, None, None])

    # ------------------------------------------------------------------
    # U tendency
    # advect2_mom_xy (x-direction, 2D branch j=1):
    #   fu(i,j) = 0.25*(u1(ic,j,k)+u1(i,j,k))*(u(i,j,k)+u(ic,j,k))
    #   dudt -= igu*(fu(i,j)-fu(ib,j))
    # Here u1 lives on cell centres (same staggering as U on a C-grid with
    # a redundant east column).  u1_c = u1[:,:,:nx], u1_cp = u1[:,:,1:nx+1]
    # is u1 at the cell to the east (ic face = i+1 in Fortran 1-based).
    # ------------------------------------------------------------------
    U_c   = U[:, :, :nx]
    u1_c  = u1[:, :, :nx]
    u1_cp = u1[:, :, 1:nx+1]          # u1 at eastern neighbour (ic = i+1)

    # x-flux of U: F_{i+1/2} = 0.25*(u1_i + u1_{i+1})*(U_i + U_{i+1})
    fu_x = 0.25 * (u1_c + u1_cp) * (U_c + jnp.roll(U_c, -1, axis=2))
    dU_x = -(fu_x - jnp.roll(fu_x, +1, axis=2)) / gu3

    # y-flux of U (3D gSAM): fu(i,j)=0.25*(v1(i,jc,k)+v1(ib,jc,k))*(u(i,j,k)+u(i,jc,k))
    # v1 has shape (nz, ny+1, nx+1); ib = i-1 → roll(v1, +1, axis=2)
    # The face j+1/2 uses v1 at (i, jc=j+1) and (ib=i-1, jc).
    # v1_jc_i  = v1[:, 1:ny+1, :nx]   (at j+1, column i)
    # v1_jc_im1= v1[:, 1:ny+1, :nx] rolled +1 in x → v1[:, 1:ny+1, :] roll +1 sliced
    v1_jc    = v1[:, 1:ny+1, :nx]                        # shape (nz, ny, nx)
    v1_jc_xm = jnp.roll(v1[:, 1:ny+1, :nx], +1, axis=2) # v1 at (ib, jc)
    # U at (i,j) and (i,jc): U_c and U_c shifted +1 in y
    U_c_yn   = jnp.concatenate([U_c[:, 1:, :], U_c[:, -1:, :]], axis=1)  # U at j+1 (periodic/edge)
    fu_y_n   = 0.25 * (v1_jc + v1_jc_xm) * (U_c + U_c_yn)               # flux at j+1/2
    fu_y_s   = jnp.concatenate([jnp.zeros_like(fu_y_n[:, :1, :]),
                                 fu_y_n[:, :-1, :]], axis=1)               # flux at j-1/2
    dU_y = -(fu_y_n - fu_y_s) / gu3

    # z-flux of U: fuz(k) = 0.25*(w1(i,j,k)+w1(i-1,j,k))*(u(i,j,k)+u(i,j,kb))
    # w1 shape (nz+1, ny, nx+1); w1(i-1,j,k) → roll w1 +1 in x
    # Face at k (between k-1 and k), for k=2..nzm (interior).
    # w1_k_i   = w1[1:nz, :, :nx]   (level k, column i)
    # w1_k_im1 = roll(w1[1:nz, :, :nx], +1, axis=2)
    w1_int    = w1[1:nz, :, :nx]                       # shape (nz-1, ny, nx)
    w1_int_xm = jnp.roll(w1[1:nz, :, :nx], +1, axis=2)
    U_kb      = U_c[0:nz-1, :, :]                      # U at k-1
    U_k       = U_c[1:nz,   :, :]                      # U at k
    fuz_int   = 0.25 * (w1_int + w1_int_xm) * (U_k + U_kb)  # shape (nz-1, ny, nx)
    fuz_full  = jnp.concatenate([jnp.zeros_like(fuz_int[:1]),
                                  fuz_int,
                                  jnp.zeros_like(fuz_int[:1])], axis=0)   # (nz+1, ny, nx)
    # tendency: dudt -= igu*(fuz(kc)-fuz(k))  where kc=k+1
    dU_z = -(fuz_full[1:nz+1] - fuz_full[0:nz]) / gu3

    dU = dU_x + dU_y + dU_z

    # ------------------------------------------------------------------
    # V tendency — V_prog lives at interior y-faces j=1..ny-1 (0-based)
    # advect2_mom_xy x-direction (3D):
    #   fv(i,j)=0.25*(u1(ic,j,k)+u1(ic,jb,k))*(v(i,j,k)+v(ic,j,k))
    # ic=i+1 (east face), jb=j-1.
    # v1 is on (nz, ny+1, nx+1); V_prog is V[:, 1:ny, :].
    # u1 at (ic, j, k): u1[:, 1:ny, 1:nx+1]   (east column of interior rows)
    # u1 at (ic, jb, k): u1[:, 0:ny-1, 1:nx+1]
    # ------------------------------------------------------------------
    V_prog = V[:, 1:ny, :]                               # (nz, ny-1, nx+1)
    V_c    = V_prog[:, :, :nx]                           # (nz, ny-1, nx)
    V_cp   = jnp.roll(V_c, -1, axis=2)                   # V at i+1

    u1_e_j  = u1[:, 1:ny,   1:nx+1]   # u1 at (ic=i+1, j,  k): (nz, ny-1, nx)
    u1_e_jb = u1[:, 0:ny-1, 1:nx+1]   # u1 at (ic=i+1, jb, k): (nz, ny-1, nx)
    fv_x    = 0.25 * (u1_e_j + u1_e_jb) * (V_c + V_cp)
    dV_x    = -(fv_x - jnp.roll(fv_x, +1, axis=2)) / gv3_int

    # y-flux of V: fv(i,j)=0.25*(v1(i,jc,k)+v1(i,j,k))*(v(i,j,k)+v(i,jc,k))
    # jc=j+1; v1 at j → v1[:, 1:ny, :nx], v1 at jc → v1[:, 2:ny+1, :nx]
    v1_j  = v1[:, 1:ny,   :nx]   # v1 at j   (nz, ny-1, nx)
    v1_jc2 = v1[:, 2:ny+1, :nx]  # v1 at j+1 (nz, ny-1, nx)
    V_jc  = V[:, 2:ny+1, :nx]    # V at jc=j+1 (nz, ny-1, nx)
    fv_y_n = 0.25 * (v1_jc2 + v1_j) * (V_c + V_jc)
    fv_y_s = jnp.concatenate([jnp.zeros_like(fv_y_n[:, :1, :]),
                               fv_y_n[:, :-1, :]], axis=1)
    dV_y = -(fv_y_n - fv_y_s) / gv3_int

    # z-flux of V: fvz(k)=0.25*(w1(i,j,k)+w1(i,jb,k))*(v(i,j,k)+v(i,j,kb))
    # w1 at (i,j,k) = w1[1:nz, 1:ny, :nx]; w1 at (i,jb=j-1,k) = w1[1:nz, 0:ny-1, :nx]
    w1_j_int  = w1[1:nz, 1:ny,   :nx]   # (nz-1, ny-1, nx)
    w1_jb_int = w1[1:nz, 0:ny-1, :nx]
    V_k       = V_c[1:nz,   :, :]        # V_prog at k
    V_kb      = V_c[0:nz-1, :, :]        # V_prog at k-1
    fvz_int   = 0.25 * (w1_j_int + w1_jb_int) * (V_k + V_kb)
    fvz_full  = jnp.concatenate([jnp.zeros_like(fvz_int[:1]),
                                  fvz_int,
                                  jnp.zeros_like(fvz_int[:1])], axis=0)
    dV_z = -(fvz_full[1:nz+1] - fvz_full[0:nz]) / gv3_int

    dV = dV_x + dV_y + dV_z

    # ------------------------------------------------------------------
    # W tendency — W_core lives at interior z-faces k=1..nzm-1 (0-based)
    # advect2_mom_xy x-direction:
    #   fw(i,j)=0.25*(u1(ic,j,k)+u1(ic,j,kcu))*(w(i,j,kc)+w(ic,j,kc))
    # kc=k+1, kcu=min(kc,nzm); W_core = W[1:nz, :, :]; its level index maps
    # to kc=1..nzm-1 in Fortran 1-based → kc in 0-based = 1..nz-1.
    # u1(ic,j,k)   = u1[k-1, :, 1:nx+1]  (k-1 because k goes 1..nzm in Fortran
    #                                       and W_core[0] = W[1])
    # u1(ic,j,kcu) = u1[min(k,nzm-1), :, 1:nx+1]  → u1[k, :, 1:nx+1] clamped
    # For W_core level l (0-based, l=0..nz-2): k_f = l+1, kc_f = l+2, kcu_f=min(l+2,nzm)
    #   u1 index for k  : l   (0-based)
    #   u1 index for kcu: min(l+1, nz-1)
    # ------------------------------------------------------------------
    W_core = W[1:nz, :, :]                  # (nz-1, ny, nx+1)
    W_c    = W_core[:, :, :nx]              # (nz-1, ny, nx)
    W_cp   = jnp.roll(W_c, -1, axis=2)     # W at i+1

    # u1 at (ic=i+1, j, k) and (ic, j, kcu):
    # l = 0..nz-2; k-index in u1 = l, kcu-index = min(l+1, nz-1)
    u1_e_k    = u1[0:nz-1, :, 1:nx+1]      # (nz-1, ny, nx)
    u1_e_kcu  = u1[1:nz,   :, 1:nx+1]      # (nz-1, ny, nx); clamp handled by pad below
    # clamp: at top level (l=nz-2), kcu should saturate at nzm=nz-1 (0-based nz-2 in u1)
    # u1 has shape (nz, ...) so u1[1:nz] goes up to u1[nz-1] which is fine (nz-1 = nzm in 0-based)
    fw_x   = 0.25 * (u1_e_k + u1_e_kcu) * (W_c + W_cp)
    dW_x   = -(fw_x - jnp.roll(fw_x, +1, axis=2)) / gw3_int

    # y-flux of W: fw(i,j)=0.25*(v1(i,jc,k)+v1(i,jc,kcu))*(w(i,j,kc)+w(i,jc,kc))
    # v1 at (i,jc=j+1,k) = v1[l, j+1, :nx]; v1 at (i,jc,kcu) = v1[l+1, j+1, :nx] clamped
    v1_jc_k   = v1[0:nz-1, 1:ny+1, :nx]   # (nz-1, ny, nx)
    v1_jc_kcu = v1[1:nz,   1:ny+1, :nx]   # (nz-1, ny, nx)
    # W is (nz+1, ny, nx+1) — no ghost cell in y; Neumann-pad north before j+1 access
    W_pad_y   = jnp.pad(W, ((0,0), (0,1), (0,0)), mode='edge')
    W_jc      = W_pad_y[1:nz, 1:ny+1, :nx]  # W at jc=j+1 (nz-1, ny, nx)
    fw_y_n    = 0.25 * (v1_jc_k + v1_jc_kcu) * (W_c + W_jc)
    fw_y_s    = jnp.concatenate([jnp.zeros_like(fw_y_n[:, :1, :]),
                                  fw_y_n[:, :-1, :]], axis=1)
    dW_y = -(fw_y_n - fw_y_s) / gw3_int

    # z-flux of W: fwz(k)=0.25*(w1(i,j,kc)+w1(i,j,k))*(w(i,j,kc)+w(i,j,k))
    # kc=k+1; for interior W faces at level m (0-based, m=1..nzm-1):
    # W at kc=m+1 = W[m+1], W at k=m = W[m]
    # w1 at kc = w1[m+1], w1 at k = w1[m]
    # fwz lives at half-levels between W levels; we need fwz at k and kb=k-1
    # for dwdt(k) = -(fwz(k)-fwz(kb))/igw
    # W_core[l] = W[l+1]; l=0..nz-2 (interior W levels)
    # fwz at level l (between W[l] and W[l+1]):
    #   fwz[l] = 0.25*(w1[l+1]+w1[l])*(W[l+1]+W[l])
    # tendency for W_core[l]: -(fwz[l] - fwz[l-1])/gw3_int[l]
    # Boundary: fwz at l=-1 (bottom) = 0, fwz at l=nz-1 (top) = 0
    w1_kc = w1[1:nz,   :, :nx]    # w1 at kc (nz-1, ny, nx)
    w1_k  = w1[0:nz-1, :, :nx]    # w1 at k  (nz-1, ny, nx)
    W_kc  = W[1:nz,    :, :nx]    # W  at kc (nz-1, ny, nx)
    W_k   = W[0:nz-1,  :, :nx]    # W  at k  (nz-1, ny, nx)
    fwz   = 0.25 * (w1_kc + w1_k) * (W_kc + W_k)   # (nz-1, ny, nx)
    fwz_full = jnp.concatenate([jnp.zeros_like(fwz[:1]),
                                 fwz,
                                 jnp.zeros_like(fwz[:1])], axis=0)   # (nz+1, ny, nx)
    # dwdt(k) -= igw*(fwz(k)-fwz(kb)); W_core[l] = W[l+1]:
    # fwz at l=W_core level → fwz_full[l+1] (upper face), fwz_full[l] (lower face)
    dW_z = -(fwz_full[1:nz] - fwz_full[0:nz-1]) / gw3_int

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
    """One timestep of 2nd-order centered momentum advection on a lat-lon C-grid."""
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
