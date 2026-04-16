"""
Coriolis + metric acceleration for a global lat-lon grid.
Port of gSAM coriolis.f90 with dolatlon and docoriolisz branches.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def coriolis_tend(
    U:      jax.Array,   # (nz, ny, nx+1)
    V:      jax.Array,   # (nz, ny+1, nx)
    W:      jax.Array,   # (nz+1, ny, nx)
    metric: dict,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Spherical Coriolis + metric + f' coupling tendency."""
    fcory   = metric["fcory"]
    fcorzy  = metric["fcorzy"]
    tanr    = metric["tanr"]
    mu      = metric["cos_lat"]
    cos_v   = metric["cos_v"]
    ady     = metric["ady"]
    dz      = metric["dz"]

    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1

    adyv_int = 0.5 * (ady[:-1] + ady[1:])
    adyv = jnp.concatenate([ady[:1], adyv_int, ady[-1:]])
    imuv = jnp.where(cos_v > 0.0, 1.0 / cos_v, 0.0)

    dzw_int = 0.5 * (dz[:-1] + dz[1:])
    dzw = jnp.concatenate([dz[:1], dzw_int, dz[-1:]])

    U_left  = U[:, :, :nx]
    U_right = U[:, :, 1:nx + 1]

    f3    = fcory[None, :, None]
    fz3   = fcorzy[None, :, None]
    tanr3 = tanr[None, :, None]
    mu3   = mu[None, :, None]

    V_s = V[:, :ny,     :]
    V_n = V[:, 1:ny + 1, :]
    V_s_w = jnp.roll(V_s, +1, axis=-1)
    V_n_w = jnp.roll(V_n, +1, axis=-1)

    adyv_s = adyv[None, :ny,     None]
    adyv_n = adyv[None, 1:ny + 1, None]
    ady3   = ady[None, :, None]

    v_bar = (adyv_s * (V_s + V_s_w) + adyv_n * (V_n + V_n_w)) / (4.0 * ady3)
    dU = (f3 + U_left * tanr3) * v_bar

    def _q(u):
        return (f3 + u * tanr3) * mu3 * u

    q_row = _q(U_left) + _q(U_right)
    q_n = q_row[:, 1:ny,     :]
    q_s = q_row[:, 0:ny - 1, :]

    imuv_int = imuv[None, 1:ny, None]
    dV_int = -0.25 * imuv_int * (q_n + q_s)

    dV = jnp.concatenate([
        jnp.zeros((nz, 1, nx)),
        dV_int,
        jnp.zeros((nz, 1, nx)),
    ], axis=1)

    W_west = jnp.roll(W, +1, axis=-1)
    W_sum_lo = W     [:nz, :, :] + W_west[:nz, :, :]
    W_sum_hi = W     [1:,  :, :] + W_west[1:,  :, :]

    dzw_lo = dzw[:nz][:, None, None]
    dzw_hi = dzw[1:][:, None, None]
    dz3    = dz[:, None, None]

    dU_z = -0.25 * fz3 * (dzw_lo * W_sum_lo + dzw_hi * W_sum_hi) / dz3
    dU = dU + dU_z

    U_sum = U_left + U_right
    U_sum_above = U_sum[1:, :, :]
    U_sum_below = U_sum[:-1, :, :]

    dW_int = 0.25 * fz3 * (U_sum_above + U_sum_below)

    dW = jnp.concatenate([
        jnp.zeros((1, ny, nx)),
        dW_int,
        jnp.zeros((1, ny, nx)),
    ], axis=0)

    return dU, dV, dW
