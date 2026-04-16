"""
Polar velocity damping for lat-lon grids.
Matches gSAM damping.f90 dodamping_poles branch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def pole_damping(
    U: jax.Array,        # (nz, ny, nx+1)  zonal velocity on east faces
    V: jax.Array,        # (nz, ny+1, nx)  meridional velocity on north faces
    lat_rad: jax.Array,  # (ny,) mass-cell latitudes in radians
    dx: float,           # equatorial grid spacing (m) = R * dlon_rad
    dy,                  # meridional spacing (unused; kept for API symmetry). Post-Gap-8 callers pass metric["dy_lat"] (ny array) or metric["dy_lat_ref"] (scalar); both work.
    dt: float,           # timestep (s)
    cu: float = 0.3,     # Courant-number cap (gSAM damping_u_cu default)
    pres: jax.Array | None = None,  # (nz,) pressure (Pa) for upper-level damping
    p_upper: float = 7000.0,        # threshold (Pa): full damping above this (gSAM: 70 hPa = 7000 Pa)
) -> tuple[jax.Array, jax.Array]:
    """Apply gSAM-style polar and upper-level velocity damping."""
    cos_lat = jnp.cos(lat_rad)
    sin2    = 1.0 - cos_lat ** 2

    tau_lat = sin2 ** 200

    umax = cu * dx * cos_lat / dt

    if pres is not None:
        tau_upper = jnp.where(pres < p_upper, 1.0, 0.0)
        tau_u_2d = jnp.maximum(tau_lat[None, :], tau_upper[:, None])
    else:
        tau_u_2d = jnp.broadcast_to(tau_lat[None, :], (U.shape[0], len(lat_rad)))

    tau_u  = tau_u_2d[:, :, None]
    umax_u = umax[None, :, None]
    U_clipped = jnp.clip(U, -umax_u, umax_u)
    U_new = (U + U_clipped * tau_u) / (1.0 + tau_u)

    tau_v_int = 0.5 * (tau_lat[:-1] + tau_lat[1:])
    umax_v_int = 0.5 * (umax[:-1] + umax[1:])

    tau_v_1d = jnp.concatenate([jnp.ones(1), tau_v_int, jnp.ones(1)])
    umax_v   = jnp.concatenate([jnp.zeros(1), umax_v_int, jnp.zeros(1)])

    if pres is not None:
        tau_v_2d = jnp.maximum(tau_v_1d[None, :], tau_upper[:, None])     # (nz, ny+1)
    else:
        tau_v_2d = jnp.broadcast_to(tau_v_1d[None, :], (V.shape[0], V.shape[1]))

    tau_v3  = tau_v_2d[:, :, None]    # (nz, ny+1, 1)
    umax_v3 = umax_v[None, :, None]   # (1, ny+1, 1)
    V_clipped = jnp.clip(V, -umax_v3, umax_v3)        # (nz, ny+1, nx)
    V_new = (V + V_clipped * tau_v3) / (1.0 + tau_v3)  # (nz, ny+1, nx)

    return U_new, V_new


@jax.jit
def spectral_polar_filter(
    U: jax.Array,              # (nz, ny, nx+1)  zonal velocity
    V: jax.Array,              # (nz, ny+1, nx)  meridional velocity
    mask_u: jax.Array,         # (ny,    nm)  float64 0/1 mask for U rows
    mask_v: jax.Array,         # (ny-1,  nm)  float64 0/1 mask for interior V rows
) -> tuple[jax.Array, jax.Array]:
    """Spectral polar filter. Zeroes high zonal wavenumbers at each latitude."""
    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1
    nm = nx // 2 + 1

    U_main = U[:, :, :nx]
    U_hat  = jnp.fft.rfft(U_main, axis=2)
    U_hat  = U_hat * mask_u[None, :, :]
    U_filt = jnp.fft.irfft(U_hat, n=nx, axis=2)
    U_new  = jnp.concatenate([U_filt, U_filt[:, :, :1]], axis=2)

    V_int  = V[:, 1:-1, :]
    V_hat  = jnp.fft.rfft(V_int, axis=2)
    V_hat  = V_hat * mask_v[None, :, :]
    V_filt = jnp.fft.irfft(V_hat, n=nx, axis=2)
    V_new  = jnp.concatenate([V[:, :1, :], V_filt, V[:, -1:, :]], axis=1)

    return U_new, V_new


@jax.jit
def spectral_scalar_filter(
    field: jax.Array,          # (nz, ny, nx) scalar on mass-cell grid
    mask: jax.Array,           # (ny, nm) float64 0/1 mask (same as mask_u)
) -> jax.Array:
    """Spectral polar filter for scalar fields (mass-cell grid)."""
    nz, ny, nx = field.shape
    nm = nx // 2 + 1

    f_hat  = jnp.fft.rfft(field, axis=2)
    f_hat  = f_hat * mask[None, :, :]
    f_filt = jnp.fft.irfft(f_hat, n=nx, axis=2)

    return f_filt


def build_polar_filter_masks(
    lat_rad: "np.ndarray",   # (ny,) mass-cell latitudes in radians
    nx: int,
) -> tuple["jax.Array", "jax.Array"]:
    """Precompute polar-filter masks for U rows and interior V rows."""
    import numpy as np

    nm = nx // 2 + 1
    m  = np.arange(nm)

    cos_u    = np.cos(lat_rad)
    m_cut_u  = np.floor(nx / 2.0 * cos_u).astype(int)
    mask_u_np = (m[None, :] <= m_cut_u[:, None]).astype(np.float64)

    lat_v   = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    cos_v   = np.cos(lat_v)
    m_cut_v = np.floor(nx / 2.0 * cos_v).astype(int)
    mask_v_np = (m[None, :] <= m_cut_v[:, None]).astype(np.float64)

    return jnp.array(mask_u_np), jnp.array(mask_v_np)


@jax.jit
def gsam_w_sponge(
    W: jax.Array,        # (nz+1, ny, nx)
    z: jax.Array,        # (nz,) cell-centre heights (m)
    nub: float = 0.6,
    taudamp_max: float = 0.333,
    dtn: float = 1.0,    # current AB-weighted sub-timestep (s)
    dt: float = 1.0,     # base timestep (s)
) -> jax.Array:
    """gSAM-exact W-only Rayleigh sponge at model top (damping.f90:33-50)."""
    nz = z.shape[0]
    dz_half_bot = (z[1] - z[0]) * 0.5 if nz > 1 else z[0]
    dz_half_top = (z[-1] - z[-2]) * 0.5 if nz > 1 else z[0]
    zi_bot = z[0] - dz_half_bot
    zi_top = z[-1] - dz_half_top
    nu = (z - zi_bot) / (zi_top - zi_bot)
    nu_excess = jnp.clip((nu - nub) / (1.0 - nub), 0.0, 1.0)
    zzz = 100.0 * nu_excess ** 2
    tau_max = dtn / dt
    taudamp = jnp.where(nu > nub, tau_max * taudamp_max * zzz / (1.0 + zzz), 0.0)

    taudamp_w = jnp.concatenate([
        taudamp[:1],
        0.5 * (taudamp[:-1] + taudamp[1:]),
        taudamp[-1:],
    ])
    W_new = W / (1.0 + taudamp_w[:, None, None])
    return W_new


@jax.jit
def top_sponge(
    U: jax.Array,      # (nz, ny, nx+1)
    V: jax.Array,      # (nz, ny+1, nx)
    W: jax.Array,      # (nz+1, ny, nx)
    z: jax.Array,      # (nz,)  cell-centre heights (m)
    dt: float,
    z_sponge: float,   # height above which sponge starts (m)
    tau_sponge: float, # minimum damping time scale at model top (s)
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rayleigh damping sponge layer at the model top."""
    nz = z.shape[0]
    z_top = z[-1]

    frac    = jnp.clip((z - z_sponge) / (z_top - z_sponge), 0.0, 1.0)
    alpha   = jnp.sin(0.5 * jnp.pi * frac) ** 2

    factor  = 1.0 / (1.0 + alpha * dt / tau_sponge)

    U_new = U * factor[:, None, None]
    V_new = V * factor[:, None, None]

    alpha_w = jnp.concatenate([alpha[:1], 0.5 * (alpha[:-1] + alpha[1:]), alpha[-1:]])
    factor_w = 1.0 / (1.0 + alpha_w * dt / tau_sponge)
    W_new = W * factor_w[:, None, None]

    return U_new, V_new, W_new
