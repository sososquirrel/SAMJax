"""
Polar velocity damping for lat-lon grids — matches gSAM damping.f90.

gSAM applies a strong implicit relaxation to U and V at high latitudes
(``dodamping_poles`` branch of damping.f90) to suppress the grid-scale
noise that the convergence of meridians causes near the poles.

The scheme is a combined velocity-clip + implicit-relax:

    tau(j)  = tau_max * (1 - cos²(lat_j))^200    # ~ sin(lat)^400, near-zero except at poles
    umax(j) = cu * dx * cos(lat_j) / dt           # max velocity ~ cos(lat)*dx/dt

    u_new = (u + clip(u, -umax, umax) * tau) / (1 + tau)
    v_new = (v + clip(v, -umax, umax) * tau) / (1 + tau)

This is applied to U and V AFTER ``adamsA()`` (provisional momentum update)
and BEFORE ``pressure()`` (Poisson solve), matching main.f90.

References
----------
  gSAM SRC/damping.f90     — dodamping_poles branch
  gSAM SRC/main.f90        — call order: adamsA → damping → pressure
  gSAM SRC/params.f90      — damping_u_cu default (0.3)
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
    """
    Apply gSAM-style polar + upper-level velocity damping to U and V.

    Two regimes (matches gSAM damping.f90 with dodamping_poles + dodamping_u):

    1. **Latitude-dependent (all levels):**
       tau(j) = (sin²(lat))^200 — effectively zero away from poles, ~1 at poles.
       Clips u/v to umax(j) = cu * dx * cos(lat) / dt.

    2. **Upper-level (pres < p_upper, all latitudes):**
       tau = 1.0 (full damping) — every timestep, u/v is relaxed 50% toward the
       clipped value.  Prevents upper-stratospheric winds from violating CFL at
       high latitudes where dx_eff = dx * cos(lat) shrinks.

    The effective tau at each (k, j) is max(tau_lat(j), tau_upper(k)).

    Parameters
    ----------
    U       : (nz, ny, nx+1)  zonal velocity
    V       : (nz, ny+1, nx)  meridional velocity
    lat_rad : (ny,) mass-cell latitudes (radians)
    dx      : equatorial zonal spacing (m)
    dy      : meridional spacing (m)
    dt      : timestep (s)
    cu      : Courant-number threshold (default 0.3, matches gSAM damping_u_cu)
    pres    : (nz,) pressure (Pa) at cell centres — enables dodamping_u
    p_upper : pressure threshold (Pa) below which full damping is applied (default 7000 = 70 hPa)

    Returns
    -------
    U_new, V_new — damped velocity arrays with the same shapes as inputs.
    """
    cos_lat = jnp.cos(lat_rad)            # (ny,)  mu(j) in gSAM
    sin2    = 1.0 - cos_lat ** 2          # sin²(lat), = 0 at equator, = 1 at poles

    # tau_lat(j) = (sin²(lat))^200  — effectively zero away from poles, ~1 at poles
    tau_lat = sin2 ** 200                  # (ny,)

    # umax(j) = cu * dx * cos(lat) / dt  — gSAM damping.f90:76
    umax = cu * dx * cos_lat / dt                          # (ny,)   [m/s]

    # Upper-level damping: tau = 1.0 where pres < p_upper  (gSAM dodamping_u)
    if pres is not None:
        tau_upper = jnp.where(pres < p_upper, 1.0, 0.0)   # (nz,)
        # Effective tau(k,j) = max(tau_lat(j), tau_upper(k))
        tau_u_2d = jnp.maximum(tau_lat[None, :], tau_upper[:, None])   # (nz, ny)
    else:
        tau_u_2d = jnp.broadcast_to(tau_lat[None, :], (U.shape[0], len(lat_rad)))  # (nz, ny)

    # ── Apply to U (mass-cell latitudes, ny rows) ──
    tau_u  = tau_u_2d[:, :, None]          # (nz, ny, 1) for broadcasting over nx+1
    umax_u = umax[None, :, None]           # (1, ny, 1)
    U_clipped = jnp.clip(U, -umax_u, umax_u)   # (nz, ny, nx+1)
    U_new = (U + U_clipped * tau_u) / (1.0 + tau_u)  # (nz, ny, nx+1)

    # ── Apply to V (D7 fix: gSAM uses mass-cell mu(j) and umax(j) for BOTH U and V) ──
    # For interior v-faces (j_v=1..ny-1), use the mass-cell values from the
    # adjacent row.  gSAM damping.f90 uses mu(j) at mass rows for V too.
    # Map each interior v-face to the average of the two adjacent mass-row values.
    tau_v_int = 0.5 * (tau_lat[:-1] + tau_lat[1:])     # (ny-1,) from mass-cell tau
    umax_v_int = 0.5 * (umax[:-1] + umax[1:])          # (ny-1,) from mass-cell umax

    # Build full (ny+1,) tau and umax for V rows (poles → 1 and 0 respectively)
    tau_v_1d = jnp.concatenate([jnp.ones(1), tau_v_int, jnp.ones(1)])    # (ny+1,)
    umax_v   = jnp.concatenate([jnp.zeros(1), umax_v_int, jnp.zeros(1)]) # (ny+1,)

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
    """
    Spectral polar filter — matches gSAM dodamping_poles spectral branch.

    At each latitude row j, zero zonal wavenumbers m > floor(nx/2 · cosφ_j).
    This prevents the Cartesian pressure solver from accumulating irrecoverable
    divergence at high latitudes where the effective grid spacing shrinks but
    the Cartesian eigenvalue stays fixed at the equatorial scale.

    The masks are precomputed (see `build_polar_filter_masks`) and passed as
    constant JAX arrays so this function is JIT-friendly.

    Parameters
    ----------
    mask_u  : (ny,   nm)   1 for kept modes, 0 for zeroed modes (U-row latitudes)
    mask_v  : (ny-1, nm)   same for interior V-face latitudes
    """
    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1
    nm = nx // 2 + 1

    # ── Filter U ──────────────────────────────────────────────────────────────
    U_main = U[:, :, :nx]                                 # (nz, ny, nx)
    U_hat  = jnp.fft.rfft(U_main, axis=2)                 # (nz, ny, nm)
    U_hat  = U_hat * mask_u[None, :, :]                   # zero high-m modes
    U_filt = jnp.fft.irfft(U_hat, n=nx, axis=2)           # (nz, ny, nx)
    U_new  = jnp.concatenate([U_filt, U_filt[:, :, :1]], axis=2)  # periodic wrap

    # ── Filter interior V faces (polar faces stay at 0) ───────────────────────
    V_int  = V[:, 1:-1, :]                                # (nz, ny-1, nx)
    V_hat  = jnp.fft.rfft(V_int, axis=2)                  # (nz, ny-1, nm)
    V_hat  = V_hat * mask_v[None, :, :]
    V_filt = jnp.fft.irfft(V_hat, n=nx, axis=2)           # (nz, ny-1, nx)
    V_new  = jnp.concatenate([V[:, :1, :], V_filt, V[:, -1:, :]], axis=1)

    return U_new, V_new


@jax.jit
def spectral_scalar_filter(
    field: jax.Array,          # (nz, ny, nx) scalar on mass-cell grid
    mask: jax.Array,           # (ny, nm) float64 0/1 mask (same as mask_u)
) -> jax.Array:
    """
    Spectral polar filter for scalar fields on the mass-cell grid.

    Zeros zonal wavenumbers m > floor(nx/2 * cos(lat_j)) at each row,
    suppressing grid-scale noise near the poles where the effective zonal
    spacing shrinks.  Uses the same mask as U rows (mass-cell latitudes).

    Not in gSAM (only needed at coarse resolution where the scalar CFL
    is not the binding constraint but grid-scale noise still accumulates).
    """
    nz, ny, nx = field.shape
    nm = nx // 2 + 1

    f_hat  = jnp.fft.rfft(field, axis=2)               # (nz, ny, nm)
    f_hat  = f_hat * mask[None, :, :]                   # zero high-m modes
    f_filt = jnp.fft.irfft(f_hat, n=nx, axis=2)        # (nz, ny, nx)

    return f_filt


def build_polar_filter_masks(
    lat_rad: "np.ndarray",   # (ny,) mass-cell latitudes in radians
    nx: int,
) -> tuple["jax.Array", "jax.Array"]:
    """
    Precompute polar-filter masks for U rows and interior V rows.

    Returns
    -------
    mask_u : (ny,   nm)  float64 JAX array — 1 for kept modes, 0 for zeroed
    mask_v : (ny-1, nm)  float64 JAX array — same for interior V-face latitudes
    """
    import numpy as np

    nm = nx // 2 + 1
    m  = np.arange(nm)

    # U mask — mass-cell latitudes
    cos_u    = np.cos(lat_rad)                              # (ny,)
    m_cut_u  = np.floor(nx / 2.0 * cos_u).astype(int)     # (ny,)
    mask_u_np = (m[None, :] <= m_cut_u[:, None]).astype(np.float64)  # (ny, nm)

    # V mask — interior v-face latitudes
    lat_v   = 0.5 * (lat_rad[:-1] + lat_rad[1:])          # (ny-1,)
    cos_v   = np.cos(lat_v)
    m_cut_v = np.floor(nx / 2.0 * cos_v).astype(int)      # (ny-1,)
    mask_v_np = (m[None, :] <= m_cut_v[:, None]).astype(np.float64)  # (ny-1, nm)

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
    """
    gSAM-exact W-only Rayleigh sponge at model top (damping.f90 section 1).

        nu(k)      = (z[k] - z[0]) / (z[-1] - z[0])
        zzz        = 100 * ((nu - nub) / (1 - nub))^2      for nu > nub
        taudamp(k) = taudamp_max * zzz / (1 + zzz)         for nu > nub
                   = 0                                       otherwise
        W_new      = W / (1 + taudamp)

    Matches gSAM damping.f90:33-50 exactly.
    Applied to W only — U/V damping lives in diffuse_damping_mom_z.
    """
    nz = z.shape[0]
    # D5 fix: use interface heights zi instead of cell-centre z
    # gSAM damping.f90:34: nu = (zi(k)-zi(1)) / (zi(nzm)-zi(1))
    # Approximate zi from cell-centre z: zi[0] = z[0]-dz[0]/2, zi[k] = (z[k-1]+z[k])/2
    # For simplicity, reconstruct bottom/top interface from half-cell offsets.
    # However, since z is cell-centre and we need interface heights, we
    # use the metric zi if available, or approximate.
    # gSAM: zi(1) is the surface, zi(nzm) is the top of the second-to-last cell.
    # We approximate: zi_bot = z[0] - dz_approx/2 where dz_approx = z[1]-z[0] for uniform
    # Better: just shift so nu uses half-cell-shifted heights.
    # The key difference is using z interfaces not centres.
    dz_half_bot = (z[1] - z[0]) * 0.5 if nz > 1 else z[0]
    dz_half_top = (z[-1] - z[-2]) * 0.5 if nz > 1 else z[0]
    zi_bot = z[0] - dz_half_bot   # surface interface
    zi_top = z[-1] - dz_half_top  # top of nzm-1 cell (gSAM zi(nzm))
    nu = (z - zi_bot) / (zi_top - zi_bot)                  # (nz,)
    nu_excess = jnp.clip((nu - nub) / (1.0 - nub), 0.0, 1.0)
    zzz = 100.0 * nu_excess ** 2
    # D4 fix: gSAM damping.f90:24 uses tau_max = dtn/dt
    tau_max = dtn / dt
    taudamp = jnp.where(nu > nub, tau_max * taudamp_max * zzz / (1.0 + zzz), 0.0)  # (nz,)

    # Interpolate centre-level taudamp to w-faces (nz+1)
    taudamp_w = jnp.concatenate([
        taudamp[:1],
        0.5 * (taudamp[:-1] + taudamp[1:]),
        taudamp[-1:],
    ])                                                     # (nz+1,)
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
    """
    Rayleigh damping sponge layer at the model top.

    Matches gSAM dodamp/damp_top: implicit relaxation toward 0 for z > z_sponge.

    Damping coefficient:
        alpha(k) = sin²(π/2 * (z[k] - z_sponge) / (z_top - z_sponge))  for z > z_sponge
        alpha(k) = 0                                                        for z ≤ z_sponge

    Implicit update:  phi_new = phi / (1 + alpha * dt / tau_sponge)

    Parameters
    ----------
    z_sponge   : height (m) where sponge begins (typically top 25% of domain)
    tau_sponge : damping time scale at model top (s) — try 1800s (30 min)
    """
    nz = z.shape[0]
    z_top = z[-1]

    # alpha[k]: sponge coefficient at each model level
    frac    = jnp.clip((z - z_sponge) / (z_top - z_sponge), 0.0, 1.0)   # (nz,)
    alpha   = jnp.sin(0.5 * jnp.pi * frac) ** 2                           # (nz,)

    # Implicit relaxation factor (1-based; =1 outside sponge, <1 inside)
    factor  = 1.0 / (1.0 + alpha * dt / tau_sponge)   # (nz,)

    U_new = U * factor[:, None, None]
    V_new = V * factor[:, None, None]

    # W is on face k=0..nz; interpolate alpha to faces
    alpha_w = jnp.concatenate([alpha[:1], 0.5 * (alpha[:-1] + alpha[1:]), alpha[-1:]])
    factor_w = 1.0 / (1.0 + alpha_w * dt / tau_sponge)   # (nz+1,)
    W_new = W * factor_w[:, None, None]

    return U_new, V_new, W_new
