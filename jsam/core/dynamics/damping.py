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
    dt: float,           # base timestep (s)
    cu: float = 0.3,     # Courant-number cap (gSAM damping_u_cu default)
    pres: jax.Array | None = None,  # (nz,) pressure (Pa) for upper-level damping
    p_upper: float = 7000.0,        # threshold (Pa): full damping above this (gSAM: 70 hPa = 7000 Pa)
    dtn: float | None = None,       # AB-weighted sub-timestep (s); defaults to dt
) -> tuple[jax.Array, jax.Array]:
    """Apply gSAM-style polar and upper-level velocity damping.

    Matches damping.f90 dodamping_poles exactly:
      - tau_max = dtn/dt  (Fix 5.7: upper-level uses tau_max, not 1.0)
      - tauy(j) and umax(j) applied at mass-cell j to both U and V
        without face interpolation (Fix 5.8: V uses mass-cell tau directly)
      - The extra north V-face (j=ny) is not damped (matches Fortran loop j=1..ny)
    """
    if dtn is None:
        dtn = dt
    tau_max = dtn / dt

    cos_lat = jnp.cos(lat_rad)
    sin2    = 1.0 - cos_lat ** 2

    # tauy and umax at mass-cell latitudes (Fortran: tauy(j), umax(j))
    tau_lat = tau_max * sin2 ** 200                  # (ny,)
    umax    = cu * dx * cos_lat / dtn               # (ny,) — gSAM uses dtn in denominator

    # ---- U (nz, ny, nx+1) ------------------------------------------------
    # Upper-level: use tau_max above p_upper (Fix 5.7)
    if pres is not None:
        tau_upper = jnp.where(pres < p_upper, tau_max, 0.0)  # (nz,)
        tau_u_2d = jnp.maximum(tau_lat[None, :], tau_upper[:, None])  # (nz, ny)
    else:
        tau_u_2d = jnp.broadcast_to(tau_lat[None, :], (U.shape[0], len(lat_rad)))

    tau_u  = tau_u_2d[:, :, None]   # (nz, ny, 1)
    umax_u = umax[None, :, None]    # (1, ny, 1)
    U_clipped = jnp.clip(U, -umax_u, umax_u)
    U_new = (U + U_clipped * tau_u) / (1.0 + tau_u)

    # ---- V (nz, ny+1, nx) ------------------------------------------------
    # Fix 5.8: apply mass-cell tau_lat[j] and umax[j] directly to V[:,j,:]
    # for j=0..ny-1 (matching Fortran v(i,j,k) at mass-cell j=1..ny).
    # The extra north face V[:,ny,:] is outside the Fortran loop — leave undamped.
    tau_v_mass = tau_lat              # (ny,)  — no interpolation
    umax_v_mass = umax               # (ny,)

    if pres is not None:
        tau_v_interior_2d = jnp.maximum(
            tau_v_mass[None, :], tau_upper[:, None]
        )  # (nz, ny)
    else:
        tau_v_interior_2d = jnp.broadcast_to(
            tau_v_mass[None, :], (V.shape[0], len(lat_rad))
        )  # (nz, ny)

    # Pad a zero-tau column for the extra north V-face (j=ny) so shape is (nz, ny+1)
    tau_v_2d = jnp.concatenate(
        [tau_v_interior_2d, jnp.zeros((V.shape[0], 1))], axis=1
    )  # (nz, ny+1)
    umax_v_1d = jnp.concatenate([umax_v_mass, jnp.zeros(1)])  # (ny+1,)

    tau_v3  = tau_v_2d[:, :, None]          # (nz, ny+1, 1)
    umax_v3 = umax_v_1d[None, :, None]      # (1, ny+1, 1)
    V_clipped = jnp.clip(V, -umax_v3, umax_v3)
    V_new = jnp.where(
        tau_v3 > 0.0,
        (V + V_clipped * tau_v3) / (1.0 + tau_v3),
        V,
    )

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
def gsam_w_courant_damping(
    W: jax.Array,        # (nz+1, ny, nx)
    metric: dict,        # must contain "zi", "dz_ref", "adzw"
    dtn: float,          # AB-weighted sub-timestep (s) — CFL denominator
    damping_w_cu: float, # Courant-number cap for W (gSAM damping_w_cu)
    tau_max: float,      # sponge damping coefficient = dtn/dt
    nub: float = 0.6,    # normalized height above which top sponge is active
) -> jax.Array:
    """gSAM damping.f90 dodamping_w: clip W to a CFL-based maximum at levels
    below the top sponge (where taudamp == 0).

    Matches Fortran exactly (damping.f90 lines 54-69):
      taudamp(k) is zero only for k where nu <= nub (below the sponge).
      For those levels: wmax = damping_w_cu * dz * adzw(k) / dtn
      Soft-clip: W_new = (W + clip(W,-wmax,wmax)*tau_max) / (1+tau_max)

    Index mapping (Fortran 1-indexed k=1..nzm):
      Fortran w(i,j,k) -> Python W[k] for k=1..nz  (W-face k)
      Fortran dz scalar base spacing -> metric["dz_ref"]
      Fortran adzw(k)  -> Python metric["adzw"][k-1]  (zi declared as zi(nz), 1-indexed)
      Fortran zi(k)    -> Python zi[k-1]              (zi declared as zi(nz), 1-indexed)
      taudamp(k)==0    <=>  nu[k] = (zi[k-1]-zi[0])/(zi[-1]-zi[0]) <= nub
    """
    zi     = metric["zi"]      # (nz+1,)
    dz_ref = metric["dz_ref"]  # scalar base grid spacing (m)
    adzw   = metric["adzw"]    # (nz+1,) — stretch factor at W-faces

    # Fortran zi is declared as zi(nz), 1-indexed (no zi(0)).
    # Python metric["zi"] has shape (nz+1,); Fortran zi(nz) has nz elements.
    # Fortran nzm = nz - 1.  Fortran zi(k) for k=1..nzm -> Python zi[k-1] = zi[0..nzm-1] = zi[:-2].
    # Fortran zi(1) = Python zi[0].
    # Fortran zi(nzm) = Python zi[nzm-1] = Python zi[-2]  (NOT zi[-1]).
    # nu for Fortran k=1..nzm: nu = (zi(k)-zi(1))/(zi(nzm)-zi(1))
    zi_lo = zi[0]    # Fortran zi(1): ground interface
    zi_hi = zi[-2]   # Fortran zi(nzm): top mass-level interface (Python nz+1 array -> index nz-1 = -2)
    nu = (zi[:-1] - zi_lo) / (zi_hi - zi_lo)   # shape (nz,), indices 0..nz-1

    # sponge_active: taudamp(k) > 0 <=> nu > nub  (shape nz,)
    sponge_active = nu > nub   # (nz,)

    # wmax at each W-face k=1..nzm (Python index 0..nzm-1):
    #   Fortran: wmax1 = damping_w_cu * dz * adzw(k) / dtn
    #   Fortran dz is a scalar base spacing; adzw(k) is the W-face stretch ratio.
    #   Fortran adzw(k) 1-indexed -> Python adzw[k-1] = adzw[:-1].
    #   Cell W-face thickness = dz_ref * adzw[k-1].
    wmax = damping_w_cu * dz_ref * adzw[:-1] / dtn   # shape (nz,)

    # Soft-clip: W_new = (W + clip(W,-wmax,wmax)*tau_max) / (1+tau_max)
    # Only applied where sponge is NOT active (taudamp==0).
    W_inner = W[1:]   # shape (nz, ny, nx) — faces k=1..nz
    wmax3   = wmax[:, None, None]           # (nz, 1, 1)
    active3 = sponge_active[:, None, None]  # (nz, 1, 1)

    W_clipped = jnp.clip(W_inner, -wmax3, wmax3)
    W_damped  = (W_inner + W_clipped * tau_max) / (1.0 + tau_max)

    # Only replace where sponge is NOT active
    W_inner_new = jnp.where(active3, W_inner, W_damped)

    # Reassemble: W[0] (ground BC) is never touched
    W_new = jnp.concatenate([W[:1], W_inner_new], axis=0)
    return W_new


@jax.jit
def gsam_w_sponge(
    W: jax.Array,        # (nz+1, ny, nx)
    zi: jax.Array,       # (nz+1,) interface heights (m); zi[0]=ground, zi[nz]=top
    nub: float = 0.6,
    taudamp_max: float = 0.333,
    dtn: float = 1.0,    # current AB-weighted sub-timestep (s)
    dt: float = 1.0,     # base timestep (s)
) -> jax.Array:
    """gSAM-exact W-only Rayleigh sponge at model top (damping.f90:31-50).

    Matches Fortran exactly:
      - Fix 5.9: nu computed using interface heights zi (not cell centres).
        Fortran: nu = (zi(k)-zi(1))/(zi(nzm)-zi(1)) for k=1..nzm.
        Fortran zi declared as zi(nz) 1-indexed: zi(k) -> Python zi[k-1].
        Python metric["zi"] has shape (nz+1,); Fortran zi(nzm) = Python zi[-2].
        JSam: nu[k-1] = (zi[k-1] - zi[0]) / (zi[-2] - zi[0]) for k=1..nzm.
      - Fix 5.10: taudamp(k) applied directly to W[k] without face averaging.
        Fortran: w(k) /= 1+taudamp(k)  for k=1..nzm.
        JSam: W[k] /= 1+taudamp[k-1]  for k=1..nz (W[0] unchanged).
    """
    nz = zi.shape[0] - 1   # number of mass levels (Python zi has nz+1 elements)
    # Fortran loop k=1..nzm (1-indexed): nu = (zi(k)-zi(1))/(zi(nzm)-zi(1))
    # Fortran zi is declared as zi(nz), 1-indexed; Python zi has shape (nz+1,).
    # Fortran nzm = nz - 1.
    # Fortran zi(k) for k=1..nzm -> Python zi[k-1] = zi[0..nzm-1] = zi[:-2].
    # Fortran zi(1) = Python zi[0].
    # Fortran zi(nzm) = Python zi[nzm-1] = Python zi[-2]  (NOT zi[-1]).
    # Fortran w(k) for k=1..nzm = JSam W[k] for k=1..nz.
    zi_lo = zi[0]    # Fortran zi(1): height of lowest interface
    zi_hi = zi[-2]   # Fortran zi(nzm): top mass-level interface (Python index nz-1 = -2)
    nu = (zi[:-1] - zi_lo) / (zi_hi - zi_lo)   # shape (nz,), for W-faces 1..nz
    nu_excess = jnp.clip((nu - nub) / (1.0 - nub), 0.0, 1.0)
    zzz = 100.0 * nu_excess ** 2
    tau_max = dtn / dt
    taudamp = jnp.where(nu > nub, tau_max * taudamp_max * zzz / (1.0 + zzz), 0.0)

    # Apply taudamp[k-1] (0-indexed) to W[k] for k=1..nz.
    # W[0]=ground BC (not touched).  Prepend zero for W[0].
    taudamp_w = jnp.concatenate([jnp.zeros(1), taudamp])  # shape (nz+1,)
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
