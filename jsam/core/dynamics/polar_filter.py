"""
Latitude-dependent Fourier polar filter for lat-lon grids.

At high latitudes the zonal grid spacing shrinks as cos(lat), so zonal waves
with wavenumber m > m_max(lat) = floor(nx/2 * cos(lat)) are sub-grid in
physical space.  The filter zeroes those modes, smoothing the divergence that
the lat-lon geometry creates at high latitudes and preventing the pressure
solver from producing spurious O(100 m/s) vertical velocities at the domain
walls.

This matches gSAM's lat-lon polar filter (applied to U and V after each
momentum advance, before the pressure solve).

References
----------
  gSAM SRC/damping.f90  — dodamping_poles branch
  Jablonowski & Williamson (2011) Rev. Geophys. review of lat-lon models
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(2,))
def polar_fourier_filter(
    field: jax.Array,   # (nz, ny, nx) or similar 3-D array, x is axis=-1
    lat_rad: jax.Array, # (ny,) latitude of mass-cell rows in radians
    nx: int,
) -> jax.Array:
    """
    Apply a latitude-dependent zonal Fourier truncation.

    For each row j, zero spectral modes m > m_max(j) where
        m_max(j) = max(1, floor(nx/2 * |cos(lat_j)|))

    This reduces the effective zonal resolution to the physical grid spacing
    cos(lat)*dx, preventing aliasing-driven divergence at high latitudes.

    Works for any leading dimension (nz, or nz+1 for W-faces).  The latitude
    axis (second from last, axis=-2) must have length ny.

    Args:
        field   : (..., ny, nx) real array
        lat_rad : (ny,) latitude in radians
        nx      : number of zonal grid points (may differ from field.shape[-1]
                  for staggered-U grids with nx+1 points; in that case pass
                  nx = field.shape[-1] - 1 and the extra column is handled
                  outside this routine)

    Returns:
        Filtered array of the same shape as ``field``.
    """
    nspec = nx // 2 + 1                     # number of rfft coefficients

    # Spectral mask: mask[j, m] = 1 if m <= m_max(j), else 0
    cos_lat = jnp.abs(jnp.cos(lat_rad))    # (ny,)
    m_max   = jnp.maximum(1, jnp.floor(nx / 2 * cos_lat).astype(int))  # (ny,)
    m_arr   = jnp.arange(nspec)            # (nspec,)
    mask    = (m_arr[None, :] <= m_max[:, None]).astype(field.dtype)  # (ny, nspec)

    # Broadcast mask over leading dimensions
    # field shape: (..., ny, nx)  →  hat shape: (..., ny, nspec)
    hat      = jnp.fft.rfft(field, axis=-1)       # (..., ny, nspec) complex
    hat_filt = hat * mask[..., :]                   # broadcast over leading dims
    return jnp.fft.irfft(hat_filt, n=nx, axis=-1)  # (..., ny, nx)
