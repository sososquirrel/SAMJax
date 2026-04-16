"""
Latitude-dependent Fourier polar filter for lat-lon grids.
Zeroes zonal modes m > floor(nx/2 * cos(lat)) to suppress grid-scale divergence.
Matches gSAM dodamping_poles branch.
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
    Apply latitude-dependent zonal Fourier truncation.
    Zero spectral modes m > max(1, floor(nx/2 * |cos(lat)|)) at each row.
    """
    nspec = nx // 2 + 1
    cos_lat = jnp.abs(jnp.cos(lat_rad))
    m_max   = jnp.maximum(1, jnp.floor(nx / 2 * cos_lat).astype(int))
    m_arr   = jnp.arange(nspec)
    mask    = (m_arr[None, :] <= m_max[:, None]).astype(field.dtype)
    hat      = jnp.fft.rfft(field, axis=-1)
    hat_filt = hat * mask[..., :]
    return jnp.fft.irfft(hat_filt, n=nx, axis=-1)
