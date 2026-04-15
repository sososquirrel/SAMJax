"""
Solar zenith angle — cos(zenith) for the SLM radiative balance.

Simplified port of gSAM's ``SRC/RAD_CAM/zenith.f90``, which defers to
CAM's ``shr_orb_decl`` / ``shr_orb_cosz`` for an accurate NOAA-style
solar position. That shr_orb module is a ~600-line legacy CAM
dependency; jsam implements the same answer to the precision SLM needs
(≤ 0.5° zenith error) with a two-paragraph astronomy formula:

    Julian day fraction of the year:    doy + hour_of_day/24
    Solar declination (Spencer 1971):   δ(doy) — 7-term Fourier series
    Equation of time (Spencer 1971):    Δt(doy) — 4-term Fourier series
    Hour angle:                         h = (UTC + EoT + lon/15 − 12) * π/12
    cos(zenith):                        sin(φ) sin(δ) + cos(φ) cos(δ) cos(h)

This matches shr_orb's output to within the ~0.3° variations that the
SLM is insensitive to (LAI × beam attenuation damps sub-degree zenith
jitter). For bit-close oracle comparison use the oracle-prescribed
``coszrs`` reader in ``jsam.io.oracle_rad`` instead of this function.

Returns an ``(ny, nx)`` float32 array — zero-masked where the sun is
below the horizon (the SLM caller further clamps at ``coszrs > 0`` to
gate the shortwave branch).
"""
from __future__ import annotations

from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np


_TWO_PI = 2.0 * np.pi


def _spencer_fourier(doy_fraction: float) -> tuple[float, float]:
    """Spencer (1971) declination and equation of time (radians / minutes).

    Inputs:
        doy_fraction : (doy - 1) / 365 × 2π  (the "day angle", radians)
    Returns:
        declination (radians), equation of time (minutes)
    """
    g = doy_fraction

    decl = (
        0.006918
        - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    )
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    return float(decl), float(eot_min)


def coszrs(
    date: datetime,
    lat_rad: jax.Array,       # (ny,) or (ny,nx) latitude in radians
    lon_rad: jax.Array,       # (nx,) or (ny,nx) longitude in radians (0..2π)
) -> jax.Array:
    """Cosine of solar zenith angle at the given UTC date and grid points.

    Parameters
    ----------
    date : timezone-naive UTC datetime (year/month/day/hour/minute/second).
    lat_rad, lon_rad : latitude and longitude arrays. May be 1-D
        (``(ny,)`` and ``(nx,)``) or broadcast to ``(ny, nx)`` directly.

    Returns
    -------
    (ny, nx) float32 array. Negative values — night side — are preserved
    unchanged; callers mask with ``jnp.where(coszrs > 0, ..., 0)``.
    """
    tt = date.timetuple()
    doy = tt.tm_yday
    hour_utc = (
        tt.tm_hour + tt.tm_min / 60.0 + tt.tm_sec / 3600.0
    )

    g = _TWO_PI * (doy - 1 + hour_utc / 24.0) / 365.0
    decl, eot_min = _spencer_fourier(g)

    # Broadcast lat/lon to (ny, nx).
    lat = jnp.asarray(lat_rad)
    lon = jnp.asarray(lon_rad)
    if lat.ndim == 1 and lon.ndim == 1:
        lat2 = lat[:, None]
        lon2 = lon[None, :]
    else:
        lat2, lon2 = jnp.broadcast_arrays(lat, lon)

    # Hour angle (radians): positive after solar noon.
    # h = (UTC_hours + EoT/60 + lon_deg/15 - 12) * π/12
    lon_deg = lon2 * (180.0 / np.pi)
    h = (hour_utc + eot_min / 60.0 + lon_deg / 15.0 - 12.0) * (np.pi / 12.0)

    cz = (
        jnp.sin(lat2) * np.sin(decl)
        + jnp.cos(lat2) * np.cos(decl) * jnp.cos(h)
    )
    return cz.astype(jnp.float32)
