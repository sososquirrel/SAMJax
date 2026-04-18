"""
Saturation vapour pressure and specific humidity — Magnus / IFS Cy47r3.

Bit-exact port of ``gSAM SRC/sat.f90``. Formulas are expressed in hPa for
esat and use pressure in millibar (hPa) in the qsat denominator, matching
the Fortran caller convention.

Pressure convention: arguments ``p`` are in millibar (hPa) — the same
units the Fortran code expects. When coupling to jsam (which carries
pressure in Pa), the caller must divide by 100 before passing p here.

All functions are elementwise and broadcast over any input shape.
"""
from __future__ import annotations

import jax.numpy as jnp


# Magnus coefficients (IFS Cy47r3, as in gSAM sat.f90)
_E0 = 6.1121      # hPa at T0
_T0 = 273.16      # K
_AW = 17.502
_TW = 32.19
_AI = 22.587
_TI = -0.7


def esatw(t):
    """Saturation vapour pressure over liquid water (hPa)."""
    return _E0 * jnp.exp(_AW * (t - _T0) / (t - _TW))


def esati(t):
    """Saturation vapour pressure over ice (hPa)."""
    return _E0 * jnp.exp(_AI * (t - _T0) / (t - _TI))


def qsatw(t, p):
    """Saturation specific humidity over liquid water.

    p in millibar (hPa); output in kg/kg.
    """
    es = esatw(t)
    return 0.622 * es / jnp.maximum(jnp.maximum(es, p - es), 1.0e-30)


def qsati(t, p):
    """Saturation specific humidity over ice. p in hPa."""
    es = esati(t)
    return 0.622 * es / jnp.maximum(jnp.maximum(es, p - es), 1.0e-30)


def dtesatw(t):
    """d(esatw)/dT (hPa/K)."""
    es = esatw(t)
    # d/dT [ aw*(t-T0)/(t-Tw) ] = aw*(T0-Tw)/(t-Tw)^2
    return es * _AW * (_T0 - _TW) / (t - _TW) ** 2


def dtesati(t):
    es = esati(t)
    return es * _AI * (_T0 - _TI) / (t - _TI) ** 2


def dtqsatw(t, p):
    es = esatw(t)
    d = dtesatw(t)
    denom = jnp.maximum(es, p - es)
    return 0.622 * (d * denom - es * d * jnp.where(p - es > es, -1.0, 1.0)) / denom ** 2


def dtqsati(t, p):
    es = esati(t)
    d = dtesati(t)
    denom = jnp.maximum(es, p - es)
    return 0.622 * (d * denom - es * d * jnp.where(p - es > es, -1.0, 1.0)) / denom ** 2
