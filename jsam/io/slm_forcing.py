"""
Per-step radiative forcing builder for the Simple Land Model.

Given the model grid, current date, and atmospheric state, return an
:class:`~jsam.core.physics.slm.SLMRadInputs` NamedTuple holding the six
surface radiation fields the SLM needs:

    sw_dir_vis, sw_dif_vis   (visible direct / diffuse, W/m²)
    sw_dir_nir, sw_dif_nir   (near-IR direct / diffuse, W/m²)
    lwds                     (surface down-welling LW, W/m²)
    coszrs                   (cosine of solar zenith angle)

Two sources are combined:

* **LW**: surface down-welling is computed from the currently-ported
  RRTMG_LW by ``compute_qrad_and_lwds_rrtmg``. This is exact for the
  gSAM IRMA configuration.

* **SW**: jsam does not yet port RRTMG_SW, and gSAM's debug dumps do
  not include the four SW surface bands, so bit-close SLM validation
  against the oracle is not currently possible. The helper below
  supplies a *prescribed* clear-sky SW profile computed from a simple
  top-of-atmosphere-minus-Beer's-law model:

      S0 = 1361.0 W/m²                       solar constant
      cos(θ) from :func:`coszrs` (zenith.py)
      tau_clear = 0.75                       clear-sky bulk transmission
      sw_total = S0 * cos(θ) * tau_clear     when cos(θ) > 0, else 0

  The 4-band split uses the SW-band fractions from CAM (visible and
  near-IR each carry roughly half the total, with ~85% direct / 15%
  diffuse under clear sky):

      sw_dir_vis = 0.43 * sw_total
      sw_dif_vis = 0.07 * sw_total
      sw_dir_nir = 0.43 * sw_total
      sw_dif_nir = 0.07 * sw_total

  Callers that need true oracle-prescribed SW should override this
  NamedTuple with fields read from a future extended gSAM debug dump.
"""
from __future__ import annotations

from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from jsam.core.physics.slm import SLMRadInputs
from jsam.core.physics.slm.zenith import coszrs as _coszrs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_S0:              float = 1361.0    # solar constant (W/m²)
_TAU_CLEARSKY:    float = 0.75
_FRAC_DIR_BAND:   float = 0.43      # 86% direct × 0.5 (per band)
_FRAC_DIF_BAND:   float = 0.07      # 14% diffuse × 0.5 (per band)


def build_slm_rad_inputs(
    date: datetime,
    lat_rad: jax.Array,
    lon_rad: jax.Array,
    lwds: jax.Array,
) -> SLMRadInputs:
    """Assemble the six-field :class:`SLMRadInputs` for one step.

    Parameters
    ----------
    date : UTC datetime for the current model step.
    lat_rad : (ny,) latitude array in radians.
    lon_rad : (nx,) longitude array in radians (0..2π).
    lwds : (ny, nx) surface down-welling LW flux from RRTMG_LW (W/m²).
           Use :func:`compute_qrad_and_lwds_rrtmg` to obtain this.

    Returns
    -------
    :class:`SLMRadInputs`
    """
    cz = _coszrs(date, lat_rad, lon_rad)
    cz_pos = jnp.maximum(cz, 0.0)

    sw_total = _S0 * cz_pos * _TAU_CLEARSKY

    sw_dir_vis = _FRAC_DIR_BAND * sw_total
    sw_dif_vis = _FRAC_DIF_BAND * sw_total
    sw_dir_nir = _FRAC_DIR_BAND * sw_total
    sw_dif_nir = _FRAC_DIF_BAND * sw_total

    return SLMRadInputs(
        sw_dir_vis=sw_dir_vis.astype(jnp.float32),
        sw_dif_vis=sw_dif_vis.astype(jnp.float32),
        sw_dir_nir=sw_dir_nir.astype(jnp.float32),
        sw_dif_nir=sw_dif_nir.astype(jnp.float32),
        lwds      =jnp.asarray(lwds, dtype=jnp.float32),
        coszrs    =cz.astype(jnp.float32),
    )
