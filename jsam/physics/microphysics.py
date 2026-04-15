"""
Single-moment bulk microphysics (SAM-equivalent).

Hydrometeor categories: QC (cloud liquid), QI (cloud ice),
QR (rain), QS (snow), QG (graupel).

TODO (Phase 1):
  - Autoconversion (Kessler)
  - Accretion (rain collecting cloud)
  - Ice nucleation (temperature threshold)
  - Sedimentation (rain, snow, graupel)
  - Evaporation/sublimation

Reference: SAM — micro_sam1mom.f90
"""
from __future__ import annotations
import jax.numpy as jnp
from jsam.core.state import ModelState
from jsam.core.grid.latlon import LatLonGrid


def microphysics_tendency(
    state: ModelState,
    grid: LatLonGrid,
    dt: float,
) -> ModelState:
    """
    Returns updated state after applying microphysics for one time step dt.
    All tendencies (dQV/dt, dQC/dt, ..., dTABS/dt) computed here.
    Placeholder.
    """
    raise NotImplementedError("microphysics_tendency: TODO")
