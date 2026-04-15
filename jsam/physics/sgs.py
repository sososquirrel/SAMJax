"""
Subgrid-scale (SGS) turbulence scheme.

Uses a 1.5-order TKE-based scheme (same as SAM): prognostic TKE equation,
eddy diffusivity proportional to sqrt(TKE) * mixing length.

TODO (Phase 1):
  - TKE production (shear + buoyancy)
  - TKE dissipation (ε ~ TKE^(3/2) / l)
  - Eddy diffusivity Km, Kh from TKE and mixing length
  - Smagorinsky fallback for LES mode

Reference: SAM — sgs.f90
"""
from __future__ import annotations
import jax.numpy as jnp
from jsam.core.state import ModelState
from jsam.core.grid.latlon import LatLonGrid


def sgs_tendency(
    state: ModelState,
    grid: LatLonGrid,
    dt: float,
) -> ModelState:
    """Returns state incremented by SGS diffusion tendencies. Placeholder."""
    raise NotImplementedError("sgs_tendency: TODO")
