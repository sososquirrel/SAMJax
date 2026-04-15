"""
Simple Land Model (SLM) — JAX port of gSAM SRC/SLM/.

This sub-package reproduces the behaviour of gSAM's Simple Land Model as
used by ``CASES/IRMA/prm_debug500``: cold-start initialisation from binary
inputs in ``GLOBAL_DATA/BIN_D/``, 16-class IGBP land parameters, Cosby 1984
soil hydraulics, Monin-Obukhov surface layer, and 9-layer Thomas tridiagonal
soil water + temperature diffusion.

Public API (populated in phases):
    SLMParams  — scalar constants (port of slm_params.f90)
    SLMStatic  — frozen (ny,nx) land-surface parameters
    SLMState   — prognostic SLM state (soilt, soilw, canopy, snow)
    slm_proc   — main step entry point (Phase 3)
"""
from jsam.core.physics.slm.params import SLMParams
from jsam.core.physics.slm.state import SLMStatic, SLMState
from jsam.core.physics.slm.run_slm import slm_proc, SLMRadInputs

__all__ = ["SLMParams", "SLMStatic", "SLMState", "slm_proc", "SLMRadInputs"]
