"""
ModelState — the complete prognostic state of jsam, represented as a JAX pytree.

All arrays have shape (nz, ny, nx) unless noted.
Staggered velocities (C-grid Arakawa):
  U: (nz, ny, nx+1) — zonal,    staggered east face
  V: (nz, ny+1, nx) — meridional, staggered north face
  W: (nz+1, ny, nx) — vertical,  staggered top face
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp


@dataclass
class ModelState:
    """Prognostic variables. Registered as a JAX pytree."""

    # Momentum (C-grid staggered)
    U: jax.Array   # zonal wind        (nz, ny, nx+1)  m/s
    V: jax.Array   # meridional wind   (nz, ny+1, nx)  m/s
    W: jax.Array   # vertical wind     (nz+1, ny, nx)  m/s

    # Thermodynamics
    TABS: jax.Array   # absolute temperature  (nz, ny, nx)  K
    QV:   jax.Array   # water vapour          (nz, ny, nx)  kg/kg
    QC:   jax.Array   # cloud liquid          (nz, ny, nx)  kg/kg
    QI:   jax.Array   # cloud ice             (nz, ny, nx)  kg/kg
    QR:   jax.Array   # rain                  (nz, ny, nx)  kg/kg
    QS:   jax.Array   # snow                  (nz, ny, nx)  kg/kg
    QG:   jax.Array   # graupel               (nz, ny, nx)  kg/kg

    # Subgrid TKE
    TKE: jax.Array   # turbulent kinetic energy  (nz, ny, nx)  m²/s²

    # Previous-step pressure fields for Adams-Bashforth adamsB correction.
    # gSAM nadams=3 uses a 3-level rotating pressure buffer: p(na)=current,
    # p(nb)=prev, p(nc)=prev-prev.  adamsB applies bt*∇p_prev + ct*∇p_pprev.
    p_prev:  jax.Array | None = None   # (nz, ny, nx), p_{n-1}
    p_pprev: jax.Array | None = None   # (nz, ny, nx), p_{n-2}

    # Simulation time — stored as JAX scalars (dynamic pytree leaves) so that
    # incrementing them does NOT change the pytree aux data and trigger full
    # recompilation of every JIT-compiled function that takes ModelState.
    nstep: jax.Array | int = 0        # scalar int — use jnp.int32(0) at init
    time:  jax.Array | float = 0.0    # scalar float — use jnp.float64(0.0) at init

    # ── pytree registration ───────────────────────────────────────────────────
    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "U", "V", "W", "TABS", "QV", "QC", "QI", "QR", "QS", "QG", "TKE",
        "p_prev", "p_pprev", "nstep", "time",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ()

    def tree_flatten(self):
        children = [getattr(self, f) for f in self._dynamic_fields]
        aux = {f: getattr(self, f) for f in self._static_fields}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(cls._dynamic_fields, children))
        kwargs.update(aux)
        return cls(**kwargs)

    @classmethod
    def zeros(cls, nz: int, ny: int, nx: int) -> "ModelState":
        """Initialise all fields to zero on a (nz, ny, nx) grid."""
        return cls(
            U=jnp.zeros((nz, ny, nx + 1)),
            V=jnp.zeros((nz, ny + 1, nx)),
            W=jnp.zeros((nz + 1, ny, nx)),
            TABS=jnp.zeros((nz, ny, nx)),
            QV=jnp.zeros((nz, ny, nx)),
            QC=jnp.zeros((nz, ny, nx)),
            QI=jnp.zeros((nz, ny, nx)),
            QR=jnp.zeros((nz, ny, nx)),
            QS=jnp.zeros((nz, ny, nx)),
            QG=jnp.zeros((nz, ny, nx)),
            TKE=jnp.zeros((nz, ny, nx)),
            p_prev =jnp.zeros((nz, ny, nx)),
            p_pprev=jnp.zeros((nz, ny, nx)),
            nstep=jnp.int32(0),
            time=jnp.float64(0.0),
        )

    @classmethod
    def from_gsam_nc(cls, path_3d: str) -> "ModelState":
        """
        Initialise state from a gSAM 3D NetCDF output file (e.g. IRMA t=0).
        Useful for initialising a jsam run from gSAM restart output.
        """
        import netCDF4 as nc
        import numpy as np

        with nc.Dataset(path_3d) as ds:
            def load(v):
                arr = ds.variables[v][0]   # drop time dim
                return jnp.array(np.array(arr))

            nz, ny, nx = ds.variables["U"][0].shape

            # gSAM stores velocities on mass grid; we keep them there for now
            # and handle staggering later in the grid module
            U_mass = load("U")
            V_mass = load("V")
            W_mass = load("W")

            return cls(
                U=(jnp.zeros((nz, ny, nx + 1))
                   .at[:, :, :-1].set(U_mass)
                   .at[:, :, -1].set(U_mass[:, :, 0])),
                V=(jnp.zeros((nz, ny + 1, nx))
                   .at[:, 1:-1, :].set(0.5 * (V_mass[:, :-1, :] + V_mass[:, 1:, :]))),
                W=(jnp.zeros((nz + 1, ny, nx))
                   .at[1:-1, :, :].set(0.5 * (W_mass[:-1, :, :] + W_mass[1:, :, :]))),
                TABS=load("TABS"),
                QV=load("QV"),
                QC=load("QC"),
                QI=load("QI"),
                QR=load("QR"),
                QS=load("QS"),
                QG=load("QG"),
                # F3 fix: gSAM initialises SGS TKE to 0 at run start.
                # TKH is eddy diffusivity (m²/s), not kinetic energy (m²/s²).
                # Use the "TKE" variable if the file has one, otherwise zeros.
                TKE=(load("TKE") if "TKE" in ds.variables else jnp.zeros((nz, ny, nx))),
                p_prev =jnp.zeros((nz, ny, nx)),
                p_pprev=jnp.zeros((nz, ny, nx)),
                nstep=jnp.int32(0),
                time=jnp.float64(0.0),
            )


# Register with JAX
jax.tree_util.register_pytree_node(
    ModelState,
    ModelState.tree_flatten,
    ModelState.tree_unflatten,
)
