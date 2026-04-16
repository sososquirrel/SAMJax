"""Prescribed radiation forcing (linear time/height interpolation of precomputed profiles).
Applies dQrad(z,t) to TABS each timestep. Time and height interpolation are JAX-differentiable.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsam.core.state import ModelState


# ---------------------------------------------------------------------------
# Forcing table (JAX pytree)
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class RadForcing:
    """Radiation forcing: qrad(ntime,nz_prof)[K/s], z_prof(nz_prof)[m], t_prof(ntime)[s]."""
    qrad:   jax.Array   # (ntime, nz_prof)
    z_prof: jax.Array   # (nz_prof,)
    t_prof: jax.Array   # (ntime,)

    @classmethod
    def constant(cls, qrad_profile: jax.Array, z_prof: jax.Array) -> "RadForcing":
        """Time-invariant forcing: profile repeated at t=0 and t=1e30."""
        qrad2 = jnp.stack([qrad_profile, qrad_profile], axis=0)
        t2    = jnp.array([0.0, 1.0e30])
        return cls(qrad=qrad2, z_prof=z_prof, t_prof=t2)

    @classmethod
    def zeros(cls, nz_prof: int) -> "RadForcing":
        """Zero heating rate on a trivial 2-level z grid."""
        z = jnp.array([0.0, 1.0e5])
        q = jnp.zeros((2, nz_prof))
        return cls(
            qrad   = jnp.zeros((2, 2)),
            z_prof = jnp.array([0.0, 1.0e5]),
            t_prof = jnp.array([0.0, 1.0e30]),
        )

    def tree_flatten(self):
        return (self.qrad, self.z_prof, self.t_prof), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


# ---------------------------------------------------------------------------
# Interpolation helpers (differentiable)
# ---------------------------------------------------------------------------

def _interp1d(x: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    """Linear interp of fp at xp onto x; clamps outside [xp[0],xp[-1]]."""
    idx = jnp.searchsorted(xp, x, side="right")
    idx = jnp.clip(idx, 1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    dx = x1 - x0
    w  = jnp.where(dx > 0.0, (x - x0) / dx, 0.0)
    w  = jnp.clip(w, 0.0, 1.0)
    return f0 + w * (f1 - f0)


def _interp_time(t: float | jax.Array, forcing: RadForcing) -> jax.Array:
    """
    Interpolate ``forcing.qrad`` linearly to time ``t``.
    Returns shape (nz_prof,).
    """
    t_prof = forcing.t_prof   # (ntime,)
    qrad   = forcing.qrad     # (ntime, nz_prof)

    idx = jnp.searchsorted(t_prof, t, side="right")
    idx = jnp.clip(idx, 1, t_prof.shape[0] - 1)
    t0, t1 = t_prof[idx - 1], t_prof[idx]
    q0, q1 = qrad[idx - 1, :], qrad[idx, :]
    dt = t1 - t0
    w  = jnp.where(dt > 0.0, (t - t0) / dt, 0.0)
    w  = jnp.clip(w, 0.0, 1.0)
    return q0 + w * (q1 - q0)   # (nz_prof,)


def qrad_on_model_grid(
    t: float | jax.Array,
    forcing: RadForcing,
    z_model: jax.Array,          # (nz,) m
) -> jax.Array:
    """
    Return Q_rad [K/s] on the model's z grid at model time t [s].
    Shape: (nz,).
    """
    # 1. Time-interpolate to (nz_prof,)
    q_prof = _interp_time(t, forcing)
    # 2. Vertically interpolate to each model level
    q_model = jax.vmap(lambda z: _interp1d(z, forcing.z_prof, q_prof))(z_model)
    return q_model   # (nz,)


# ---------------------------------------------------------------------------
# Top-level radiation step
# ---------------------------------------------------------------------------

@jax.jit
def rad_proc(state: ModelState, metric: dict, forcing: RadForcing, dt: float) -> ModelState:
    """Apply Q_rad(z, t) * dt to TABS."""
    z_model = metric["z"]                              # (nz,)
    qrad    = qrad_on_model_grid(state.time, forcing, z_model)  # (nz,)

    # Broadcast to (nz, ny, nx)
    nz, ny, nx = state.TABS.shape
    dTABS = qrad[:, None, None] * jnp.ones((1, ny, nx))

    return ModelState(
        U     = state.U,
        V     = state.V,
        W     = state.W,
        TABS  = state.TABS + dt * dTABS,
        QV    = state.QV,
        QC    = state.QC,
        QI    = state.QI,
        QR    = state.QR,
        QS    = state.QS,
        QG    = state.QG,
        TKE   = state.TKE,
        p_prev = state.p_prev, p_pprev = state.p_pprev,
        nstep = state.nstep,
        time  = state.time,
    )
