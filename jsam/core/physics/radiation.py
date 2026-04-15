"""
Prescribed radiation forcing for jsam.

For ERA5-initialized simulations the full RRTMG scheme is unnecessary and
inconsistent (ERA5 uses ECMWF's own radiative transfer).  Instead we
interpolate a precomputed column-averaged heating-rate profile Q_rad(z, t)
— derived from ERA5 or any other source — onto the model grid and apply it
as a TABS tendency.  This mirrors gSAM's ``doradforcing`` option in
forcing.f90.

The heating-rate table is horizontally uniform but can be time-varying.
Time and height interpolation are both linear so the module is
JAX-differentiable end-to-end.

API
---
    from jsam.core.physics.radiation import RadForcing, rad_proc

    # Build once from ERA5-derived data
    forcing = RadForcing(
        qrad=jnp.array(qrad_Kpers),   # (ntime, nz_prof)  K/s
        z_prof=jnp.array(z_m),        # (nz_prof,)  m  (monotone increasing)
        t_prof=jnp.array(t_s),        # (ntime,)  s  (monotone increasing)
    )

    # Inside the time-step loop
    new_state = rad_proc(state, metric, forcing, dt)

Notes
-----
- ``qrad`` must be in K/s (not K/step).  ERA5 ``mttswr + mttlwr`` fields
  (W/m²) must be divided by ``rho * cp`` before use.
- ``t_prof`` is in model-seconds elapsed since the simulation epoch.
  Clamp behaviour: beyond the last snapshot the last value is held.
- Only TABS is modified; all other fields pass through unchanged.
- nstep and time are NOT incremented here (advance_scalars owns that).
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
    """
    Time-varying column radiation forcing.

    Attributes
    ----------
    qrad   : (ntime, nz_prof)  radiative heating rate [K/s]
    z_prof : (nz_prof,)        height of forcing levels [m], monotone increasing
    t_prof : (ntime,)          time of each snapshot [s],    monotone increasing
    """
    qrad:   jax.Array   # (ntime, nz_prof)
    z_prof: jax.Array   # (nz_prof,)
    t_prof: jax.Array   # (ntime,)

    @classmethod
    def constant(
        cls,
        qrad_profile: jax.Array,   # (nz_prof,) K/s
        z_prof: jax.Array,         # (nz_prof,) m
    ) -> "RadForcing":
        """
        Time-invariant forcing: a single profile repeated at t=0 and t=1e30.
        """
        qrad2 = jnp.stack([qrad_profile, qrad_profile], axis=0)   # (2, nz_prof)
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
    """
    Linear interpolation of fp[i] at xp[i] onto scalar x.
    Clamps to boundary values outside [xp[0], xp[-1]].
    Differentiable w.r.t. x and fp.
    """
    # Find the right interval: idx is the *right* endpoint index (1..n-1)
    idx = jnp.searchsorted(xp, x, side="right")
    idx = jnp.clip(idx, 1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    # weight: 0 at x0, 1 at x1
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
def rad_proc(
    state:   ModelState,
    metric:  dict,
    forcing: RadForcing,
    dt:      float,
) -> ModelState:
    """
    Apply prescribed radiative heating for one timestep.

    Adds ``Q_rad(z, state.time) * dt`` to TABS.  All other fields and
    nstep/time are passed through unchanged.

    Parameters
    ----------
    state   : current ModelState
    metric  : grid metric dict (must contain ``"z"`` key, shape (nz,))
    forcing : RadForcing with time-varying heating profile
    dt      : timestep [s]

    Returns
    -------
    new ModelState with updated TABS
    """
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
