"""Adaptive timestep CFL check. Matches gSAM kurant.f90."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_cfl(
    U:      jax.Array,
    V:      jax.Array,
    W:      jax.Array,
    metric: dict,
    dt:     float,
) -> jax.Array:
    """Global-max advective CFL at the given dt."""
    dx      = metric["dx_lon"]
    dy      = metric["dy_lat"]
    cos_lat = metric["cos_lat"]
    dz_ref  = metric["dz_ref"]
    adzw    = metric["adzw"]

    U_abs = jnp.maximum(jnp.abs(U[:, :, :-1]), jnp.abs(U[:, :, 1:]))
    V_abs = jnp.maximum(jnp.abs(V[:, :-1, :]), jnp.abs(V[:, 1:, :]))
    W_abs = jnp.maximum(jnp.abs(W[:-1, :, :]), jnp.abs(W[1:, :, :]))

    dx_j = dx * jnp.maximum(cos_lat, 1e-6)
    idx  = dt / dx_j
    idy  = dt / dy
    idz  = dt / (dz_ref * adzw[:U_abs.shape[0]])

    cfl_u = U_abs * idx[None, :, None]
    cfl_v = V_abs * idy[None, :, None]
    cfl_w = W_abs * idz[:, None, None]

    cfl_cell = jnp.sqrt(cfl_u * cfl_u + cfl_v * cfl_v + cfl_w * cfl_w)
    return jnp.max(cfl_cell)


def kurant_dt(
    U:       jax.Array,
    V:       jax.Array,
    W:       jax.Array,
    metric:  dict,
    dt_ref:  float,
    cfl_max: float = 0.7,
) -> tuple[jax.Array, jax.Array]:
    """Return (dtn, cfl_adv) — CFL-limited timestep and diagnostic CFL."""
    cfl_adv = compute_cfl(U, V, W, metric, dt_ref)
    dtn     = jnp.minimum(dt_ref, dt_ref * cfl_max / (cfl_adv + 1e-10))
    return dtn, cfl_adv
