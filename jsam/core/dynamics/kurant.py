"""Adaptive timestep CFL check. Matches gSAM kurant.f90."""
from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import numpy as np


def compute_cfl(
    U:      jax.Array,
    V:      jax.Array,
    W:      jax.Array,
    metric: dict,
    dt:     float,
) -> jax.Array:
    """Global-max advective CFL at the given dt. Matches gSAM kurant.f90."""
    dx      = metric["dx_lon"]
    dy      = metric["dy_lat"]
    cos_lat = metric["cos_lat"]
    dz_ref  = metric["dz_ref"]
    adzw    = metric["adzw"]

    # gSAM uses single face per cell: u(i,j,k)=east face, v(i,j,k)=north face, w(i,j,k)=top face
    U_abs = jnp.abs(U[:, :, 1:])    # eastern face of each cell, matches gSAM u(i,j,k)
    V_abs = jnp.abs(V[:, 1:, :])    # northern face of each cell, matches gSAM v(i,j,k)
    W_abs = jnp.abs(W[1:, :, :])    # top face of each cell, matches gSAM w(i,j,k)

    dx_j = dx * jnp.maximum(cos_lat, 1e-6)
    idx  = dt / dx_j
    idy  = dt / dy
    idz  = dt / (dz_ref * adzw[:U_abs.shape[0]])

    cfl_u = U_abs * idx[None, :, None]
    cfl_v = V_abs * idy[None, :, None]
    cfl_w = W_abs * idz[:, None, None]

    cfl_cell = jnp.sqrt(cfl_u * cfl_u + cfl_v * cfl_v + cfl_w * cfl_w)

    if os.environ.get("JSAM_KURANT_DEBUG", "") == "1":
        cfl_np = np.asarray(cfl_cell)
        idx_flat = int(np.argmax(cfl_np))
        k, j, i = np.unravel_index(idx_flat, cfl_np.shape)
        cfl_u_np = float(np.asarray(cfl_u)[k, j, i])
        cfl_v_np = float(np.asarray(cfl_v)[k, j, i])
        cfl_w_np = float(np.asarray(cfl_w)[k, j, i])
        u_np = float(np.asarray(U_abs)[k, j, i])
        v_np = float(np.asarray(V_abs)[k, j, i])
        w_np = float(np.asarray(W_abs)[k, j, i])
        dx_j_np = float(np.asarray(dx_j)[j])
        dy_np = float(np.asarray(dy)[j])
        dzw_np = float(np.asarray(adzw[:U_abs.shape[0]])[k] * dz_ref)
        cos_np = float(np.asarray(cos_lat)[j])
        print(f"  [kurant] dt={dt:.3f}s  max|cfl|={cfl_np.max():.4f}  "
              f"@(k={k},j={j},i={i})  cos_lat={cos_np:.4f}", flush=True)
        print(f"           cfl_u={cfl_u_np:.4f} (|U|={u_np:.2f}, dx_j={dx_j_np:.1f}m)  "
              f"cfl_v={cfl_v_np:.4f} (|V|={v_np:.2f}, dy={dy_np:.1f}m)  "
              f"cfl_w={cfl_w_np:.4f} (|W|={w_np:.3f}, dz_w={dzw_np:.1f}m)",
              flush=True)

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
