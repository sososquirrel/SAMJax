"""
Adaptive timestep CFL check.

Matches gSAM SRC/kurant.f90 lines 19-47 exactly.  The Fortran kernel is

    do k = 1,nzm
     idz = dtn/(dz*adzw(k))              ! dz scalar ref, adzw(k) per-level stretch
     do j=1,ny
      idx = imu(j)*dtn/dx                ! imu(j) = 1/cos(lat_j)
      idy = YES3D*dtn/(dy*ady(j))        ! dy scalar ref, ady(j) per-row stretch
      do i=1,nx
       cflz1    = abs(w(i,j,k))*idz
       cflh1_sq = (u(i,j,k)*idx)**2 + (v(i,j,k)*idy)**2
       cfll     = sqrt(cflh1_sq + cflz1**2)
       cfl      = max(cfl, cfll)
      end do
     end do
    end do

i.e. the per-cell advective Courant number is

    cfl_cell = sqrt(
        (|u|*dt / (dx*cos(lat)))**2
      + (|v|*dt / (dy_ref*ady(j)))**2
      + (|w|*dt / (dz_ref*adzw(k)))**2
    )

Note `dz_ref*adzw(k)` is the W-face center-to-center spacing
(z(k)-z(k-1)), NOT the mass-cell thickness zi(k+1)-zi(k); these
coincide only on a uniform vertical grid.  Likewise `dy_ref*ady(j)`
equals jsam's per-row `dy_lat[j]` by construction.

The reduced timestep is

    dtn = min(dt_ref, dt_ref * cfl_max / (cfl_adv + eps))

gSAM defaults from params.f90: cfl_max = 0.7, ncycle_max = 4.
"""
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
    """
    Global-max advective CFL at the given dt.

    U, V, W are the full staggered prognostic arrays:
        U: (nz, ny, nx+1)
        V: (nz, ny+1, nx)
        W: (nz+1, ny, nx)

    Each velocity component is max-reduced onto mass points (worst-case
    over the two neighbouring faces), then combined with the spherical
    grid spacing to form the cell Courant number.
    """
    dx      = metric["dx_lon"]     # scalar (m) — equatorial zonal spacing
    dy      = metric["dy_lat"]     # (ny,)  — per-row meridional spacing (= dy_ref*ady)
    cos_lat = metric["cos_lat"]    # (ny,)
    dz_ref  = metric["dz_ref"]     # scalar — reference vertical spacing
    adzw    = metric["adzw"]       # (nz+1,) — per-level stretched factor (use first nz)

    U_abs = jnp.maximum(jnp.abs(U[:, :, :-1]), jnp.abs(U[:, :, 1:]))   # (nz,ny,nx)
    V_abs = jnp.maximum(jnp.abs(V[:, :-1, :]), jnp.abs(V[:, 1:, :]))   # (nz,ny,nx)
    W_abs = jnp.maximum(jnp.abs(W[:-1, :, :]), jnp.abs(W[1:, :, :]))   # (nz,ny,nx)

    dx_j = dx * jnp.maximum(cos_lat, 1e-6)             # guard against cos=0 at poles
    idx  = dt / dx_j                                   # (ny,)
    idy  = dt / dy                                     # (ny,)
    idz  = dt / (dz_ref * adzw[:U_abs.shape[0]])       # (nz,) — dz*adzw(k), k=1..nzm

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
    """
    Return (dtn, cfl_adv).

    dtn is the advective-CFL-limited timestep, capped at dt_ref.
    cfl_adv is the global-max advective CFL diagnostic at dt=dt_ref.
    """
    cfl_adv = compute_cfl(U, V, W, metric, dt_ref)
    dtn     = jnp.minimum(dt_ref, dt_ref * cfl_max / (cfl_adv + 1e-10))
    return dtn, cfl_adv
