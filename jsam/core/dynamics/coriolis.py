"""
Coriolis + metric acceleration for a global lat-lon grid.

Port of gSAM SRC/coriolis.f90, dolatlon=.true. + docoriolisz=.true. branch.

Two physical contributions, summed into (dU, dV, dW):

  1. Standard spherical Coriolis + metric (curvature) term — gSAM
     coriolis.f90 lines 23-42, dolatlon branch:

       dU/dt += +(fcory[j] + U·tanr[j]) · V_bar             [ady-weighted]
       dV/dt += -(fcory[j] + U·tanr[j]) · mu[j] · U / cos(lat_v)  [4-corner avg]

  2. f'-coupling between W and U (the 2Ω·cosφ "cosine Coriolis"
     parameter) — gSAM coriolis.f90 lines 56-74, docoriolisz branch:

       dU/dt += -0.25 · fcorzy[j] · (1/adz[k]) ·
                  ( adzw[k]   · (W[k]   + W_west[k])
                  + adzw[k+1] · (W[k+1] + W_west[k+1]) )

       dW/dt += +0.25 · fcorzy[j] ·
                  ( U[i,j,k] + U[i+1,j,k] + U[i,j,k-1] + U[i+1,j,k-1] )

       applied at interior W-faces only (k=1..nz-1); W[0] and W[nz]
       remain zero (rigid lid/surface BC).

Array conventions (jsam, 0-indexed)
-------------------------------------
  U : (nz, ny, nx+1)   — zonal velocity at east faces
  V : (nz, ny+1, nx)   — meridional velocity at north faces (poles = 0)
  W : (nz+1, ny, nx)   — vertical velocity at top faces (W[0]=W[nz]=0)
  lat_rad : (ny,)      — mass-cell latitudes
  cos_v   : (ny+1,)    — cos(lat) at v-faces (from build_metric)
  fcory   : (ny,)      — 2*Omega*sin(lat)            standard Coriolis
  fcorzy  : (ny,)      — 2*Omega*cos(lat)            cosine Coriolis (f')
  tanr    : (ny,)      — tan(lat)/R at mass-cell latitudes
  dz      : (nz,)      — mass-cell thicknesses (gSAM adz·dz_ref, m)

References
----------
  gSAM SRC/coriolis.f90   — dolatlon=.true., docoriolis=.true., docoriolisz=.true.
  gSAM SRC/params.f90     — Omega = 7.2921e-5 rad/s
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def coriolis_tend(
    U:      jax.Array,   # (nz, ny, nx+1)
    V:      jax.Array,   # (nz, ny+1, nx)
    W:      jax.Array,   # (nz+1, ny, nx)
    metric: dict,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Spherical Coriolis + metric (curvature) + f'-coupling tendency.

    Returns
    -------
    dU : (nz, ny, nx)   — tendency at U west-faces of mass cells 0..nx-1
                          (periodic x; caller wraps to full nx+1 buffer)
    dV : (nz, ny+1, nx) — tendency at V north-faces, zero at poles
    dW : (nz+1, ny, nx) — tendency at W faces, zero at k=0 and k=nz
    """
    fcory   = metric["fcory"]    # (ny,) 2Ω sinφ
    fcorzy  = metric["fcorzy"]   # (ny,) 2Ω cosφ  (docoriolisz)
    tanr    = metric["tanr"]     # (ny,) tan(lat)/R
    mu      = metric["cos_lat"]  # (ny,) cos(lat)
    cos_v   = metric["cos_v"]    # (ny+1,) ady-weighted = gSAM muv
    ady     = metric["ady"]      # (ny,)   dy_per_row / dy_ref
    dz      = metric["dz"]       # (nz,)   cell thickness (m)

    nz, ny, nx_p1 = U.shape
    nx = nx_p1 - 1

    # ------------------------------------------------------------------
    # adyv[j_v] at v-faces (ny+1,) — pole entries unused (V=0 there).
    # ------------------------------------------------------------------
    adyv_int = 0.5 * (ady[:-1] + ady[1:])                 # (ny-1,)
    adyv = jnp.concatenate([ady[:1], adyv_int, ady[-1:]]) # (ny+1,)
    imuv = 1.0 / jnp.clip(cos_v, 1e-6, None)              # (ny+1,)

    # ------------------------------------------------------------------
    # w-face spacings dzw[k] (ny+1,) matching gSAM adzw·dz_ref.
    #   dzw[0]  = dz[0]     (w-face at k=0 collocates surface, half-cell)
    #   dzw[k]  = 0.5*(dz[k-1]+dz[k])  for k=1..nz-1
    #   dzw[nz] = dz[nz-1]  (w-face at top)
    # ------------------------------------------------------------------
    dzw_int = 0.5 * (dz[:-1] + dz[1:])                    # (nz-1,)
    dzw = jnp.concatenate([dz[:1], dzw_int, dz[-1:]])     # (nz+1,)

    # ------------------------------------------------------------------
    # U staggered slices
    # ------------------------------------------------------------------
    U_left  = U[:, :, :nx]       # west face of mass cell i     (nz, ny, nx)
    U_right = U[:, :, 1:nx + 1]  # east face of mass cell i     (nz, ny, nx)

    f3    = fcory[None, :, None]    # (1, ny, 1)
    fz3   = fcorzy[None, :, None]   # (1, ny, 1)
    tanr3 = tanr[None, :, None]
    mu3   = mu[None, :, None]

    # ------------------------------------------------------------------
    # 1a. Standard (f + u·tanr)·V_bar → dU
    # ------------------------------------------------------------------
    V_s = V[:, :ny,     :]                           # (nz, ny, nx)  south v-face
    V_n = V[:, 1:ny + 1, :]                          # (nz, ny, nx)  north v-face
    V_s_w = jnp.roll(V_s, +1, axis=-1)               # V at (j  , i-1)
    V_n_w = jnp.roll(V_n, +1, axis=-1)               # V at (j+1, i-1)

    adyv_s = adyv[None, :ny,     None]
    adyv_n = adyv[None, 1:ny + 1, None]
    ady3   = ady[None, :, None]

    v_bar = (adyv_s * (V_s + V_s_w) + adyv_n * (V_n + V_n_w)) / (4.0 * ady3)
    dU = (f3 + U_left * tanr3) * v_bar               # (nz, ny, nx)

    # ------------------------------------------------------------------
    # 1b. Standard -0.25·imuv·Σ(f + u·tanr)·mu·u → dV
    # ------------------------------------------------------------------
    def _q(u):
        return (f3 + u * tanr3) * mu3 * u

    q_row = _q(U_left) + _q(U_right)                 # (nz, ny, nx)
    q_n = q_row[:, 1:ny,     :]                      # north mass row j_v
    q_s = q_row[:, 0:ny - 1, :]                      # south mass row j_v - 1

    imuv_int = imuv[None, 1:ny, None]                # (1, ny-1, 1)
    dV_int = -0.25 * imuv_int * (q_n + q_s)          # (nz, ny-1, nx)

    dV = jnp.concatenate([
        jnp.zeros((nz, 1, nx)),
        dV_int,
        jnp.zeros((nz, 1, nx)),
    ], axis=1)                                       # (nz, ny+1, nx)

    # ------------------------------------------------------------------
    # 2a. docoriolisz contribution to dU:
    #
    #   dU -= 0.25 · fcorzy · (1/dz[k]) ·
    #         ( dzw[k]   · (W[k]   + W_west[k])
    #         + dzw[k+1] · (W[k+1] + W_west[k+1]) )
    #
    # W_west at the west face of mass cell i uses mass cells (i, i-1):
    #   W_west = 0.5*(W[i, j, k] + W[i-1, j, k])  — wait, the gSAM formula
    #   is `w(i,j,k) + w(i-1,j,k)` (sum, not average); the 0.25 prefactor
    #   already accounts for the factor-of-2 from the two mass cells and
    #   the factor-of-2 from the two w-faces.
    # ------------------------------------------------------------------
    W_west = jnp.roll(W, +1, axis=-1)                # W at mass cell i-1
    W_sum_lo = W     [:nz, :, :] + W_west[:nz, :, :] # faces k=0..nz-1 (bottom of cell k)
    W_sum_hi = W     [1:,  :, :] + W_west[1:,  :, :] # faces k=1..nz   (top of cell k)

    dzw_lo = dzw[:nz][:, None, None]                 # (nz, 1, 1) — bottom face metric
    dzw_hi = dzw[1:][:, None, None]                  # (nz, 1, 1) — top face metric
    dz3    = dz[:, None, None]                       # (nz, 1, 1)

    dU_z = -0.25 * fz3 * (dzw_lo * W_sum_lo + dzw_hi * W_sum_hi) / dz3
    dU = dU + dU_z                                   # (nz, ny, nx)

    # ------------------------------------------------------------------
    # 2b. docoriolisz contribution to dW at interior faces k=1..nz-1:
    #
    #   dW[k] += 0.25 · fcorzy ·
    #            ( U[i,j,k] + U[i+1,j,k] + U[i,j,k-1] + U[i+1,j,k-1] )
    #
    # The 4-point average straddles the w-face: two zonal faces of the
    # cell above (k) and two of the cell below (k-1).  In jsam, mass cell
    # k=0..nz-1 has its bottom w-face at index k and its top at index k+1.
    # So the w-face at interior index f=1..nz-1 sits between cells k=f-1
    # (below) and k=f (above).
    # ------------------------------------------------------------------
    U_sum = U_left + U_right                          # (nz, ny, nx)
    U_sum_above = U_sum[1:, :, :]                     # cell k   (nz-1, ny, nx)
    U_sum_below = U_sum[:-1, :, :]                    # cell k-1 (nz-1, ny, nx)

    dW_int = 0.25 * fz3 * (U_sum_above + U_sum_below) # (nz-1, ny, nx)

    dW = jnp.concatenate([
        jnp.zeros((1, ny, nx)),
        dW_int,
        jnp.zeros((1, ny, nx)),
    ], axis=0)                                        # (nz+1, ny, nx)

    return dU, dV, dW
