"""
Pressure solver for the anelastic equations on a lat-lon grid.
Solves ∇²p' = RHS via rfft-x + sparse-LU-(y,z) (spherical, matching gSAM pressure_gmg).

SPHERICAL SOLVER — matches gSAM pressure_gmg (used for dolatlon=T, doflat=F):
  - RHS: Spherical anelastic divergence with cos(lat)/imu factors (matches gSAM press_rhs.f90)
  - Solver: rfft in x + sparse LU factorisation per zonal mode m in (y,z)
  - Zonal eigenvalue: α_m(j) = [-4sin²(πm/nx)/(dλ R)²] × cos²(φ_j)  (Cartesian × cos², matches gSAM)
  - Meridional operator: (1/cosφ) d/dφ(cosφ dP/dφ) / R²  (true spherical Laplacian)
  - Vertical operator: rhow(k)/(rho(k)*adz(k)*adzw(k)*dz_ref²)  (anelastic)
  - Pressure history: p_prev/p_pprev returned unchanged; actual pressure separate
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jsam.core.state import ModelState
from jsam.core.grid.latlon import LatLonGrid, EARTH_RADIUS
from jsam.core.physics.microphysics import G_GRAV, CP
from jsam.core.dynamics.boundaries import bound_uv

def build_metric(grid: LatLonGrid, polar_filter: bool = False) -> dict:
    """Precompute metric factors for press_rhs and press_grad."""
    lat_rad = np.deg2rad(grid.lat)               # file lat — used only for Coriolis (gSAM line 497)
    lon_rad = np.deg2rad(grid.lon)
    dlon_rad = np.deg2rad(grid.dlon)

    # cos_lat = cos(y_gl) — geometric cell-center (midpoint of v-faces).
    # gSAM setgrid.f90:224 uses mu_gl = cos(y_gl) for ALL metric factors
    # (kurant, imu, muv, tanr).  Only Coriolis (fcory/fcorzy, line 497) uses
    # the file lat(j) directly.
    cos_lat = np.array(grid.cos_lat)             # (ny,) now y_gl-based

    dy_per_row = np.array(grid.dy_per_row)
    dy_ref     = float(grid.dy_ref)
    ady        = np.array(grid.ady)

    muv_interior = (ady[:-1] * cos_lat[1:] + ady[1:] * cos_lat[:-1]) / (ady[:-1] + ady[1:])
    lat_v_boundary = np.deg2rad(grid.lat_v[[0, -1]])
    cos_v = np.concatenate([
        [np.cos(lat_v_boundary[0])],
        muv_interior,
        [np.cos(lat_v_boundary[1])],
    ])

    dx_lon = EARTH_RADIUS * dlon_rad

    imu = 1.0 / np.clip(cos_lat, 1e-6, None)

    rho = grid.rho
    dz  = grid.dz
    nz  = len(rho)
    zi   = np.asarray(grid.zi)
    z1d  = np.asarray(grid.z)
    dz_ref = float(zi[1] - zi[0])
    adz    = np.asarray(dz, dtype=np.float64) / dz_ref
    adzw   = np.empty(len(rho) + 1)
    adzw[0]  = 1.0
    adzw[1:-1] = (z1d[1:] - z1d[:-1]) / dz_ref
    adzw[-1] = adzw[-2]
    rhow = np.zeros(nz + 1)
    adz_below = adz[:-1]
    adz_above = adz[1:]
    rhow[1:-1] = (rho[:-1] * adz_above + rho[1:] * adz_below) \
                 / (adz_below + adz_above)
    rhow[0]  = 2.0 * rho[0]    - rhow[1]
    rhow[-1] = 2.0 * rho[nz-1] - rhow[-2]

    p_surf = 101325.0
    p_face = np.zeros(len(rho) + 1)
    p_face[0] = p_surf
    for k in range(len(rho)):
        p_face[k + 1] = p_face[k] - rho[k] * G_GRAV * dz[k]
    pres = 0.5 * (p_face[:-1] + p_face[1:])

    gamaz = G_GRAV * grid.z / CP

    OMEGA = 4.0 * np.pi / 86400.0 / 2.0
    # Coriolis: gSAM setgrid.f90:497,501 uses sin/cos of lat(j) (file lat), NOT y_gl
    fcory  = 2.0 * OMEGA * np.sin(lat_rad)
    fcorzy = 2.0 * OMEGA * np.cos(lat_rad)
    # tanr: gSAM setgrid.f90:232 uses ±sqrt(1-mu²)/mu/R, i.e. tan(y_gl)/R
    # (mu = cos(y_gl) = our cos_lat).  Not tan(lat_file)/R.
    lat_center_rad = np.deg2rad(np.array(grid.lat_center))
    tanr   = np.tan(lat_center_rad) / EARTH_RADIUS

    m = {
        "imu": jnp.array(imu),
        "cos_v": jnp.array(cos_v),
        "dx_lon": float(dx_lon),
        "dy_lat": jnp.array(dy_per_row),
        "dy_lat_ref": float(dy_ref),
        "ady": jnp.array(ady),
        "rho": jnp.array(rho),
        "rhow": jnp.array(rhow),
        "dz": jnp.array(dz),
        "adz": jnp.array(adz),
        "adzw": jnp.array(adzw),
        "dz_ref": float(dz_ref),
        "cos_lat": jnp.array(cos_lat),
        "lat_rad": jnp.array(lat_rad),
        "lon_rad": jnp.array(lon_rad),
        "dlon_rad": float(dlon_rad),
        "nx": int(len(grid.lon)),
        "pres": jnp.array(pres),
        "presi": jnp.array(p_face),
        "gamaz": jnp.array(gamaz),
        "z": jnp.array(grid.z),
        "zi": jnp.array(zi),
        "fcory": jnp.array(fcory),
        "fcorzy": jnp.array(fcorzy),
        "tanr": jnp.array(tanr),
    }

    return m


def _press_rhs_core(U, V, W, metric, dt, debug_tag=None):
    """Common body.  If ``debug_tag`` is not None, prints per-direction
    max|div| with that tag."""
    dx      = metric["dx_lon"]
    dy      = metric["dy_lat"]
    rho     = metric["rho"]
    rhow    = metric["rhow"]
    dz      = metric["dz"]
    imu     = metric["imu"]
    cos_v   = metric["cos_v"]
    cos_lat = metric["cos_lat"]

    rho3   = rho[:, None, None]
    rhow3  = rhow[:, None, None]
    dz3    = dz[:, None, None]
    dy3    = dy[None, :, None]

    # Port of gSAM press_rhs.f90:19-43 (IRMA: dowally=True, dowallx=False):
    #   dowally: V(:, south_pole, :) = 0
    #   bound_uv: U[nx]=U[0], V[ny]=V[south]=0
    V = V.at[:, 0, :].set(0.0)
    U, V = bound_uv(U, V)

    div_u = imu[None, :, None] * (U[:, :, 1:] - U[:, :, :-1]) / dx
    div_v = (cos_v[None, 1:, None] * V[:, 1:, :]
           - cos_v[None, :-1, None] * V[:, :-1, :]) / (dy3 * cos_lat[None, :, None])
    div_w = (rhow3[1:] * W[1:] - rhow3[:-1] * W[:-1]) / (rho3 * dz3)

    if debug_tag is not None:
        import numpy as _np
        du = float(jnp.max(jnp.abs(div_u)))
        dv = float(jnp.max(jnp.abs(div_v)))
        dw = float(jnp.max(jnp.abs(div_w)))
        # Locate the worst div_w cell
        _dw_np = _np.array(div_w)
        _idx = _np.unravel_index(_np.argmax(_np.abs(_dw_np)), _dw_np.shape)
        print(f"  [press_rhs {debug_tag}] max|div_u|={du:.3e}  "
              f"max|div_v|={dv:.3e}  max|div_w|={dw:.3e}  "
              f"worst_dw@(k={_idx[0]},j={_idx[1]},i={_idx[2]})", flush=True)

    return (div_u + div_v + div_w) / dt


_press_rhs_jit = jax.jit(_press_rhs_core, static_argnames=("debug_tag",))


def press_rhs(U, V, W, metric, dt):
    """RHS of pressure Poisson — spherical anelastic divergence.

    If ``JSAM_PRESS_DEBUG=1``, runs an un-jitted version that prints
    per-direction max|div| to isolate which direction dominates the
    residual divergence.
    """
    import os as _os
    if _os.environ.get("JSAM_PRESS_DEBUG", "") == "1":
        return _press_rhs_core(U, V, W, metric, dt, debug_tag="")
    return _press_rhs_jit(U, V, W, metric, dt, debug_tag=None)


def _helmholtz_op(
    P_flat: jax.Array,
    m: int,
    metric: dict,
    ny: int,
    nz: int,
) -> jax.Array:
    """Apply Helmholtz operator H_m for zonal wavenumber m."""
    P = P_flat.reshape(ny, nz)

    cos_lat  = metric["cos_lat"]
    dy_row   = metric["dy_lat"]
    rho      = metric["rho"]
    rhow     = metric["rhow"]
    dz       = metric["dz"]
    dlon_rad = metric["dlon_rad"]

    R = EARTH_RADIUS

    nx_grid = metric["nx"]
    sin2 = jnp.sin(jnp.pi * m / nx_grid) ** 2
    # Scalar alpha (scaled form of spherical Poisson; matches gSAM pressure_gmg).
    alpha_m = -4.0 * sin2 / (dlon_rad * R) ** 2

    zonal = alpha_m * P

    cos_v = metric["cos_v"]
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])

    dP_north = P[1:, :] - P[:-1, :]
    flux_n = cos_v[1:-1, None] * dP_north / dy_v_int[:, None]

    flux_south = jnp.zeros((1, nz))
    flux_north = jnp.zeros((1, nz))
    flux = jnp.concatenate([flux_south, flux_n, flux_north], axis=0)

    L_y = (flux[1:, :] - flux[:-1, :]) / (dy_row[:, None] * cos_lat[:, None])

    adzw   = metric["adzw"]
    dz_ref = metric["dz_ref"]
    dPz = P[:, 1:] - P[:, :-1]
    flux_z_int = rhow[1:-1, None].T * dPz / (adzw[1:-1] * dz_ref)

    flux_z_bot = jnp.zeros((ny, 1))
    flux_z_top = jnp.zeros((ny, 1))
    flux_z = jnp.concatenate([flux_z_bot, flux_z_int, flux_z_top], axis=1)

    L_z = (flux_z[:, 1:] - flux_z[:, :-1]) / (dz[None, :] * rho[None, :])

    result = zonal + L_y + L_z
    return result.ravel()


def _build_Lz_matrix(metric: dict, nz: int) -> jax.Array:
    """Build nz×nz tridiagonal vertical Laplacian matrix."""
    rho  = metric["rho"]
    rhow = metric["rhow"]
    dz   = metric["dz"]
    adzw = metric["adzw"]
    dz_ref = float(metric["dz_ref"])

    rho_np  = np.array(rho)
    rhow_np = np.array(rhow)
    dz_np   = np.array(dz)
    adzw_np = np.array(adzw)

    L = np.zeros((nz, nz))
    for k in range(nz):
        coeff = 1.0 / (rho_np[k] * dz_np[k])
        if k > 0:
            c_lo = rhow_np[k] / (adzw_np[k] * dz_ref)
            L[k, k - 1] += c_lo * coeff
            L[k, k]     -= c_lo * coeff
        if k < nz - 1:
            c_hi = rhow_np[k + 1] / (adzw_np[k + 1] * dz_ref)
            L[k, k + 1] += c_hi * coeff
            L[k, k]     -= c_hi * coeff

    return jnp.array(L)


def _compute_alpha_m(m: int, metric: dict) -> jax.Array:
    """Return the zonal eigenvalue α_m (scalar) for the scaled spherical form.

    gSAM pressure_gmg multiplies both sides of ∇²p = RHS by cos²(lat), which
    moves the cos²(lat) factor from the zonal term to the meridional/vertical
    operators and the RHS.  In the scaled form, the zonal eigenvalue is simply
    the Cartesian FFT eigenvalue:  α_m = -4 sin²(πm/nx) / (R dλ)².

    Matches pressure_gmg.f90 line 245:
        alpha = (2 − 2·cos(2π m / nx)) / dx²
    where dx = R·dλ (constant).
    """
    dlon_rad = metric["dlon_rad"]
    nx_grid  = metric["nx"]
    sin2 = np.sin(np.pi * m / nx_grid) ** 2
    alpha = -4.0 * sin2 / (dlon_rad * EARTH_RADIUS) ** 2  # scalar
    return alpha  # float


def _build_Ly_matrix(metric: dict, ny: int) -> np.ndarray:
    """
    Build the ny×ny tridiagonal meridional Laplacian matrix (neg-semidef).

    Matches the L_y term in _helmholtz_op exactly, including non-uniform
    spacing (gSAM dyvar grid):

      flux_p[j_face] = cos_v[j_face] * (P[j] - P[j-1]) / dy_v[j_face]
      L_y[j]         = (flux_p[j+1] - flux_p[j]) / (dy_row[j] * cos_lat[j])

    where dy_v[j_face] = 0.5*(dy_row[j-1] + dy_row[j]) is the distance
    between adjacent mass points.  Neumann at the poles → boundary fluxes
    are zero.  On a uniform grid this reduces to the 1/dy² form.
    """
    cos_lat = np.array(metric["cos_lat"])
    cos_v   = np.array(metric["cos_v"])
    dy_row  = np.array(metric["dy_lat"])
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])

    L = np.zeros((ny, ny))
    for j in range(ny):
        inv_row = 1.0 / (dy_row[j] * cos_lat[j])
        if j > 0:
            c = cos_v[j] / dy_v_int[j - 1] * inv_row
            L[j, j - 1] += c
            L[j, j]     -= c
        if j < ny - 1:
            c = cos_v[j + 1] / dy_v_int[j] * inv_row
            L[j, j + 1] += c
            L[j, j]     -= c
    return L


def _build_Hm_matrix(m: int, metric: dict, ny: int, nz: int) -> np.ndarray:
    """
    Build the full (ny*nz)×(ny*nz) Helmholtz matrix for zonal wavenumber m.

    H_m = diag(α_m) ⊗ I_nz  +  L_y ⊗ I_nz  +  I_ny ⊗ L_z

    Flattening: row-major, index j*nz+k ↔ (j, k).
    H_m is negative (semi-)definite; for m≥1 it is negative definite.
    For m=0 it has a one-dimensional null space (constant field).
    """
    L_z_np = np.array(_build_Lz_matrix(metric, nz))
    L_y_np = _build_Ly_matrix(metric, ny)
    alpha  = float(_compute_alpha_m(m, metric))  # scalar
    I_ny = np.eye(ny)
    I_nz = np.eye(nz)
    I_full = np.eye(ny * nz)
    return (alpha * I_full
          + np.kron(L_y_np, I_nz)
          + np.kron(I_ny, L_z_np))


def _make_vertical_precond(m: int, metric: dict, ny: int, nz: int):
    """
    Build and return a function that applies the inverse of the vertical
    preconditioner M_j = -(alpha_m(j)*I + L_z) for each latitude j.

    M_j is positive definite for m > 0 (alpha_m < 0, L_z neg-semidef → -M_j pos-def).
    The preconditioner is applied to the NEGATED system (-H_m * p = -b).

    Returns:
        precond(r) — applies M^{-1} r, where r has shape (ny*nz,)
    """
    L_z = _build_Lz_matrix(metric, nz)
    alpha = float(_compute_alpha_m(m, metric))  # scalar
    I_nz = jnp.eye(nz)
    # Broadcast scalar alpha across ny rows
    alpha_rows = jnp.full((int(L_z.shape[0]) if False else 1,), alpha)  # placeholder
    M_batch = -(alpha * I_nz[None, :, :] + L_z[None, :, :])
    M_inv = jax.vmap(jnp.linalg.inv)(M_batch)

    def precond(r: jax.Array) -> jax.Array:
        R = r.reshape(ny, nz)
        Q = jax.vmap(lambda Mi, ri: Mi @ ri)(M_inv, R)
        return Q.ravel()

    return precond


# ---------------------------------------------------------------------------
# 3.  Preconditioned CG (PCG) solver  (Python for-loop, eager execution)
# ---------------------------------------------------------------------------

def _pcg_solve(
    op,
    b: jax.Array,
    precond,
    tol: float = 1e-7,  # gSAM gmg_precision typical value
    maxiter: int = 200,
) -> jax.Array:
    """Preconditioned Conjugate Gradient solver (eager execution)."""
    tol2 = tol ** 2 * float(jnp.dot(b, b))
    if tol2 == 0.0:
        return jnp.zeros_like(b)

    x  = jnp.zeros_like(b)
    r  = b
    z  = precond(r)
    p  = z
    rz = float(jnp.dot(r, z))

    for _ in range(maxiter):
        rr = float(jnp.dot(r, r))
        if rr <= tol2:
            break
        Ap  = op(p)
        pAp = float(jnp.dot(p, Ap))
        if pAp <= 0:
            break
        alpha  = rz / pAp
        x      = x + alpha * p
        r      = r - alpha * Ap
        z      = precond(r)
        rz_new = float(jnp.dot(r, z))
        beta   = rz_new / rz if rz > 0 else 0.0
        p      = z + beta * p
        rz     = rz_new

    return x


# ---------------------------------------------------------------------------
# 3b.  Spherical pressure solver: sparse LU per zonal mode  (numpy/scipy)
# ---------------------------------------------------------------------------

def _build_Hm_sparse(m: int, metric_np: dict, ny: int, nz: int):
    """
    Build H_m matching gSAM gmg_discr.f90 (mu_discr='B', rho_discr='B').

    System: [alpha_scalar*I + L_y_B + cos²(lat)*L_z_B_base] * P = cos²(lat)*RHS

    L_y 'B' off-diagonal (j→j-1):
      c_lo_B[j] = 2*(mu[j-1]*dy[j] + mu[j]*dy[j-1]) * mu[j] / ((dy[j-1]+dy[j])² * dy[j])
    L_z 'B' base off-diagonal (k→k-1), scaled by cos²(lat_j) per row:
      d_lo_base[k] = 2*(rho[k-1]*dz[k] + rho[k]*dz[k-1]) / ((dz[k-1]+dz[k])² * dz[k]*rho[k])
    alpha = Cartesian x-eigenvalue, scalar (no lat dependence) — gSAM pressure_gmg.f90.

    Row/column index: row = j*nz + k  (j outer, k inner).
    """
    import scipy.sparse as sp

    mu       = np.maximum(np.asarray(metric_np["cos_lat"], np.float64), 1e-6)  # (ny,)
    dy       = np.asarray(metric_np["dy_lat"],  np.float64)   # (ny,) physical cell widths
    rho      = np.asarray(metric_np["rho"],     np.float64)   # (nz,) cell-centre density
    dz       = np.asarray(metric_np["dz"],      np.float64)   # (nz,) physical cell heights
    dlon_rad = float(metric_np["dlon_rad"])
    nx_grid  = int(metric_np["nx"])

    # Scalar alpha — same for all j (gSAM pressure_gmg.f90:245).
    # gmg.f90:297 subtracts alpha from every row's diagonal uniformly (no polar
    # zeroing), so match that exactly.
    sin2         = np.sin(np.pi * m / nx_grid) ** 2
    alpha_scalar = -4.0 * sin2 / (dlon_rad * EARTH_RADIUS) ** 2  # (float)
    alpha_per_j        = np.full(ny, alpha_scalar, dtype=np.float64)

    # ── L_y 'B' stencil ─────────────────────────────────────────────────────
    # Interface between j-1 and j: 'B' average of mu, times mu[j] (center)
    # c_lo_B[j] = 2*(mu[j-1]*dy[j] + mu[j]*dy[j-1]) * mu[j] / ((dy[j-1]+dy[j])² * dy[j])
    # c_hi_B[j] = 2*(mu[j]*dy[j+1] + mu[j+1]*dy[j]) * mu[j] / ((dy[j]+dy[j+1])² * dy[j])
    dy_sum  = dy[:-1] + dy[1:]                           # (ny-1,) lo-hi sums
    num_ly  = mu[:-1] * dy[1:] + mu[1:] * dy[:-1]       # (ny-1,) 'B' numerator (shared)
    c_lo_B  = np.zeros(ny)
    c_hi_B  = np.zeros(ny)
    c_lo_B[1:]  = 2.0 * num_ly * mu[1:]  / (dy_sum ** 2 * dy[1:])
    c_hi_B[:-1] = 2.0 * num_ly * mu[:-1] / (dy_sum ** 2 * dy[:-1])

    # ── L_z 'B' base stencil (before cos² scaling) ──────────────────────────
    # d_lo_base[k] = 2*(rho[k-1]*dz[k]+rho[k]*dz[k-1]) / ((dz[k-1]+dz[k])²*dz[k]*rho[k])
    # d_hi_base[k] = 2*(rho[k]*dz[k+1]+rho[k+1]*dz[k]) / ((dz[k]+dz[k+1])²*dz[k]*rho[k])
    dz_sum      = dz[:-1] + dz[1:]                       # (nz-1,)
    num_lz      = rho[:-1] * dz[1:] + rho[1:] * dz[:-1] # (nz-1,)
    d_lo_base   = np.zeros(nz)
    d_hi_base   = np.zeros(nz)
    d_lo_base[1:]  = 2.0 * num_lz / (dz_sum ** 2 * dz[1:]  * rho[1:])
    d_hi_base[:-1] = 2.0 * num_lz / (dz_sum ** 2 * dz[:-1] * rho[:-1])

    # cos²(lat) scales L_z per row (gSAM: mu_cent(j)² factor in L_z 'B')
    cos2 = mu ** 2   # (ny,)

    # ── COO assembly ─────────────────────────────────────────────────────────
    j_idx = np.arange(ny)
    k_idx = np.arange(nz)
    jj, kk = np.meshgrid(j_idx, k_idx, indexing='ij')  # (ny, nz)
    row_all = (jj * nz + kk).ravel()                   # (ny*nz,)

    # Diagonal: alpha[j] + L_y_diag[j] + cos²[j]*L_z_diag_base[k]
    diag_L_y      = alpha_per_j - (c_lo_B + c_hi_B)    # (ny,)
    diag_Lz_base  = -(d_lo_base + d_hi_base)            # (nz,)
    diag_vals = (diag_L_y[:, None] + cos2[:, None] * diag_Lz_base[None, :]).ravel()

    r_list = [row_all]
    c_list = [row_all]
    v_list = [diag_vals]

    # ±k (vertical), off-diagonals scaled by cos²[j]
    d_lo_2d = (cos2[:, None] * d_lo_base[None, :]).ravel()  # (ny*nz,)
    d_hi_2d = (cos2[:, None] * d_hi_base[None, :]).ravel()

    mask_lo = kk.ravel() > 0
    r_list.append(row_all[mask_lo]);  c_list.append(row_all[mask_lo] - 1)
    v_list.append(d_lo_2d[mask_lo])

    mask_hi = kk.ravel() < nz - 1
    r_list.append(row_all[mask_hi]);  c_list.append(row_all[mask_hi] + 1)
    v_list.append(d_hi_2d[mask_hi])

    # ±j (meridional), L_y 'B'
    mask_jlo = jj.ravel() > 0
    r_list.append(row_all[mask_jlo]);  c_list.append(row_all[mask_jlo] - nz)
    v_list.append(c_lo_B[jj.ravel()[mask_jlo]])

    mask_jhi = jj.ravel() < ny - 1
    r_list.append(row_all[mask_jhi]);  c_list.append(row_all[mask_jhi] + nz)
    v_list.append(c_hi_B[jj.ravel()[mask_jhi]])

    rows_arr = np.concatenate(r_list).astype(np.int32)
    cols_arr = np.concatenate(c_list).astype(np.int32)
    vals_arr = np.concatenate(v_list)

    return sp.csc_matrix((vals_arr, (rows_arr, cols_arr)), shape=(ny * nz, ny * nz))


# Module-level cache: keyed by (ny, nz, nm).  Valid for one grid per session.
_SOLVER_CACHE: dict = {}


class _LUSolver:
    """Wraps a scipy SuperLU object for in-memory caching.

    Uses lu.solve(b) directly — avoids manually reimplementing the
    permutation logic, which is error-prone.
    Not picklable (disk cache disabled intentionally).
    """
    __slots__ = ('_lu',)

    def __init__(self, lu):
        self._lu = lu

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self._lu.solve(b)



def _get_sparse_solvers(metric_np: dict, ny: int, nz: int, nm: int):
    """
    Build (and cache) sparse LU factorizations of H_m for all zonal modes.

    Returns a list of nm _LUSolver objects: lu.solve(b) → x such that H_m x = b.
    LU factors (L, U, perm_r, perm_c — all picklable) are cached to disk keyed
    on grid geometry; set JSAM_LU_CACHE_DIR to override the cache directory.
    """
    import scipy.sparse.linalg as spla
    import scipy.sparse as sp
    from concurrent.futures import ThreadPoolExecutor
    import os

    key = (ny, nz, nm)
    if key in _SOLVER_CACHE:
        return _SOLVER_CACHE[key]

    n_workers = min(nm, int(os.environ.get("JSAM_LU_WORKERS", os.cpu_count() or 4)))

    def _build_mode(m):
        Hm = _build_Hm_sparse(m, metric_np, ny, nz)
        if m == 0:
            Hm = Hm - sp.eye(ny * nz, format="csc") * 1e-10
        return Hm
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        hm_list = list(pool.map(_build_mode, range(nm)))

    def _factorize(Hm):
        return _LUSolver(spla.splu(Hm))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        solvers = list(pool.map(_factorize, hm_list))
    del hm_list

    print("  [pressure] Factorisations ready.", flush=True)

    _SOLVER_CACHE[key] = solvers
    return solvers


# ---------------------------------------------------------------------------
# 4.  Thomas tridiagonal solver (vectorised over all modes)
# ---------------------------------------------------------------------------

def _thomas_batch_numpy(
    a: np.ndarray,       # (nz,) sub-diagonal;   a[0] is unused
    b_batch: np.ndarray, # (M, nz) per-mode diagonal
    c: np.ndarray,       # (nz,) super-diagonal; c[nz-1] is unused
    d_batch: np.ndarray, # (M, nz) RHS
) -> np.ndarray:         # (M, nz) solution
    """
    Solve M tridiagonal systems a[k]*p[k-1] + b[k]*p[k] + c[k]*p[k+1] = d[k]
    simultaneously via the Thomas algorithm.  O(M*nz) — no loops over modes.
    """
    M, nz = d_batch.shape
    b = b_batch.copy()   # work arrays; do not modify caller's arrays
    d = d_batch.copy()
    w = np.zeros((M, nz))

    # Forward sweep
    w[:, 0] = c[0] / b[:, 0]
    d[:, 0] /= b[:, 0]
    for k in range(1, nz):
        denom = b[:, k] - a[k] * w[:, k - 1]
        if k < nz - 1:
            w[:, k] = c[k] / denom
        d[:, k] = (d[:, k] - a[k] * d[:, k - 1]) / denom

    # Backward substitution
    p = np.empty_like(d)
    p[:, -1] = d[:, -1]
    for k in range(nz - 2, -1, -1):
        p[:, k] = d[:, k] - w[:, k] * p[:, k + 1]
    return p


# ---------------------------------------------------------------------------
# 5.  Poisson solver — FFT in x + DCT-II in y + Thomas in z  (Cartesian)
#     Kept for reference / test use.  Active solver is solve_pressure below.
# ---------------------------------------------------------------------------

def _solve_pressure_cartesian(
    rhs: jax.Array,         # (nz, ny, nx)
    metric: dict,
    **kwargs,
) -> jax.Array:             # (nz, ny, nx)
    """
    Solve ∇²p' = rhs using FFT in x + DCT-II in y + Thomas in z.

    Matches gSAM pressure_big.f90 (dolatlon=F or doflat=T).  NOT the active
    solver for global lat-lon runs — use _solve_pressure_spherical for that.

    Cartesian eigenvalues:
      λ_x[m] = (2*cos(2πm/nx) − 2) / dx²
      λ_y[n] = (2*cos(πn/ny)  − 2) / dy²   (uses scalar dy_ref — wrong for sphere)
    """
    from scipy.fft import dct, idct

    rhs_np = np.array(rhs, dtype=np.float64)   # (nz, ny, nx)
    nz, ny, nx = rhs_np.shape
    nm = nx // 2 + 1

    dx   = float(metric["dx_lon"])
    # Cartesian reference solver: DCT-II diagonalises only constant-coefficient
    # operators, so it requires a scalar dy.  Use the reference dy (mid-latitude)
    # — this solver is kept only for testing; the active solver is spherical.
    dy   = float(metric["dy_lat_ref"])
    rho    = np.array(metric["rho"],   dtype=np.float64)   # (nz,)
    rhow   = np.array(metric["rhow"],  dtype=np.float64)   # (nz+1,)
    dz_    = np.array(metric["dz"],    dtype=np.float64)   # (nz,)
    adzw_  = np.array(metric["adzw"],  dtype=np.float64)   # (nz+1,)
    dz_ref_ = float(metric["dz_ref"])

    # ── Eigenvalues ────────────────────────────────────────────────────────────
    m_arr = np.arange(nm, dtype=np.float64)
    lam_x = (2.0 * np.cos(2.0 * np.pi * m_arr / nx) - 2.0) / dx**2   # (nm,)

    k_arr = np.arange(ny, dtype=np.float64)
    lam_y = (2.0 * np.cos(np.pi * k_arr / ny) - 2.0) / dy**2          # (ny,)

    # ── Vertical tridiagonal coefficients (L_z, same for all (m,k)) ──────────
    # Matches gSAM pressure_big.f90 exactly (Fortran k → Python l):
    #   a(k) = rhow(k)   / (rho(k) * adz(k) * adzw(k)   * dz^2)
    #   c(k) = rhow(k+1) / (rho(k) * adz(k) * adzw(k+1) * dz^2)
    # In Python (0-based, dz[l] = adz[l]*dz_ref):
    #   a_sub[l] = rhow[l]   / (rho[l] * adz[l] * adzw[l]   * dz_ref^2)
    #   c_sup[l] = rhow[l+1] / (rho[l] * adz[l] * adzw[l+1] * dz_ref^2)
    a_sub = np.zeros(nz)   # sub-diagonal;   a_sub[0]   unused
    c_sup = np.zeros(nz)   # super-diagonal; c_sup[-1]  unused
    for l in range(nz):
        inv = 1.0 / (rho[l] * dz_[l])   # dz_[l] = adz[l]*dz_ref
        if l > 0:
            a_sub[l] = rhow[l]     / (adzw_[l]     * dz_ref_) * inv
        if l < nz - 1:
            c_sup[l] = rhow[l + 1] / (adzw_[l + 1] * dz_ref_) * inv
    Lz_diag = -(a_sub + c_sup)   # (nz,)

    # ── Per-(k,m) diagonal for Thomas: shape (ny*nm, nz) ─────────────────────
    # Outer index = k (meridional DCT mode), inner = m (zonal rfft mode)
    lam_km = lam_x[None, :] + lam_y[:, None]           # (ny, nm)
    b_diag = Lz_diag[None, :] + lam_km.ravel()[:, None] # (ny*nm, nz)

    # (k=0, m=0) singularity: shift diagonal by −1 so Thomas returns p≈0
    b_diag[0] -= 1.0

    # ── 1. rfft in x ──────────────────────────────────────────────────────────
    rhs_x = np.fft.rfft(rhs_np, axis=2)   # (nz, ny, nm) complex
    p_hat = np.zeros_like(rhs_x)
    is_nyquist = (nx % 2 == 0)

    for is_imag in (False, True):
        if is_imag:
            part = rhs_x.imag.copy()
            # m=0 and Nyquist coefficients are always real
            part[:, :, 0] = 0.0
            if is_nyquist:
                part[:, :, -1] = 0.0
        else:
            part = rhs_x.real.copy()

        # ── 2. DCT-II in y (Neumann BCs) ──────────────────────────────────────
        b_yk = dct(part, type=2, norm='ortho', axis=1)   # (nz, ny, nm)

        # ── 3. Thomas solve — all (k, m) simultaneously ───────────────────────
        # Layout: (ny*nm, nz) with k outer, m inner
        b_flat = b_yk.transpose(1, 2, 0).reshape(ny * nm, nz).copy()

        # Zero RHS for singular (k=0, m=0) mode
        b_flat[0] = 0.0

        p_flat = _thomas_batch_numpy(a_sub, b_diag, c_sup, b_flat)  # (ny*nm, nz)
        p_flat[0] = 0.0   # enforce p=0 for global-mean mode

        # ── 4. iDCT-II in y ────────────────────────────────────────────────────
        p_yk    = p_flat.reshape(ny, nm, nz).transpose(2, 0, 1)   # (nz, ny, nm)
        p_part  = idct(p_yk, type=2, norm='ortho', axis=1)         # (nz, ny, nm)

        if is_imag:
            p_hat += 1j * p_part
        else:
            p_hat += p_part.astype(complex)

    # ── 5. irfft in x ─────────────────────────────────────────────────────────
    p = np.fft.irfft(p_hat, n=nx, axis=2)   # (nz, ny, nx)
    return jnp.array(p)


# ---------------------------------------------------------------------------
# 5b. Poisson solver — spherical Helmholtz via rfft in x + sparse LU in (y,z)
# ---------------------------------------------------------------------------

def solve_pressure(
    rhs: jax.Array,   # (nz, ny, nx)
    metric: dict,
    **kwargs,
) -> jax.Array:       # (nz, ny, nx)
    """
    Solve the anelastic Poisson equation for pressure.

    Matches gSAM pressure_gmg (dolatlon=T, doflat=F):
      - rfft in x → nm complex modes
      - RHS × cos²(lat) per mode (matching gSAM: ff = rhs · mu_gl²)
      - Per-mode (y,z) solve — geometric multigrid (GMG) by default,
        matching gSAM's own solver path.  Set ``JSAM_USE_LU_PRESSURE=1``
        to fall back to the cached sparse-LU backend.
      - irfft in x → physical pressure
    """
    import os
    if os.environ.get("JSAM_USE_LU_PRESSURE", "").strip() == "1":
        return _solve_pressure_spherical(rhs, metric, **kwargs)
    return _solve_pressure_spherical_gmg(rhs, metric, **kwargs)


def _solve_pressure_spherical(
    rhs: jax.Array,   # (nz, ny, nx)
    metric: dict,
    **kwargs,
) -> jax.Array:       # (nz, ny, nx)
    """
    Solve the spherical anelastic Poisson equation (ACTIVE SOLVER).

    Matches gSAM pressure_gmg (used for dolatlon=T, doflat=F).

    Algorithm: rfft in x + sparse LU in (y,z) per zonal mode m:
      1. rfft in x  → nm complex modes
      2. For each mode m: solve (ny*nz) spherical Helmholtz system H_m p_m = rhs_m
         H_m includes true spherical α_m(j) eigenvalue (lat-dependent).
         LU factorisations cached on first call.
      3. irfft in x  → physical pressure field
    """
    rhs_np = np.array(rhs, dtype=np.float64)   # (nz, ny, nx)
    nz, ny, nx = rhs_np.shape
    nm = nx // 2 + 1

    # Convert metric to plain numpy dict for scipy
    metric_np: dict = {}
    for k, v in metric.items():
        if isinstance(v, jax.Array):
            metric_np[k] = np.array(v, dtype=np.float64)
        else:
            metric_np[k] = v

    # Build / retrieve cached sparse LU factorisations for all nm modes
    solvers = _get_sparse_solvers(metric_np, ny, nz, nm)

    # cos²(lat) for RHS pre-scaling — matches gSAM: ff(k,j) = rhs(k,j) * mu_gl(j)²
    cos2_ny = metric_np["cos_lat"] ** 2   # (ny,)

    # ── 1. rfft in x ─────────────────────────────────────────────────────────
    rhs_x = np.fft.rfft(rhs_np, axis=2)   # (nz, ny, nm) complex

    p_hat = np.zeros_like(rhs_x)          # (nz, ny, nm) complex

    # ── 2. Sparse solve for each zonal mode m — parallel over modes ───────────
    from concurrent.futures import ThreadPoolExecutor
    import os
    n_solve_workers = int(os.environ.get("JSAM_LU_WORKERS", os.cpu_count() or 4))

    def _solve_mode(m):
        lu    = solvers[m]
        rhs_m = rhs_x[:, :, m]          # (nz, ny)
        # Pre-multiply by cos²(lat) matching gSAM: ff(k,j) = rhs(k,j) * mu_gl(j)²
        rhs_m = rhs_m * cos2_ny[np.newaxis, :]
        rhs_real = np.ascontiguousarray(rhs_m.real.T).ravel()
        rhs_imag = np.ascontiguousarray(rhs_m.imag.T).ravel()
        if m == 0:
            rhs_real -= rhs_real.mean()
        p_real = lu.solve(rhs_real)
        p_imag = lu.solve(rhs_imag)
        return m, (p_real + 1j * p_imag).reshape(ny, nz).T

    with ThreadPoolExecutor(max_workers=n_solve_workers) as pool:
        for m, p_m in pool.map(_solve_mode, range(nm)):
            p_hat[:, :, m] = p_m

    # ── 3. irfft in x ────────────────────────────────────────────────────────
    p = np.fft.irfft(p_hat, n=nx, axis=2)   # (nz, ny, nx)

    # Remove global mean (barotropic mode; no physical meaning for p')
    p -= p.mean()

    return jnp.array(p)


# ---------------------------------------------------------------------------
# 4b.  Spherical solver — GMG backend (matches gSAM pressure_gmg.f90 exactly)
# ---------------------------------------------------------------------------

# Cache: (ny, nz) → (matrices, Ps, Rs, alphas_list).  The hierarchy / operators
# depend only on the grid geometry, so one build per grid serves every mode
# and every time-step.
_GMG_CACHE: dict = {}

# Warm-start buffer, keyed by (ny, nz, nm).  For each zonal mode m we keep
# the previous time-step's solution (real + imag halves) so the next call
# to gmg_solve can start from there rather than zero — matches gSAM's
# ``pfy(:,:,i)`` save array in pressure_gmg.f90.  Reset per grid.
_PFY_CACHE: dict = {}


def _get_gmg_infra(metric_np: dict, ny: int, nz: int, nm: int):
    """Build (and cache) GMG hierarchy + level matrices + P/R operators.

    Returns (matrices, Ps, Rs, alphas) where ``alphas[m]`` is the zonal
    eigenvalue for Fourier mode m (negative in jsam sign convention).
    """
    from jsam.core.dynamics.gmg import (
        level0_from_metric, build_grid_hierarchy,
        build_all_level_matrices, build_all_transfer_operators, alpha_m,
    )

    key = (ny, nz, nm)
    if key in _GMG_CACHE:
        return _GMG_CACHE[key]

    l0       = level0_from_metric(metric_np)
    levels   = build_grid_hierarchy(l0, coarse_size=8, max_levels=20)
    matrices = build_all_level_matrices(levels)
    Ps, Rs   = build_all_transfer_operators(levels)

    dlon_rad = float(metric_np["dlon_rad"])
    alphas   = [alpha_m(m, dlon_rad, EARTH_RADIUS) for m in range(nm)]

    print(f"  [pressure/gmg] Hierarchy: {len(levels)} levels  finest {ny}×{nz}",
          flush=True)
    _GMG_CACHE[key] = (matrices, Ps, Rs, alphas)
    return _GMG_CACHE[key]


def _solve_pressure_spherical_gmg(
    rhs: jax.Array,    # (nz, ny, nx)
    metric: dict,
    tol: float = 1e-10,
    max_v_cyc: int = 100,   # gSAM pressure_gmg.f90:199 uses v_cyc=100
    **kwargs,
) -> jax.Array:        # (nz, ny, nx)
    """Spherical anelastic pressure solver using GMG per zonal mode.

    Exact port of gSAM ``pressure_gmg.f90`` (dolatlon=T, doflat=F) —
    same rfft-in-x + per-mode (y,z) solve structure as the LU backend,
    but replaces the LU factorisation with geometric multigrid.

    Algorithm:
      1. rfft in x  → nm complex modes on the (y, z) plane
      2. For each mode m:
           pre-scale rhs_m *= cos²(lat)     (gSAM ff = rhs · mu²)
           call gmg_solve for (A + α_m · I) p = rhs_m (real + imag)
      3. irfft in x  → physical pressure

    Convergence control:
      * ``tol``         — relative residual tolerance (gSAM default 1e-10)
      * ``max_v_cyc``   — max V-cycles per real/imag solve

    Near-singular modes (small m) may stagnate before reaching ``tol``
    — gSAM accepts this too (error_code=2).  The caller ignores the
    error code since stagnation at √cond·ε_machine is physically harmless.
    """
    from jsam.core.dynamics.gmg import VCycleOpts, VCycleState, gmg_solve

    rhs_np = np.array(rhs, dtype=np.float64)        # (nz, ny, nx)
    nz_arr, ny_arr, nx = rhs_np.shape
    nm = nx // 2 + 1

    # Convert metric to plain numpy dict for the numpy/scipy GMG kernels
    metric_np: dict = {}
    for k, v in metric.items():
        if isinstance(v, jax.Array):
            metric_np[k] = np.array(v, dtype=np.float64)
        else:
            metric_np[k] = v

    matrices, Ps, Rs, alphas = _get_gmg_infra(metric_np, ny_arr, nz_arr, nm)

    cos2_ny = metric_np["cos_lat"] ** 2              # (ny,)

    # ── 1. rfft in x ────────────────────────────────────────────────────────
    rhs_x = np.fft.rfft(rhs_np, axis=2)              # (nz, ny, nm) complex
    p_hat = np.zeros_like(rhs_x)                     # (nz, ny, nm) complex

    # ── 2. Per-mode GMG (parallel over modes) ───────────────────────────────
    from concurrent.futures import ThreadPoolExecutor
    import os
    n_workers = int(os.environ.get(
        "JSAM_LU_WORKERS", os.cpu_count() or 4
    ))

    opts_default = VCycleOpts()                       # gSAM defaults

    # m=0 singular Laplacian: gSAM-exact — alpha=0 + remov_null='cycl'.
    opts_m0 = VCycleOpts(remov_null="cycl")

    # Warm-start buffer: keeps the previous call's solution per mode so
    # gmg_solve starts from a good initial guess (matches gSAM's pfy(:,:,i)
    # save array in pressure_gmg.f90).  Keyed by (ny, nz, nm).
    pfy_key = (ny_arr, nz_arr, nm)
    pfy = _PFY_CACHE.setdefault(
        pfy_key,
        {m: {"re": np.zeros((ny_arr, nz_arr), dtype=np.float64),
             "im": np.zeros((ny_arr, nz_arr), dtype=np.float64)}
         for m in range(nm)},
    )

    _press_debug = os.environ.get("JSAM_PRESS_DEBUG", "") == "1"
    _target_modes = {0, 1, 2, 10, 100, 500, nm - 1}

    def _solve_mode(m):
        alpha = alphas[m]
        opts  = opts_m0 if m == 0 else opts_default
        rhs_m  = rhs_x[:, :, m] * cos2_ny[np.newaxis, :]
        rhs_r  = np.ascontiguousarray(rhs_m.real.T)
        rhs_i  = np.ascontiguousarray(rhs_m.imag.T)
        # gSAM pressure_gmg.f90 passes ff directly to gmg_solve without any
        # mean subtraction.  Arithmetic-mean subtraction would push the RHS
        # out of range(A): the 'B' stencil is non-symmetric, so null(A^T) is
        # not constants — the physical RHS is in range(A) via mass
        # conservation (cos·Δy·Δz·ρ-weighted zero sum), not arithmetic-zero.
        state = VCycleState.empty_for(matrices)
        res_r = gmg_solve(matrices, Ps, Rs, rhs_r, alpha,
                          tol=tol, max_v_cyc=max_v_cyc,
                          sol=pfy[m]["re"], opts=opts, state=state)
        res_i = gmg_solve(matrices, Ps, Rs, rhs_i, alpha,
                          tol=tol, max_v_cyc=max_v_cyc,
                          sol=pfy[m]["im"], opts=opts, state=state)
        pfy[m]["re"] = res_r.sol.copy()
        pfy[m]["im"] = res_i.sol.copy()
        if _press_debug and m in _target_modes:
            rhs_rmax = max(np.abs(rhs_r).max(), np.abs(rhs_i).max())
            p_rmax   = max(np.abs(res_r.sol).max(), np.abs(res_i.sol).max())
            print(f"    [mode m={m:4d}] alpha={alpha:+.3e}  "
                  f"|rhs_m|={rhs_rmax:.3e}  |p_m|={p_rmax:.3e}  "
                  f"re_iters={res_r.iter} err={res_r.error_code}  "
                  f"im_iters={res_i.iter} err={res_i.error_code}",
                  flush=True)
        return m, (res_r.sol + 1j * res_i.sol).T

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for m, p_m in pool.map(_solve_mode, range(nm)):
            p_hat[:, :, m] = p_m

    # ── 3. irfft in x ───────────────────────────────────────────────────────
    p = np.fft.irfft(p_hat, n=nx, axis=2)             # (nz, ny, nx)

    if _press_debug:
        # Separate m=0 (zonal mean) contribution from rest
        p_m0 = np.fft.irfft(
            p_hat * (np.arange(nm) == 0)[None, None, :], n=nx, axis=2
        )
        p_m_gt0 = p - p_m0
        print(f"    [p after irfft] max|p|={np.abs(p).max():.3e}  "
              f"max|p_m=0|={np.abs(p_m0).max():.3e}  "
              f"max|p_m>0|={np.abs(p_m_gt0).max():.3e}  "
              f"p.mean={p.mean():+.3e}", flush=True)

    # Remove global mean (barotropic mode)
    p -= p.mean()

    return jnp.array(p)


# ---------------------------------------------------------------------------
# 5.  Pressure gradient application  (matches press_grad.f90)
# ---------------------------------------------------------------------------

def _apply_pressure_gradient_core(U, V, W, p, metric, dt, igam2, debug_tag=None):
    dx     = metric["dx_lon"]
    dy_row = metric["dy_lat"]
    imu    = metric["imu"]

    p_west = jnp.roll(p, 1, axis=2)
    dp_dx  = imu[None, :, None] * (p - p_west) / dx

    dy_v_int  = 0.5 * (dy_row[:-1] + dy_row[1:])
    dp_dy_int = (p[:, 1:, :] - p[:, :-1, :]) / dy_v_int[None, :, None]

    dz_ref  = metric["dz_ref"]
    adzw    = metric["adzw"]
    dz_face = dz_ref * adzw[1:-1]
    dp_dz_int = (p[1:, :, :] - p[:-1, :, :]) / dz_face[:, None, None]

    if debug_tag is not None:
        import numpy as _np
        gx = float(jnp.max(jnp.abs(dp_dx)))
        gy = float(jnp.max(jnp.abs(dp_dy_int)))
        gz = float(jnp.max(jnp.abs(dp_dz_int)))
        _gz_np = _np.array(dp_dz_int)
        _idx = _np.unravel_index(_np.argmax(_np.abs(_gz_np)), _gz_np.shape)
        print(f"  [press_grad {debug_tag}] max|dp_dx|={gx:.3e}  "
              f"max|dp_dy|={gy:.3e}  max|dp_dz|={gz:.3e}  "
              f"worst_gz@(k={_idx[0]},j={_idx[1]},i={_idx[2]})  "
              f"dt·igam2·gz={dt*igam2*gz:.3e}", flush=True)

    U_new = U.at[:, :, 1:-1].add(-dt * dp_dx[:, :, 1:])
    U_new = U_new.at[:, :, 0].add(-dt * dp_dx[:, :, 0])
    V_new = V.at[:, 1:-1, :].add(-dt * dp_dy_int)
    W_new = W.at[1:-1, :, :].add(-dt * igam2 * dp_dz_int)
    return U_new, V_new, W_new


_apply_pressure_gradient_jit = jax.jit(
    _apply_pressure_gradient_core, static_argnames=("debug_tag",)
)


def apply_pressure_gradient(U, V, W, p, metric, dt, igam2=1.0):
    """Subtract spherical pressure gradient from tentative velocities.

    If ``JSAM_PRESS_DEBUG=1``, runs un-jitted with per-direction
    max|dp/d*| prints.
    """
    import os as _os
    if _os.environ.get("JSAM_PRESS_DEBUG", "") == "1":
        return _apply_pressure_gradient_core(U, V, W, p, metric, dt, igam2,
                                             debug_tag="")
    return _apply_pressure_gradient_jit(U, V, W, p, metric, dt, igam2,
                                        debug_tag=None)


# ---------------------------------------------------------------------------
# 6.  Full pressure step  (convenience wrapper)
# ---------------------------------------------------------------------------

def adams_b(
    state:   ModelState,
    p_prev:  jax.Array,             # (nz, ny, nx) p_{n-1}
    metric:  dict,
    dt:      float,
    bt:      float         = -0.5,  # AB coefficient for ∇p_{n-1}
    p_pprev: jax.Array | None = None,  # (nz, ny, nx) p_{n-2}, None on AB2 step
    ct:      float         = 0.0,   # AB3 coefficient for ∇p_{n-2}
    igam2:   float         = 1.0,   # 1/gamma_RAVE² (=1 for incompressible)
) -> ModelState:
    """
    Apply previous-step pressure gradient correction (adamsB.f90).

    Matches gSAM adamsB.f90:
        u -= dt * (bt*(p_nb[i]-p_nb[i-1]) + ct*(p_nc[i]-p_nc[i-1])) * imu/dx
        w -= dt * igam2 * (bt*(p_nb[k]-p_nb[k-1]) + ct*(p_nc[k]-p_nc[k-1])) /dz

    This is called AFTER adamsA (advance_momentum) and BEFORE the new pressure
    solve, correcting for the lagged pressure gradient in the AB3 scheme.

    For the AB2 bootstrap step (ct=0), this reduces to a single-term correction
    with coefficient bt (= -0.5 for constant dt).
    """
    # apply_pressure_gradient does: U -= dt_eff * grad(p).
    U_new, V_new, W_new = apply_pressure_gradient(
        state.U, state.V, state.W, p_prev, metric, bt * dt, igam2,
    )
    if p_pprev is not None and ct != 0.0:
        U_new, V_new, W_new = apply_pressure_gradient(
            U_new, V_new, W_new, p_pprev, metric, ct * dt, igam2,
        )
    U_new = U_new.at[:, :, -1].set(U_new[:, :, 0])   # restore periodicity
    return ModelState(
        U=U_new, V=V_new, W=W_new,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )


def pressure_step(
    state: ModelState,
    grid: LatLonGrid,
    metric: dict,
    dt: float,
    n_iter: int = 1,
    at: float = 1.0,
    rel_tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple["ModelState", jax.Array]:
    """
    Full pressure correction step with Richardson iteration.

      1. Compute anelastic divergence (RHS)
      2. Solve Poisson equation for p'
      3. Subtract ∇p' from (U, V, W)
      4. Repeat 1–3 on the residual until |rhs| / |rhs_initial| < rel_tol
         (capped at ``max_iter`` iterations)

    The 'B' Helmholtz stencil in solve_pressure is close to but not exactly the
    consistent discretisation of press_rhs + press_grad, so a single LU solve
    leaves ~4% residual divergence.  Richardson iteration with the LU as
    preconditioner converges (slowly — factor ~0.5–0.8 per iter), but is needed
    to reach gSAM's pressure_gmg tolerance (eps=1e-10 at nstep≤3).

    The ``n_iter`` argument is a hard floor — the loop runs at least n_iter
    iterations even if tolerance is reached earlier.  Use ``rel_tol`` /
    ``max_iter`` for adaptive convergence (default: iterate until residual
    drops 6 orders of magnitude or 50 iters, whichever first).

    PRESSURE HISTORY MANAGEMENT (differs from gSAM):
      - gSAM: Manages pressure in rotating buffer p(:,:,:,na/nb/nc)
      - JAX: Returns p_prev and p_pprev unchanged; actual pressure as separate p_total
      - Caller must manage pressure history update outside this function
    """
    U_cur, V_cur, W_cur = state.U, state.V, state.W
    p_total = jnp.zeros((state.TABS.shape[0], state.TABS.shape[1],
                          state.TABS.shape[2]))

    dt_eff = at * dt
    import os as _os_diag
    _debug = _os_diag.environ.get("JSAM_PRESS_DEBUG") == "1"
    _rhs0_abs = None
    _iter = 0
    while True:
        rhs = press_rhs(U_cur, V_cur, W_cur, metric, dt_eff)
        _rhs_abs = float(jnp.max(jnp.abs(rhs)))
        _rhs_rms = float(jnp.sqrt(jnp.mean(rhs ** 2)))
        if _rhs0_abs is None:
            _rhs0_abs = _rhs_abs
            _rhs0_rms = _rhs_rms
        if _debug:
            _W_before = float(jnp.max(jnp.abs(W_cur)))
            _W_bot = float(jnp.max(jnp.abs(W_cur[0, :, :])))
            _W_top = float(jnp.max(jnp.abs(W_cur[-1, :, :])))
            _U_cyc = float(jnp.max(jnp.abs(U_cur[:, :, -1] - U_cur[:, :, 0])))
            _V_sp  = float(jnp.max(jnp.abs(V_cur[:, 0, :])))
            _V_np  = float(jnp.max(jnp.abs(V_cur[:, -1, :])))
            print(f"  [press diag] W[0]_max={_W_bot:.3e}  W[-1]_max={_W_top:.3e}  "
                  f"|U[nx]-U[0]|_max={_U_cyc:.3e}  V[south]_max={_V_sp:.3e}  V[north]_max={_V_np:.3e}",
                  flush=True)
        # Stop conditions: (a) reached max_iter, OR (b) reached n_iter floor AND
        # residual is below tolerance.
        _converged = (_rhs_abs / max(_rhs0_abs, 1e-30)) < rel_tol
        if _iter >= max_iter or (_iter >= n_iter and _converged):
            if _debug:
                print(f"  [pressure] STOP iter={_iter} |rhs|={_rhs_abs:.3e} "
                      f"rel={_rhs_abs/max(_rhs0_abs,1e-30):.3e} "
                      f"rms_rel={_rhs_rms/max(_rhs0_rms,1e-30):.3e} "
                      f"|W|={_W_before:.3f}", flush=True)
            break
        p_inc = solve_pressure(rhs, metric)
        if _debug:
            _p_abs = float(jnp.max(jnp.abs(p_inc)))
            _p_rms = float(jnp.sqrt(jnp.mean(p_inc ** 2)))
            print(f"  [pressure iter={_iter}] |rhs_in|={_rhs_abs:.3e} "
                  f"(rms={_rhs_rms:.3e})  |p|={_p_abs:.3e} (rms={_p_rms:.3e})  "
                  f"rel={_rhs_abs/max(_rhs0_abs,1e-30):.3e} "
                  f"rms_rel={_rhs_rms/max(_rhs0_rms,1e-30):.3e}  "
                  f"|W|_before={_W_before:.3f}", flush=True)
        p_total = p_total + p_inc
        U_cur, V_cur, W_cur = apply_pressure_gradient(
            U_cur, V_cur, W_cur, p_inc, metric, dt_eff,
        )
        U_cur = U_cur.at[:, :, -1].set(U_cur[:, :, 0])
        _iter += 1

    # gSAM pressure.f90 clips pp (physical pressure for output) but NOT p (the
    # velocity-correction pressure stored here).  No clip applied — matches gSAM.

    new_state = ModelState(
        U=U_cur, V=V_cur, W=W_cur,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, p_total
