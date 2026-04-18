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

def build_metric(grid: LatLonGrid, polar_filter: bool = False) -> dict:
    """Precompute metric factors for press_rhs and press_grad."""
    lat_rad = np.deg2rad(grid.lat)
    lon_rad = np.deg2rad(grid.lon)
    dlon_rad = np.deg2rad(grid.dlon)

    cos_lat = np.cos(lat_rad)

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
    fcory  = 2.0 * OMEGA * np.sin(lat_rad)
    fcorzy = 2.0 * OMEGA * np.cos(lat_rad)
    tanr   = np.tan(lat_rad) / EARTH_RADIUS

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


@jax.jit
def press_rhs(
    U: jax.Array,
    V: jax.Array,
    W: jax.Array,
    metric: dict,
    dt: float,
) -> jax.Array:
    """RHS of pressure Poisson — spherical anelastic divergence."""
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

    div_u = imu[None, :, None] * (U[:, :, 1:] - U[:, :, :-1]) / dx

    div_v = (cos_v[None, 1:, None] * V[:, 1:, :]
           - cos_v[None, :-1, None] * V[:, :-1, :]) / (dy3 * cos_lat[None, :, None])

    div_w = (rhow3[1:] * W[1:] - rhow3[:-1] * W[:-1]) / (rho3 * dz3)

    return (div_u + div_v + div_w) / dt


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
    alpha_m = -4.0 * sin2 / (dlon_rad * R * cos_lat) ** 2

    zonal = alpha_m[:, None] * P

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
    """Return the zonal eigenvalue α_m(j) for each latitude j."""
    cos_lat  = metric["cos_lat"]   # (ny,) JAX
    dlon_rad = metric["dlon_rad"]
    nx_grid  = metric["nx"]
    sin2 = np.sin(np.pi * m / nx_grid) ** 2
    alpha = -4.0 * sin2 / (dlon_rad * EARTH_RADIUS * np.array(cos_lat)) ** 2
    return jnp.array(alpha)  # (ny,)


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
    alpha  = np.array(_compute_alpha_m(m, metric))
    I_ny = np.eye(ny)
    I_nz = np.eye(nz)
    return (np.kron(np.diag(alpha), I_nz)
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
    alpha = _compute_alpha_m(m, metric)
    I_nz = jnp.eye(nz)
    M_batch = -(alpha[:, None, None] * I_nz[None, :, :] + L_z[None, :, :])
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
    Build H_m as a scipy CSC sparse matrix of size (ny*nz, ny*nz).

    H_m is the discrete spherical Helmholtz operator for zonal wavenumber m:
      H_m[P](j,k) = alpha_m(j)*P(j,k) + L_y[P](j,k) + L_z[P](j,k)

    Row/column index: row = j*nz + k  (j outer, k inner).

    Non-zeros per row: up to 5 (self, ±j neighbour, ±k neighbour).
    Fully vectorised — no Python loops over (j, k).
    """
    import scipy.sparse as sp

    cos_lat_c = np.maximum(np.asarray(metric_np["cos_lat"], np.float64), 1e-6)
    cos_v     = np.asarray(metric_np["cos_v"],   np.float64)
    dy_row    = np.asarray(metric_np["dy_lat"],  np.float64)
    rho       = np.asarray(metric_np["rho"],     np.float64)
    rhow      = np.asarray(metric_np["rhow"],    np.float64)
    dz        = np.asarray(metric_np["dz"],      np.float64)
    adzw      = np.asarray(metric_np.get("adzw", np.ones(nz + 1)), np.float64)
    dz_ref_val = float(metric_np.get("dz_ref", dz[0]))
    dlon_rad  = float(metric_np["dlon_rad"])
    nx_grid   = int(metric_np["nx"])

    sin2            = np.sin(np.pi * m / nx_grid) ** 2
    # Cartesian x-eigenvalue × cos²(lat) per row — matches gSAM pressure_gmg.f90:
    #   alpha = (2 - 2*cos(2πm/nx)) / dx²   (Cartesian, then scaled by cos²(lat_j))
    alpha_cartesian = -4.0 * sin2 / (dlon_rad * EARTH_RADIUS) ** 2       # (scalar)
    alpha_m         = alpha_cartesian * cos_lat_c ** 2                     # (ny,)

    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])   # (ny-1,)
    inv_row  = 1.0 / (dy_row * cos_lat_c)           # (ny,)

    c_lo        = np.zeros(ny)
    c_hi        = np.zeros(ny)
    c_lo[1:]    = cos_v[1:-1]  / dy_v_int * inv_row[1:]
    c_hi[:-1]   = cos_v[1:-1] / dy_v_int * inv_row[:-1]

    inv_rho_dz  = 1.0 / (rho * dz)                  # (nz,)
    d_lo        = np.zeros(nz)
    d_hi        = np.zeros(nz)
    d_lo[1:]    = rhow[1:-1]  / (adzw[1:-1]  * dz_ref_val) * inv_rho_dz[1:]
    d_hi[:-1]   = rhow[1:-1]  / (adzw[1:-1]  * dz_ref_val) * inv_rho_dz[:-1]

    # Build COO arrays fully vectorised over (j, k)
    j_idx = np.arange(ny)   # (ny,)
    k_idx = np.arange(nz)   # (nz,)
    jj, kk = np.meshgrid(j_idx, k_idx, indexing='ij')   # (ny, nz) each
    row_all = (jj * nz + kk).ravel()                     # (ny*nz,)

    diag_j = alpha_m - (c_lo + c_hi)   # (ny,)
    diag_k = -(d_lo + d_hi)            # (nz,)
    diag_vals = (diag_j[:, None] + diag_k[None, :]).ravel()  # (ny*nz,)

    r_list = [row_all]
    c_list = [row_all]
    v_list = [diag_vals]

    # ±k neighbours (vertical)
    mask_lo = kk.ravel() > 0
    r_list.append(row_all[mask_lo]);  c_list.append(row_all[mask_lo] - 1)
    v_list.append(d_lo[kk.ravel()[mask_lo]])

    mask_hi = kk.ravel() < nz - 1
    r_list.append(row_all[mask_hi]);  c_list.append(row_all[mask_hi] + 1)
    v_list.append(d_hi[kk.ravel()[mask_hi]])

    # ±j neighbours (meridional)
    mask_jlo = jj.ravel() > 0
    r_list.append(row_all[mask_jlo]);  c_list.append(row_all[mask_jlo] - nz)
    v_list.append(c_lo[jj.ravel()[mask_jlo]])

    mask_jhi = jj.ravel() < ny - 1
    r_list.append(row_all[mask_jhi]);  c_list.append(row_all[mask_jhi] + nz)
    v_list.append(c_hi[jj.ravel()[mask_jhi]])

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
    import hashlib, pickle, os, time
    from pathlib import Path

    key = (ny, nz, nm)
    if key in _SOLVER_CACHE:
        return _SOLVER_CACHE[key]

    n_workers = min(nm, int(os.environ.get("JSAM_LU_WORKERS", os.cpu_count() or 4)))

    # ── Disk cache: keyed on (ny, nz, nm) + a hash of the grid metric arrays ──
    cache_dir  = os.environ.get("JSAM_LU_CACHE_DIR", "/glade/work/sabramian/jsam_lu_cache")
    _hash_data = b"".join(
        np.asarray(metric_np[k], dtype=np.float64).tobytes()
        for k in ("dy_lat", "cos_lat", "cos_v", "rho", "rhow", "dz")
        if k in metric_np
    )
    _hash_data += f"{ny}_{nz}_{nm}_v2_cartesian_alpha".encode()  # v2: Cartesian×cos² alpha
    cache_key  = hashlib.md5(_hash_data).hexdigest()
    cache_file = Path(cache_dir) / f"lu_{cache_key}.pkl"

    if False and cache_file.exists():  # cache load disabled
        print(f"  [pressure] Loading LU cache ({nm} modes)...", flush=True)
        t0 = time.time()
        with open(cache_file, 'rb') as f:
            solvers = pickle.load(f)
        print(f"  [pressure] LU cache loaded in {time.time()-t0:.1f}s", flush=True)
        _SOLVER_CACHE[key] = solvers
        return solvers

    print(f"  [pressure] Building H_m matrices ({nm} modes × {ny}×{nz})...", flush=True)
    def _build_mode(m):
        Hm = _build_Hm_sparse(m, metric_np, ny, nz)
        if m == 0:
            Hm = Hm - sp.eye(ny * nz, format="csc") * 1e-10
        return Hm
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        hm_list = list(pool.map(_build_mode, range(nm)))

    print(f"  [pressure] Factorising {nm} H_m matrices in parallel...", flush=True)
    def _factorize(Hm):
        return _LUSolver(spla.splu(Hm))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        solvers = list(pool.map(_factorize, hm_list))
    del hm_list

    print("  [pressure] Factorisations ready.", flush=True)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(solvers, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [pressure] LU factors cached → {cache_file}", flush=True)
    except Exception as e:
        print(f"  [pressure] LU cache write failed: {e}", flush=True)

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
      - RHS × cos²(lat) per mode (matching gSAM: ff = rhs * mu_gl²)
      - Sparse LU in (y,z) per mode with Cartesian α × cos²(lat_j) diagonal
      - irfft in x → physical pressure
    """
    return _solve_pressure_spherical(rhs, metric, **kwargs)


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
# 5.  Pressure gradient application  (matches press_grad.f90)
# ---------------------------------------------------------------------------

@jax.jit
def apply_pressure_gradient(
    U: jax.Array,    # (nz, ny, nx+1)
    V: jax.Array,    # (nz, ny+1, nx)
    W: jax.Array,    # (nz+1, ny, nx)
    p: jax.Array,    # (nz, ny, nx)
    metric: dict,
    dt: float,
    igam2: float = 1.0,   # 1/gamma_RAVE² (=1 for incompressible)
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Subtract spherical pressure gradient from tentative velocities.

    Consistent with the spherical press_rhs and the spherical Helmholtz
    solver (FFT-x + sparse-LU in (y,z)):

      U -= dt * imu(j) * (p(i) - p(i-1)) / dx
      V -= dt * (p(j) - p(j-1)) / dy_v
      W -= dt * igam2 * (p(k) - p(k-1)) / (dz_ref * adzw(k))
    """
    dx     = metric["dx_lon"]
    dy_row = metric["dy_lat"]   # (ny,) per-row mass-cell dy
    imu    = metric["imu"]      # (ny,)  = 1/cos(lat)

    # ── Zonal: imu * dp/dλ at east faces  (spherical) ────────────────────────
    p_west = jnp.roll(p, 1, axis=2)
    dp_dx  = imu[None, :, None] * (p - p_west) / dx             # (nz, ny, nx)
    U_new = U.at[:, :, 1:-1].add(-dt * dp_dx[:, :, 1:])
    U_new = U_new.at[:, :, 0].add(-dt * dp_dx[:, :, 0])

    # ── Meridional: dp/dφ at interior v-faces.
    # Non-uniform: dy_v_face[j] = 0.5*(dy_row[j-1]+dy_row[j])
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])                 # (ny-1,)
    dp_dy_int = (p[:, 1:, :] - p[:, :-1, :]) / dy_v_int[None, :, None]  # (nz, ny-1, nx)
    V_new = V.at[:, 1:-1, :].add(-dt * dp_dy_int)

    # ── Vertical: dp/dz at w-faces — use center-to-center spacing adzw*dz_ref
    # Matches gSAM press_grad.f90: rdz = 1./(dz*adzw(k)) where dz=dz_ref.
    # Must be consistent with _build_Lz_matrix and _helmholtz_op which also
    # use adzw (D8 fix).
    dz_ref = metric["dz_ref"]                                     # scalar
    adzw   = metric["adzw"]                                       # (nz+1,)
    dz_face = dz_ref * adzw[1:-1]                                 # (nz-1,) center-to-center
    dp_dz_int = (p[1:, :, :] - p[:-1, :, :]) / dz_face[:, None, None]  # (nz-1, ny, nx)
    W_new = W.at[1:-1, :, :].add(-dt * igam2 * dp_dz_int)

    return U_new, V_new, W_new


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
) -> tuple["ModelState", jax.Array]:
    """
    Full pressure correction step with Richardson iteration.

      1. Compute anelastic divergence (RHS)
      2. Solve Poisson equation for p'
      3. Subtract ∇p' from (U, V, W)
      4. Repeat 1–3 on the residual for ``n_iter`` total iterations

    The effective time step for the pressure gradient is ``at * dt``, matching
    gSAM press_grad.f90 which applies ``u -= dt3(na) * at * ∇p``.

    PRESSURE HISTORY MANAGEMENT (differs from gSAM):
      - gSAM: Manages pressure in rotating buffer p(:,:,:,na/nb/nc)
      - JAX: Returns p_prev and p_pprev unchanged; actual pressure as separate p_total
      - Caller must manage pressure history update outside this function
    """
    U_cur, V_cur, W_cur = state.U, state.V, state.W
    p_total = jnp.zeros((state.TABS.shape[0], state.TABS.shape[1],
                          state.TABS.shape[2]))

    dt_eff = at * dt
    for _iter in range(n_iter):
        rhs = press_rhs(U_cur, V_cur, W_cur, metric, dt_eff)
        p_inc = solve_pressure(rhs, metric)
        print(f"  [pressure diag] iter={_iter}"
              f" W_in=[{float(jnp.min(W_cur)):.3e},{float(jnp.max(W_cur)):.3e}]"
              f" rhs_max={float(jnp.max(jnp.abs(rhs))):.3e}"
              f" p_max={float(jnp.max(jnp.abs(p_inc))):.3e}", flush=True)
        p_total = p_total + p_inc
        U_cur, V_cur, W_cur = apply_pressure_gradient(
            U_cur, V_cur, W_cur, p_inc, metric, dt_eff,
        )
        U_cur = U_cur.at[:, :, -1].set(U_cur[:, :, 0])
        print(f"  [pressure diag] iter={_iter}"
              f" W_out=[{float(jnp.min(W_cur)):.3e},{float(jnp.max(W_cur)):.3e}]", flush=True)

    # gSAM pressure.f90:83-84: clamp pressure perturbation to ±15% of reference
    pres_ref = metric["pres"][:, None, None]   # (nz, 1, 1)
    p_total = jnp.clip(p_total, -0.15 * pres_ref, 0.15 * pres_ref)

    new_state = ModelState(
        U=U_cur, V=V_cur, W=W_cur,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, p_total
