"""
Pressure solver for the anelastic equations on a lat-lon grid.
Solves ∇²p' = RHS via rfft-x + sparse-LU in (y,z) per zonal mode.
Matches gSAM pressure_big.f90 + press_rhs.f90 + press_grad.f90.
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
        "gamaz": jnp.array(gamaz),
        "z": jnp.array(grid.z),
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
    tol: float = 1e-5,
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
    """
    import scipy.sparse as sp

    cos_lat  = metric_np["cos_lat"]
    cos_lat_c = np.maximum(cos_lat, 1e-6)
    cos_v    = metric_np["cos_v"]
    dy_row   = np.asarray(metric_np["dy_lat"], dtype=np.float64)
    rho      = metric_np["rho"]
    rhow     = metric_np["rhow"]
    dz       = metric_np["dz"]
    dlon_rad = float(metric_np["dlon_rad"])
    nx_grid  = int(metric_np["nx"])

    R = EARTH_RADIUS
    sin2    = np.sin(np.pi * m / nx_grid) ** 2
    alpha_m = -4.0 * sin2 / (dlon_rad * R * cos_lat_c) ** 2   # (ny,)

    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])

    c_lo = np.zeros(ny)   # coefficient multiplying P[j-1,:]
    c_hi = np.zeros(ny)   # coefficient multiplying P[j+1,:]
    for j in range(ny):
        inv_row = 1.0 / (dy_row[j] * cos_lat_c[j])
        if j > 0:
            c_lo[j] = cos_v[j] / dy_v_int[j - 1] * inv_row
        if j < ny - 1:
            c_hi[j] = cos_v[j + 1] / dy_v_int[j] * inv_row

    adzw = np.asarray(metric_np.get("adzw", np.ones(nz + 1)), dtype=np.float64)
    dz_ref_val = float(metric_np.get("dz_ref", dz[0]))
    d_lo = np.zeros(nz)
    d_hi = np.zeros(nz)
    for k in range(nz):
        c = 1.0 / (rho[k] * dz[k])
        if k > 0:
            d_lo[k] = rhow[k] / (adzw[k] * dz_ref_val) * c
        if k < nz - 1:
            d_hi[k] = rhow[k + 1] / (adzw[k + 1] * dz_ref_val) * c

    n = ny * nz
    rows, cols, vals = [], [], []
    for j in range(ny):
        diag_j = alpha_m[j] - (c_lo[j] + c_hi[j])
        for k in range(nz):
            row = j * nz + k
            diag_k = -(d_lo[k] + d_hi[k])
            rows.append(row); cols.append(row)
            vals.append(diag_j + diag_k)
            if k > 0:
                rows.append(row); cols.append(row - 1); vals.append(d_lo[k])
            if k < nz - 1:
                rows.append(row); cols.append(row + 1); vals.append(d_hi[k])
            if j > 0:
                rows.append(row); cols.append(row - nz); vals.append(c_lo[j])
            if j < ny - 1:
                rows.append(row); cols.append(row + nz); vals.append(c_hi[j])

    H = sp.csc_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    return H


# Module-level cache: keyed by (ny, nz, nm).  Valid for one grid per session.
_SOLVER_CACHE: dict = {}


def _get_sparse_solvers(metric_np: dict, ny: int, nz: int, nm: int):
    """
    Build (and cache) sparse LU factorizations of H_m for all zonal modes.

    Returns a list of nm callables: solve_m(b) → x such that H_m x = b.
    The factorization is precomputed once; each solve is O(ny*nz*bandwidth).
    """
    import scipy.sparse.linalg as spla

    key = (ny, nz, nm)
    if key not in _SOLVER_CACHE:
        print(f"  [pressure] Factorising spherical Helmholtz operators "
              f"({nm} modes × {ny}×{nz})...", flush=True)
        import scipy.sparse as sp
        solvers = []
        for m in range(nm):
            Hm = _build_Hm_sparse(m, metric_np, ny, nz)
            if m == 0:
                # H_0 = L_y + L_z has a true 1-D null space (constant field).
                # Negative shift preserves the sign (all eigenvalues ≤ 0)
                # while making the null-space eigenvalue strictly negative.
                Hm = Hm - sp.eye(ny * nz, format="csc") * 1e-10
            # m >= 1: H_m is strictly negative definite — SuperLU handles
            # any conditioning fine without regularisation.  Earlier code
            # subtracted 1e-7*I here, but that shift is comparable to (or
            # larger than) the matrix scale on small/coarse grids and
            # ruins the round-trip residual (4.78e-5 vs 2.7e-10 with no
            # shift on the LAM-equator test fixture).
            lu = spla.splu(Hm)
            solvers.append(lu)
        _SOLVER_CACHE[key] = solvers
        print("  [pressure] Factorisations ready.", flush=True)
    return _SOLVER_CACHE[key]


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

    Matches gSAM pressure_big.f90:
      - Cartesian eigenvalues (same approximation gSAM uses globally):
          λ_x[m] = (2*cos(2πm/nx) − 2) / dx²
          λ_y[k] = (2*cos(πk/ny)  − 2) / dy²
      - DCT-II in y handles Neumann (wall) BCs at south/north edges
      - Thomas tridiagonal solve in z for each (m, k) mode pair
      - (m=0, k=0) singularity: zero RHS + diagonal shift of −1 → p = 0

    Why Cartesian and not spherical:
      DCT-II diagonalises only constant-coefficient operators.  The spherical
      y-operator (1/R²cosφ) d/dφ(cosφ dP/dφ) has variable coefficients and
      cannot be diagonalised by DCT-II without an expensive eigenvalue solve.
      The Cartesian approximation is exact near the equator and is the same
      approximation gSAM uses for all global runs; the polar filter compensates
      at high latitudes.

    Numerical stability advantage over the 2D sparse-LU approach:
      For each (m,k) the Thomas system diagonal is L_z_diag[l] + λ_x + λ_y.
      Even for near-zero (m=0,k=1) or (m=1,k=0) eigenvalues (λ ≈ 4e-14 m⁻²),
      L_z_diag dominates (~8e-6 m⁻²) at every row, so Thomas never encounters
      a near-zero pivot.  The 2D LU approach sees the (y=1,z=0) barotropic
      eigenvector as a near-zero eigenvalue of the full matrix and amplifies it
      by ~10¹³.
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
    rho  = np.array(metric["rho"],  dtype=np.float64)   # (nz,)
    rhow = np.array(metric["rhow"], dtype=np.float64)   # (nz+1,)
    dz_  = np.array(metric["dz"],   dtype=np.float64)   # (nz,)

    # ── Eigenvalues ────────────────────────────────────────────────────────────
    m_arr = np.arange(nm, dtype=np.float64)
    lam_x = (2.0 * np.cos(2.0 * np.pi * m_arr / nx) - 2.0) / dx**2   # (nm,)

    k_arr = np.arange(ny, dtype=np.float64)
    lam_y = (2.0 * np.cos(np.pi * k_arr / ny) - 2.0) / dy**2          # (ny,)

    # ── Vertical tridiagonal coefficients (L_z, same for all (m,k)) ──────────
    a_sub = np.zeros(nz)   # sub-diagonal;   a_sub[0]   unused
    c_sup = np.zeros(nz)   # super-diagonal; c_sup[-1]  unused
    for l in range(nz):
        inv = 1.0 / (rho[l] * dz_[l])
        if l > 0:
            a_sub[l] = rhow[l]     / dz_[l - 1] * inv
        if l < nz - 1:
            c_sup[l] = rhow[l + 1] / dz_[l]     * inv
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
    Solve the spherical anelastic Poisson equation.

    Uses FFT in x + sparse LU in (y,z) per zonal mode — solves the true
    spherical Helmholtz operator, consistent with the spherical press_rhs
    and apply_pressure_gradient.
    """
    return _solve_pressure_spherical(rhs, metric, **kwargs)


def _solve_pressure_spherical(
    rhs: jax.Array,   # (nz, ny, nx)
    metric: dict,
    **kwargs,
) -> jax.Array:       # (nz, ny, nx)
    """
    Solve the spherical anelastic Poisson equation via:
      1. rfft in x  → nm complex modes
      2. For each zonal mode m: solve the (ny×nz) spherical Helmholtz system
             H_m p_m = rhs_m
         where H_m includes the true cos(lat) meridional operator and the
         spectral zonal eigenvalue α_m(j) = -4sin²(πm/nx)/(dλ R cosφ_j)².
         Uses cached sparse LU factorisation built once per grid.
      3. irfft in x  → physical pressure field

    Consistent with the spherical press_rhs (imu div) and spherical
    apply_pressure_gradient (imu grad).  For grids near the equator the result
    is numerically identical to the Cartesian DCT+Thomas solver.
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

    # ── 1. rfft in x ─────────────────────────────────────────────────────────
    rhs_x = np.fft.rfft(rhs_np, axis=2)   # (nz, ny, nm) complex

    p_hat = np.zeros_like(rhs_x)          # (nz, ny, nm) complex

    # ── 2. Sparse solve for each zonal mode m ────────────────────────────────
    for m, lu in enumerate(solvers):
        # Extract RHS for this mode: shape (nz, ny)
        rhs_m = rhs_x[:, :, m]   # (nz, ny)

        # H_m uses row = j*nz + k  (j outer, k inner).
        # Transpose (nz,ny) → (ny,nz), then flatten j-outer k-inner.
        rhs_real = np.ascontiguousarray(rhs_m.real.T).ravel()   # (ny*nz,)
        rhs_imag = np.ascontiguousarray(rhs_m.imag.T).ravel()   # (ny*nz,)

        # m=0: remove global mean from RHS to decouple the null-space component
        if m == 0:
            rhs_real -= rhs_real.mean()
            # imag part of m=0 is zero (rfft m=0 is always real)

        p_real = lu.solve(rhs_real)   # (ny*nz,)
        p_imag = lu.solve(rhs_imag)

        # Reshape back: (ny*nz,) j-outer k-inner → (ny,nz) → transpose → (nz,ny)
        p_m = (p_real + 1j * p_imag).reshape(ny, nz).T   # (nz, ny)
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
    gSAM press_grad.f90 which applies ``u -= dt3(na) * at * ∇p``.  The
    stored pressure is then the TRUE p (same scaling gSAM uses), so that
    adamsB's bt, ct coefficients apply cleanly.
    """
    U_cur, V_cur, W_cur = state.U, state.V, state.W
    p_total = jnp.zeros((state.TABS.shape[0], state.TABS.shape[1],
                          state.TABS.shape[2]))

    dt_eff = at * dt
    for _ in range(n_iter):
        rhs = press_rhs(U_cur, V_cur, W_cur, metric, dt_eff)
        p_inc = solve_pressure(rhs, metric)
        p_total = p_total + p_inc
        U_cur, V_cur, W_cur = apply_pressure_gradient(
            U_cur, V_cur, W_cur, p_inc, metric, dt_eff,
        )
        U_cur = U_cur.at[:, :, -1].set(U_cur[:, :, 0])

    new_state = ModelState(
        U=U_cur, V=V_cur, W=W_cur,
        TABS=state.TABS, QV=state.QV, QC=state.QC,
        QI=state.QI, QR=state.QR, QS=state.QS, QG=state.QG,
        TKE=state.TKE,
        p_prev=state.p_prev, p_pprev=state.p_pprev,
        nstep=state.nstep, time=state.time,
    )
    return new_state, p_total
