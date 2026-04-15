"""
Pressure solver for the anelastic equations on a lat-lon grid.

Solves the Poisson equation:
    ∇²p' = RHS

where RHS = (1/dt) * ∇·(ρ u*)  / ρ   (anelastic divergence of tentative velocity)

Algorithm (matching gSAM global solver):
  1. Compute RHS via press_rhs  [matches press_rhs.f90 exactly]
  2. rfft in longitude (periodic) → nm independent 2-D problems in (lat, z)
  3. For each zonal wavenumber m: solve H_m p_m = b_m via BiCGSTAB where
     H_m uses the spherical Helmholtz operator (cosφ-weighted L_y + L_z)
     This matches gSAM's global GMG solver which also uses the spherical metric.
  4. irfft back to physical space
  5. Apply pressure gradient to correct velocity  [matches press_grad.f90]

References:
  gSAM SRC/press_rhs.f90    — RHS computation
  gSAM SRC/pressure_big.f90 — FFT + cosine + Thomas solver (we replicate this)
  gSAM SRC/press_grad.f90   — pressure gradient application
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jsam.core.state import ModelState
from jsam.core.grid.latlon import LatLonGrid, EARTH_RADIUS
from jsam.core.physics.microphysics import G_GRAV, CP

# ---------------------------------------------------------------------------
# Grid precomputation helpers (called once, results passed to JIT functions)
# ---------------------------------------------------------------------------

def build_metric(grid: LatLonGrid, polar_filter: bool = False) -> dict:
    """
    Precompute all metric factors needed by press_rhs and press_grad.
    Returns a dict of JAX arrays (can be treated as static/constant).

    Parameters
    ----------
    grid         : LatLonGrid
    polar_filter : IGNORED — kept only for backwards-compatibility with
                   older callers.  gSAM has no spectral polar filter; the
                   only pole conditioning is the implicit CFL-based pole
                   damping in ``damping.f90`` (already folded into
                   ``diffuse_damping_mom_z``).  Any value passed here is
                   silently dropped and no ``pfmask_u`` / ``pfmask_v``
                   entries are added to the metric dict.

    Matches gSAM notation:
      imu(j)  = 1/cos(lat_j)                  [zonal metric inverse]
      muv(j)  = cos(lat at v north-face j)     [v-face cosine]
      rhow(k) = density at w-faces (level interfaces)
    """
    lat_rad = np.deg2rad(grid.lat)           # (ny,)
    lon_rad = np.deg2rad(grid.lon)           # (nx,)
    dlon_rad = np.deg2rad(grid.dlon)

    # cos(lat) at mass-point latitudes
    cos_lat = np.cos(lat_rad)                # (ny,)

    # Per-row meridional spacing (m), shape (ny,). On dyvar grids this
    # ranges 15–110 km; on a uniform grid it is a constant. gSAM's ady(j)
    # is the ratio dy_per_row[j] / dy_ref (see setgrid.f90:222-256).
    dy_per_row = np.array(grid.dy_per_row)        # (ny,)
    dy_ref     = float(grid.dy_ref)               # scalar (mid-latitude)
    ady        = np.array(grid.ady)               # (ny,) from LatLonGrid (gSAM setgrid.f90)

    # cos_v at v-faces.
    # Interior (1..ny-1): gSAM ady-weighted muv (setgrid.f90:255) —
    #   muv(j) = (ady(j-1)*mu(j) + ady(j)*mu(j-1)) / (ady(j-1)+ady(j))
    # Boundary (0, ny): take cos of the actual boundary v-face latitude.
    # For a global grid that v-face sits at ±90° so cos_v = 0; for a
    # limited-area grid it is extrapolated half a row outside the mass
    # points and cos_v is non-zero.
    muv_interior = (ady[:-1] * cos_lat[1:] + ady[1:] * cos_lat[:-1]) / (ady[:-1] + ady[1:])
    lat_v_boundary = np.deg2rad(grid.lat_v[[0, -1]])
    cos_v = np.concatenate([
        [np.cos(lat_v_boundary[0])],
        muv_interior,
        [np.cos(lat_v_boundary[1])],
    ])                                                      # (ny+1,)

    # Physical grid spacings
    dx_lon = EARTH_RADIUS * dlon_rad    # equatorial zonal spacing (m)

    # imu(j) = 1/cos(lat_j)  (zonal metric, matches gSAM)
    imu = 1.0 / np.clip(cos_lat, 1e-6, None)   # (ny,) guard against poles

    # Density at w-faces (vertical interfaces) — matches gSAM setdata.f90:473-484
    # dolatlon branch exactly (non-uniform dz needs the adz cross-weighted form):
    #
    #   do k=2,nzm
    #     rhow(k) = (rho(k-1)*adz(k) + rho(k)*adz(k-1)) / (adz(k)+adz(k-1))
    #   end do
    #   rhow(1)  = 2*rho(1)  - rhow(2)
    #   rhow(nz) = 2*rho(nzm)- rhow(nzm)
    #
    # 1-indexed gSAM → 0-indexed Python: interior face k_f=1..nz-1 straddles
    # mass cells k_f-1 (below) and k_f (above).
    rho = grid.rho                                                    # (nz,)
    dz  = grid.dz                                                     # (nz,)
    nz  = len(rho)
    # gSAM stretched-grid factors (setgrid.f90):
    #   dz_ref    = zi[1] - zi[0]              (scalar, bottom layer thickness)
    #   adz[k]    = dz[k] / dz_ref             (cell-thickness ratio, shape (nz,))
    #   adzw[0]   = 1
    #   adzw[k]   = (z[k]-z[k-1])/dz_ref       for k=1..nz-1
    #   adzw[nz]  = adzw[nz-1]                 (wraparound)
    zi   = np.asarray(grid.zi)                 # (nz+1,)
    z1d  = np.asarray(grid.z)                  # (nz,)
    dz_ref = float(zi[1] - zi[0])
    adz    = np.asarray(dz, dtype=np.float64) / dz_ref
    adzw   = np.empty(len(rho) + 1)
    adzw[0]  = 1.0
    adzw[1:-1] = (z1d[1:] - z1d[:-1]) / dz_ref
    adzw[-1] = adzw[-2]
    rhow = np.zeros(nz + 1)
    # interior: k_f = 1..nz-1  →  rho_below=rho[k_f-1], rho_above=rho[k_f]
    adz_below = adz[:-1]                                              # (nz-1,)  adz(k_f-1)
    adz_above = adz[1:]                                               # (nz-1,)  adz(k_f)
    rhow[1:-1] = (rho[:-1] * adz_above + rho[1:] * adz_below) \
                 / (adz_below + adz_above)
    # bottom / top linear extrapolation (gSAM lines 483-484)
    rhow[0]  = 2.0 * rho[0]    - rhow[1]
    rhow[-1] = 2.0 * rho[nz-1] - rhow[-2]

    # Hydrostatic base-state pressure (Pa) by integrating dp = -rho*g*dz from surface.
    # p_face[0] = p_surf (standard sea-level pressure).
    p_surf = 101325.0   # Pa
    p_face = np.zeros(len(rho) + 1)
    p_face[0] = p_surf
    for k in range(len(rho)):
        p_face[k + 1] = p_face[k] - rho[k] * G_GRAV * dz[k]
    pres = 0.5 * (p_face[:-1] + p_face[1:])   # cell-centre pressure (nz,) Pa

    # gamaz[k] = g * z[k] / cp  (liquid-ice static energy height term)
    gamaz = G_GRAV * grid.z / CP   # (nz,) K

    # Coriolis + metric arrays (gSAM coriolis.f90)
    OMEGA = 7.2921e-5    # rad/s  (gSAM params.f90)
    fcory  = 2.0 * OMEGA * np.sin(lat_rad)         # (ny,) f-parameter
    fcorzy = 2.0 * OMEGA * np.cos(lat_rad)         # (ny,) f' = 2Ω cosφ  (docoriolisz)
    tanr   = np.tan(lat_rad) / EARTH_RADIUS        # (ny,) tan(lat)/R  metric term

    m = {
        "imu":     jnp.array(imu),         # (ny,)
        "cos_v":   jnp.array(cos_v),       # (ny+1,) ady-weighted (gSAM muv)
        "dx_lon":  float(dx_lon),          # scalar (m)
        "dy_lat":     jnp.array(dy_per_row),   # (ny,) m   — per-row (Gap 8)
        "dy_lat_ref": float(dy_ref),           # scalar reference dy (gSAM `dy`)
        "ady":        jnp.array(ady),          # (ny,) ratio dy_per_row / dy_ref
        "rho":     jnp.array(rho),         # (nz,)
        "rhow":    jnp.array(rhow),        # (nz+1,)
        "dz":      jnp.array(dz),          # (nz,)
        "adz":    jnp.array(adz),      # (nz,) gSAM adz(k) = dz(k)/dz_ref
        "adzw":   jnp.array(adzw),     # (nz+1,) gSAM adzw(k) — normalized center spacing
        "dz_ref": float(dz_ref),       # scalar — gSAM dz (bottom-layer thickness)
        "cos_lat": jnp.array(cos_lat),     # (ny,)
        "lat_rad": jnp.array(lat_rad),     # (ny,) mass-cell latitudes (rad)
        "lon_rad": jnp.array(lon_rad),     # (nx,) mass-cell longitudes (rad)
        "dlon_rad": float(dlon_rad),
        "nx":       int(len(grid.lon)),    # actual zonal grid size (for FFT eigenvalues)
        "pres":    jnp.array(pres),        # (nz,) Pa  hydrostatic base-state pressure
        "gamaz":   jnp.array(gamaz),       # (nz,) K   g*z/cp  (static energy height term)
        "z":       jnp.array(grid.z),      # (nz,) m   cell-centre heights
        "fcory":   jnp.array(fcory),       # (ny,) 2*Omega*sin(lat)
        "fcorzy":  jnp.array(fcorzy),      # (ny,) 2*Omega*cos(lat)  [docoriolisz]
        "tanr":    jnp.array(tanr),        # (ny,) tan(lat)/R  metric term
    }

    # gSAM has no spectral polar filter — the `polar_filter` kwarg above
    # is accepted for API compatibility but never emits pfmask_u/pfmask_v.
    return m


# ---------------------------------------------------------------------------
# 1.  RHS — anelastic divergence  (matches press_rhs.f90)
# ---------------------------------------------------------------------------

@jax.jit
def press_rhs(
    U: jax.Array,   # (nz, ny, nx+1)  zonal velocity at east faces
    V: jax.Array,   # (nz, ny+1, nx)  meridional velocity at north faces
    W: jax.Array,   # (nz+1, ny, nx)  vertical velocity at top faces
    metric: dict,
    dt: float,
) -> jax.Array:     # (nz, ny, nx)
    """
    RHS of the pressure Poisson equation — spherical anelastic divergence.

    Consistent with the spherical Helmholtz solver and spherical gradient:

      RHS(i,j,k) = (1/dt) * [
          imu(j) * (U(i+1) - U(i)) / dx                          [spherical]
        + (cos_v(j+1)*V(j+1) - cos_v(j)*V(j)) / (dy * cos(j))   [spherical]
        + (rhow(k+1)*W(k+1) - rhow(k)*W(k)) / (rho(k)*dz(k))   [vertical]
      ]

    where imu(j) = 1/cos(lat_j), cos_v(j) = cos(lat_v_face_j).
    """
    dx      = metric["dx_lon"]
    dy      = metric["dy_lat"]    # (ny,) per-row
    rho     = metric["rho"]       # (nz,)
    rhow    = metric["rhow"]      # (nz+1,)
    dz      = metric["dz"]        # (nz,)
    imu     = metric["imu"]       # (ny,)   = 1/cos(lat)
    cos_v   = metric["cos_v"]     # (ny+1,)
    cos_lat = metric["cos_lat"]   # (ny,)

    rho3   = rho[:, None, None]
    rhow3  = rhow[:, None, None]
    dz3    = dz[:, None, None]
    dy3    = dy[None, :, None]    # (1, ny, 1)

    # Spherical zonal divergence: (1/(R cosφ)) d(U)/dλ
    div_u = imu[None, :, None] * (U[:, :, 1:] - U[:, :, :-1]) / dx

    # Spherical meridional divergence: (1/(R cosφ)) d(cosφ V)/dφ
    div_v = (cos_v[None, 1:, None] * V[:, 1:, :]
           - cos_v[None, :-1, None] * V[:, :-1, :]) / (dy3 * cos_lat[None, :, None])

    # Anelastic vertical divergence (same as Cartesian — no metric)
    div_w = (rhow3[1:] * W[1:] - rhow3[:-1] * W[:-1]) / (rho3 * dz3)

    return (div_u + div_v + div_w) / dt


# ---------------------------------------------------------------------------
# 2.  Helmholtz operator for one zonal wavenumber m   (used inside PCG)
# ---------------------------------------------------------------------------

def _helmholtz_op(
    P_flat: jax.Array,     # (ny*nz,)  real or complex
    m: int,
    metric: dict,
    ny: int,
    nz: int,
) -> jax.Array:            # (ny*nz,)
    """
    Apply the Helmholtz operator H_m to P for zonal wavenumber m:

      H_m[P](j,k) = α_m(j) * P(j,k)    [spectral zonal eigenvalue]
                  + L_y[P](j,k)          [meridional Laplacian with cos(lat)]
                  + L_z[P](j,k)          [vertical Laplacian with density]

    Boundary conditions (Neumann everywhere → zero normal gradient):
      L_y: dP/dlat = 0 at south and north poles (j=0 and j=ny-1)
      L_z: dP/dz   = 0 at bottom (k=0) and top (k=nz-1)

    Note: H_m is negative (semi-)definite. The solver uses the negated system.
    """
    P = P_flat.reshape(ny, nz)

    cos_lat  = metric["cos_lat"]   # (ny,)
    dy_row   = metric["dy_lat"]    # (ny,) per-row mass-cell spacing
    rho      = metric["rho"]       # (nz,)
    rhow     = metric["rhow"]      # (nz+1,)
    dz       = metric["dz"]        # (nz,)
    dlon_rad = metric["dlon_rad"]

    R = EARTH_RADIUS

    # ── Spectral zonal eigenvalue α_m(j) ─────────────────────────────────────
    # FD eigenvalue: -4 sin²(π m / nx) / (dλ * R cos φ)²
    # Use metric["nx"] (actual grid size) so this is correct for both global
    # and limited-area domains.  (For global grids nx ≈ 2π/dlon_rad anyway.)
    nx_grid = metric["nx"]
    sin2 = jnp.sin(jnp.pi * m / nx_grid) ** 2
    alpha_m = -4.0 * sin2 / (dlon_rad * R * cos_lat) ** 2   # (ny,)   [m^-2]

    zonal = alpha_m[:, None] * P   # (ny, nz)

    # ── Meridional Laplacian: (1/(R²cos φ)) d/dφ(cos φ dP/dφ) ──────────────
    # Non-uniform form (gSAM-compatible):
    #   flux_p[j_face] = cos_v[j_face] * (P[j] - P[j-1]) / dy_v[j_face]
    #   L_y P[j] = (flux_p[j+1] - flux_p[j]) / (dy_row[j] * cos_lat[j])
    # where dy_v[j_face] = 0.5*(dy_row[j-1]+dy_row[j]) is the distance
    # between mass points j-1 and j.  On a uniform grid dy_v = dy_row and
    # this reduces to the 1/dy² form.
    cos_v = metric["cos_v"]                                  # (ny+1,)
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])              # (ny-1,)

    dP_north = P[1:, :] - P[:-1, :]                          # (ny-1, nz)
    flux_n = cos_v[1:-1, None] * dP_north / dy_v_int[:, None]

    # Neumann BC: zero flux at south (j=0) and north (j=ny-1) faces
    flux_south = jnp.zeros((1, nz))
    flux_north = jnp.zeros((1, nz))
    flux = jnp.concatenate([flux_south, flux_n, flux_north], axis=0)  # (ny+1, nz)

    L_y = (flux[1:, :] - flux[:-1, :]) / (dy_row[:, None] * cos_lat[:, None])  # (ny, nz)

    # ── Vertical Laplacian: (1/ρ) d/dz(ρ_w dP/dz) ───────────────────────────
    dPz = P[:, 1:] - P[:, :-1]                           # (ny, nz-1) at z-faces 1..nz-1
    flux_z_int = rhow[1:-1, None].T * dPz / dz[:-1]      # (ny, nz-1)  interior faces

    # Neumann at bottom and top: zero flux
    flux_z_bot = jnp.zeros((ny, 1))
    flux_z_top = jnp.zeros((ny, 1))
    flux_z = jnp.concatenate([flux_z_bot, flux_z_int, flux_z_top], axis=1)  # (ny, nz+1)

    L_z = (flux_z[:, 1:] - flux_z[:, :-1]) / (dz[None, :] * rho[None, :])  # (ny, nz)

    result = zonal + L_y + L_z
    return result.ravel()


def _build_Lz_matrix(metric: dict, nz: int) -> jax.Array:
    """
    Build the nz×nz tridiagonal vertical Laplacian matrix (same for all latitudes).

    L_z[k, k'] is the (1/ρ) d/dz(ρ_w dP/dz) operator with Neumann BCs.
    This matrix is negative semi-definite.
    """
    rho  = metric["rho"]    # (nz,) JAX array
    rhow = metric["rhow"]   # (nz+1,)
    dz   = metric["dz"]     # (nz,)

    # Convert to numpy for matrix construction
    rho_np  = np.array(rho)
    rhow_np = np.array(rhow)
    dz_np   = np.array(dz)

    L = np.zeros((nz, nz))
    for k in range(nz):
        coeff = 1.0 / (rho_np[k] * dz_np[k])
        if k > 0:
            c_lo = rhow_np[k] / dz_np[k - 1]   # flux at bottom face of k
            # Actually: flux[k] = rhow[k] * (P[k]-P[k-1]) / dz[k-1] is NOT right
            # We use: flux[l] = rhow[l] * (P[l] - P[l-1]) / dz[l-1] for l=1..nz-1
            # But in _helmholtz_op we use: flux_z_int = rhow[1:-1] * dPz / dz[:-1]
            # dPz[l] = P[l+1] - P[l] for l=0..nz-2
            # So flux_z_int[l] = rhow[l+1] * (P[l+1]-P[l]) / dz[l]  (face between l and l+1)
            # L_z[k] = (flux_z_int[k] - flux_z_int[k-1]) / (rho[k] * dz[k])
            # flux_z_int[k-1] = rhow[k] * (P[k]-P[k-1]) / dz[k-1]
            c_lo = rhow_np[k] / dz_np[k - 1]
            L[k, k - 1] += c_lo * coeff
            L[k, k]     -= c_lo * coeff
        if k < nz - 1:
            # flux_z_int[k] = rhow[k+1] * (P[k+1]-P[k]) / dz[k]
            c_hi = rhow_np[k + 1] / dz_np[k]
            L[k, k + 1] += c_hi * coeff
            L[k, k]     -= c_hi * coeff

    return jnp.array(L)   # (nz, nz)


def _compute_alpha_m(m: int, metric: dict) -> jax.Array:
    """Return the zonal eigenvalue α_m(j) for each latitude j."""
    cos_lat  = metric["cos_lat"]   # (ny,) JAX
    dlon_rad = metric["dlon_rad"]
    nx_grid  = metric["nx"]        # actual number of zonal grid points
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
    cos_lat = np.array(metric["cos_lat"])            # (ny,)
    cos_v   = np.array(metric["cos_v"])              # (ny+1,)
    dy_row  = np.array(metric["dy_lat"])             # (ny,) per-row

    # dy_v[j_face] for j_face = 1..ny-1 (interior v-faces)
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])      # (ny-1,)

    L = np.zeros((ny, ny))
    for j in range(ny):
        inv_row = 1.0 / (dy_row[j] * cos_lat[j])
        if j > 0:
            # flux_p[j]  contributes  -cos_v[j]*(P[j]-P[j-1])/dy_v[j] * inv_row
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
    L_z_np = np.array(_build_Lz_matrix(metric, nz))   # (nz, nz)
    L_y_np = _build_Ly_matrix(metric, ny)              # (ny, ny)
    alpha  = np.array(_compute_alpha_m(m, metric))     # (ny,)

    I_ny = np.eye(ny)
    I_nz = np.eye(nz)

    return (np.kron(np.diag(alpha), I_nz)   # α_m(j) * I_nz per latitude
          + np.kron(L_y_np, I_nz)           # meridional Laplacian
          + np.kron(I_ny, L_z_np))          # vertical Laplacian


def _make_vertical_precond(m: int, metric: dict, ny: int, nz: int):
    """
    Build and return a function that applies the inverse of the vertical
    preconditioner M_j = -(alpha_m(j)*I + L_z) for each latitude j.

    M_j is positive definite for m > 0 (alpha_m < 0, L_z neg-semidef → -M_j pos-def).
    The preconditioner is applied to the NEGATED system (-H_m * p = -b).

    Returns:
        precond(r) — applies M^{-1} r, where r has shape (ny*nz,)
    """
    L_z = _build_Lz_matrix(metric, nz)            # (nz, nz)
    alpha = _compute_alpha_m(m, metric)            # (ny,) negative values

    # M[j] = -(alpha_m[j]*I + L_z) = positive definite nz×nz matrix
    # Build (ny, nz, nz) batch of M matrices
    I_nz = jnp.eye(nz)
    M_batch = -(alpha[:, None, None] * I_nz[None, :, :] + L_z[None, :, :])  # (ny, nz, nz)

    # Precompute M^{-1} for each latitude via vmap
    M_inv = jax.vmap(jnp.linalg.inv)(M_batch)    # (ny, nz, nz)

    def precond(r: jax.Array) -> jax.Array:
        # r: (ny*nz,) → (ny, nz)
        R = r.reshape(ny, nz)
        # Apply M^{-1}[j] to R[j] for each j
        Q = jax.vmap(lambda Mi, ri: Mi @ ri)(M_inv, R)   # (ny, nz)
        return Q.ravel()

    return precond


# ---------------------------------------------------------------------------
# 3.  Preconditioned CG (PCG) solver  (Python for-loop, eager execution)
# ---------------------------------------------------------------------------

def _pcg_solve(
    op,                      # callable: (n,) → (n,), positive definite
    b: jax.Array,            # (n,) RHS
    precond,                 # callable: (n,) → (n,), applies M^{-1}
    tol: float = 1e-5,
    maxiter: int = 200,
) -> jax.Array:              # (n,)
    """
    Preconditioned Conjugate Gradient solver (Python for-loop, eager).

    Avoids jax.lax.while_loop to prevent per-mode JIT compilation overhead
    (33 modes × compile time ≈ several minutes). Eager PCG is fast for the
    small (ny×nz) systems we solve here.

    Solves A * x = b where A is symmetric positive definite, M^{-1} ≈ A^{-1}.
    Stops when ||r||₂ < tol * ||b||₂  or after maxiter iterations.
    """
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

    cos_lat  = metric_np["cos_lat"]                    # (ny,)
    cos_lat_c = np.maximum(cos_lat, 1e-6)
    cos_v    = metric_np["cos_v"]                      # (ny+1,)
    dy_row   = np.asarray(metric_np["dy_lat"], dtype=np.float64)   # (ny,) per-row
    rho      = metric_np["rho"]                        # (nz,)
    rhow     = metric_np["rhow"]                       # (nz+1,)
    dz       = metric_np["dz"]                         # (nz,)
    dlon_rad = float(metric_np["dlon_rad"])
    nx_grid  = int(metric_np["nx"])

    R = EARTH_RADIUS
    sin2    = np.sin(np.pi * m / nx_grid) ** 2
    alpha_m = -4.0 * sin2 / (dlon_rad * R * cos_lat_c) ** 2   # (ny,)

    # Non-uniform L_y coefficients (variable meridional spacing, gSAM dyvar).
    #   flux_p[j_face] = cos_v[j_face] * (P[j] - P[j-1]) / dy_v[j_face]
    #   L_y[j]         = (flux_p[j+1] - flux_p[j]) / (dy_row[j] * cos_lat[j])
    # → L_y[j, j-1] =  cos_v[j]   / (dy_v[j]   * dy_row[j] * cos_lat[j])
    #   L_y[j, j+1] =  cos_v[j+1] / (dy_v[j+1] * dy_row[j] * cos_lat[j])
    #   L_y[j, j]   = -(L_y[j, j-1] + L_y[j, j+1])
    # where dy_v[j_face] = 0.5*(dy_row[j-1]+dy_row[j]).  On uniform grids
    # dy_v == dy_row and this reduces to the old 1/dy² form exactly.
    dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])        # (ny-1,)

    c_lo = np.zeros(ny)   # coefficient multiplying P[j-1,:]
    c_hi = np.zeros(ny)   # coefficient multiplying P[j+1,:]
    for j in range(ny):
        inv_row = 1.0 / (dy_row[j] * cos_lat_c[j])
        if j > 0:
            c_lo[j] = cos_v[j] / dy_v_int[j - 1] * inv_row
        if j < ny - 1:
            c_hi[j] = cos_v[j + 1] / dy_v_int[j] * inv_row

    # Precompute L_z coefficients for each k (same for all j)
    # L_z[k, k-1] = d_lo[k],  L_z[k, k+1] = d_hi[k],  L_z[k, k] = -d_lo[k]-d_hi[k]
    d_lo = np.zeros(nz)   # coefficient multiplying P[:,k-1]
    d_hi = np.zeros(nz)   # coefficient multiplying P[:,k+1]
    for k in range(nz):
        c = 1.0 / (rho[k] * dz[k])
        if k > 0:
            d_lo[k] = rhow[k] / (dz[k - 1]) * c
        if k < nz - 1:
            d_hi[k] = rhow[k + 1] / dz[k] * c

    # Build COO arrays
    n = ny * nz
    rows, cols, vals = [], [], []

    for j in range(ny):
        diag_j = alpha_m[j] - (c_lo[j] + c_hi[j])   # zonal + L_y diagonal
        for k in range(nz):
            row = j * nz + k
            diag_k = -(d_lo[k] + d_hi[k])             # L_z diagonal
            # Diagonal
            rows.append(row); cols.append(row)
            vals.append(diag_j + diag_k)
            # Vertical neighbours
            if k > 0:
                rows.append(row); cols.append(row - 1); vals.append(d_lo[k])
            if k < nz - 1:
                rows.append(row); cols.append(row + 1); vals.append(d_hi[k])
            # Meridional neighbours
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
      V -= dt * (p(j) - p(j-1)) / dy
      W -= dt * igam2 * (p(k) - p(k-1)) / dz(k)
    """
    dx   = metric["dx_lon"]
    dy_row = metric["dy_lat"]   # (ny,) per-row mass-cell dy
    dz   = metric["dz"]      # (nz,)
    imu  = metric["imu"]     # (ny,)  = 1/cos(lat)

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

    # ── Vertical: dp/dz at top faces  (same as Cartesian) ────────────────────
    dz3 = dz[:, None, None]
    dp_dz_int = (p[1:, :, :] - p[:-1, :, :]) / dz3[:-1]        # (nz-1, ny, nx)
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
