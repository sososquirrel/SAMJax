"""
Lat-lon Arakawa C-grid — Phase 1 grid for jsam.

Layout (same as gSAM global):
  - Scalar fields:   (nz, ny, nx)    at cell centres
  - U (zonal):       (nz, ny, nx+1)  at east faces
  - V (meridional):  (nz, ny+1, nx)  at north faces
  - W (vertical):    (nz+1, ny, nx)  at top faces

Horizontal indexing: [j, i] = [lat, lon], 0-based, periodic in i.
"""

from __future__ import annotations
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .base import Grid

# Earth radius
EARTH_RADIUS = 6371229.0   # m


@dataclass(frozen=True)
class LatLonGrid(Grid):
    """
    Regular lat-lon grid matching gSAM global layout.

    Parameters
    ----------
    lat : 1D array of cell-centre latitudes  (degrees, size ny)
    lon : 1D array of cell-centre longitudes (degrees, size nx)
    z   : 1D array of cell-centre heights    (m, size nz)
    zi  : 1D array of cell-face heights      (m, size nz+1)
    rho : 1D array of base-state density     (kg/m³, size nz)
    """
    lat:  np.ndarray   # (ny,)   degrees
    lon:  np.ndarray   # (nx,)   degrees
    z:    np.ndarray   # (nz,)   m
    zi:   np.ndarray   # (nz+1,) m
    rho:  np.ndarray   # (nz,)   kg/m³

    # ── Shape ────────────────────────────────────────────────────────────────
    @property
    def nx(self): return len(self.lon)

    @property
    def ny(self): return len(self.lat)

    @property
    def nz(self): return len(self.z)

    # ── Spacings ─────────────────────────────────────────────────────────────
    @property
    def dlon(self) -> float:
        """Longitude spacing in degrees."""
        return float(self.lon[1] - self.lon[0])

    @property
    def dlat(self) -> float:
        """Latitude spacing at the pole row (degrees).

        On a non-uniform (dyvar) grid this is just `lat[1]-lat[0]`, which
        happens to be the polar-row spacing.  Prefer `dy_per_row` / `dy_ref`
        for any meridional metric factor — `dlat` is only kept for legacy
        consumers that want a representative scalar spacing.
        """
        return float(self.lat[1] - self.lat[0])

    @property
    def dx(self) -> np.ndarray:
        """Zonal grid spacing (m), shape (ny,) — varies with latitude."""
        return EARTH_RADIUS * np.cos(np.deg2rad(self.lat)) * np.deg2rad(self.dlon)

    @property
    def lat_v(self) -> np.ndarray:
        """V-face latitudes (degrees), shape (ny+1,).

        Interior v-faces are midway between mass-cell latitudes.  For a
        global grid (first mass point within ~1° of the pole — gSAM
        doglobal=.true.) the boundary faces are pinned at ±90° to match
        `latv[0]=-π/2, latv[ny]=+π/2`.  For limited-domain grids (tests
        with interior ranges) they are extrapolated half the first/last
        interior spacing outside the mass rows.
        """
        interior = 0.5 * (self.lat[:-1] + self.lat[1:])                 # (ny-1,)
        is_global = (abs(self.lat[0]) > 85.0) and (abs(self.lat[-1]) > 85.0)
        if is_global:
            south = -90.0
            north = 90.0
        else:
            south = self.lat[0]  - (interior[0]  - self.lat[0])
            north = self.lat[-1] + (self.lat[-1] - interior[-1])
        return np.concatenate([[south], interior, [north]])              # (ny+1,)

    @property
    def dy_per_row(self) -> np.ndarray:
        """True meridional spacing of each mass row (m), shape (ny,).

        dy_per_row[j] = R * (lat_v[j+1] - lat_v[j]) * π/180

        On a uniform grid this is constant; on the dyvar grid it ranges
        0.133°..0.986° → ~15..110 km.
        """
        return EARTH_RADIUS * np.deg2rad(np.diff(self.lat_v))

    @property
    def dy_ref(self) -> float:
        """Reference meridional spacing (m) — distance between the two
        mid-latitude mass-cell centres.

        Matches gSAM setgrid.f90:238  `dy = y_gl(ny_gl/2+1) - y_gl(ny_gl/2)`.
        Since y_gl[j] = 0.5*(yv_gl[j]+yv_gl[j+1]), this telescopes to
        0.5*(dy_per_row[ny/2 - 1] + dy_per_row[ny/2])  (Python 0-index, ny even).
        On uniform grids this equals dy_per_row[j] for all j.
        """
        dy_arr = self.dy_per_row
        ny = len(dy_arr)
        if ny < 2:
            return float(dy_arr[0])
        return float(0.5 * (dy_arr[ny // 2 - 1] + dy_arr[ny // 2]))

    @property
    def ady(self) -> np.ndarray:
        """Per-row ratio dy_per_row / dy_ref (gSAM ady(j)), shape (ny,).

        Equals 1.0 on uniform grids.  All non-uniform metric corrections go
        through this factor (gSAM setgrid.f90:240 pattern).
        """
        return self.dy_per_row / self.dy_ref

    @property
    def dy(self) -> np.ndarray:
        """Meridional spacing per row (m), shape (ny,).

        API change 2026-04-12 (Gap 8): was a float, now an ndarray.  All
        consumers must broadcast over the ny axis.  On uniform grids every
        entry is equal; on dyvar grids the tropical entries are ~15–30 km
        while the polar entries are ~110 km.
        """
        return self.dy_per_row

    @property
    def dz(self) -> np.ndarray:
        """Vertical spacing (m), shape (nz,)."""
        return np.diff(self.zi)

    @property
    def cos_lat(self) -> np.ndarray:
        """cos(lat) weights, shape (ny,)."""
        return np.cos(np.deg2rad(self.lat))

    @property
    def area_weights(self) -> np.ndarray:
        """Cell area weights (proportional to cos(lat)), shape (ny, nx)."""
        return np.outer(self.cos_lat, np.ones(self.nx))

    # ── Differential operators (JAX arrays) ──────────────────────────────────

    def divergence(self, u, v):
        """
        Horizontal divergence of (u, v) on the C-grid.
        u: (nz, ny, nx+1)  — u at east faces
        v: (nz, ny+1, nx)  — v at north faces
        Returns: (nz, ny, nx)

        Matches gSAM driver exactly:
            div(i,j,k) = (U(i+1,j,k)-U(i,j,k))/dx(j) + (V(i,j+1,k)-V(i,j,k))/dy(j)
        Verified against matching_tests/test_operators (divergence_zero_v_const,
        divergence_linear_u): max_abs = 0.0.
        """
        dx = jnp.array(self.dx)[None, :, None]   # (1, ny, 1)
        dy = jnp.array(self.dy)[None, :, None]   # (1, ny, 1) — per-row

        du_dx = (u[:, :, 1:] - u[:, :, :-1]) / dx              # (nz, ny, nx)
        dv_dy = (v[:, 1:, :] - v[:, :-1, :]) / dy              # (nz, ny, nx)

        return du_dx + dv_dy

    def gradient(self, phi):
        """
        Horizontal gradient of scalar phi (nz, ny, nx) on mass grid.
        Returns: (dphi_dx, dphi_dy) each (nz, ny, nx)
        Uses centred differences with periodic zonal BC.
        """
        dx = jnp.array(self.dx)[None, :, None]   # (1, ny, 1)

        phi_px = jnp.roll(phi, -1, axis=2)  # periodic east
        phi_mx = jnp.roll(phi, +1, axis=2)  # periodic west
        dphi_dx = (phi_px - phi_mx) / (2.0 * dx)

        # Non-uniform meridional centred difference:
        # dphi/dy[j] ≈ (phi[j+1] - phi[j-1]) / (dy_v[j] + dy_v[j+1])
        # where dy_v[j] = 0.5*(dy_per_row[j-1] + dy_per_row[j]) is the
        # distance between adjacent mass points.  At the poles we fall back
        # to one-sided differences via edge padding.
        dy_row = np.array(self.dy_per_row)                      # (ny,)
        dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])              # (ny-1,) between mass pts
        # Build a (ny,) denominator: sum of neighbour v-face distances,
        # edge-padded at the boundaries.
        dy_v_north = np.concatenate([dy_v_int, dy_v_int[-1:]])   # (ny,) dy_v[j+1]
        dy_v_south = np.concatenate([dy_v_int[:1], dy_v_int])    # (ny,) dy_v[j]
        denom_y = jnp.array(dy_v_north + dy_v_south)[None, :, None]  # (1, ny, 1)

        phi_py = jnp.concatenate([phi[:, 1:, :], phi[:, -1:, :]], axis=1)
        phi_my = jnp.concatenate([phi[:, :1, :], phi[:, :-1, :]], axis=1)
        dphi_dy = (phi_py - phi_my) / denom_y

        return dphi_dx, dphi_dy

    def laplacian(self, phi):
        """Horizontal Laplacian of phi (nz, ny, nx). Returns (nz, ny, nx)."""
        dphi_dx, dphi_dy = self.gradient(phi)
        d2_dx2, _ = self.gradient(dphi_dx)
        _, d2_dy2 = self.gradient(dphi_dy)
        return d2_dx2 + d2_dy2

    def vertical_divergence(self, w):
        """
        Vertical divergence of w (nz+1, ny, nx) → (nz, ny, nx).
        Uses base-state density for anelastic form: (1/rho) d(rho*w)/dz.
        """
        dz = jnp.array(self.dz)[:, None, None]   # (nz, 1, 1)
        rho = jnp.array(self.rho)[:, None, None]  # (nz, 1, 1)
        return (w[1:] - w[:-1]) / dz / rho

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_gsam_nc(cls, path_3d: str) -> "LatLonGrid":
        """Build grid from a gSAM 3D NetCDF output file."""
        import netCDF4 as nc

        with nc.Dataset(path_3d) as ds:
            return cls(
                lat=np.array(ds.variables["lat"][:]),
                lon=np.array(ds.variables["lon"][:]),
                z=np.array(ds.variables["z"][:]),
                zi=np.array(ds.variables["zi"][:]),
                rho=np.array(ds.variables["rho"][:]),
            )
