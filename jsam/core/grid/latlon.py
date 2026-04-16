"""Lat-lon Arakawa C-grid. Scalars at centres: (nz,ny,nx); U at east: (nz,ny,nx+1);
V at north: (nz,ny+1,nx); W at top: (nz+1,ny,nx). Periodic in zonal."""

from __future__ import annotations
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .base import Grid

# Earth radius
EARTH_RADIUS = 6371229.0   # m


@dataclass(frozen=True)
class LatLonGrid(Grid):
    """Regular lat-lon grid."""
    lat:  np.ndarray   # (ny,) degrees
    lon:  np.ndarray   # (nx,) degrees
    z:    np.ndarray   # (nz,) m
    zi:   np.ndarray   # (nz+1,) m
    rho:  np.ndarray   # (nz,) kg/m³

    @property
    def nx(self): return len(self.lon)

    @property
    def ny(self): return len(self.lat)

    @property
    def nz(self): return len(self.z)
    @property
    def dlon(self) -> float:
        """Longitude spacing (degrees)."""
        return float(self.lon[1] - self.lon[0])

    @property
    def dlat(self) -> float:
        """Latitude spacing (degrees)."""
        return float(self.lat[1] - self.lat[0])

    @property
    def dx(self) -> np.ndarray:
        """Zonal grid spacing (m), shape (ny,) — varies with latitude."""
        return EARTH_RADIUS * np.cos(np.deg2rad(self.lat)) * np.deg2rad(self.dlon)

    @property
    def lat_v(self) -> np.ndarray:
        """V-face latitudes (degrees), shape (ny+1,)."""
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
        """Meridional spacing per row (m), shape (ny,)."""
        return EARTH_RADIUS * np.deg2rad(np.diff(self.lat_v))

    @property
    def dy_ref(self) -> float:
        """Reference meridional spacing (m)."""
        dy_arr = self.dy_per_row
        ny = len(dy_arr)
        if ny < 2:
            return float(dy_arr[0])
        return float(0.5 * (dy_arr[ny // 2 - 1] + dy_arr[ny // 2]))

    @property
    def ady(self) -> np.ndarray:
        """Per-row spacing ratio dy_per_row / dy_ref, shape (ny,)."""
        return self.dy_per_row / self.dy_ref

    @property
    def dy(self) -> np.ndarray:
        """Meridional spacing per row (m), shape (ny,)."""
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

    def divergence(self, u, v):
        """Divergence of (u,v). u: (nz,ny,nx+1), v: (nz,ny+1,nx). Returns (nz,ny,nx)."""
        dx = jnp.array(self.dx)[None, :, None]   # (1, ny, 1)
        dy = jnp.array(self.dy)[None, :, None]   # (1, ny, 1) — per-row

        du_dx = (u[:, :, 1:] - u[:, :, :-1]) / dx              # (nz, ny, nx)
        dv_dy = (v[:, 1:, :] - v[:, :-1, :]) / dy              # (nz, ny, nx)

        return du_dx + dv_dy

    def gradient(self, phi):
        """Gradient of phi (nz,ny,nx). Returns (dphi_dx, dphi_dy) each (nz,ny,nx)."""
        dx = jnp.array(self.dx)[None, :, None]   # (1, ny, 1)

        phi_px = jnp.roll(phi, -1, axis=2)  # periodic east
        phi_mx = jnp.roll(phi, +1, axis=2)  # periodic west
        dphi_dx = (phi_px - phi_mx) / (2.0 * dx)

        dy_row = np.array(self.dy_per_row)
        dy_v_int = 0.5 * (dy_row[:-1] + dy_row[1:])
        dy_v_north = np.concatenate([dy_v_int, dy_v_int[-1:]])
        dy_v_south = np.concatenate([dy_v_int[:1], dy_v_int])
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
        """Vertical divergence w (nz+1,ny,nx) → (nz,ny,nx). Anelastic: (1/rho)*d(rho*w)/dz."""
        dz = jnp.array(self.dz)[:, None, None]   # (nz, 1, 1)
        rho = jnp.array(self.rho)[:, None, None]  # (nz, 1, 1)
        return (w[1:] - w[:-1]) / dz / rho

    @classmethod
    def from_gsam_nc(cls, path_3d: str) -> "LatLonGrid":
        """Build grid from gSAM 3D NetCDF file."""
        import netCDF4 as nc

        with nc.Dataset(path_3d) as ds:
            return cls(
                lat=np.array(ds.variables["lat"][:]),
                lon=np.array(ds.variables["lon"][:]),
                z=np.array(ds.variables["z"][:]),
                zi=np.array(ds.variables["zi"][:]),
                rho=np.array(ds.variables["rho"][:]),
            )
