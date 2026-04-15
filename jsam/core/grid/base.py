"""Abstract base class for all jsam grids."""

from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as jnp


class Grid(ABC):
    """
    Abstract grid. Subclasses implement lat-lon, hexagonal, or triangular
    meshes. All operators (div, grad, curl) must be defined on the grid.
    """

    @property
    @abstractmethod
    def nx(self) -> int: ...

    @property
    @abstractmethod
    def ny(self) -> int: ...

    @property
    @abstractmethod
    def nz(self) -> int: ...

    @property
    @abstractmethod
    def dx(self): ...  # horizontal spacing (m), may be 2D array for lat-lon

    @property
    @abstractmethod
    def dz(self): ...  # vertical spacing (m), 1D array

    @abstractmethod
    def divergence(self, u, v) -> jnp.ndarray:
        """Horizontal divergence of (u, v) on the grid."""
        ...

    @abstractmethod
    def gradient(self, phi) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Horizontal gradient (dphi/dx, dphi/dy) on the grid."""
        ...

    @abstractmethod
    def laplacian(self, phi) -> jnp.ndarray:
        """Horizontal Laplacian on the grid."""
        ...
