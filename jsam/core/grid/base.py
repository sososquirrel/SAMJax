"""Abstract base class for all jsam grids."""

from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as jnp


class Grid(ABC):
    """Abstract grid for operators (div, grad, curl, laplacian)."""

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
    def dx(self): ...  # horizontal spacing (m)

    @property
    @abstractmethod
    def dz(self): ...  # vertical spacing (m)

    @abstractmethod
    def divergence(self, u, v) -> jnp.ndarray: ...

    @abstractmethod
    def gradient(self, phi) -> tuple[jnp.ndarray, jnp.ndarray]: ...

    @abstractmethod
    def laplacian(self, phi) -> jnp.ndarray: ...
