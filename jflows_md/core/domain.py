"""Mixed Euclidean/periodic domain metadata."""

from __future__ import annotations

import operator

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Mixed_Domain(eqx.Module):
    """A contiguous product domain ``R^p x T^q``.

    The Euclidean block is always first and the periodic torsion block last.
    This convention is frozen in every molecular CoordinateSpec.
    """

    euclidean_dim: int = eqx.field(static=True)
    periodic_dim: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(self, euclidean_dim: int, periodic_dim: int):
        if isinstance(euclidean_dim, bool) or isinstance(periodic_dim, bool):
            raise ValueError("mixed-domain dimensions must be integers, not booleans")
        try:
            euclidean_dim = operator.index(euclidean_dim)
            periodic_dim = operator.index(periodic_dim)
        except TypeError as exc:
            raise ValueError("mixed-domain dimensions must be integers") from exc
        if euclidean_dim < 0 or periodic_dim < 0 or euclidean_dim + periodic_dim <= 0:
            raise ValueError("invalid mixed-domain dimensions")
        self.euclidean_dim = euclidean_dim
        self.periodic_dim = periodic_dim
        self.dimension = self.euclidean_dim + self.periodic_dim

    def _validate(self, x: Array, name: str) -> None:
        if x.ndim < 1 or x.shape[-1] != self.dimension:
            raise ValueError(
                f"{name} must have last dimension {self.dimension}, got {x.shape}"
            )

    def wrap(self, x: Array) -> Array:
        self._validate(x, "mixed-domain coordinates")
        if self.periodic_dim == 0:
            return x
        euclidean = x[..., : self.euclidean_dim]
        torsions = x[..., self.euclidean_dim :]
        torsions = jnp.mod(torsions + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return jnp.concatenate((euclidean, torsions), axis=-1)

    def displacement(self, x: Array, y: Array) -> Array:
        """Return the shortest tangent displacement ``x-y``."""

        self._validate(x, "x")
        self._validate(y, "y")
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x.shape} and {y.shape}"
            )
        delta = x - y
        if self.periodic_dim == 0:
            return delta
        euclidean = delta[..., : self.euclidean_dim]
        torsions = delta[..., self.euclidean_dim :]
        torsions = jnp.mod(torsions + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return jnp.concatenate((euclidean, torsions), axis=-1)
