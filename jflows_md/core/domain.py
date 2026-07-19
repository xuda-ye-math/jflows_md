"""Mixed Euclidean/periodic domain."""

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
        self.euclidean_dim = int(euclidean_dim)
        self.periodic_dim = int(periodic_dim)
        self.dimension = int(euclidean_dim + periodic_dim)

    def wrap(self, x: Array) -> Array:
        if self.periodic_dim == 0:
            return x
        euclidean = x[..., : self.euclidean_dim]
        torsions = x[..., self.euclidean_dim :]
        torsions = jnp.mod(torsions + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return jnp.concatenate((euclidean, torsions), axis=-1)

    def displacement(self, x: Array, y: Array) -> Array:
        """Return the shortest tangent displacement ``x-y``."""
        delta = x - y
        if self.periodic_dim == 0:
            return delta
        euclidean = delta[..., : self.euclidean_dim]
        torsions = delta[..., self.euclidean_dim :]
        torsions = jnp.mod(torsions + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        return jnp.concatenate((euclidean, torsions), axis=-1)
