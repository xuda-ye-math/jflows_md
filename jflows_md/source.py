"""Gaussian-Euclidean x uniform-torus molecular source."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential

from .core.domain import Mixed_Domain


__all__ = ["Molecular_Source"]


class Molecular_Source(Potential):
    domain: Mixed_Domain
    mean: Array
    variance: Array

    def __init__(
        self,
        domain: Mixed_Domain,
        mean: Array | list[float] | None = None,
        variance: Array | list[float] | None = None,
    ):
        self.domain = domain
        dtype = jnp.asarray(0.0).dtype
        self.mean = (
            jnp.zeros(domain.euclidean_dim)
            if mean is None
            else jnp.asarray(mean, dtype=dtype)
        )
        self.variance = (
            jnp.ones(domain.euclidean_dim)
            if variance is None
            else jnp.asarray(variance, dtype=dtype)
        )

    @classmethod
    def from_spec(
        cls,
        domain: Mixed_Domain,
        spec: Mapping,
    ) -> "Molecular_Source":
        return cls(domain, spec["source_mean"], spec["source_variance"])

    def __call__(self, q: Array) -> Array:
        euclidean = q[..., : self.domain.euclidean_dim]
        difference = euclidean - self.mean
        return 0.5 * jnp.sum(difference**2 / self.variance, axis=-1)

    def samples(self, key: Array, N: int) -> Array:
        """Draw ``N`` samples from the mixed-domain source."""
        gaussian_key, torus_key = jax.random.split(key)
        euclidean = self.mean + jnp.sqrt(self.variance) * jax.random.normal(
            gaussian_key, (N, self.domain.euclidean_dim), dtype=self.mean.dtype
        )
        periodic = jax.random.uniform(
            torus_key,
            (N, self.domain.periodic_dim),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=self.mean.dtype,
        )
        return jnp.concatenate((euclidean, periodic), axis=-1)
