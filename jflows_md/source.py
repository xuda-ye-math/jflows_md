"""Gaussian-Euclidean x uniform-torus molecular source."""

from __future__ import annotations

from collections.abc import Mapping
import operator

import equinox as eqx
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
        default_dtype = jnp.asarray(0.0).dtype
        mean_array = jnp.zeros(domain.euclidean_dim) if mean is None else jnp.asarray(mean)
        variance_array = (
            jnp.ones(domain.euclidean_dim) if variance is None else jnp.asarray(variance)
        )
        if jnp.issubdtype(mean_array.dtype, jnp.integer):
            mean_array = mean_array.astype(default_dtype)
        if jnp.issubdtype(variance_array.dtype, jnp.integer):
            variance_array = variance_array.astype(default_dtype)
        if not jnp.issubdtype(mean_array.dtype, jnp.floating) or not jnp.issubdtype(
            variance_array.dtype, jnp.floating
        ):
            raise ValueError("source mean and variance must be real-valued")
        dtype = jnp.result_type(mean_array.dtype, variance_array.dtype)
        self.mean = mean_array.astype(dtype)
        self.variance = variance_array.astype(dtype)
        if self.mean.shape != (domain.euclidean_dim,):
            raise ValueError("source mean has the wrong shape")
        if self.variance.shape != (domain.euclidean_dim,):
            raise ValueError("source variance has the wrong shape")
        if not bool(jnp.all(jnp.isfinite(self.mean))):
            raise ValueError("source mean must be finite")
        if not bool(jnp.all(jnp.isfinite(self.variance) & (self.variance > 0))):
            raise ValueError("source variance must be finite and positive")

    @classmethod
    def from_spec(cls, domain: Mixed_Domain, spec: Mapping) -> "Molecular_Source":
        return cls(domain, spec.get("source_mean"), spec.get("source_variance"))

    def __call__(self, q: Array) -> Array:
        self.domain._validate(q, "molecular source coordinates")
        if q.ndim != 2:
            raise ValueError(f"molecular source coordinates must have shape [N, d], got {q.shape}")
        euclidean = q[..., : self.domain.euclidean_dim]
        return 0.5 * jnp.sum((euclidean - self.mean) ** 2 / self.variance, axis=-1)

    def samples(self, key: Array, N: int | None = None, *, n: int | None = None) -> Array:
        """Draw samples; ``n=`` remains as a compatibility alias for ``N=``."""

        if N is None:
            N = n
        elif n is not None:
            raise ValueError("pass only one of N or n")
        if N is None:
            raise TypeError("missing required sample count N")
        if isinstance(N, bool):
            raise ValueError("N must be a positive integer, not a boolean")
        try:
            N = operator.index(N)
        except TypeError as exc:
            raise ValueError(f"N must be a positive integer, got {N!r}") from exc
        if N < 1:
            raise ValueError("N must be a positive integer")
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
