"""Gaussian-Euclidean x uniform-torus molecular source."""

from __future__ import annotations

from collections.abc import Mapping
import math
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
    defensive_scale2: Array
    defensive_weight: float = eqx.field(static=True)
    defensive_df: float = eqx.field(static=True)

    def __init__(
        self,
        domain: Mixed_Domain,
        mean: Array | list[float] | None = None,
        variance: Array | list[float] | None = None,
        *,
        defensive_weight: float = 0.0,
        defensive_df: float = 3.0,
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
        try:
            defensive_weight = float(defensive_weight)
            defensive_df = float(defensive_df)
        except (TypeError, ValueError) as exc:
            raise ValueError("defensive source controls must be real scalars") from exc
        if not math.isfinite(defensive_weight) or not 0.0 <= defensive_weight < 1.0:
            raise ValueError("defensive_weight must lie in [0, 1)")
        if not math.isfinite(defensive_df) or defensive_df <= 2.0:
            raise ValueError("defensive_df must be finite and greater than two")
        self.defensive_weight = defensive_weight
        self.defensive_df = defensive_df
        self.defensive_scale2 = self.variance * (
            (defensive_df - 2.0) / defensive_df
        )
        if self.mean.shape != (domain.euclidean_dim,):
            raise ValueError("source mean has the wrong shape")
        if self.variance.shape != (domain.euclidean_dim,):
            raise ValueError("source variance has the wrong shape")
        if not bool(jnp.all(jnp.isfinite(self.mean))):
            raise ValueError("source mean must be finite")
        if not bool(jnp.all(jnp.isfinite(self.variance) & (self.variance > 0))):
            raise ValueError("source variance must be finite and positive")

    @classmethod
    def from_spec(
        cls,
        domain: Mixed_Domain,
        spec: Mapping,
        *,
        defensive_weight: float = 0.0,
        defensive_df: float = 3.0,
    ) -> "Molecular_Source":
        return cls(
            domain,
            spec.get("source_mean"),
            spec.get("source_variance"),
            defensive_weight=defensive_weight,
            defensive_df=defensive_df,
        )

    def __call__(self, q: Array) -> Array:
        self.domain._validate(q, "molecular source coordinates")
        if q.ndim != 2:
            raise ValueError(f"molecular source coordinates must have shape [N, d], got {q.shape}")
        euclidean = q[..., : self.domain.euclidean_dim]
        difference = euclidean - self.mean
        gaussian_energy = 0.5 * jnp.sum(difference ** 2 / self.variance, axis=-1)
        if self.defensive_weight == 0.0:
            return gaussian_energy
        gaussian_log_density = -gaussian_energy - 0.5 * jnp.sum(
            jnp.log(2.0 * jnp.pi * self.variance)
        )
        df = self.defensive_df
        student_log_normalizer = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
            - 0.5 * jnp.log(self.defensive_scale2)
        )
        student_log_density = jnp.sum(
            student_log_normalizer
            - 0.5 * (df + 1.0) * jnp.log1p(
                difference ** 2 / (df * self.defensive_scale2)
            ),
            axis=-1,
        )
        log_density = jnp.logaddexp(
            math.log1p(-self.defensive_weight) + gaussian_log_density,
            math.log(self.defensive_weight) + student_log_density,
        )
        return -log_density

    def samples(self, key: Array, N: int) -> Array:
        """Draw ``N`` samples from the mixed-domain source."""
        if isinstance(N, bool):
            raise ValueError("N must be a positive integer, not a boolean")
        try:
            N = operator.index(N)
        except TypeError as exc:
            raise ValueError(f"N must be a positive integer, got {N!r}") from exc
        if N < 1:
            raise ValueError("N must be a positive integer")
        if self.defensive_weight == 0.0:
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
        (
            gaussian_key,
            student_normal_key,
            student_gamma_key,
            choice_key,
            torus_key,
        ) = jax.random.split(key, 5)
        euclidean = self.mean + jnp.sqrt(self.variance) * jax.random.normal(
            gaussian_key, (N, self.domain.euclidean_dim), dtype=self.mean.dtype
        )
        student_normal = jax.random.normal(
            student_normal_key,
            (N, self.domain.euclidean_dim),
            dtype=self.mean.dtype,
        )
        chi_square = 2.0 * jax.random.gamma(
            student_gamma_key,
            self.defensive_df / 2.0,
            shape=(N, self.domain.euclidean_dim),
            dtype=self.mean.dtype,
        )
        student = self.mean + jnp.sqrt(self.defensive_scale2) * (
            student_normal / jnp.sqrt(chi_square / self.defensive_df)
        )
        defensive = jax.random.bernoulli(
            choice_key, self.defensive_weight, shape=(N, 1)
        )
        euclidean = jnp.where(defensive, student, euclidean)
        periodic = jax.random.uniform(
            torus_key,
            (N, self.domain.periodic_dim),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=self.mean.dtype,
        )
        return jnp.concatenate((euclidean, periodic), axis=-1)
