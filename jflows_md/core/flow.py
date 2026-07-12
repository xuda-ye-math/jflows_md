"""Low-level mixed Euclidean/torus spline coupling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from math import pi
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.flow import CircularRQSTransform, MonotonicRQSTransform, Transform

from .checks import integer
from .domain import Mixed_Domain


class Mixed_Domain_Wrap(Transform):
    """Canonical representative map for a mixed Euclidean/torus domain."""

    domain: Mixed_Domain
    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __call__(self, x: Array) -> Array:
        return self.domain.wrap(x)

    def _inverse(self, y: Array) -> Array:
        return self.domain.wrap(y)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.zeros(x.shape[:-1], dtype=x.dtype)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        return self(x), jnp.zeros(x.shape[:-1], dtype=x.dtype)


class Mixed_Spline_Coupling(Transform):
    """One coupling layer with ordinary and circular RQS outputs.

    Parameters are conditioned on unchanged coordinates. Euclidean inputs enter
    raw; periodic inputs enter as cosine/sine pairs, making the conditioner
    invariant to the chosen torsion representative.
    """

    network: eqx.nn.MLP
    domain: Mixed_Domain
    condition_indices: tuple[int, ...] = eqx.field(static=True)
    transform_indices: tuple[int, ...] = eqx.field(static=True)
    euclidean_indices: tuple[int, ...] = eqx.field(static=True)
    periodic_indices: tuple[int, ...] = eqx.field(static=True)
    euclidean_positions: tuple[int, ...] = eqx.field(static=True)
    periodic_positions: tuple[int, ...] = eqx.field(static=True)
    bins: int = eqx.field(static=True)
    euclidean_bound: float = eqx.field(static=True)
    slope: float = eqx.field(static=True)
    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(
        self,
        key: Array,
        domain: Mixed_Domain,
        condition_indices: Sequence[int],
        *,
        bins: int,
        euclidean_bound: float,
        hidden_features: tuple[int, ...],
        slope: float,
        activation: Callable[[Array], Array],
    ):
        condition = tuple(
            sorted(integer("condition index", index, minimum=0) for index in condition_indices)
        )
        if len(set(condition)) != len(condition) or any(
            index >= domain.dimension for index in condition
        ):
            raise ValueError("condition indices must be unique and lie in the domain")
        transformed = tuple(index for index in range(domain.dimension) if index not in condition)
        if not condition or not transformed:
            raise ValueError("a mixed spline coupling needs nonempty condition and transform sets")
        bins = integer("bins", bins, minimum=2)
        if (
            not math.isfinite(float(euclidean_bound))
            or euclidean_bound <= 0
            or not math.isfinite(float(slope))
            or not 0 < slope < 1
        ):
            raise ValueError("invalid mixed spline configuration")
        hidden_features = tuple(
            integer("hidden feature width", width, minimum=1)
            for width in hidden_features
        )
        if not hidden_features or len(set(hidden_features)) != 1:
            raise ValueError("hidden_features must contain one repeated positive width")

        euclidean = tuple(index for index in transformed if index < domain.euclidean_dim)
        periodic = tuple(index for index in transformed if index >= domain.euclidean_dim)
        self.condition_indices = condition
        self.transform_indices = transformed
        self.euclidean_indices = euclidean
        self.periodic_indices = periodic
        self.euclidean_positions = tuple(transformed.index(index) for index in euclidean)
        self.periodic_positions = tuple(transformed.index(index) for index in periodic)
        self.domain = domain
        self.bins = bins
        self.euclidean_bound = float(euclidean_bound)
        self.slope = float(slope)

        in_size = sum(1 if index < domain.euclidean_dim else 2 for index in condition)
        out_size = len(transformed) * 3 * bins
        self.network = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=int(hidden_features[0]),
            depth=len(hidden_features),
            activation=activation,
            key=key,
        )

    def _features(self, x: Array) -> Array:
        features = []
        for index in self.condition_indices:
            value = x[..., index : index + 1]
            if index < self.domain.euclidean_dim:
                features.append(value)
            else:
                features.extend((jnp.cos(value), jnp.sin(value)))
        return jnp.concatenate(features, axis=-1)

    def _parameters(self, x: Array) -> Array:
        features = self._features(x)
        flat = features.reshape((-1, features.shape[-1]))
        output = jax.vmap(self.network)(flat)
        shape = (*features.shape[:-1], len(self.transform_indices), 3, self.bins)
        return output.reshape(shape)

    @staticmethod
    def _sum_ladj(ladj: Array) -> Array:
        return jnp.sum(ladj, axis=-1)

    def _apply(self, x: Array, *, inverse: bool) -> tuple[Array, Array]:
        parameters = self._parameters(x)
        result = x
        ladj = jnp.zeros(x.shape[:-1], dtype=x.dtype)

        if self.euclidean_indices:
            positions = self.euclidean_positions
            params = parameters[..., positions, :, :]
            transform = MonotonicRQSTransform(
                params[..., 0, :],
                params[..., 1, :],
                params[..., 2, :-1],
                bound=self.euclidean_bound,
                slope=self.slope,
            )
            values = x[..., self.euclidean_indices]
            values, local_ladj = (
                transform.inv.call_and_ladj(values)
                if inverse
                else transform.call_and_ladj(values)
            )
            result = result.at[..., self.euclidean_indices].set(values)
            ladj = ladj + self._sum_ladj(local_ladj)

        if self.periodic_indices:
            positions = self.periodic_positions
            params = parameters[..., positions, :, :]
            transform = CircularRQSTransform(
                params[..., 0, :],
                params[..., 1, :],
                params[..., 2, :],
                bound=pi,
                slope=self.slope,
            )
            values = x[..., self.periodic_indices]
            values, local_ladj = (
                transform.inv.call_and_ladj(values)
                if inverse
                else transform.call_and_ladj(values)
            )
            result = result.at[..., self.periodic_indices].set(values)
            ladj = ladj + self._sum_ladj(local_ladj)

        return result, ladj

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        return self._apply(x, inverse=False)

    def __call__(self, x: Array) -> Array:
        return self._apply(x, inverse=False)[0]

    def _inverse(self, y: Array) -> Array:
        return self._apply(y, inverse=True)[0]

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return self._apply(x, inverse=False)[1]

    def inv_and_ladj(self, y: Array) -> tuple[Array, Array]:
        return self._apply(y, inverse=True)


def zero_coupling(coupling: Mixed_Spline_Coupling) -> Mixed_Spline_Coupling:
    """Return an exactly identity-initialized coupling."""

    return eqx.tree_at(
        lambda layer: (
            layer.network.layers[-1].weight,
            layer.network.layers[-1].bias,
        ),
        coupling,
        replace_fn=jnp.zeros_like,
    )
