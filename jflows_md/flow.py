"""Mixed-domain flows that do not modify :mod:`jflows`."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.flow import ComposedTransform, Flow

from .core.checks import integer
from .core.domain import Mixed_Domain
from .core.flow import Mixed_Domain_Wrap, Mixed_Spline_Coupling, zero_coupling


__all__ = ["Mixed_Identity", "Mixed_NSF"]


def _balanced_condition_indices(key: Array, domain: Mixed_Domain) -> tuple[int, ...]:
    """Split each nontrivial domain block across a half-size coupling mask.

    A purely random half-mask can put every torsion on the same side. In a
    two-layer complementary pair those torsions are then always transformed
    together and can never condition one another. The balanced strategy keeps
    the random ordering within each block but, whenever a block has at least
    two coordinates, places members on both sides of the coupling.
    """

    size = domain.dimension // 2
    euclidean = domain.euclidean_dim
    periodic = domain.periodic_dim
    candidates = []
    for euclidean_count in range(max(0, size - periodic), min(euclidean, size) + 1):
        periodic_count = size - euclidean_count
        penalty = 0.0
        if euclidean >= 2 and euclidean_count in (0, euclidean):
            penalty += 100.0
        if periodic >= 2 and periodic_count in (0, periodic):
            penalty += 100.0
        if euclidean:
            penalty += abs(euclidean_count / euclidean - 0.5)
        if periodic:
            penalty += abs(periodic_count / periodic - 0.5)
        candidates.append((penalty, euclidean_count, periodic_count))
    _, euclidean_count, periodic_count = min(candidates)
    euclidean_key, periodic_key = jax.random.split(key)
    euclidean_order = np.asarray(
        jax.random.permutation(euclidean_key, euclidean)
    )
    periodic_order = np.asarray(
        jax.random.permutation(periodic_key, periodic)
    ) + euclidean
    selected = np.concatenate(
        (euclidean_order[:euclidean_count], periodic_order[:periodic_count])
    )
    return tuple(sorted(int(value) for value in selected))


class Mixed_Identity(Flow):
    domain: Mixed_Domain

    def __call__(self, x: Array) -> Array:
        return self.domain.wrap(x)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        return self(x), jnp.zeros(x.shape[:-1], dtype=x.dtype)

    def inv(self, y: Array) -> Array:
        return self.domain.wrap(y)

    def inv_and_ladj(self, y: Array) -> tuple[Array, Array]:
        return self.inv(y), jnp.zeros(y.shape[:-1], dtype=y.dtype)

    def t(self) -> ComposedTransform:
        return ComposedTransform(Mixed_Domain_Wrap(self.domain))

    def zeros(self) -> "Mixed_Identity":
        return self


class Mixed_NSF(Flow):
    """Spline coupling flow on ``R^p x T^q``.

    Ordinary RQS with identity tails acts on Euclidean coordinates; circular
    C1 RQS acts on torsions. Alternating coupling masks allow both coordinate
    types to condition one another while never linearly mixing their domains.
    """

    domain: Mixed_Domain
    couplings: tuple[Mixed_Spline_Coupling, ...]
    bins: int = eqx.field(static=True)
    transforms: int = eqx.field(static=True)
    euclidean_bound: float = eqx.field(static=True)
    mask_strategy: str = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        domain: Mixed_Domain,
        *,
        bins: int = 8,
        transforms: int = 6,
        euclidean_bound: float = 5.0,
        hidden_features: tuple[int, ...] = (64, 64),
        slope: float = 1e-3,
        activation=jax.nn.silu,
        mask_strategy: str = "random",
        condition_mask=None,
    ):
        if not isinstance(domain, Mixed_Domain):
            raise TypeError("domain must be a Mixed_Domain")
        if domain.dimension < 2:
            raise ValueError("Mixed_NSF requires a domain of dimension at least two")
        bins = integer("bins", bins, minimum=2)
        transforms = integer("transforms", transforms, minimum=2)
        if not math.isfinite(float(euclidean_bound)) or euclidean_bound <= 0:
            raise ValueError("euclidean_bound must be positive and finite")
        if not math.isfinite(float(slope)) or not 0 < slope < 1:
            raise ValueError("slope must be finite and lie in (0, 1)")
        if not hidden_features:
            raise ValueError("hidden_features must be nonempty")
        if mask_strategy not in ("random", "balanced"):
            raise ValueError("mask_strategy must be 'random' or 'balanced'")
        hidden_features = tuple(
            integer("hidden feature width", width)
            for width in hidden_features
        )
        if len(set(hidden_features)) != 1:
            raise ValueError("hidden_features must contain one repeated positive width")
        self.domain = domain
        self.bins = bins
        self.transforms = transforms
        self.euclidean_bound = float(euclidean_bound)
        self.mask_strategy = mask_strategy
        keys = jax.random.split(key, (transforms + 1) // 2)
        explicit_masks = None
        if condition_mask is not None:
            explicit_masks = np.asarray(condition_mask, dtype=bool)
            expected = (transforms, domain.dimension)
            if explicit_masks.shape != expected:
                raise ValueError(
                    f"condition_mask must have shape {expected}, got "
                    f"{explicit_masks.shape}"
                )
            counts = explicit_masks.sum(axis=1)
            if np.any(counts == 0) or np.any(counts == domain.dimension):
                raise ValueError(
                    "every condition_mask row must condition and transform "
                    "at least one coordinate"
                )
        layers = []
        for index in range(transforms):
            pair = index // 2
            if explicit_masks is not None:
                condition = tuple(
                    int(value) for value in np.flatnonzero(explicit_masks[index])
                )
            elif mask_strategy == "balanced":
                first = _balanced_condition_indices(keys[pair], domain)
                condition = first if index % 2 == 0 else tuple(
                    value for value in range(domain.dimension) if value not in first
                )
            else:
                permutation = np.asarray(
                    jax.random.permutation(keys[pair], domain.dimension)
                )
                first = tuple(
                    int(value)
                    for value in permutation[: domain.dimension // 2]
                )
                condition = first if index % 2 == 0 else tuple(
                    value for value in range(domain.dimension) if value not in first
                )
            layers.append(
                Mixed_Spline_Coupling(
                    jax.random.fold_in(keys[pair], index + 1),
                    domain,
                    condition,
                    bins=bins,
                    euclidean_bound=euclidean_bound,
                    hidden_features=hidden_features,
                    slope=slope,
                    activation=activation,
                )
            )
        self.couplings = tuple(layers)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        value = self.domain.wrap(x)
        ladj = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for coupling in self.couplings:
            value, local_ladj = coupling.call_and_ladj(value)
            ladj = ladj + local_ladj
        return self.domain.wrap(value), ladj

    def __call__(self, x: Array) -> Array:
        return self.call_and_ladj(x)[0]

    def inv_and_ladj(self, y: Array) -> tuple[Array, Array]:
        value = self.domain.wrap(y)
        ladj = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for coupling in reversed(self.couplings):
            value, local_ladj = coupling.inv_and_ladj(value)
            ladj = ladj + local_ladj
        return self.domain.wrap(value), ladj

    def inv(self, y: Array) -> Array:
        return self.inv_and_ladj(y)[0]

    def t(self) -> ComposedTransform:
        wrap = Mixed_Domain_Wrap(self.domain)
        return ComposedTransform(wrap, *self.couplings, wrap)

    def zeros(self) -> "Mixed_NSF":
        return eqx.tree_at(
            lambda flow: flow.couplings,
            self,
            replace=tuple(zero_coupling(layer) for layer in self.couplings),
        )
