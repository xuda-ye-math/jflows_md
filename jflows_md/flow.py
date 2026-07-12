"""Mixed-domain flows that do not modify :mod:`jflows`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.flow import ComposedTransform, Flow

from .core.domain import Mixed_Domain
from .core.flow import Mixed_Domain_Wrap, Mixed_Spline_Coupling, zero_coupling


__all__ = ["Mixed_Identity", "Mixed_NSF"]


def conditioner_features(x: Array, domain: Mixed_Domain) -> Array:
    """Raw Euclidean values plus cosine/sine embeddings of torsions."""

    euclidean = x[..., : domain.euclidean_dim]
    periodic = x[..., domain.euclidean_dim :]
    return jnp.concatenate((euclidean, jnp.cos(periodic), jnp.sin(periodic)), axis=-1)


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
    ):
        if domain.dimension < 2:
            raise ValueError("Mixed_NSF requires a domain of dimension at least two")
        if transforms < 2:
            raise ValueError("Mixed_NSF needs at least two coupling transforms")
        self.domain = domain
        self.bins = int(bins)
        self.transforms = int(transforms)
        self.euclidean_bound = float(euclidean_bound)
        keys = jax.random.split(key, (transforms + 1) // 2)
        layers = []
        for index in range(transforms):
            pair = index // 2
            permutation = np.asarray(jax.random.permutation(keys[pair], domain.dimension))
            first = tuple(int(value) for value in permutation[: domain.dimension // 2])
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
