#!/usr/bin/env python
"""Default-float32 and rematerialized molecular trainer smoke test."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md import Mixed_NSF, Molecular_Source  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import train_molecular_forward_KLX_G  # noqa: E402


class Toy_Target(Potential):
    domain: Mixed_Domain

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum(value[:, :2] ** 2, axis=-1) - 0.2 * jnp.cos(
            value[:, 2]
        )


def main() -> None:
    assert not jax.config.x64_enabled
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Target(domain)
    samples = source.samples(jax.random.key(40), N=12)
    flow = Mixed_NSF(
        jax.random.key(41),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    trained, ess, kept, updated = train_molecular_forward_KLX_G(
        samples,
        source,
        target,
        flow,
        n_batch=6,
        steps=1,
        lr=1e-3,
        checkpoint=True,
        e_clip=1000.0,
        g_clip=100.0,
        seed=42,
    )
    jax.block_until_ready((trained, ess, kept, updated))
    assert samples.dtype == jnp.float32 and ess.dtype == jnp.float32
    assert ess.shape == kept.shape == updated.shape == (1,)
    assert bool(jnp.isfinite(ess).all() & jnp.isfinite(kept).all())
    assert bool(updated[0])
    assert all(
        bool(jnp.isfinite(leaf).all())
        for leaf in jax.tree.leaves(trained)
        if isinstance(leaf, jax.Array)
    )
    print("PASS default-float32 checkpointed molecular trainer")


if __name__ == "__main__":
    main()
