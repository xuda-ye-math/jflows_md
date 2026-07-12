#!/usr/bin/env python
"""Default-float32 and rematerialized molecular trainer smoke test."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows.train import Monitor  # noqa: E402
from jflows.utils import compute_ESS_log  # noqa: E402
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


class Hard_Wall_Target(Potential):
    """Finite on half the batch and +inf on the other half."""

    domain: Mixed_Domain

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain

    def __call__(self, value):
        finite = 0.5 * jnp.sum(value**2, axis=-1)
        return jnp.where(value[:, 0] > 0.0, jnp.inf, finite)


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
    messages = []
    trained, ess, kept, updated = train_molecular_forward_KLX_G(
        samples,
        samples,
        source,
        target,
        flow,
        n_batch=6,
        steps=2,
        lr=1e-3,
        checkpoint=True,
        e_clip=1000.0,
        g_clip=100.0,
        monitor=Monitor(1, "[molecular] ", messages.append),
        seed=42,
    )
    jax.block_until_ready((trained, ess, kept, updated))
    jax.effects_barrier()
    assert samples.dtype == jnp.float32 and ess.dtype == jnp.float32
    assert ess.shape == kept.shape == updated.shape == (2,)
    trainer_key = jax.random.fold_in(jax.random.key(31), 42)
    _, source_key, _ = jax.random.split(jax.random.fold_in(trainer_key, 1), 3)
    source_batch = samples[
        jax.random.choice(
            source_key, samples.shape[0], (6,), replace=False
        )
    ]
    proposal, inverse_ladj = flow.inv_and_ladj(source_batch)
    expected_first_ess = compute_ESS_log(
        source(source_batch) - target(proposal) + inverse_ladj
    )
    assert bool(jnp.allclose(ess[0], expected_first_ess, atol=1e-6))
    assert all(
        leaf.dtype == jnp.float32
        for leaf in jax.tree.leaves(trained)
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)
    )
    assert bool(jnp.isfinite(ess).all() & jnp.isfinite(kept).all())
    assert bool(updated[0])
    assert len(messages) == 2
    for step, message in enumerate(messages, start=1):
        assert message.startswith(
            f"[molecular] step {step:>5d}   loss = "
        )
        assert message.endswith(f"ESS = {float(ess[step - 1]):.4f}")
    assert all(
        bool(jnp.isfinite(leaf).all())
        for leaf in jax.tree.leaves(trained)
        if isinstance(leaf, jax.Array)
    )

    hard_samples = jnp.asarray(
        [
            [-1.0, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
            [1.0, 0.0, 0.5],
        ],
        dtype=samples.dtype,
    )
    _, hard_ess, _, _ = train_molecular_forward_KLX_G(
        hard_samples,
        hard_samples,
        source,
        Hard_Wall_Target(domain),
        flow,
        n_batch=4,
        steps=1,
        lr=1e-3,
        seed=43,
    )
    assert 0.0 < float(hard_ess[0]) <= 1.0
    print("PASS default-float32 checkpointed molecular trainer")


if __name__ == "__main__":
    main()
