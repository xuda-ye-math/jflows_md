#!/usr/bin/env python
"""Default-float32 molecular trainer smoke test."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows.train import Monitor  # noqa: E402
from jflows_md.utils.screen import compute_ESS_log  # noqa: E402  (the screened ESS the trainers report)
from jflows_md import Mixed_NSF, Molecular_Source  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import (  # noqa: E402
    _adam,
    _clip_global,
    _learning_rate,
    train_forward_KLX_G,
)


class Toy_Target(Potential):
    domain: Mixed_Domain

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum(value[:, :2] ** 2, axis=-1) - 0.2 * jnp.cos(
            value[:, 2]
        )


def main() -> None:
    assert not jax.config.x64_enabled

    params = (jnp.asarray([1.0], dtype=jnp.float32),)
    moments = jax.tree.map(jnp.zeros_like, params)
    rejected = _adam(
        params,
        moments,
        moments,
        (jnp.asarray([jnp.nan], dtype=jnp.float32),),
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1e-3),
        float("inf"),
        jnp.asarray(True),
    )
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(rejected[:3]),
            (params[0], moments[0], moments[0]),
        )
    )
    assert int(rejected[3]) == 0

    overflowed = _adam(
        params,
        moments,
        moments,
        (jnp.asarray([3e30], dtype=jnp.float32),),
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1e-3),
        float("inf"),
        jnp.asarray(True),
    )
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(overflowed[:3]),
            (params[0], moments[0], moments[0]),
        )
    )
    assert int(overflowed[3]) == 0
    clean_gradient = (jnp.asarray([1.0], dtype=jnp.float32),)
    recovered = _adam(
        overflowed[0],
        overflowed[1],
        overflowed[2],
        clean_gradient,
        jnp.asarray(0.0),
        overflowed[3],
        jnp.asarray(1e-3),
        float("inf"),
        jnp.asarray(True),
    )
    clean = _adam(
        params,
        moments,
        moments,
        clean_gradient,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1e-3),
        float("inf"),
        jnp.asarray(True),
    )
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(jax.tree.leaves(recovered), jax.tree.leaves(clean))
    )
    clipped = _clip_global(
        (jnp.asarray([3e30, -3e30], dtype=jnp.float32),), 1.0
    )
    clipped_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in clipped))
    assert bool(jnp.isfinite(clipped_norm)) and float(clipped_norm) <= 1.000001
    assert float(_learning_rate(1.0, 4, 1)) == 0.25
    assert float(_learning_rate(1.0, 4, 4)) == 1.0

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Target(domain)
    samples = source.samples(jax.random.key(40), 12)
    flow = Mixed_NSF(
        jax.random.key(41),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    messages = []
    trained, ess = train_forward_KLX_G(
        samples,
        source,
        target,
        flow,
        domain,
        6,
        2,
        1e-3,
        1,
        1e-3,
        1,
        1,
        monitor=Monitor(1, "[molecular] ", messages.append),
        checkpoint=True,
        seed=42,
    )
    jax.block_until_ready((trained, ess))
    jax.effects_barrier()
    assert samples.dtype == ess.dtype == jnp.float32
    assert ess.shape == (2,)
    trainer_key = jax.random.fold_in(jax.random.key(31), 42)
    index_key, _ = jax.random.split(jax.random.fold_in(trainer_key, 1))
    source_batch = samples[
        jax.random.choice(index_key, samples.shape[0], (6,), replace=False)
    ]
    proposal, ladj = flow.inv_and_ladj(source_batch)
    proposal = domain.wrap(proposal)
    expected = compute_ESS_log(source(source_batch) - target(proposal) + ladj)
    assert bool(jnp.allclose(ess[0], expected, atol=1e-6))
    assert len(messages) == 2
    assert all("[t: 0.000000 -> 1.000000]" in message for message in messages)
    assert all(
        bool(jnp.isfinite(leaf).all())
        for leaf in jax.tree.leaves(trained)
        if isinstance(leaf, jax.Array)
    )
    print("PASS default-float32 molecular trainer")


if __name__ == "__main__":
    main()
