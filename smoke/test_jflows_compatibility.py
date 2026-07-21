#!/usr/bin/env python
"""Public compatibility checks between committed jflows and jflows_md."""

from __future__ import annotations

import inspect
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jflows.flow as jflows_flow  # noqa: E402
from jflows.flow import (  # noqa: E402
    CircularRQSTransform,
    ComposedTransform,
    Flow,
    MonotonicRQSTransform,
    NSF,
    Transform,
)
from jflows.potential import Nlog_Gaussian  # noqa: E402
from jflows.boltzmann import boltzmann_identity as jflows_identity  # noqa: E402
from jflows.train import train_forward_KLX_G  # noqa: E402
from jflows.utils import (  # noqa: E402
    annealed_importance_sampling,
    importance_weights_log,
    langevin,
    sequential_monte_carlo,
)
from jflows_md import (  # noqa: E402
    Mixed_NSF,
    Molecular_Source,
    boltzmann_identity as molecular_identity,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402


def _assert_signature(function, expected: tuple[str, ...]) -> None:
    assert tuple(inspect.signature(function).parameters) == expected


def main() -> None:
    # jflows_md uses only this explicitly public extension of jflows.flow.
    required = {
        "CircularRQSTransform",
        "ComposedTransform",
        "Flow",
        "MonotonicRQSTransform",
        "Transform",
    }
    assert required <= set(jflows_flow.__all__)
    assert issubclass(Transform, object)
    assert all(
        value is not None
        for value in (
            CircularRQSTransform,
            ComposedTransform,
            Flow,
            MonotonicRQSTransform,
        )
    )

    # Preserve the committed jflows public signatures and compilation scheme.
    _assert_signature(
        langevin,
        ("key", "samples", "potential", "dt", "steps", "adjust", "taming", "chunks"),
    )
    _assert_signature(
        sequential_monte_carlo,
        (
            "key",
            "samples",
            "source",
            "target",
            "ladder",
            "mc_dt",
            "mc_steps",
            "adjust",
            "taming",
            "chunks",
        ),
    )
    assert callable(train_forward_KLX_G)
    assert callable(langevin)
    assert callable(sequential_monte_carlo)
    assert callable(annealed_importance_sampling)
    generic_identity = inspect.signature(jflows_identity).parameters
    mixed_identity = inspect.signature(molecular_identity).parameters
    assert tuple(generic_identity)[:6] == tuple(mixed_identity)[:6] == (
        "x_valid", "source", "target", "ladder", "mc_dt", "mc_steps",
    )
    assert {"monitor", "bg_param", "chunks", "seed"} <= set(
        generic_identity
    ) & set(mixed_identity)
    assert {"rg_param_0", "rg_param_1", "mc_image_radius"} <= set(
        mixed_identity
    )
    assert not ({
        "flow", "pool_size", "batch_size", "train_steps", "lr",
    } & (set(generic_identity) | set(mixed_identity)))

    source = Nlog_Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0])
    target = Nlog_Gaussian(mean=[0.3, -0.2], variance=[0.8, 1.2])
    samples = source.samples(jax.random.key(1), 16)
    flow = NSF(
        jax.random.key(2),
        a=[-5.0, -5.0],
        b=[5.0, 5.0],
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()

    for direction in ("F", "G"):
        full = importance_weights_log(
            samples, source, target, flow, direction, chunks=1
        )
        chunked = importance_weights_log(
            samples, source, target, flow, direction, chunks=4
        )
        assert bool(jnp.allclose(chunked, full, rtol=1e-5, atol=1e-5))

    particles, level_ess = sequential_monte_carlo(
        jax.random.key(3),
        samples,
        source,
        target,
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
        adjust=False,
        chunks=2,
    )
    assert particles.shape == samples.shape and level_ess.shape == (2,)
    assert bool(jnp.isfinite(particles).all() & jnp.isfinite(level_ess).all())

    proposed = annealed_importance_sampling(
        jax.random.key(4),
        samples,
        source,
        target,
        flow,
        "G",
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
        adjust=False,
        chunks=2,
    )
    assert proposed.shape == samples.shape and bool(jnp.isfinite(proposed).all())

    trained, history = train_forward_KLX_G(
        samples,
        source,
        target,
        flow,
        batch_size=4,
        train_steps=1,
        lr=1e-3,
        ladder=1,
        mc_dt=1e-3,
        mc_steps=1,
        coeff_lambda=1.0,
        mc_adjust=False,
        seed=5,
    )
    jax.block_until_ready((trained, history))
    assert history.shape == (1,) and bool(jnp.isfinite(history).all())

    domain = Mixed_Domain(2, 1)
    mixed_source = Molecular_Source(domain)
    mixed_samples = mixed_source.samples(jax.random.key(6), N=8)
    mixed_flow = Mixed_NSF(
        jax.random.key(7),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    mapped, ladj = mixed_flow.call_and_ladj(mixed_samples)
    recovered, inverse_ladj = mixed_flow.inv_and_ladj(mapped)
    assert isinstance(mixed_flow, Flow)
    assert bool(jnp.allclose(recovered, mixed_samples, atol=1e-5))
    assert bool(jnp.allclose(ladj, -inverse_ladj, atol=1e-5))

    print("PASS committed jflows behavior and public jflows_md compatibility")


if __name__ == "__main__":
    main()
