#!/usr/bin/env python
"""Regression tests for eager molecular chunk and ladder controllers."""

from __future__ import annotations

import inspect
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows.utils import resample  # noqa: E402
from jflows_md import Mixed_Identity  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.utils import (  # noqa: E402
    _mixed_mala_chunk,
    annealed_importance_sampling,
    mixed_mala,
    mixed_quench_and_temper,
    potential_space_smc,
    sequential_monte_carlo,
)


class Toy_Mixed_Potential(Potential):
    domain: Mixed_Domain
    center: jax.Array

    def __init__(self, domain: Mixed_Domain, center):
        self.domain = domain
        self.center = jnp.asarray(center)

    def __call__(self, value):
        euclidean = value[:, : self.domain.euclidean_dim] - self.center[:2]
        torsion = value[:, 2] - self.center[2]
        return 0.5 * jnp.sum(euclidean**2, axis=-1) - jnp.cos(torsion)


def main() -> None:
    assert not hasattr(mixed_mala, "lower")
    assert not hasattr(sequential_monte_carlo, "lower")
    assert not hasattr(annealed_importance_sampling, "lower")
    assert hasattr(_mixed_mala_chunk, "lower")
    assert tuple(inspect.signature(mixed_mala).parameters)[4:] == (
        "dt",
        "steps",
        "image_radius",
        "chunks",
    )
    assert tuple(inspect.signature(sequential_monte_carlo).parameters)[4:7] == (
        "ladder",
        "mc_dt",
        "mc_steps",
    )
    assert "mc_image_radius" in inspect.signature(
        mixed_quench_and_temper
    ).parameters
    assert "mc_image_radius" in inspect.signature(
        annealed_importance_sampling
    ).parameters

    domain = Mixed_Domain(2, 1)
    source = Toy_Mixed_Potential(domain, [0.0, 0.0, 0.0])
    target = Toy_Mixed_Potential(domain, [0.3, -0.2, 0.6])
    initial = domain.wrap(jax.random.normal(jax.random.key(10), (12, 3)))

    single_key = jax.random.key(101)
    single_result = mixed_mala(
        single_key, initial, target, domain, dt=1e-2, steps=2, chunks=1
    )
    single_manual = _mixed_mala_chunk(
        single_key,
        initial,
        target,
        domain,
        mc_dt=1e-2,
        mc_steps=2,
        image_radius=3,
    )
    jax.block_until_ready((single_result, single_manual))
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(single_result, single_manual)
    )

    key = jax.random.key(11)
    result, acceptance = mixed_mala(
        key, initial, target, domain, dt=1e-2, steps=2, chunks=3
    )
    parts = jnp.array_split(initial, 3, axis=0)
    keys = jax.random.split(key, 3)
    manual = [
        _mixed_mala_chunk(
            part_key,
            part,
            target,
            domain,
            mc_dt=1e-2,
            mc_steps=2,
            image_radius=3,
        )
        for part_key, part in zip(keys, parts)
    ]
    manual_result = jnp.concatenate([item[0] for item in manual], axis=0)
    manual_acceptance = sum(item[1] / 3 for item in manual)
    jax.block_until_ready((manual_result, manual_acceptance))
    assert bool(jnp.array_equal(result, manual_result))
    assert bool(jnp.array_equal(acceptance, manual_acceptance))

    uniform = sequential_monte_carlo(
        jax.random.key(12),
        initial,
        source,
        target,
        ladder=2,
        mc_dt=1e-2,
        mc_steps=1,
        chunks=3,
    )
    explicit = potential_space_smc(
        jax.random.key(12),
        initial,
        source,
        target,
        t_list=(0.5, 1.0),
        mc_dt=1e-2,
        mc_steps=1,
        chunks=3,
    )
    jax.block_until_ready((uniform, explicit))
    assert all(bool(jnp.array_equal(left, right)) for left, right in zip(uniform, explicit))

    identity = Mixed_Identity(domain)
    ais_key = jax.random.key(13)
    ais = annealed_importance_sampling(
        ais_key,
        initial,
        source,
        target,
        identity,
        ladder=2,
        mc_dt=1e-2,
        mc_steps=1,
        chunks=3,
    )
    assert ais.shape == initial.shape and bool(jnp.isfinite(ais).all())
    manual_ais = identity.inv(initial)
    for level in range(1, 3):
        latent, ladj_g = identity.call_and_ladj(manual_ais)
        log_weight = (
            -target(manual_ais) + source(latent) - ladj_g
        ) / 2
        weight = jnp.exp(log_weight - log_weight.max())
        resample_key, mala_key = jax.random.split(
            jax.random.fold_in(ais_key, level)
        )
        manual_ais = resample(
            resample_key, manual_ais, weight, N=manual_ais.shape[0]
        )
        # Score-free AIS intentionally always rejuvenates at the target.
        manual_ais = mixed_mala(
            mala_key,
            manual_ais,
            target,
            domain,
            dt=1e-2,
            steps=1,
            chunks=3,
        )[0]
    assert bool(jnp.array_equal(ais, manual_ais))

    classical_smc = sequential_monte_carlo(
        ais_key,
        initial,
        source,
        target,
        ladder=2,
        mc_dt=1e-2,
        mc_steps=1,
        chunks=3,
    )[0]
    assert float(jnp.max(jnp.abs(ais - classical_smc))) > 1e-6

    for bad_chunk in (0, initial.shape[0] + 1):
        try:
            mixed_mala(
                jax.random.key(14),
                initial,
                target,
                domain,
                dt=1e-2,
                chunks=bad_chunk,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"chunks={bad_chunk} must fail")
    print("PASS jflows_md eager SMC and score-free target-rejuvenated AIS")


if __name__ == "__main__":
    main()
