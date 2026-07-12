#!/usr/bin/env python
"""Tiny synthetic smoke test for mixed-domain KL+X BG training."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.boltzmann import (  # noqa: E402
    _operation_key,
    molecular_boltzmann_forward_KLX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_NSF  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402
from jflows_md.utils import annealed_importance_sampling  # noqa: E402


class Toy_Molecular_Target(Potential):
    """Smooth shifted Gaussian/von-Mises target with molecular metadata."""

    domain: Mixed_Domain
    center: jax.Array

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain
        self.center = jnp.asarray([0.2, -0.1, 0.4])

    def __call__(self, x):
        euclidean = x[:, :2] - self.center[:2]
        torsion = x[:, 2] - self.center[2]
        return 0.5 * jnp.sum(euclidean**2, axis=-1) - 0.2 * jnp.cos(torsion)

    def reference_internal(self):
        return self.center


def main() -> None:
    base_key = jax.random.key(299)
    operation_keys = [
        _operation_key(base_key, namespace, stage, attempt)
        for namespace in (1, 2, 3)
        for stage in (1, 2)
        for attempt in (1, 2)
    ]
    key_words = {
        tuple(map(int, jax.random.key_data(key))) for key in operation_keys
    }
    assert len(key_words) == len(operation_keys)

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Molecular_Target(domain)
    x_valid = source.samples(jax.random.key(300), 32)
    flow = Mixed_NSF(
        jax.random.key(301),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()

    particles, stages = molecular_boltzmann_forward_KLX_G(
        x_valid,
        source,
        target,
        flow,
        n_pool=16,
        n_batch=8,
        steps=2,
        lr=1e-3,
        ladder=2,
        mc_step=1e-3,
        mc_iters=1,
        coeff_lambda=1.0,
        bg_param={
            "t_safe": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 0.0,
            "max_stages": 1,
            "max_retry": 1,
        },
        e_clip=1000.0,
        g_clip=100.0,
        seed=302,
    )
    jax.block_until_ready(particles)

    assert particles.shape == x_valid.shape
    assert bool(jnp.isfinite(particles).all())
    assert len(stages) == 1 and stages[0]["t"] == 1.0
    assert stages[0]["ess_samples"] == x_valid.shape[0]
    assert 0.0 < stages[0]["ess"] <= 1.0
    assert 0.0 < stages[0]["trained_ess"] <= 1.0
    assert 0.0 < stages[0]["identity_ess"] <= 1.0
    assert stages[0]["selected"] in ("trained", "identity")
    assert stages[0]["ess_history"].shape == (2,)
    assert stages[0]["kept_history"].shape == (2,)
    assert stages[0]["update_history"].shape == (2,)
    assert stages[0]["smc_ess"].shape == (2,)
    assert stages[0]["smc_acceptance"].shape == (2, 1)
    assert stages[0]["mala_acceptance"].shape == (1,)
    assert bool(
        jnp.isfinite(stages[0]["ess_history"]).all()
        & jnp.isfinite(stages[0]["kept_history"]).all()
    )
    leaves = eqx.filter(stages[0]["flow"], eqx.is_inexact_array)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(leaves))

    ais_flow = Mixed_NSF(
        jax.random.key(303),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    )
    ais_samples = annealed_importance_sampling(
        jax.random.key(304),
        x_valid,
        source,
        target,
        ais_flow,
        ladder=2,
        step=1e-3,
        iters=1,
        chunk=2,
    )
    jax.block_until_ready(ais_samples)
    assert ais_samples.shape == x_valid.shape
    assert bool(jnp.isfinite(ais_samples).all())
    assert bool(
        jnp.all((ais_samples[:, 2] >= -jnp.pi) & (ais_samples[:, 2] < jnp.pi))
    )
    print(
        "PASS mixed BG training and G-native score-free AIS: "
        f"stage_ESS={stages[0]['ess']:.3f} N={stages[0]['ess_samples']}"
    )


if __name__ == "__main__":
    main()
