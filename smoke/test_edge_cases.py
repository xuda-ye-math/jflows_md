#!/usr/bin/env python
"""Raw mixed-domain sampler equation checks."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402
from jflows_md.utils import (  # noqa: E402
    mixed_mala_step,
    potential_space_smc,
    sequential_monte_carlo,
    wrapped_normal_relative_error_bound,
)


class Shifted(Potential):
    domain: Mixed_Domain
    shift: jax.Array

    def __init__(self, domain, shift):
        self.domain = domain
        self.shift = jnp.asarray(shift)

    def __call__(self, samples):
        delta = self.domain.displacement(samples, self.shift)
        return 0.5 * jnp.sum(delta**2, axis=-1)


def main() -> None:
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Shifted(domain, [0.2, -0.1, 0.4])
    samples = source.samples(jax.random.key(1), 12)
    stepped, accepted = mixed_mala_step(
        jax.random.key(2), samples, target, domain, dt=1e-4, image_radius=3
    )
    assert stepped.shape == samples.shape and accepted.shape == (12,)
    assert wrapped_normal_relative_error_bound(1e-4, 3) < 1e-12
    smc, ess, acceptance = sequential_monte_carlo(
        jax.random.key(3),
        samples,
        source,
        target,
        ladder=2,
        mc_dt=1e-4,
        mc_steps=1,
        domain=domain,
        chunks=2,
    )
    explicit, explicit_ess, _ = potential_space_smc(
        jax.random.key(3),
        samples,
        source,
        target,
        (0.5, 1.0),
        mc_dt=1e-4,
        mc_steps=1,
        domain=domain,
        chunks=2,
    )
    assert smc.shape == explicit.shape == samples.shape
    assert ess.shape == explicit_ess.shape == (2,)
    assert acceptance.shape == (2, 1)
    print("PASS raw mixed-domain sampler equations")


if __name__ == "__main__":
    main()
