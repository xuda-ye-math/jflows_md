#!/usr/bin/env python
"""Stereochemical-support and mixed-domain sampling-utility smoke tests."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential, potential_from  # noqa: E402
from jflows_md import Mixed_Identity, Molecular_Potential  # noqa: E402
from jflows_md.core.chirality import signed_volume  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.system import Molecular_Bundle  # noqa: E402
from jflows_md.utils import mixed_mala, sequential_monte_carlo  # noqa: E402


NAMES = (
    "adp_ff96_obc1",
    "glycerol_gaff2_am1bcc_obc1",
    "diethanolamine_gaff2_am1bcc_obc1",
)


class Toy_Mixed_Potential(Potential):
    domain: Mixed_Domain
    shift: jax.Array

    def __init__(self, domain: Mixed_Domain, shift):
        self.domain = domain
        self.shift = jnp.asarray(shift)

    def __call__(self, x):
        euclidean = (
            x[:, : self.domain.euclidean_dim]
            - self.shift[: self.domain.euclidean_dim]
        )
        torsion = (
            x[:, self.domain.euclidean_dim :]
            - self.shift[self.domain.euclidean_dim :]
        )
        return 0.5 * jnp.sum(euclidean**2, axis=-1) - jnp.sum(
            jnp.cos(torsion), axis=-1
        )


def main() -> None:
    for index, name in enumerate(NAMES):
        bundle = Molecular_Bundle.load(name)
        potential = Molecular_Potential.from_bundle(name)
        source = potential.source()
        q = source.samples(jax.random.key(100 + index), 512)
        x = potential.cartesian(q)
        diagnostic_atoms = bundle.coordinates["diagnostic_chirality_atoms"]
        volume = signed_volume(x, *diagnostic_atoms)
        if name == "adp_ff96_obc1":
            assert bool(jnp.all(volume > 0))
            assert bool(jnp.all(potential.support_mask(x)))
        else:
            assert bool(jnp.any(volume > 0) & jnp.any(volume < 0))
            assert bool(jnp.all(potential.support_mask(x)))

        reference = potential.cartesian(potential.reference_internal()[None])
        mirrored = reference.at[..., 0].multiply(-1.0)
        parity_error = float(
            jnp.max(
                jnp.abs(
                    potential.forcefield(reference)
                    - potential.forcefield(mirrored)
                )
            )
        )
        assert parity_error < 1e-4, (name, parity_error)
        if name == "adp_ff96_obc1":
            assert not bool(potential.support_mask(mirrored)[0])
        else:
            assert bool(potential.support_mask(mirrored)[0])
        print(f"PASS support {name}: parity_error={parity_error:.2e}")

    domain = Mixed_Domain(2, 1)
    target = potential_from(
        lambda x: 0.5 * jnp.sum(x[:, :2] ** 2, axis=-1)
        - 2.0 * jnp.cos(x[:, 2])
    )
    initial = domain.wrap(jax.random.normal(jax.random.key(90), (64, 3)))
    identity = Mixed_Identity(domain)
    roundtrip = domain.displacement(identity.inv(identity(initial)), initial)
    assert float(jnp.max(jnp.abs(roundtrip))) == 0
    result, acceptance = mixed_mala(
        jax.random.key(91),
        initial,
        target,
        domain,
        step=1e-2,
        iters=3,
        chunk=2,
    )
    jax.block_until_ready((result, acceptance))
    assert result.shape == initial.shape and acceptance.shape == (3,)
    assert bool(jnp.isfinite(result).all() & jnp.isfinite(acceptance).all())
    assert bool(jnp.all((result[:, 2] >= -jnp.pi) & (result[:, 2] < jnp.pi)))
    assert bool(jnp.all((acceptance >= 0) & (acceptance <= 1)))
    print(f"PASS mixed MALA: mean_acceptance={float(acceptance.mean()):.3f}")

    source = Toy_Mixed_Potential(domain, [0.0, 0.0, 0.0])
    target = Toy_Mixed_Potential(domain, [0.4, -0.2, 0.7])
    particles, ess, smc_acceptance = sequential_monte_carlo(
        jax.random.key(92),
        initial,
        source,
        target,
        ladder=2,
        step=1e-2,
        iters=2,
    )
    jax.block_until_ready((particles, ess, smc_acceptance))
    assert particles.shape == initial.shape
    assert ess.shape == (2,) and smc_acceptance.shape == (2, 2)
    assert bool(jnp.isfinite(particles).all() & jnp.isfinite(ess).all())
    assert bool(jnp.all((ess > 0) & (ess <= 1)))
    assert bool(jnp.all((particles[:, 2] >= -jnp.pi) & (particles[:, 2] < jnp.pi)))
    print(f"PASS potential-space SMC: min_ESS={float(ess.min()):.3f}")


if __name__ == "__main__":
    main()
