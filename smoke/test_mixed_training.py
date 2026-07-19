#!/usr/bin/env python
"""Tiny KLX/KLXX and sharpening-enabled Boltzmann smoke test."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.boltzmann import (  # noqa: E402
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_NSF  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402
from jflows_md.train import train_forward_KLXX_G  # noqa: E402


class Toy_Regularized(Potential):
    domain: Mixed_Domain
    center: jax.Array
    rg_param: jax.Array

    def __init__(self, domain, center, rg_param):
        self.domain = domain
        self.center = center
        self.rg_param = jnp.asarray(rg_param)

    def __call__(self, x):
        delta = self.domain.displacement(x, self.center)
        radial = 0.5 * jnp.sum(delta**2, axis=-1)
        return (1.0 + 0.01 * self.rg_param[0]) * radial + self.rg_param[1] * (
            1.0 - jnp.cos(x[:, -1])
        )


class Toy_Molecular_Target(Potential):
    domain: Mixed_Domain
    center: jax.Array

    def __init__(self, domain):
        self.domain = domain
        self.center = jnp.asarray([0.2, -0.1, 0.4])

    def __call__(self, x):
        return Toy_Regularized(self.domain, self.center, (0.0, 0.0))(x)

    def regularized(self, rg_param):
        return Toy_Regularized(self.domain, self.center, rg_param)


def _flow(domain, seed):
    return Mixed_NSF(
        jax.random.key(seed),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()


def _controls():
    return {
        "rg_param_0": (5.0, 0.2),
        "rg_param_1": (10.0, 0.1),
        "bg_param": {
            "t_safe": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 0.0,
            "max_stages": 1,
            "max_retry": 1,
        },
        "chunks": 1,
        "mc_image_radius": 3,
    }


def _check_stage(particles, stages):
    assert particles.shape == (16, 3)
    assert len(stages) == 1
    stage = stages[0]
    assert stage["t_start"] == 0.0 and stage["t"] == 1.0
    np.testing.assert_allclose(stage["rg_start"], (5.0, 0.2), rtol=0, atol=1e-7)
    np.testing.assert_allclose(stage["rg_end"], (10.0, 0.1), rtol=0, atol=1e-7)
    assert 0.0 < stage["sharpen_ess"] <= 1.0
    assert stage["sharpen_mala_acceptance"].shape == (1,)
    assert stage["batch_ess_hist"].shape == (1, 1)
    assert stage["sharpen_ess_hist"].shape == (1,)
    assert stage["valid_sample_count"] == 16
    assert stage["selected"] in ("trained", "identity")
    assert isinstance(stage["flow"], eqx.Module)
    assert stage["objective"] in ("forward_klx", "forward_klxx")
    assert bool(jnp.isfinite(particles).all())


def main() -> None:
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Molecular_Target(domain)
    samples = source.samples(jax.random.key(300), 16)

    hat = source.samples(jax.random.key(301), 16)
    trained, ess = train_forward_KLXX_G(
        samples,
        samples,
        hat,
        source,
        target.regularized((5.0, 0.2)),
        _flow(domain, 302),
        domain,
        4,
        1,
        1e-3,
        mc_steps=0,
        seed=303,
    )
    jax.block_until_ready((trained, ess))
    assert ess.shape == (1,)

    particles, stages = boltzmann_forward_KLX_G(
        samples,
        source,
        target,
        _flow(domain, 304),
        8,
        4,
        1,
        1e-3,
        1,
        1e-3,
        1,
        seed=305,
        **_controls(),
    )
    _check_stage(particles, stages)

    particles, stages = boltzmann_forward_KLXX_G(
        samples,
        source,
        target,
        _flow(domain, 306),
        8,
        4,
        1,
        1e-3,
        1,
        0.0,
        1e-2,
        0,
        1e-3,
        1,
        seed=307,
        **_controls(),
    )
    _check_stage(particles, stages)
    print("PASS molecular KLX/KLXX linear sharpening")


if __name__ == "__main__":
    main()
