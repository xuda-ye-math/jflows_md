#!/usr/bin/env python
"""Identity-only molecular Boltzmann computation and sharpening smoke test."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
import jflows_md  # noqa: E402
import jflows_md.boltzmann as boltzmann  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402


class Regularized_Quadratic(Potential):
    domain: Mixed_Domain
    scale: jax.Array

    def __init__(self, domain, rg_param):
        self.domain = domain
        self.scale = jnp.asarray(1.0 + rg_param[0])

    def __call__(self, samples):
        return 0.5 * self.scale * jnp.sum(samples**2, axis=-1)


class Sharpening_Target(Potential):
    domain: Mixed_Domain

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, samples):
        return 0.5 * jnp.sum(samples**2, axis=-1)

    def regularized(self, rg_param):
        return Regularized_Quadratic(self.domain, rg_param)


STAGE_KEYS = {
    "t",
    "t_start",
    "rg_start",
    "rg_end",
    "population_rg",
    "valid_selected_ess",
    "valid_identity_ess",
    "valid_sample_count",
    "selected",
    "t_hist",
    "valid_identity_ess_hist",
    "sharpen_ess_hist",
    "attempt_status_hist",
    "selection_history",
    "smc_ess",
    "smc_acceptance",
    "mala_acceptance",
    "sharpen_ess",
    "sharpen_mala_acceptance",
    "objective",
    "elapsed_seconds",
}


def main() -> None:
    assert jflows_md.boltzmann_identity is boltzmann.boltzmann_identity
    parameters = inspect.signature(boltzmann.boltzmann_identity).parameters
    assert tuple(parameters) == (
        "x_valid",
        "source",
        "target",
        "ladder",
        "mc_dt",
        "mc_steps",
        "rg_param_0",
        "rg_param_1",
        "monitor",
        "bg_param",
        "chunks",
        "mc_image_radius",
        "seed",
    )
    assert not {
        "flow",
        "pool_size",
        "batch_size",
        "train_steps",
        "lr",
        "checkpoint",
        "initialize_from_identity",
    } & set(parameters)

    domain = Mixed_Domain(2, 0)
    source = Molecular_Source(domain)
    target = Sharpening_Target(domain)
    samples = source.samples(jax.random.key(10), 512)
    controls = {
        "ladder": 2,
        "mc_dt": 1e-3,
        "mc_steps": 1,
        "rg_param_0": (0.0, 0.0),
        "rg_param_1": (3.0, 0.0),
        "bg_param": {
            "t_safe": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 0.0,
            "max_stages": 1,
            "max_retry": 1,
        },
        "chunks": 4,
        "seed": 11,
    }

    trainer = boltzmann._training_attempt
    mala = boltzmann.mixed_mala
    mala_energies = []

    def unexpected_training(*args, **kwargs):
        del args, kwargs
        raise AssertionError("boltzmann_identity called a flow trainer")

    def observed_mala(*args, **kwargs):
        probe = jnp.ones((1, 2))
        mala_energies.append(float(args[2](probe)[0]))
        return mala(*args, **kwargs)

    boltzmann._training_attempt = unexpected_training
    boltzmann.mixed_mala = observed_mala
    try:
        population, stages = boltzmann.boltzmann_identity(
            samples, source, target, **controls
        )
    finally:
        boltzmann._training_attempt = trainer
        boltzmann.mixed_mala = mala

    repeated, repeated_stages = boltzmann.boltzmann_identity(
        samples, source, target, **controls
    )
    assert bool(jnp.array_equal(population, repeated))
    assert [stage["t"] for stage in repeated_stages] == [1.0]
    assert jnp.allclose(jnp.asarray(mala_energies), jnp.asarray((1.0, 4.0)))
    assert len(stages) == 1
    stage = stages[0]
    assert set(stage) == STAGE_KEYS
    assert stage["objective"] == "identity"
    assert stage["selected"] == "identity"
    assert stage["valid_sample_count"] == samples.shape[0]
    assert stage["valid_selected_ess"] == stage["valid_identity_ess"]
    assert stage["valid_identity_ess"] > 0.999
    assert 0.0 < stage["sharpen_ess"] < 0.99
    assert stage["rg_start"] == (0.0, 0.0)
    assert stage["rg_end"] == (3.0, 0.0)
    assert stage["population_rg"] == stage["rg_end"]
    assert stage["sharpen_ess_hist"].shape == (1,)
    assert stage["mala_acceptance"].shape == (1,)
    assert stage["sharpen_mala_acceptance"].shape == (1,)
    assert not any("flow" in key or "trained" in key for key in stage)
    assert bool(jnp.isfinite(population).all())
    print("PASS molecular identity Boltzmann computation with sharpening")


if __name__ == "__main__":
    main()
