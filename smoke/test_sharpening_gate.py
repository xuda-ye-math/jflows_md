#!/usr/bin/env python
"""Combined flow and sharpening ESS acceptance gate."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
import jflows_md.boltzmann as boltzmann  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_Identity  # noqa: E402
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


def _run(samples, source, target, flow, *, max_retry, tau_ess=None):
    policy = {
        "t_safe": 1.0,
        "shrink_factor": 0.7,
        "tau_smc": 0.0,
        "max_stages": 1,
        "max_retry": max_retry,
    }
    if tau_ess is not None:
        policy["tau_ess"] = tau_ess
    return boltzmann.boltzmann_forward_KLX_G(
        samples,
        source,
        target,
        flow,
        0,
        32,
        1,
        1e-3,
        1,
        1e-3,
        0,
        rg_param_0=(0.0, 0.0),
        rg_param_1=(100.0, 0.0),
        bg_param=policy,
        chunks=4,
        seed=11,
    )


def main() -> None:
    domain = Mixed_Domain(2, 0)
    source = Molecular_Source(domain)
    target = Sharpening_Target(domain)
    samples = source.samples(jax.random.key(10), 4096)
    flow = Mixed_Identity(domain)

    trainer = boltzmann.train_forward_KLX_G
    ess_function = boltzmann.compute_ESS_log
    boltzmann.train_forward_KLX_G = lambda *args, **kwargs: (
        args[4],
        jnp.ones((1,)),
    )
    try:
        unchanged, failed = _run(
            samples, source, target, flow, max_retry=1
        )
        default_particles, default_stages = _run(
            samples, source, target, flow, max_retry=10
        )
        _, relaxed_stages = _run(
            samples, source, target, flow, max_retry=10, tau_ess=0.45
        )
        boltzmann.compute_ESS_log = lambda _: jnp.asarray(jnp.nan)
        invalid_validation_particles, invalid_validation_stages = _run(
            samples, source, target, flow, max_retry=1
        )
        recovery_values = iter((jnp.nan, jnp.nan, 1.0, 1.0, 1.0))
        boltzmann.compute_ESS_log = lambda _: jnp.asarray(
            next(recovery_values)
        )
        _, recovery_stages = _run(
            samples, source, target, flow, max_retry=2
        )
        sharpening_values = iter((1.0, 1.0, jnp.nan))
        boltzmann.compute_ESS_log = lambda _: jnp.asarray(
            next(sharpening_values)
        )
        invalid_sharpen_particles, invalid_sharpen_stages = _run(
            samples, source, target, flow, max_retry=1
        )
    finally:
        boltzmann.train_forward_KLX_G = trainer
        boltzmann.compute_ESS_log = ess_function

    assert failed == []
    assert bool(jnp.array_equal(unchanged, samples))

    assert len(default_stages) == 1
    stage = default_stages[0]
    assert stage["t_hist"][0] == 1.0
    assert stage["t"] < 1.0
    assert stage["valid_identity_ess_hist"][0] > 0.999
    assert stage["attempt_status_hist"][0] == "rejected"
    assert stage["attempt_status_hist"][-1] == "accepted"
    assert stage["sharpen_ess_hist"].shape == stage["t_hist"].shape
    assert stage["sharpen_ess_hist"][0] < 0.6
    assert stage["sharpen_ess_hist"][-1] == stage["sharpen_ess"]
    assert stage["sharpen_ess"] >= 0.6
    assert bool(jnp.isfinite(default_particles).all())

    assert len(relaxed_stages) == 1
    relaxed = relaxed_stages[0]
    assert relaxed["t"] > stage["t"]
    assert 0.45 <= relaxed["sharpen_ess"] < 0.6
    assert invalid_validation_stages == []
    assert invalid_sharpen_stages == []
    assert bool(jnp.array_equal(invalid_validation_particles, samples))
    assert bool(jnp.array_equal(invalid_sharpen_particles, samples))
    assert len(recovery_stages) == 1
    recovery = recovery_stages[0]
    assert recovery["attempt_status_hist"] == ("rejected", "accepted")
    assert bool(jnp.isnan(recovery["sharpen_ess_hist"][0]))
    assert recovery["sharpen_ess_hist"][1] == 1.0
    print("PASS combined flow and sharpening ESS gate")


if __name__ == "__main__":
    main()
