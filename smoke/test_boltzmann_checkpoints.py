#!/usr/bin/env python
"""Focused molecular stage-checkpoint controller regressions."""

from __future__ import annotations

from contextlib import contextmanager
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
import jflows_md.boltzmann as bg  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import (  # noqa: E402
    train_molecular_forward_KLX_G,
    train_molecular_forward_KLXX_G,
)


class Probe_Potential(Potential):
    domain: Mixed_Domain

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum(value**2, axis=-1)

    def reference_internal(self):
        return jnp.zeros((1,))


class Probe_Flow(eqx.Module):
    shift: jax.Array

    def zeros(self):
        return Probe_Flow(jnp.zeros_like(self.shift))

    def inv(self, value):
        return value + self.shift

    def inv_and_ladj(self, value):
        return self.inv(value), jnp.zeros(value.shape[0], dtype=value.dtype)


@contextmanager
def patched_driver(*, trainer, identity_weights, importance_weights=None):
    originals = {
        "train_molecular_forward_KLX_G": bg.train_molecular_forward_KLX_G,
        "sequential_monte_carlo": bg.sequential_monte_carlo,
        "mixed_mala": bg.mixed_mala,
        "_importance_weights_g": bg._importance_weights_g,
        "_chunked_identity_weights": bg._chunked_identity_weights,
        "_chunked_inverse": bg._chunked_inverse,
    }

    def smc(key, samples, source, target, *, ladder, iters, **kwargs):
        del key, source, target, kwargs
        return (
            samples,
            jnp.ones((ladder,)),
            jnp.ones((ladder, iters)),
        )

    def mala(key, samples, target, domain, *, iters, **kwargs):
        del key, target, domain, kwargs
        return samples, jnp.ones((iters,))

    if importance_weights is None:
        def importance_weights(samples, source, target, flow, chunk):
            del source, target, flow, chunk
            return jnp.zeros((samples.shape[0],), dtype=samples.dtype)

    try:
        bg.train_molecular_forward_KLX_G = trainer
        bg.sequential_monte_carlo = smc
        bg.mixed_mala = mala
        bg._importance_weights_g = importance_weights
        bg._chunked_identity_weights = identity_weights
        bg._chunked_inverse = lambda flow, samples, chunk: flow.inv(samples)
        yield
    finally:
        for name, value in originals.items():
            setattr(bg, name, value)


def common_inputs():
    domain = Mixed_Domain(1, 0)
    potential = Probe_Potential(domain)
    samples = jnp.linspace(-1.0, 1.0, 8)[:, None]
    flow = Probe_Flow(jnp.zeros((1,)))
    return samples, potential, flow


def expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_legacy_warm_start_and_tie() -> None:
    samples, potential, flow = common_inputs()
    seen = []

    def trainer(pool, previous, current, initial, n_batch, steps, lr, **kwargs):
        del pool, previous, current, n_batch, lr, kwargs
        seen.append(float(initial.shift[0]))
        trained = Probe_Flow(initial.shift + 1.0)
        return (
            trained,
            jnp.ones((steps,)),
            jnp.ones((steps,)),
            jnp.ones((steps,), dtype=bool),
        )

    def identity_weights(samples, source, target, chunk):
        del source, target, chunk
        return jnp.zeros((samples.shape[0],), dtype=samples.dtype)

    with patched_driver(trainer=trainer, identity_weights=identity_weights):
        _, stages = bg.molecular_boltzmann_forward_KLX_G(
            samples,
            potential,
            potential,
            flow,
            n_pool=4,
            n_batch=2,
            steps=1,
            lr=1e-3,
            ladder=1,
            mc_step=1e-3,
            mc_iters=1,
            selection_steps=(),
            bg_param={
                "t_safe": 0.4,
                "enlarge_factor": 1.5,
                "tau_smc": 0.0,
                "tau_ess": 0.0,
                "max_stages": 2,
                "max_retry": 1,
            },
        )

    assert seen == [0.0, 1.0], seen
    assert len(stages) == 2 and stages[-1]["t"] == 1.0
    assert all(stage["selected_checkpoint"] == "final" for stage in stages)
    assert all(stage["selected_step"] == 1 for stage in stages)
    assert all(
        tuple(stage["checkpoint_labels"]) == ("identity", "final")
        for stage in stages
    )


def test_finite_snapshot_rescues_nonfinite_final() -> None:
    samples, potential, flow = common_inputs()

    def trainer(pool, previous, current, initial, n_batch, steps, lr, **kwargs):
        del pool, previous, current, initial, n_batch, lr, kwargs
        snapshot = Probe_Flow(jnp.ones((1,)))
        final = Probe_Flow(jnp.full((1,), jnp.nan))
        return (
            final,
            jnp.ones((steps,)),
            jnp.ones((steps,)),
            jnp.ones((steps,), dtype=bool),
            (snapshot,),
        )

    def identity_weights(samples, source, target, chunk):
        del source, target, chunk
        values = jnp.full((samples.shape[0],), -10.0, dtype=samples.dtype)
        return values.at[0].set(0.0)

    def importance_weights(samples, source, target, flow, chunk):
        del source, target, chunk
        scale = jnp.abs(flow.shift[0] - 1.0)
        return -scale * jnp.arange(samples.shape[0], dtype=samples.dtype)

    with patched_driver(
        trainer=trainer,
        identity_weights=identity_weights,
        importance_weights=importance_weights,
    ):
        _, stages = bg.molecular_boltzmann_forward_KLX_G(
            samples,
            potential,
            potential,
            flow,
            n_pool=4,
            n_batch=2,
            steps=2,
            lr=1e-3,
            ladder=1,
            mc_step=1e-3,
            mc_iters=1,
            selection_steps=(1,),
            bg_param={
                "t_safe": 1.0,
                "tau_smc": 0.0,
                "tau_ess": 0.0,
                "max_stages": 1,
                "max_retry": 1,
            },
        )

    stage = stages[0]
    assert tuple(stage["checkpoint_labels"]) == (
        "identity",
        "warm_start",
        "checkpoint",
        "final",
    )
    assert tuple(map(int, stage["checkpoint_steps"])) == (-1, 0, 1, 2)
    assert stage["selected_checkpoint"] == "checkpoint"
    assert stage["selected_step"] == 1
    assert stage["final_ess"] == 0.0
    assert stage["ess"] > max(
        float(stage["checkpoint_ess"][0]),
        float(stage["checkpoint_ess"][1]),
        float(stage["checkpoint_ess"][3]),
    )
    assert bool(jnp.isfinite(stage["flow"].shift).all())


def test_warm_start_uniquely_wins() -> None:
    samples, potential, _ = common_inputs()
    flow = Probe_Flow(jnp.full((1,), 2.0))

    def trainer(pool, previous, current, initial, n_batch, steps, lr, **kwargs):
        del pool, previous, current, initial, n_batch, lr, kwargs
        snapshot = Probe_Flow(jnp.ones((1,)))
        final = Probe_Flow(jnp.full((1,), 3.0))
        return (
            final,
            jnp.ones((steps,)),
            jnp.ones((steps,)),
            jnp.ones((steps,), dtype=bool),
            (snapshot,),
        )

    def identity_weights(samples, source, target, chunk):
        del source, target, chunk
        values = jnp.full((samples.shape[0],), -10.0, dtype=samples.dtype)
        return values.at[0].set(0.0)

    def importance_weights(samples, source, target, candidate, chunk):
        del source, target, chunk
        scale = jnp.abs(candidate.shift[0] - 2.0)
        return -scale * jnp.arange(samples.shape[0], dtype=samples.dtype)

    with patched_driver(
        trainer=trainer,
        identity_weights=identity_weights,
        importance_weights=importance_weights,
    ):
        _, stages = bg.molecular_boltzmann_forward_KLX_G(
            samples,
            potential,
            potential,
            flow,
            n_pool=4,
            n_batch=2,
            steps=2,
            lr=1e-3,
            ladder=1,
            mc_step=1e-3,
            mc_iters=1,
            selection_steps=(1,),
            bg_param={
                "t_safe": 1.0,
                "tau_smc": 0.0,
                "tau_ess": 0.0,
                "max_stages": 1,
                "max_retry": 1,
            },
        )

    stage = stages[0]
    assert stage["selected_checkpoint"] == "warm_start"
    assert stage["selected_step"] == 0
    assert float(stage["flow"].shift[0]) == 2.0
    assert stage["ess"] > max(
        float(stage["checkpoint_ess"][0]),
        float(stage["checkpoint_ess"][2]),
        float(stage["checkpoint_ess"][3]),
    )


def test_sparse_checkpoint_limit() -> None:
    samples, potential, flow = common_inputs()
    too_many = tuple(range(1, 34))
    expect_value_error(
        lambda: train_molecular_forward_KLX_G(
            samples,
            potential,
            potential,
            flow,
            n_batch=2,
            steps=33,
            lr=1e-3,
            snapshot_steps=too_many,
        )
    )
    expect_value_error(
        lambda: train_molecular_forward_KLXX_G(
            samples,
            samples,
            samples,
            potential,
            potential,
            flow,
            potential.domain,
            n_batch=2,
            steps=33,
            lr=1e-3,
            snapshot_steps=too_many,
        )
    )
    expect_value_error(
        lambda: bg.molecular_boltzmann_forward_KLX_G(
            samples,
            potential,
            potential,
            flow,
            n_pool=4,
            n_batch=2,
            steps=33,
            lr=1e-3,
            ladder=1,
            mc_step=1e-3,
            mc_iters=1,
            selection_steps=too_many,
        )
    )


def main() -> None:
    test_legacy_warm_start_and_tie()
    test_finite_snapshot_rescues_nonfinite_final()
    test_warm_start_uniquely_wins()
    test_sparse_checkpoint_limit()
    print("PASS molecular checkpoint warm-start and nonfinite-final regressions")


if __name__ == "__main__":
    main()
