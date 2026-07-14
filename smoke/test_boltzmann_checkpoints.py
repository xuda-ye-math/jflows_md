#!/usr/bin/env python
"""Focused molecular final-versus-identity controller regressions."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
import jflows_md.boltzmann as bg  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402


class Probe_Potential(Potential):
    domain: Mixed_Domain

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum(value**2, axis=-1)

    def reference_internal(self):
        return jnp.zeros((1,))

    def samples(self, key, sample_count):
        del key
        return (10.0 + jnp.arange(sample_count))[:, None]


class Probe_Flow(eqx.Module):
    shift: jax.Array

    def zeros(self):
        return Probe_Flow(jnp.zeros_like(self.shift))

    def inv(self, value):
        return value + self.shift

    def inv_and_ladj(self, value):
        return self.inv(value), jnp.zeros(value.shape[0], dtype=value.dtype)


@contextmanager
def patched_driver(
    *,
    trainer=None,
    klxx_trainer=None,
    identity_weights,
    importance_weights=None,
    smc_sampler=None,
    qt_sampler=None,
):
    originals = {
        "train_forward_KLX_G": bg.train_forward_KLX_G,
        "train_forward_KLXX_G": bg.train_forward_KLXX_G,
        "sequential_monte_carlo": bg.sequential_monte_carlo,
        "mixed_mala": bg.mixed_mala,
        "mixed_quench_and_temper": bg.mixed_quench_and_temper,
        "_importance_weights_g": bg._importance_weights_g,
        "_chunked_identity_weights": bg._chunked_identity_weights,
        "_chunked_inverse": bg._chunked_inverse,
    }

    if smc_sampler is None:
        def smc_sampler(
            key, samples, source, target, *, ladder, mc_steps, **kwargs
        ):
            del key, source, target, kwargs
            return samples, jnp.ones((ladder,)), jnp.ones((ladder, mc_steps))

    def mala(key, samples, target, domain, *, steps, **kwargs):
        del key, target, domain, kwargs
        return samples, jnp.ones((steps,))

    if importance_weights is None:
        def importance_weights(samples, source, target, flow, chunks):
            del source, target, flow, chunks
            return jnp.zeros((samples.shape[0],), dtype=samples.dtype)

    try:
        if trainer is not None:
            bg.train_forward_KLX_G = trainer
        if klxx_trainer is not None:
            bg.train_forward_KLXX_G = klxx_trainer
        if qt_sampler is not None:
            bg.mixed_quench_and_temper = qt_sampler
        bg.sequential_monte_carlo = smc_sampler
        bg.mixed_mala = mala
        bg._importance_weights_g = importance_weights
        bg._chunked_identity_weights = identity_weights
        bg._chunked_inverse = lambda flow, samples, chunks: flow.inv(samples)
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


def controls(**overrides):
    values = {
        "t_safe": 1.0,
        "tau_smc": 0.0,
        "tau_ess": 0.0,
        "max_stages": 1,
        "max_retry": 1,
    }
    values.update(overrides)
    return values


def run_driver(samples, potential, flow, **kwargs):
    return bg.boltzmann_forward_KLX_G(
        samples,
        potential,
        potential,
        flow,
        pool_size=kwargs.pop("pool_size", 4),
        batch_size=kwargs.pop("batch_size", 2),
        train_steps=kwargs.pop("train_steps", 1),
        lr=1e-3,
        ladder=1,
        mc_dt=1e-3,
        mc_steps=1,
        **kwargs,
    )


def zero_identity_weights(samples, source, target, chunks):
    del source, target, chunks
    return jnp.zeros((samples.shape[0],), dtype=samples.dtype)


def accepting_trainer(
    target_pool,
    source_pool,
    previous,
    current,
    initial,
    batch_size,
    train_steps,
    lr,
    **kwargs,
):
    del target_pool, source_pool, previous, current, batch_size, lr, kwargs
    return (
        initial,
        jnp.ones((train_steps,)),
        jnp.ones((train_steps,)),
        jnp.ones((train_steps,), dtype=bool),
    )


def test_full_pool_and_positive_pool_paths() -> None:
    samples, potential, flow = common_inputs()
    captured = []
    lines = []

    def smc_sampler(key, pool, source, target, *, ladder, mc_steps, **kwargs):
        del key, source, target, kwargs
        captured.append(pool)
        return pool, jnp.ones((ladder,)), jnp.ones((ladder, mc_steps))

    with patched_driver(
        trainer=accepting_trainer,
        identity_weights=zero_identity_weights,
        smc_sampler=smc_sampler,
    ):
        _, full_stages = run_driver(
            samples,
            potential,
            flow,
            pool_size=0,
            monitor=SimpleNamespace(printer=lines.append),
            bg_param=controls(),
        )

    assert bool(jnp.array_equal(captured.pop(), samples))
    assert full_stages[0]["selection_pool_mode"] == "full"
    assert full_stages[0]["selection_pool_size"] == samples.shape[0]
    assert any("pool=full N=8" in line for line in lines)

    with patched_driver(
        trainer=accepting_trainer,
        identity_weights=zero_identity_weights,
        smc_sampler=smc_sampler,
    ):
        _, indexed_stages = run_driver(
            samples,
            potential,
            flow,
            pool_size=4,
            bg_param=controls(),
        )

    base_key = jax.random.fold_in(jax.random.key(41), 0)
    selection_base = bg._operation_key(base_key, 1, 1)
    draw_key = jax.random.fold_in(selection_base, 1)
    expected_indices = jax.random.randint(
        draw_key, (4,), 0, samples.shape[0]
    )
    assert bool(jnp.array_equal(captured.pop(), samples[expected_indices]))
    assert indexed_stages[0]["selection_pool_mode"] == "with_replacement"
    assert indexed_stages[0]["selection_pool_size"] == 4


def test_full_pool_klxx_uses_fresh_qt_source_population() -> None:
    samples, potential, flow = common_inputs()
    captured = {}
    lines = []

    def qt_sampler(key, initial, current, domain, **kwargs):
        del key, current, domain
        captured["qt_initial"] = initial
        captured["qt_controls"] = kwargs
        return initial, jnp.ones((kwargs["mc_steps"],))

    def klxx_trainer(
        target_pool,
        source_pool,
        hat_pool,
        previous,
        current,
        initial,
        domain,
        batch_size,
        train_steps,
        lr,
        **kwargs,
    ):
        del previous, current, domain, batch_size, lr, kwargs
        captured["target_pool"] = target_pool
        captured["source_pool"] = source_pool
        captured["hat_pool"] = hat_pool
        return (
            initial,
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,), dtype=bool),
        )

    with patched_driver(
        klxx_trainer=klxx_trainer,
        identity_weights=zero_identity_weights,
        qt_sampler=qt_sampler,
    ):
        _, stages = bg.boltzmann_forward_KLXX_G(
            samples,
            potential,
            potential,
            flow,
            pool_size=0,
            batch_size=2,
            train_steps=1,
            lr=1e-3,
            ladder=1,
            mc_dt=2e-3,
            mc_steps=1,
            mc_image_radius=3,
            melt=0.25,
            opt_alpha=0.02,
            opt_steps=3,
            chunks=2,
            monitor=SimpleNamespace(printer=lines.append),
            bg_param=controls(),
        )

    fresh = potential.samples(jax.random.key(0), samples.shape[0])
    assert bool(jnp.array_equal(captured["qt_initial"], fresh))
    assert not bool(jnp.array_equal(captured["qt_initial"], samples))
    assert bool(jnp.array_equal(captured["source_pool"], samples))
    assert bool(jnp.array_equal(captured["target_pool"], samples))
    assert bool(jnp.array_equal(captured["hat_pool"], fresh))
    assert captured["qt_controls"] == {
        "melt": 0.25,
        "opt_alpha": 0.02,
        "opt_steps": 3,
        "mc_dt": 2e-3,
        "mc_steps": 1,
        "mc_image_radius": 3,
        "chunks": 2,
    }
    assert stages[0]["selection_pool_mode"] == "full"
    assert stages[0]["selection_pool_size"] == samples.shape[0]
    assert any("source=fresh, N=8" in line for line in lines)


def test_full_pool_input_validation() -> None:
    samples, potential, flow = common_inputs()

    for overrides in (
        {"pool_size": -1},
        {"pool_size": False},
        {"pool_size": 0, "batch_size": samples.shape[0] + 1},
        {"pool_size": 0, "chunks": samples.shape[0] + 1},
    ):
        try:
            run_driver(samples, potential, flow, **overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid full-pool controls accepted: {overrides}")


def test_each_stage_attempt_starts_from_identity_and_trained_tie_wins() -> None:
    samples, potential, flow = common_inputs()
    seen = []

    def trainer(pool, source_pool, previous, current, initial, batch_size, train_steps, lr, **kwargs):
        del pool, source_pool, previous, current, batch_size, lr, kwargs
        seen.append(float(initial.shift[0]))
        return (
            Probe_Flow(initial.shift + 1.0),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,), dtype=bool),
        )

    with patched_driver(trainer=trainer, identity_weights=zero_identity_weights):
        _, stages = run_driver(
            samples,
            potential,
            flow,
            bg_param=controls(
                t_safe=0.4,
                enlarge_factor=1.5,
                max_stages=2,
            ),
        )

    assert seen == [0.0, 0.0], seen
    assert len(stages) == 2 and stages[-1]["t"] == 1.0
    assert all(stage["selected"] == "trained" for stage in stages)
    assert all(
        stage["valid_trained_ess"] == stage["valid_identity_ess"] == 1.0
        for stage in stages
    )
    assert all(stage["attempt_status_hist"] == ("accepted",) for stage in stages)
    assert all(
        bool(jnp.array_equal(stage["t_hist"], jnp.asarray([stage["t"]])))
        for stage in stages
    )
def test_only_final_endpoint_is_scored() -> None:
    samples, potential, _ = common_inputs()
    flow = Probe_Flow(jnp.full((1,), 2.0))
    scored = []

    def trainer(pool, source_pool, previous, current, initial, batch_size, train_steps, lr, **kwargs):
        del pool, source_pool, previous, current, initial, batch_size, lr, kwargs
        return (
            Probe_Flow(jnp.full((1,), 3.0)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,), dtype=bool),
        )

    def importance_weights(samples, source, target, candidate, chunks):
        del source, target, chunks
        scored.append(float(candidate.shift[0]))
        return jnp.zeros((samples.shape[0],), dtype=samples.dtype)

    with patched_driver(
        trainer=trainer,
        identity_weights=zero_identity_weights,
        importance_weights=importance_weights,
    ):
        _, stages = run_driver(
            samples, potential, flow, train_steps=2, bg_param=controls()
        )

    assert scored == [3.0], scored
    assert stages[0]["selected"] == "trained"
    assert float(stages[0]["flow"].shift[0]) == 3.0


def test_zero_updates_reaches_final_ess_gate() -> None:
    samples, potential, flow = common_inputs()
    calls, lines = [], []

    def trainer(pool, source_pool, previous, current, initial, batch_size, train_steps, lr, **kwargs):
        del pool, source_pool, previous, current, batch_size, lr, kwargs
        calls.append(True)
        return (
            initial,
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.zeros((train_steps,), dtype=bool),
        )

    with patched_driver(trainer=trainer, identity_weights=zero_identity_weights):
        _, stages = run_driver(
            samples,
            potential,
            flow,
            train_steps=2,
            monitor=SimpleNamespace(printer=lines.append),
            bg_param=controls(tau_ess=0.9, max_retry=2),
        )

    assert len(calls) == 1
    assert len(stages) == 1 and stages[0]["t"] == 1.0
    assert stages[0]["valid_selected_ess"] == 1.0
    assert stages[0]["selected"] == "trained"
    assert not bool(jnp.any(stages[0]["update_applied_hist"]))
    assert any("zero optimizer updates" in line for line in lines)
    assert any("validation ESS" in line for line in lines)
    assert not any("training rejected" in line for line in lines)


def test_identity_rescues_nonfinite_final() -> None:
    samples, potential, flow = common_inputs()
    lines = []

    def trainer(pool, source_pool, previous, current, initial, batch_size, train_steps, lr, **kwargs):
        del pool, source_pool, previous, current, initial, batch_size, lr, kwargs
        return (
            Probe_Flow(jnp.full((1,), jnp.nan)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,), dtype=bool),
        )

    with patched_driver(trainer=trainer, identity_weights=zero_identity_weights):
        _, stages = run_driver(
            samples,
            potential,
            flow,
            monitor=SimpleNamespace(printer=lines.append),
            bg_param=controls(tau_ess=0.9, max_retry=2),
        )

    stage = stages[0]
    assert stage["selected"] == "identity"
    assert stage["valid_selected_ess"] == 1.0
    assert stage["valid_trained_ess"] == 0.0
    assert bool(jnp.isfinite(stage["flow"].shift).all())
    assert any("nonfinite final trained flow" in line for line in lines)
    assert not any("training rejected" in line for line in lines)


def test_retry_smc_is_diagnostic_after_validation_rejection() -> None:
    samples, potential, flow = common_inputs()
    trainer_calls, initial_shifts, smc_calls, lines = [], [], [], []

    def trainer(pool, source_pool, previous, current, initial, batch_size, train_steps, lr, **kwargs):
        del pool, source_pool, previous, current, batch_size, lr, kwargs
        trainer_calls.append(True)
        initial_shifts.append(float(initial.shift[0]))
        return (
            Probe_Flow(jnp.asarray([float(len(trainer_calls))])),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,)),
            jnp.ones((train_steps,), dtype=bool),
        )

    def smc_sampler(key, pool, source, target, *, ladder, mc_steps, **kwargs):
        del key, source, target, kwargs
        smc_calls.append(True)
        ess = 1.0 if len(smc_calls) == 1 else 0.0
        return pool, jnp.full((ladder,), ess), jnp.ones((ladder, mc_steps))

    def peaked_identity(samples, source, target, chunks):
        del source, target, chunks
        values = jnp.full((samples.shape[0],), -10.0, dtype=samples.dtype)
        return values.at[0].set(0.0)

    def importance_weights(samples, source, target, candidate, chunks):
        del source, target, chunks
        if float(candidate.shift[0]) == 2.0:
            return jnp.zeros((samples.shape[0],), dtype=samples.dtype)
        values = jnp.full((samples.shape[0],), -10.0, dtype=samples.dtype)
        return values.at[0].set(0.0)

    with patched_driver(
        trainer=trainer,
        identity_weights=peaked_identity,
        importance_weights=importance_weights,
        smc_sampler=smc_sampler,
    ):
        _, stages = run_driver(
            samples,
            potential,
            flow,
            monitor=SimpleNamespace(printer=lines.append),
            bg_param=controls(
                shrink_factor=0.7,
                tau_smc=0.5,
                tau_ess=0.9,
                max_retry=2,
            ),
        )

    assert len(trainer_calls) == 2 and len(smc_calls) == 2
    assert initial_shifts == [0.0, 0.0], initial_shifts
    assert len(stages) == 1 and stages[0]["t"] == 0.7
    assert stages[0]["valid_selected_ess"] == 1.0
    assert stages[0]["attempt_status_hist"] == ("rejected", "accepted")
    assert stages[0]["batch_ess_hist"].shape == (2, 1)
    assert stages[0]["valid_trained_ess_hist"].shape == (2,)
    assert stages[0]["valid_identity_ess_hist"].shape == (2,)
    assert any("training rejected -> shrink" in line for line in lines)
    assert any("retry SMC ESS=0.000" in line for line in lines)
    assert any("ACCEPTED" in line for line in lines)


def test_selection_failure_is_manifested() -> None:
    samples, potential, flow = common_inputs()

    def trainer(*args, **kwargs):
        raise AssertionError("training must not start after failed SMC selection")

    def rejecting_smc(key, pool, source, target, *, ladder, mc_steps, **kwargs):
        del key, source, target, kwargs
        return pool, jnp.zeros((ladder,)), jnp.ones((ladder, mc_steps))

    with tempfile.TemporaryDirectory() as temporary:
        with patched_driver(
            trainer=trainer,
            identity_weights=zero_identity_weights,
            smc_sampler=rejecting_smc,
        ):
            _, stages = run_driver(
                samples,
                potential,
                flow,
                flow_dir=temporary,
                monitor=SimpleNamespace(printer=lambda _: None),
                bg_param=controls(tau_smc=0.9),
            )
        assert stages == []
        manifest = json.loads(
            (Path(temporary) / "attempts.json").read_text(encoding="utf-8")
        )
        assert manifest["attempts"] == []
        run_state = manifest["run_state"]
        assert run_state["status"] == "incomplete"
        assert run_state["terminal_reason"] == "smc_selection_failed"
        assert run_state["terminal_stage"] == 1
        assert 0.0 < run_state["terminal_t"] <= 1.0


def main() -> None:
    test_full_pool_and_positive_pool_paths()
    test_full_pool_klxx_uses_fresh_qt_source_population()
    test_full_pool_input_validation()
    test_each_stage_attempt_starts_from_identity_and_trained_tie_wins()
    test_only_final_endpoint_is_scored()
    test_zero_updates_reaches_final_ess_gate()
    test_identity_rescues_nonfinite_final()
    test_retry_smc_is_diagnostic_after_validation_rejection()
    test_selection_failure_is_manifested()
    print("PASS molecular final-versus-identity and validation-ESS regressions")


if __name__ == "__main__":
    main()
