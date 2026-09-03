#!/usr/bin/env python
"""Tiny molecular generator, persistence, and inference smoke test."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows.train import Monitor  # noqa: E402
from jflows_md import Mixed_NSF, Molecular_Source, run_inference  # noqa: E402
from jflows_md.boltzmann import (  # noqa: E402
    Manual_Reject,
    boltzmann_FABX_G,
    boltzmann_FAB_G,
    boltzmann_forward_KLL1_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G,
    boltzmann_identity,
    iterate_boltzmann,
)
from jflows_md.boltzmann.load import load, run, validate  # noqa: E402
from jflows_md.train import REJECT_CHECK_STEPS  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402


class Toy_Target(Potential):
    domain: Mixed_Domain

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum((value[:, :2] - 0.3) ** 2, axis=-1) - 0.2 * jnp.cos(value[:, 2])


VALID_SIZE, BATCH, STEPS = 64, 16, 2
MC = dict(mc_dt=0.05, mc_steps_1=2, mc_steps_2=3)
BG = {"t_safe": 0.5, "tau_valid": 0.0}


def main() -> None:
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Target(domain)
    x_valid = source.samples(jax.random.key(2), VALID_SIZE)
    flow = Mixed_NSF(jax.random.key(1), domain, bins=4, transforms=1, hidden_features=(8,)).zeros()
    messages = []
    monitor = Monitor(1, "[md] ", messages.append)

    _, stages, _ = boltzmann_identity(x_valid, source, target, 0.05, 3, bg_param=BG, chunks=2)
    assert stages[-1]["t"] == 1.0 and stages[0]["selected"] == "identity"
    assert stages[0]["rg_start"] is None and "sharpen_ess" not in stages[0]

    _, stages, _ = boltzmann_forward_KLX_G(
        x_valid, source, target, flow, BATCH, STEPS, 1e-3, 2, MC["mc_dt"], MC["mc_steps_1"],
        MC["mc_steps_2"], coeff_lambda=0.5, monitor=monitor, bg_param=BG, chunks=2, u_clip=50.0,
    )
    assert stages[-1]["t"] == 1.0 and stages[0]["batch_ess_hist"].shape == (1, STEPS)
    # the monitor's printer carries both the per-step lines and the stage lines
    assert any("step" in message for message in messages)
    assert any("ACCEPTED" in message for message in messages)

    _, stages, _ = boltzmann_forward_KLXX_G(
        x_valid, source, target, flow, 0, BATCH, STEPS, 1e-3, 2, 0.5, 0.1, 2,
        MC["mc_dt"], MC["mc_steps_1"], MC["mc_steps_2"], coeff_qt=0.5, bg_param=BG, chunks=2,
    )
    assert stages[-1]["t"] == 1.0 and stages[0]["objective"] == "forward_klxx"

    # manual rejection: the flag raised by the signal rejects the first attempt
    import os, signal
    with Manual_Reject() as reject:
        assert reject.command.startswith("Ctrl+C")
        calls = {"n": 0}
        class Once:
            """Raise the signal at the first chunk boundary of the first attempt."""
            def peek(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    os.kill(os.getpid(), signal.SIGINT)
                return reject.peek()
            def __call__(self):
                return reject()
        steps = 2 * REJECT_CHECK_STEPS + 3
        _, rejected_stages, seconds = boltzmann_forward_KLX_G(
            x_valid, source, target, flow, BATCH, steps, 1e-3, 2, MC["mc_dt"], MC["mc_steps_1"],
            MC["mc_steps_2"], coeff_lambda=0.5, bg_param={"t_safe": 1.0, "tau_valid": 0.0},
            chunks=2, reject_requested=Once(),
        )
    assert rejected_stages[0]["attempt_status_hist"][0] == "rejected-manual"
    # the scan stopped after the first chunk of REJECT_CHECK_STEPS steps; the rest is NaN
    first_attempt = rejected_stages[0]["batch_ess_hist"][0]
    assert first_attempt.shape == (steps,)
    assert np.isfinite(first_attempt[:REJECT_CHECK_STEPS]).all() and np.isnan(first_attempt[REJECT_CHECK_STEPS:]).all()
    assert np.isfinite(rejected_stages[0]["batch_ess_hist"][-1]).all()
    assert rejected_stages[0]["attempt_status_hist"][-1] == "accepted"
    assert rejected_stages[0]["t"] < 1.0 and np.isnan(rejected_stages[0]["valid_trained_ess_hist"][0])
    assert 0.0 < seconds < sum(stage["elapsed_seconds"] for stage in rejected_stages)
    assert seconds == sum(stage["accepted_attempt_seconds"] for stage in rejected_stages)
    _, fixed_stages, _ = boltzmann_forward_KLX_G_fixed(
        x_valid, source, target, flow, BATCH, STEPS, 1e-3, 2, MC["mc_dt"], MC["mc_steps_1"],
        MC["mc_steps_2"], (1.0,), chunks=2,
    )
    assert fixed_stages[0]["accepted_attempt_seconds"] > 0.0

    _, stages, _ = boltzmann_forward_KLL1_G(
        x_valid, source, target, flow, BATCH, STEPS, 1e-3, 2, MC["mc_dt"],
        MC["mc_steps_1"], MC["mc_steps_2"], coeff_lambda=0.5, bg_param=BG, chunks=2,
    )
    assert stages[-1]["t"] == 1.0 and stages[0]["objective"] == "forward_kll1"

    _, stages, _ = boltzmann_FAB_G(
        x_valid, source, target, flow, BATCH, STEPS, 1e-3, 2, MC["mc_dt"],
        MC["mc_steps_1"], MC["mc_steps_2"], bg_param=BG, chunks=2, u_clip=50.0,
    )
    assert stages[-1]["t"] == 1.0 and stages[0]["objective"] == "fab"

    _, stages, _ = boltzmann_FABX_G(
        x_valid, source, target, flow, 0, BATCH, STEPS, 1e-3, 2, 0.5, 0.1, 2,
        MC["mc_dt"], MC["mc_steps_1"], MC["mc_steps_2"], coeff_qt=0.5,
        bg_param=BG, chunks=2,
    )
    assert stages[-1]["t"] == 1.0 and stages[0]["objective"] == "fabx"

    try:
        iterate_boltzmann(x_valid, source, target, flow, objective="nonsense")
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("an unknown objective must be rejected")
    try:
        boltzmann_identity(x_valid, source, target, 0.05, 3, bg_param={"tau_ess": 0.0})
    except KeyError:
        pass
    else:
        raise AssertionError("a retired policy key must be rejected")

    controls = dict(
        objective="forward_klx", pool_size=0, batch_size=BATCH, steps_total=STEPS, lr=1e-3,
        ladder=1, initialize_from_identity=True, coeff_lambda=0.0, coeff_theta=1.0,
        coeff_alpha=0.5, coeff_qt=0.0, melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=None,
        bg_param=BG, chunks=2, mc_image_radius=3, checkpoint=False, **MC,
    )
    with tempfile.TemporaryDirectory() as folder:
        run_dir = Path(folder) / "run"

        def iterate(samples, template, accepted, stage):
            return iterate_boltzmann(
                samples, source, target, template, accepted_t=accepted, start_stage=stage, **controls
            )

        _, records = run(run_dir, "toy", {"valid_size": VALID_SIZE}, x_valid, flow, iterate)
        assert validate(run_dir)["status"] == "complete" and len(records) >= 1
        samples, continuation, loaded = load(run_dir, flow)
        assert samples.shape == (VALID_SIZE, 3) and len(loaded) == len(records)
        assert "flow" in loaded[0] and loaded[0]["t"] == records[0]["t"]

        manifest = run_inference(
            run_dir, Path(folder) / "inference", flow, source, target,
            sample_count=40, chunk_size=16, mc_dt=0.05, mc_steps=2, seed=3, screen_fraction=0.05,
        )
        assert manifest["status"] == "complete"
        final = np.load(Path(folder) / "inference" / manifest["inference_samples_path"])
        assert final.shape == (40, 3) and np.isfinite(final).all()
        assert all(0.0 < item["pushforward_ess"] <= 1.0 for item in manifest["stages"])
        stage_sets = [
            np.load(Path(folder) / "inference" / item["samples_path"]) for item in manifest["stages"]
        ]
        assert all(values.shape == (40, 3) and np.isfinite(values).all() for values in stage_sets)
        assert len(stage_sets) == 1 or not np.array_equal(stage_sets[0], stage_sets[-1])
        assert not any((Path(folder) / "inference" / name).exists() for name in ("work_a.npy", "work_b.npy"))
        assert all(np.load(Path(folder) / "inference" / item["log_weights_path"]).shape == (40,)
                   for item in manifest["stages"])
    print("PASS molecular generators, persistence, and inference")


if __name__ == "__main__":
    main()
