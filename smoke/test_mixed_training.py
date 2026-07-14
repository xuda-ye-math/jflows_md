#!/usr/bin/env python
"""Tiny synthetic smoke test for mixed-domain KL+X BG training."""

from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows.utils import (  # noqa: E402
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
    resample,
)
from jflows_md.boltzmann import (  # noqa: E402
    _operation_key,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_NSF  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402
from jflows_md.artifacts import load_mixed_flow  # noqa: E402
from jflows_md.train import train_forward_KLXX_G  # noqa: E402
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

    with tempfile.TemporaryDirectory() as nonempty:
        (Path(nonempty) / "keep.txt").write_text("do not overwrite\n")
        try:
            boltzmann_forward_KLX_G(
                x_valid,
                source,
                target,
                flow,
                pool_size=16,
                batch_size=8,
                train_steps=1,
                lr=1e-3,
                ladder=1,
                mc_dt=1e-3,
                mc_steps=0,
                bg_param={"t_safe": 1.0},
                flow_dir=nonempty,
            )
            raise AssertionError("nonempty flow_dir was overwritten")
        except FileExistsError:
            pass

    with tempfile.TemporaryDirectory() as temporary:
        particles, stages = boltzmann_forward_KLX_G(
            x_valid,
            source,
            target,
            flow,
            pool_size=16,
            batch_size=8,
            train_steps=2,
            lr=1e-3,
            ladder=2,
            mc_dt=1e-3,
            mc_steps=1,
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
            flow_dir=temporary,
        )
        trained_path = stages[0]["trained_flow_path_hist"][0]
        selected_path = stages[0]["selected_flow_path"]
        assert trained_path is not None and selected_path is not None
        loaded_trained = load_mixed_flow(Path(temporary) / trained_path)
        loaded_selected = load_mixed_flow(Path(temporary) / selected_path)
        assert eqx.tree_equal(loaded_trained, stages[0]["flow"]) or (
            stages[0]["selected"] == "identity"
        )
        assert eqx.tree_equal(loaded_selected, stages[0]["flow"])
        manifest = json.loads(
            (Path(temporary) / "attempts.json").read_text(encoding="utf-8")
        )
        assert manifest["run_state"]["status"] == "complete"
        assert manifest["attempts"][0]["selected_flow_path"] == selected_path
        assert manifest["attempts"][0]["valid_selected_ess"] == stages[0][
            "valid_selected_ess"
        ]
        monitor_path = Path(temporary) / manifest["attempts"][0]["monitor_path"]
        with np.load(monitor_path) as monitor:
            assert np.array_equal(
                monitor["batch_ess_hist"],
                np.asarray(stages[0]["batch_ess_hist"][0]),
            )
            assert np.array_equal(
                monitor["kept_fraction_hist"],
                np.asarray(stages[0]["kept_fraction_hist"][0]),
            )
            assert np.array_equal(
                monitor["update_applied_hist"],
                np.asarray(stages[0]["update_applied_hist"][0]),
            )
    plain_particles, plain_stages = boltzmann_forward_KLX_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=16,
        batch_size=8,
        train_steps=2,
        lr=1e-3,
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
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
    assert bool(jnp.array_equal(plain_particles, particles))
    assert eqx.tree_equal(plain_stages[0]["flow"], stages[0]["flow"])
    assert bool(
        jnp.array_equal(
            plain_stages[0]["batch_ess_hist"], stages[0]["batch_ess_hist"]
        )
    )
    assert plain_stages[0]["trained_flow_path_hist"] == (None,)
    assert plain_stages[0]["selected_flow_path"] is None
    jax.block_until_ready(particles)

    assert particles.shape == x_valid.shape
    assert bool(jnp.isfinite(particles).all())
    assert len(stages) == 1 and stages[0]["t"] == 1.0
    assert stages[0]["valid_sample_count"] == x_valid.shape[0]
    assert 0.0 < stages[0]["valid_selected_ess"] <= 1.0
    assert 0.0 < stages[0]["valid_trained_ess"] <= 1.0
    assert 0.0 < stages[0]["valid_identity_ess"] <= 1.0
    assert stages[0]["selected"] in ("trained", "identity")
    assert stages[0]["t_hist"].shape == (1,)
    assert stages[0]["batch_ess_hist"].shape == (1, 2)
    assert abs(
        stages[0]["valid_selected_ess"]
        - max(stages[0]["valid_trained_ess"], stages[0]["valid_identity_ess"])
    ) < 1e-12
    assert stages[0]["kept_fraction_hist"].shape == (1, 2)
    assert stages[0]["update_applied_hist"].shape == (1, 2)
    assert stages[0]["smc_ess"].shape == (2,)
    assert stages[0]["smc_acceptance"].shape == (2, 1)
    assert stages[0]["mala_acceptance"].shape == (1,)
    assert bool(
        jnp.isfinite(stages[0]["batch_ess_hist"]).all()
        & jnp.isfinite(stages[0]["kept_fraction_hist"]).all()
    )
    leaves = eqx.filter(stages[0]["flow"], eqx.is_inexact_array)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(leaves))

    klxx_flow = Mixed_NSF(
        jax.random.key(305),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    klxx_particles, klxx_stages = boltzmann_forward_KLXX_G(
        x_valid,
        source,
        target,
        klxx_flow,
        pool_size=16,
        batch_size=8,
        train_steps=2,
        lr=1e-3,
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
        melt=0.1,
        opt_alpha=1e-2,
        opt_steps=2,
        coeff_lambda=1.0,
        coeff_alpha=0.5,
        coeff_beta=0.5,
        bg_param={
            "t_safe": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 0.0,
            "max_stages": 1,
            "max_retry": 1,
        },
        e_clip=1000.0,
        g_clip=100.0,
        lr_warmup=2,
        seed=306,
    )
    jax.block_until_ready(klxx_particles)
    assert klxx_particles.shape == x_valid.shape
    assert bool(jnp.isfinite(klxx_particles).all())
    assert len(klxx_stages) == 1 and klxx_stages[0]["t"] == 1.0
    assert klxx_stages[0]["objective"] == "klxx"
    assert abs(
        klxx_stages[0]["valid_selected_ess"]
        - max(
            klxx_stages[0]["valid_trained_ess"],
            klxx_stages[0]["valid_identity_ess"],
        )
    ) < 1e-12
    assert klxx_stages[0]["hat_mala_acceptance"].shape == (1,)
    assert bool(jnp.isfinite(klxx_stages[0]["batch_ess_hist"]).all())

    zero_mix_flow, zero_mix_ess, _, _ = train_forward_KLXX_G(
        x_valid,
        x_valid,
        x_valid,
        source,
        target,
        klxx_flow,
        domain,
        batch_size=8,
        train_steps=1,
        lr=1e-3,
        coeff_alpha=0.0,
        coeff_beta=0.0,
        mc_steps=0,
        seed=307,
    )
    jax.block_until_ready((zero_mix_flow, zero_mix_ess))
    assert zero_mix_ess.shape == (1,) and bool(jnp.isfinite(zero_mix_ess).all())
    zero_key = jax.random.fold_in(jax.random.key(37), 307)
    zero_keys = jax.random.split(jax.random.fold_in(zero_key, 1), 7)
    zero_source_batch = x_valid[
        jax.random.choice(
            zero_keys[1], x_valid.shape[0], (8,), replace=False
        )
    ]
    zero_proposal, zero_ladj = klxx_flow.inv_and_ladj(zero_source_batch)
    zero_expected_ess = compute_ESS_log(
        source(zero_source_batch) - target(zero_proposal) + zero_ladj
    )
    assert bool(jnp.allclose(zero_mix_ess[0], zero_expected_ess, atol=1e-10))

    ais_flow = Mixed_NSF(
        jax.random.key(303),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    )
    ais_samples, initial_log_weights = annealed_importance_sampling(
        jax.random.key(304),
        x_valid,
        source,
        target,
        ais_flow,
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
        chunks=2,
        return_initial_log_weights=True,
    )
    jax.block_until_ready(ais_samples)
    expected_log_weights = importance_weights_log(
        x_valid, source, target, ais_flow, "G", chunks=2
    )
    assert bool(
        jnp.allclose(initial_log_weights, expected_log_weights, atol=1e-10)
    )
    ais_default = annealed_importance_sampling(
        jax.random.key(304),
        x_valid,
        source,
        target,
        ais_flow,
        ladder=2,
        mc_dt=1e-3,
        mc_steps=1,
        chunks=2,
    )
    assert bool(jnp.array_equal(ais_default, ais_samples))

    manual_key = jax.random.key(308)
    manual, manual_ladj = ais_flow.inv_and_ladj(x_valid)
    manual_initial = -target(manual) + source(x_valid) + manual_ladj
    for level in range(1, 4):
        if level == 1:
            full_weight = manual_initial
        else:
            latent, ladj_g = ais_flow.call_and_ladj(manual)
            full_weight = -target(manual) + source(latent) - ladj_g
        resample_key, _ = jax.random.split(jax.random.fold_in(manual_key, level))
        manual = resample(
            resample_key,
            manual,
            linear_weights_from_log(full_weight / 3.0),
            N=manual.shape[0],
        )
    public_manual = annealed_importance_sampling(
        manual_key,
        x_valid,
        source,
        target,
        ais_flow,
        ladder=3,
        mc_dt=1e-3,
        mc_steps=0,
        chunks=1,
    )
    assert bool(jnp.array_equal(public_manual, manual))
    assert ais_samples.shape == x_valid.shape
    assert bool(jnp.isfinite(ais_samples).all())
    assert bool(
        jnp.all((ais_samples[:, 2] >= -jnp.pi) & (ais_samples[:, 2] < jnp.pi))
    )
    print(
        "PASS mixed BG training and G-native score-free AIS: "
        f"stage_ESS={stages[0]['valid_selected_ess']:.3f} "
        f"N={stages[0]['valid_sample_count']}"
    )


if __name__ == "__main__":
    main()
