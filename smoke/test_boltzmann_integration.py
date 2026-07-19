#!/usr/bin/env python
"""Actual computed-stage interruption and resume equivalence."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path  # noqa: E402
from tempfile import TemporaryDirectory  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.boltzmann import iterate_boltzmann  # noqa: E402
from jflows_md.boltzmann.load import load, manifest, run  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_NSF  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402


class Regularized_Target(Potential):
    domain: Mixed_Domain
    center: jax.Array
    rg_param: jax.Array

    def __init__(self, domain, center, rg_param):
        self.domain = domain
        self.center = center
        self.rg_param = jnp.asarray(rg_param)

    def __call__(self, samples):
        delta = self.domain.displacement(samples, self.center)
        radial = 0.5 * jnp.sum(delta**2, axis=-1)
        return (1.0 + 0.01 * self.rg_param[0]) * radial


class Molecular_Target(Potential):
    domain: Mixed_Domain
    center: jax.Array

    def __init__(self, domain):
        self.domain = domain
        self.center = jnp.asarray([0.2, -0.1, 0.4])

    def __call__(self, samples):
        return Regularized_Target(
            self.domain, self.center, (0.0, 0.0)
        )(samples)

    def regularized(self, rg_param):
        return Regularized_Target(self.domain, self.center, rg_param)


def main() -> None:
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Molecular_Target(domain)
    samples = source.samples(jax.random.key(20), 16)
    flow = Mixed_NSF(
        jax.random.key(21),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    rg_param_0 = (5.0, 0.2)
    rg_param_1 = (10.0, 0.1)
    controls = {
        "objective": "forward_klx",
        "pool_size": 0,
        "batch_size": 4,
        "train_steps": 1,
        "lr": 1e-3,
        "ladder": 1,
        "mc_dt": 1e-3,
        "mc_steps": 0,
        "rg_param_0": rg_param_0,
        "rg_param_1": rg_param_1,
        "initialize_from_identity": True,
        "coeff_lambda": 1.0,
        "coeff_alpha": 0.5,
        "coeff_beta": 0.5,
        "melt": 0.0,
        "opt_alpha": 1.0,
        "opt_steps": 0,
        "monitor": None,
        "bg_param": {
            "t_safe": 0.5,
            "enlarge_factor": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 0.0,
            "max_stages": 2,
            "max_retry": 1,
        },
        "chunks": 1,
        "mc_image_radius": 3,
        "checkpoint": False,
        "seed": 22,
    }

    def iterate(current, template, accepted, stage):
        return iterate_boltzmann(
            current,
            source,
            target,
            template,
            accepted_t=accepted,
            start_stage=stage,
            **controls,
        )

    def first(current, template, accepted, stage):
        yield next(iter(iterate(current, template, accepted, stage)))

    config = {
        "rg_param_0": rg_param_0,
        "rg_param_1": rg_param_1,
        "seed": 22,
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        full, full_records = run(
            root / "full", "resume", config, samples, flow, iterate
        )
        _, partial_records = run(
            root / "resumed", "resume", config, samples, flow, first
        )
        resumed, resumed_records = run(
            root / "resumed",
            "resume",
            config,
            None,
            flow,
            iterate,
            resume=True,
        )
        loaded, continuation, loaded_records = load(root / "resumed", flow)

        assert partial_records[-1]["t"] == 0.5
        assert manifest(root / "resumed")["status"] == "complete"
        assert np.array_equal(np.asarray(resumed), np.asarray(full))
        assert np.array_equal(loaded, np.asarray(resumed))
        assert eqx.tree_equal(
            resumed_records[-1]["continuation_flow"],
            full_records[-1]["continuation_flow"],
        )
        assert eqx.tree_equal(
            continuation, resumed_records[-1]["continuation_flow"]
        )
        assert loaded_records[-1]["flow_endpoint"] == "pre_sharpen"
        assert loaded_records[-1]["sharpen_ess_hist"].shape == (1,)
        assert loaded_records[-1]["sharpen_ess_hist"][0] == (
            loaded_records[-1]["sharpen_ess"]
        )
        np.testing.assert_allclose(
            loaded_records[-1]["population_rg"],
            rg_param_1,
            rtol=0,
            atol=1e-7,
        )
    print("PASS actual combined-stage interruption/resume equivalence")


if __name__ == "__main__":
    main()
