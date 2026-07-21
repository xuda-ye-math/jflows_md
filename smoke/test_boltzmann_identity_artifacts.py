#!/usr/bin/env python
"""Flow-free molecular identity Boltzmann persistence smoke test."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.boltzmann import iterate_identity  # noqa: E402
from jflows_md.boltzmann.load import (  # noqa: E402
    load,
    load_stage_flow,
    load_training_history,
    manifest,
    run,
    validate,
)
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


RG0 = (0.0, 0.0)
RG1 = (2.0, 0.0)
BG_PARAM = {
    "t_safe": 0.5,
    "enlarge_factor": 1.0,
    "tau_smc": 0.0,
    "tau_ess": 0.0,
    "max_stages": 2,
    "max_retry": 1,
}


def main() -> None:
    domain = Mixed_Domain(2, 0)
    source = Molecular_Source(domain)
    target = Sharpening_Target(domain)
    samples = source.samples(jax.random.key(20), 64)
    config = {
        "method": "identity",
        "rg_param_0": RG0,
        "rg_param_1": RG1,
    }

    def stages(current, flow, accepted, number):
        assert flow is None
        return iterate_identity(
            current,
            source,
            target,
            ladder=1,
            mc_dt=1e-3,
            mc_steps=1,
            rg_param_0=RG0,
            rg_param_1=RG1,
            monitor=None,
            bg_param=BG_PARAM,
            chunks=2,
            mc_image_radius=3,
            seed=21,
            accepted_t=accepted,
            start_stage=number,
        )

    def first_stage(current, flow, accepted, number):
        yield next(iter(stages(current, flow, accepted, number)))

    with TemporaryDirectory() as directory:
        partial, records = run(
            directory,
            "molecular-identity-persistence",
            config,
            samples,
            None,
            first_stage,
        )
        assert partial.shape == samples.shape and len(records) == 1
        assert manifest(directory)["status"] == "exhausted"
        assert "flow_template" not in validate(directory)
        assert not list(Path(directory).rglob("*.eqx"))

        final, records = run(
            directory,
            "molecular-identity-persistence",
            config,
            None,
            None,
            stages,
            resume=True,
        )
        assert final.shape == samples.shape and bool(jnp.isfinite(final).all())
        assert len(records) == 2 and records[-1]["t"] == 1.0
        assert manifest(directory)["status"] == "complete"

        loaded, continuation, loaded_records = load(directory)
        assert bool(jnp.array_equal(loaded, final))
        assert continuation is None and len(loaded_records) == 2
        assert not list(Path(directory).rglob("*.eqx"))
        for item in manifest(directory)["stages"]:
            path = Path(directory) / item["path"] / "stage.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["population_rg"] == saved["rg_end"]
            assert not any("flow" in key or "trained" in key for key in saved)
        for record in loaded_records:
            assert record["population_rg"] == record["rg_end"]
            assert record["sharpen_ess"] > 0.0
            assert not any("flow" in key or "trained" in key for key in record)
            assert "batch_ess_hist" not in record
        history = load_training_history(directory, 2)
        assert set(history) == {
            "t_hist",
            "valid_identity_ess_hist",
            "sharpen_ess_hist",
            "smc_ess",
            "smc_acceptance",
            "mala_acceptance",
            "sharpen_mala_acceptance",
            "attempt_status_hist",
            "selection_history",
        }
        try:
            load_stage_flow(directory, 1, "selected", None)
        except ValueError as error:
            assert str(error) == "run has no stored flow"
        else:
            raise AssertionError("identity run exposed a flow artifact")
    print("PASS flow-free molecular identity persistence with sharpening")


if __name__ == "__main__":
    main()
