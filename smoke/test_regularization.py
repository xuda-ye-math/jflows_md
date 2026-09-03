#!/usr/bin/env python
"""Regularized potential ``U^{rho}`` and the diagonal regularization path of the generator, persistence, and inference."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Mixed_NSF, Molecular_Potential, run_inference  # noqa: E402
from jflows_md.boltzmann import boltzmann_forward_KLX_G_fixed, iterate_boltzmann  # noqa: E402
from jflows_md.boltzmann.load import load, run, validate  # noqa: E402

RG_0, RG_1 = (50.0, 0.25), (100.0, 0.15)
MC = dict(mc_dt=1e-4, mc_steps_1=2, mc_steps_2=3)


def check_regularized_potential(target, source) -> None:
    q = source.samples(jax.random.key(3), 32)
    reference = target.reference_internal()[None]
    strict = target.regularized((1e12, 0.0))
    soft = target.regularized(RG_1)
    # e -> inf, r -> 0 recovers the base potential
    np.testing.assert_allclose(strict(q), target(q), rtol=1e-5, atol=1e-6)
    # no compression at the reference geometry (zero excess, floor inactive)
    np.testing.assert_allclose(soft(reference), target(reference), rtol=1e-5, atol=1e-6)
    # the compression never raises an energy and lowers a high one
    assert bool(jnp.all(soft(q) <= target(q) + 1e-6))
    ne = target.domain.euclidean_dim
    high = None
    for factor in (2.0, 4.0, 8.0):
        candidate = reference.at[:, :ne].add(factor)
        if float(target.physical_energy(candidate)[0]) - float(soft.reference_energy_kj_mol) > RG_1[0]:
            high = candidate
            break
    assert high is not None, "no configuration above the threshold found"
    assert float(soft.regularized_energy(high)[0]) < float(target.physical_energy(high)[0])
    assert float(soft(high)[0]) < float(target(high)[0])
    # the pair floor keeps a collision finite
    clash = jnp.concatenate([jnp.full_like(reference[:, : target.domain.euclidean_dim], 0.0),
                             reference[:, target.domain.euclidean_dim:]], axis=1)
    assert bool(jnp.isfinite(soft(clash)).all())
    print("PASS regularized potential U^rho: limits, reference, compression, floor")


def check_regularization_path(target, source) -> None:
    domain = target.domain
    x_valid = source.samples(jax.random.key(2), 48)
    flow = Mixed_NSF(jax.random.key(1), domain, bins=4, transforms=1, hidden_features=(8,)).zeros()
    _, stages, _ = boltzmann_forward_KLX_G_fixed(
        x_valid, source, target, flow, 16, 2, 1e-3, 2, MC["mc_dt"], MC["mc_steps_1"],
        MC["mc_steps_2"], (0.5, 1.0), chunks=2, rg_param_0=RG_0, rg_param_1=RG_1,
    )
    assert [stage["t"] for stage in stages] == [0.5, 1.0]
    for stage in stages:
        assert stage["rg_start"] is not None and stage["rg_end"] is not None
        assert "sharpen_ess" not in stage
    np.testing.assert_allclose(stages[0]["rg_start"], RG_0)
    np.testing.assert_allclose(stages[0]["rg_end"], (75.0, 0.2))
    np.testing.assert_allclose(stages[1]["rg_start"], (75.0, 0.2))
    np.testing.assert_allclose(stages[1]["rg_end"], RG_1)
    # the diagonal stage targets: the population after stage 1 follows U_{0.5,0.5}, whose
    # identity weights towards U_{1,1} are those of the regularization at t = 1
    from jflows_md.boltzmann import _diagonal
    soft, _ = _diagonal(source, target, RG_0, RG_1, 0.5)
    sharp, _ = _diagonal(source, target, RG_0, RG_1, 1.0)
    q = source.samples(jax.random.key(7), 8)
    expected = 0.5 * source(q) + 0.5 * target.regularized((75.0, 0.2))(q)
    np.testing.assert_allclose(soft(q), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sharp(q), target.regularized(RG_1)(q), rtol=1e-5, atol=1e-6)
    # a fixed regularization keeps rho constant
    _, fixed, _ = boltzmann_forward_KLX_G_fixed(
        x_valid, source, target, flow, 16, 2, 1e-3, 2, MC["mc_dt"], MC["mc_steps_1"],
        MC["mc_steps_2"], (1.0,), chunks=2, rg_param_0=RG_1, rg_param_1=RG_1,
    )
    assert fixed[0]["rg_start"] == RG_1 and fixed[0]["rg_end"] == RG_1
    print("PASS diagonal regularization path on a fixed schedule; fixed regularization keeps rho")

    controls = dict(
        objective="forward_klx", pool_size=0, batch_size=16, steps_total=2, lr=1e-3, ladder=2,
        mc_dt=MC["mc_dt"], mc_steps_1=MC["mc_steps_1"], mc_steps_2=MC["mc_steps_2"],
        initialize_from_identity=True, coeff_lambda=1.0, coeff_theta=1.0, coeff_alpha=0.5,
        coeff_qt=0.0, melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=None, bg_param=None,
        chunks=2, mc_image_radius=3, checkpoint=False, seed=0, t_list=(0.5, 1.0),
        rg_param_0=RG_0, rg_param_1=RG_1,
    )
    config = {key: value for key, value in controls.items() if key != "monitor"}

    def iterate(samples, template, accepted, stage):
        return iterate_boltzmann(samples, source, target, template, accepted_t=accepted,
                                 start_stage=stage, **controls)

    with tempfile.TemporaryDirectory() as folder:
        run_dir = Path(folder) / "run"
        _, records = run(run_dir, "regularization-path", config, x_valid, flow, iterate, resume=False)
        assert validate(run_dir)["status"] == "complete" and len(records) == 2
        _, _, loaded = load(run_dir, flow)
        assert loaded[0]["rg_end"] == (75.0, 0.2) and loaded[1]["rg_start"] == (75.0, 0.2)
        manifest = run_inference(
            run_dir, Path(folder) / "inference", flow, source, target, sample_count=40,
            mc_dt=MC["mc_dt"], mc_steps=2, chunk_size=16, seed=1,
        )
        assert manifest["status"] == "complete"
        assert [tuple(item["rg_end"]) for item in manifest["stages"]] == [(75.0, 0.2), RG_1]
        assert np.load(Path(folder) / "inference" / manifest["inference_samples_path"]).shape == (40, domain.dimension)
    print("PASS regularization path persisted, loaded, and replayed by the inference scheme")


def main() -> None:
    target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
    source = target.source()
    check_regularized_potential(target, source)
    check_regularization_path(target, source)


if __name__ == "__main__":
    main()
