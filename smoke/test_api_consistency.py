#!/usr/bin/env python
"""Public molecular API and v0.5 layout checks."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows_md import (  # noqa: E402
    Mixed_Identity,
    Molecular_Potential,
    Molecular_Source,
    boltzmann_identity,
)
from jflows_md.boltzmann import (  # noqa: E402
    boltzmann_FABX_G,
    boltzmann_FAB_G,
    boltzmann_forward_KLL1_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.bundle_build import write_bundle  # noqa: E402
from jflows_md.bundle_build.builder import build_coordinate_spec  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import (  # noqa: E402
    train_FABX_G,
    train_FAB_G,
    train_forward_KLL1_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)


def main() -> None:
    coordinate_builder = inspect.signature(build_coordinate_spec).parameters
    assert tuple(coordinate_builder) == (
        "system_spec",
        "positions_nm",
        "bonds",
        "target",
        "zmatrix",
        "fixed_stereocenters",
        "signed_volume_diagnostics",
    )
    assert coordinate_builder["target"].default is None
    for name in tuple(coordinate_builder)[3:]:
        assert coordinate_builder[name].kind is inspect.Parameter.KEYWORD_ONLY

    bundle_builder = inspect.signature(write_bundle).parameters
    assert tuple(bundle_builder) == (
        "output",
        "name",
        "target",
        "prmtop_path",
        "coordinate_path",
        "model",
        "canonical_smiles",
        "expected_formula",
        "expected_charge",
        "minimize",
        "zmatrix",
        "fixed_stereocenters",
        "signed_volume_diagnostics",
    )
    assert bundle_builder["zmatrix"].default is None
    assert bundle_builder["fixed_stereocenters"].default == ()
    assert bundle_builder["signed_volume_diagnostics"].default is None
    for name in tuple(bundle_builder)[1:]:
        assert bundle_builder[name].kind is inspect.Parameter.KEYWORD_ONLY

    for trainer in (
        train_forward_KLX_G, train_forward_KLL1_G, train_FAB_G,
        train_forward_KLXX_G, train_FABX_G,
    ):
        parameters = inspect.signature(trainer).parameters
        assert "mc_steps_1" in parameters and "mc_steps_2" in parameters
        assert "mc_steps" not in parameters
        assert parameters["initialize_from_identity"].default is False
        assert "e_clip" not in parameters
        assert "energy_origin" not in parameters
        assert parameters["u_clip"].default == float("inf")
        assert parameters["g_clip"].default == float("inf")
        assert parameters["lr_warmup"].default == 0
        assert "t_start" in parameters and "t_end" in parameters
    for generator in (
        boltzmann_forward_KLX_G, boltzmann_forward_KLL1_G, boltzmann_FAB_G,
        boltzmann_forward_KLXX_G, boltzmann_FABX_G,
    ):
        parameters = inspect.signature(generator).parameters
        assert parameters["initialize_from_identity"].default is True
        assert parameters["screen_fraction"].default == 1e-4
        assert parameters["rg_param_0"].default is None and "tau_smc" not in parameters
        assert "flow_dir" not in parameters
        assert "resume" not in parameters
        assert "run_dir" not in parameters
        assert parameters["u_clip"].default == float("inf")
        assert parameters["g_clip"].default == float("inf")
        assert parameters["lr_warmup"].default == 0
    assert inspect.signature(boltzmann_forward_KLXX_G).parameters["coeff_qt"].default == 0.0
    # the FAB losses carry no target-measure coefficient
    for fab in (train_FAB_G, train_FABX_G, boltzmann_FAB_G, boltzmann_FABX_G):
        assert "coeff_lambda" not in inspect.signature(fab).parameters
    for mixture in (train_FABX_G, boltzmann_FABX_G):
        assert "coeff_theta" in inspect.signature(mixture).parameters
    identity = inspect.signature(boltzmann_identity).parameters
    assert tuple(identity) == (
        "x_valid",
        "source",
        "target",
        "mc_dt",
        "mc_steps_2",
        "monitor",
        "bg_param",
        "chunks",
        "mc_image_radius",
        "seed",
        "screen_fraction",
        "rg_param_0",
        "rg_param_1",
    )
    assert not {
        "flow", "pool_size", "batch_size", "steps_total", "lr",
        "checkpoint", "initialize_from_identity", "ladder",
    } & set(identity)

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain, mean=[0, 0], variance=[1, 1])
    samples = source.samples(jax.random.key(20), 4)
    assert samples.shape == (4, 3)
    assert samples.dtype == jnp.float32
    assert source(samples).shape == (4,)
    identity, ess = train_forward_KLX_G(
        samples,
        source,
        source,
        Mixed_Identity(domain),
        domain,
        4,
        2,
        1e-3,
        1,
        1e-2,
        1,
        1,
        g_clip=1.0,
    )
    assert identity(samples).shape == samples.shape
    assert ess.shape == (2,) and bool(jnp.isfinite(ess).all())
    target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
    assert target.source().samples(jax.random.key(21), 1).dtype == jnp.float32
    print("PASS modern minimal molecular API")


if __name__ == "__main__":
    main()
