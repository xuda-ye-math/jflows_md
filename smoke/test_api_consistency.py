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
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.bundle_build import write_bundle  # noqa: E402
from jflows_md.bundle_build.builder import build_coordinate_spec  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import train_forward_KLX_G, train_forward_KLXX_G  # noqa: E402


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

    for trainer in (train_forward_KLX_G, train_forward_KLXX_G):
        parameters = inspect.signature(trainer).parameters
        assert parameters["initialize_from_identity"].default is False
        assert "e_clip" not in parameters
        assert "energy_origin" not in parameters
        assert parameters["u_clip"].default == float("inf")
        assert parameters["g_clip"].default == float("inf")
        assert parameters["lr_warmup"].default == 0
        assert "t_start" in parameters and "t_end" in parameters
    for generator in (boltzmann_forward_KLX_G, boltzmann_forward_KLXX_G):
        parameters = inspect.signature(generator).parameters
        assert parameters["initialize_from_identity"].default is True
        assert parameters["rg_param_0"].default is inspect.Parameter.empty
        assert parameters["rg_param_1"].default is inspect.Parameter.empty
        assert "flow_dir" not in parameters
        assert "resume" not in parameters
        assert "run_dir" not in parameters
        assert parameters["u_clip"].default == float("inf")
        assert parameters["g_clip"].default == float("inf")
        assert parameters["lr_warmup"].default == 0
    identity = inspect.signature(boltzmann_identity).parameters
    assert tuple(identity) == (
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
        "flow", "pool_size", "batch_size", "train_steps", "lr",
        "checkpoint", "initialize_from_identity",
    } & set(identity)

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain, mean=[0, 0], variance=[1, 1])
    samples = source.samples(jax.random.key(20), 4)
    assert samples.shape == (4, 3)
    assert samples.dtype == jnp.float32
    assert source(samples).shape == (4,)
    identity, ess = train_forward_KLX_G(
        samples,
        samples,
        source,
        source,
        Mixed_Identity(domain),
        4,
        2,
        1e-3,
        g_clip=1.0,
    )
    assert identity(samples).shape == samples.shape
    assert ess.shape == (2,) and bool(jnp.isfinite(ess).all())
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    assert target.source().samples(jax.random.key(21), 1).dtype == jnp.float32
    print("PASS modern minimal molecular API")


if __name__ == "__main__":
    main()
