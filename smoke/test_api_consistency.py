#!/usr/bin/env python
"""Companion API, jflows dependency, shape, and float32 contract checks."""

from __future__ import annotations

import inspect
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jflows  # noqa: E402
import jflows_md  # noqa: E402
import jflows_md.utils as md_utils  # noqa: E402
from jflows.flow import (  # noqa: E402
    CircularRQSTransform,
    MonotonicRQSTransform,
    Transform,
)
from jflows.train import Monitor  # noqa: E402
from jflows_md import (  # noqa: E402
    Molecular_Potential,
    Molecular_Source,
)
from jflows_md.boltzmann import (  # noqa: E402
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import (  # noqa: E402
    train_forward_KLX_G,
    train_forward_KLXX_G,
)


def expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    assert jflows.__file__ != jflows_md.__file__
    canonical_functions = (
        md_utils.wrapped_normal_relative_error_bound,
        md_utils.mixed_mala_step,
        md_utils.mixed_mala,
        md_utils.mixed_quench_and_temper,
        md_utils.sequential_monte_carlo,
        md_utils.potential_space_smc,
        md_utils.annealed_importance_sampling,
        train_forward_KLX_G,
        train_forward_KLXX_G,
        boltzmann_forward_KLX_G,
        boltzmann_forward_KLXX_G,
        Molecular_Source.samples,
    )
    assert all(inspect.signature(function).parameters for function in canonical_functions)
    assert md_utils.smc is md_utils.sequential_monte_carlo
    assert md_utils.ais is md_utils.annealed_importance_sampling
    assert all(value is not None for value in (
        Transform,
        CircularRQSTransform,
        MonotonicRQSTransform,
    ))
    assert "checkpoint" in inspect.signature(train_forward_KLX_G).parameters
    assert "checkpoint" in inspect.signature(boltzmann_forward_KLX_G).parameters
    assert "checkpoint" in inspect.signature(train_forward_KLXX_G).parameters
    assert "checkpoint" in inspect.signature(boltzmann_forward_KLXX_G).parameters
    assert "snapshot_steps" not in inspect.signature(train_forward_KLX_G).parameters
    assert "snapshot_steps" not in inspect.signature(train_forward_KLXX_G).parameters
    assert "selection_steps" not in inspect.signature(boltzmann_forward_KLX_G).parameters
    assert "selection_steps" not in inspect.signature(boltzmann_forward_KLXX_G).parameters
    assert tuple(inspect.signature(train_forward_KLX_G).parameters)[5:8] == (
        "batch_size",
        "train_steps",
        "lr",
    )
    canonical_bg = inspect.signature(boltzmann_forward_KLX_G).parameters
    for name in (
        "pool_size",
        "batch_size",
        "train_steps",
        "mc_dt",
        "mc_steps",
        "chunks",
        "mc_image_radius",
        "flow_dir",
    ):
        assert name in canonical_bg
    assert not any(name.startswith("_") for name in canonical_bg)
    canonical_klxx = inspect.signature(boltzmann_forward_KLXX_G).parameters
    assert "opt_alpha" in canonical_klxx and "opt_steps" in canonical_klxx
    messages = []
    Monitor(1, "[molecular] ", messages.append)._emit(1, -2.0, 0.25)
    assert messages == [
        "[molecular] step     1   loss = -2.0000e+00   ESS = 0.2500"
    ]

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    samples = source.samples(jax.random.key(20), N=4)
    assert samples.dtype == jnp.float32
    assert source(samples).dtype == jnp.float32
    expect_value_error(lambda: source(jnp.zeros((4, 2))))
    expect_value_error(lambda: source(jnp.zeros((4, 4))))
    expect_value_error(lambda: source(jnp.zeros((3,))))
    expect_value_error(lambda: source.samples(jax.random.key(21), N=0))
    integer_literal_source = Molecular_Source(
        domain, mean=[0, 0], variance=[1, 1]
    )
    integer_literal_samples = integer_literal_source.samples(
        jax.random.key(23), N=2
    )
    assert integer_literal_samples.dtype == jnp.float32

    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    q = target.source().samples(jax.random.key(22), N=1)
    assert q.dtype == jnp.float32
    expect_value_error(lambda: target(jnp.zeros((1, target.dimension - 1))))
    expect_value_error(lambda: target(jnp.zeros((1, target.dimension + 1))))
    expect_value_error(lambda: target(jnp.zeros((target.dimension,))))
    print("PASS jflows API, dimensions, final-only gate, and float32 defaults")


if __name__ == "__main__":
    main()
