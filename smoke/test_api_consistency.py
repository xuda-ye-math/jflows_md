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
from jflows.flow import (  # noqa: E402
    CircularRQSTransform,
    MonotonicRQSTransform,
    Transform,
)
from jflows.train import Monitor  # noqa: E402
from jflows_md import (  # noqa: E402
    Molecular_Monitor,
    Molecular_Potential,
    Molecular_Source,
)
from jflows_md.boltzmann import (  # noqa: E402
    molecular_boltzmann_forward_KLX_G,
    molecular_boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import (  # noqa: E402
    train_molecular_forward_KLX_G,
    train_molecular_forward_KLXX_G,
)


def expect_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    assert jflows.__file__ != jflows_md.__file__
    assert all(value is not None for value in (
        Transform,
        CircularRQSTransform,
        MonotonicRQSTransform,
    ))
    assert "checkpoint" in inspect.signature(train_molecular_forward_KLX_G).parameters
    assert "checkpoint" in inspect.signature(molecular_boltzmann_forward_KLX_G).parameters
    assert "checkpoint" in inspect.signature(train_molecular_forward_KLXX_G).parameters
    assert "checkpoint" in inspect.signature(molecular_boltzmann_forward_KLXX_G).parameters
    assert "snapshot_steps" in inspect.signature(train_molecular_forward_KLX_G).parameters
    assert "snapshot_steps" in inspect.signature(train_molecular_forward_KLXX_G).parameters
    assert "selection_steps" in inspect.signature(molecular_boltzmann_forward_KLX_G).parameters
    assert "selection_steps" in inspect.signature(molecular_boltzmann_forward_KLXX_G).parameters
    assert issubclass(Molecular_Monitor, Monitor)

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    samples = source.samples(jax.random.key(20), N=4)
    legacy_samples = source.samples(jax.random.key(20), n=4)
    assert bool(jnp.array_equal(samples, legacy_samples))
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
    print("PASS jflows API, dimensions, checkpoint controls, and float32 defaults")


if __name__ == "__main__":
    main()
