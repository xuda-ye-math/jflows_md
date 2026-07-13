#!/usr/bin/env python
"""Bounded float32 compile/run smoke for the real glycerol target."""

from __future__ import annotations

import os
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows_md import Molecular_Potential  # noqa: E402
from jflows_md.utils import mixed_mala  # noqa: E402


@eqx.filter_jit
def energy_and_gradient(target, samples):
    return target(samples), target.grad(samples)


def main() -> None:
    assert not jax.config.x64_enabled
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    reference = target.reference_internal()[None]
    samples = jnp.repeat(reference, 2, axis=0)
    assert samples.dtype == jnp.float32

    started = time.perf_counter()
    energy, gradient = energy_and_gradient(target, samples)
    jax.block_until_ready((energy, gradient))
    energy_grad_s = time.perf_counter() - started
    assert energy.dtype == jnp.float32 and gradient.dtype == jnp.float32
    assert bool(jnp.isfinite(energy).all() & jnp.isfinite(gradient).all())

    angle_index = target.coordinates.n_bonds
    offset = target.coordinates.angle_offset[0]
    scale = target.coordinates.angle_scale[0]
    near_boundary = jnp.repeat(reference, 2, axis=0)
    near_boundary = near_boundary.at[:, angle_index].set(
        (jnp.asarray([-30.0, 30.0]) - offset) / scale
    )
    boundary_energy, boundary_gradient = energy_and_gradient(target, near_boundary)
    jax.block_until_ready((boundary_energy, boundary_gradient))
    assert bool(
        jnp.isfinite(boundary_energy).all()
        & jnp.isfinite(boundary_gradient).all()
    )

    saturated = jnp.repeat(reference, 2, axis=0)
    saturated = saturated.at[:, angle_index].set(
        (jnp.asarray([-100.0, 100.0]) - offset) / scale
    )
    saturated_energy = target(saturated)
    assert bool(jnp.logical_not(jnp.isnan(saturated_energy)).all())

    started = time.perf_counter()
    moved, acceptance = mixed_mala(
        jax.random.key(30),
        samples,
        target,
        target.domain,
        dt=1e-8,
        steps=1,
        chunks=2,
    )
    jax.block_until_ready((moved, acceptance))
    mala_s = time.perf_counter() - started
    assert moved.dtype == jnp.float32 and acceptance.dtype == jnp.float32
    assert bool(jnp.isfinite(moved).all() & jnp.isfinite(acceptance).all())
    assert moved.shape == samples.shape and acceptance.shape == (1,)
    print(
        "PASS float32 glycerol compile path: "
        f"energy+grad={energy_grad_s:.2f}s one-step-MALA={mala_s:.2f}s"
    )


if __name__ == "__main__":
    main()
