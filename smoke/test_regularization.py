#!/usr/bin/env python
"""Focused float32 tests for molecular energy/distance regularization."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Molecular_Potential  # noqa: E402
from jflows_md.core.forcefield import (  # noqa: E402
    Amber_OBC_Force_Field,
    _distance_with_zero_subgradient,
)
from jflows_md.potential import _minimum_finite_pair_floor_nm  # noqa: E402


def _pair_energy(
    distance,
    *,
    charge=0.0,
    sigma=0.3,
    epsilon=1.0,
    floor=0.0,
):
    x = jnp.zeros((1, 2, 3)).at[0, 1, 0].set(distance)
    return Amber_OBC_Force_Field._pair_energy_with_floor(
        x,
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([charge]),
        jnp.asarray([sigma]),
        jnp.asarray([epsilon]),
        jnp.asarray(floor),
    )[0]


def check_public_energy_map() -> None:
    potential = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    regularized = potential.regularized(
        50.0, pair_distance_floor_nm=0.1
    )
    assert regularized.energy_threshold_kj_mol.dtype == jnp.float32
    assert regularized.pair_distance_floor_nm.dtype == jnp.float32

    threshold = regularized.energy_threshold_kj_mol
    reference = regularized.reference_energy_kj_mol
    energy = reference + jnp.asarray([-10.0, 0.0, 49.0, 50.0, 51.0, 5000.0])
    expected = reference + jnp.asarray(
        [-10.0, 0.0, 49.0, 50.0, 50.0 + 50.0 * jnp.log1p(1.0 / 50.0),
         50.0 + 50.0 * jnp.log1p(4950.0 / 50.0)]
    )
    np.testing.assert_allclose(
        regularized._regularize_energy(energy), expected, rtol=2e-6, atol=2e-5
    )
    derivative = jax.grad(
        lambda value: regularized._regularize_energy(value)
    )
    np.testing.assert_allclose(derivative(reference + threshold), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        derivative(reference + threshold + jnp.asarray(1e-3)),
        1.0,
        rtol=3e-5,
        atol=3e-5,
    )

    shift = jnp.asarray(123.0)
    shifted = eqx.tree_at(
        lambda value: value.reference_energy_kj_mol,
        regularized,
        reference + shift,
    )
    np.testing.assert_allclose(
        shifted._regularize_energy(energy + shift),
        regularized._regularize_energy(energy) + shift,
        rtol=2e-6,
        atol=2e-5,
    )

    signature = inspect.signature(Molecular_Potential.regularized)
    assert tuple(signature.parameters) == (
        "self",
        "energy_threshold_kj_mol",
        "pair_distance_floor_nm",
    )
    assert signature.parameters["pair_distance_floor_nm"].default == 0.0
    for retired in (
        "energy_cut_kj_mol",
        "energy_scale_kj_mol",
        "tail_fraction",
    ):
        assert retired not in signature.parameters

    finite_overflow = float(np.finfo(np.float32).max) * 2.0
    positive_underflow = np.nextafter(0.0, 1.0, dtype=np.float64)
    invalid = (
        (True, 0.0),
        (50.0, np.bool_(False)),
        (jnp.asarray(True), 0.0),
        (50.0, jnp.asarray(False)),
        ([50.0], 0.0),
        (finite_overflow, 0.0),
        (50.0, finite_overflow),
        (50.0, positive_underflow),
        (50.0, 1e-30),
    )
    for energy_threshold, distance_floor in invalid:
        try:
            potential.regularized(
                energy_threshold,
                pair_distance_floor_nm=distance_floor,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid regularizer survived host/float32 validation: "
                f"{energy_threshold!r}, {distance_floor!r}"
            )

    smallest_float32 = float(np.nextafter(np.float32(0), np.float32(1)))
    subnormal = potential.regularized(smallest_float32)
    mapped = subnormal._regularize_energy(
        jnp.asarray([jnp.finfo(jnp.float32).max])
    )
    assert bool(jnp.isfinite(mapped).all())


def check_pair_floor() -> None:
    distance = jnp.asarray(0.31)
    raw_x = jnp.zeros((1, 2, 3)).at[0, 1, 0].set(distance)
    raw = Amber_OBC_Force_Field._pair_energy(
        raw_x,
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([-0.2]),
        jnp.asarray([0.3]),
        jnp.asarray([0.8]),
    )[0]
    floored_zero = _pair_energy(
        distance, charge=-0.2, sigma=0.3, epsilon=0.8, floor=0.0
    )
    np.testing.assert_allclose(floored_zero, raw, rtol=2e-6, atol=2e-5)

    floor = jnp.asarray(0.2)
    below = [_pair_energy(value, floor=floor) for value in (0.02, 0.10, 0.19)]
    np.testing.assert_allclose(below, below[0], rtol=0, atol=0)
    for value in (0.02, 0.10, 0.19):
        gradient = jax.grad(lambda radius: _pair_energy(radius, floor=floor))(
            jnp.asarray(value)
        )
        np.testing.assert_array_equal(gradient, jnp.asarray(0.0))
    left = jnp.nextafter(floor, jnp.asarray(0.0))
    right = jnp.nextafter(floor, jnp.asarray(jnp.inf))
    np.testing.assert_array_equal(
        jax.grad(lambda radius: _pair_energy(radius, floor=floor))(left),
        jnp.asarray(0.0),
    )
    assert bool(
        jnp.isfinite(
            jax.grad(lambda radius: _pair_energy(radius, floor=floor))(right)
        )
    )

    collision_cases = (
        (dict(charge=-1.0, sigma=0.3, epsilon=1.0), jnp.inf),
        (dict(charge=1.0, sigma=0.0, epsilon=1.0), jnp.inf),
        (dict(charge=-1.0, sigma=0.0, epsilon=1.0), -jnp.inf),
        (dict(charge=0.0, sigma=0.0, epsilon=1.0), 0.0),
        (dict(charge=0.0, sigma=0.3, epsilon=0.0), 0.0),
    )
    for kwargs, expected in collision_cases:
        value = _pair_energy(0.0, floor=0.0, **kwargs)
        if jnp.isposinf(expected):
            assert bool(jnp.isposinf(value))
        elif jnp.isneginf(expected):
            assert bool(jnp.isneginf(value))
        else:
            np.testing.assert_array_equal(value, jnp.asarray(expected))
    huge_floor = _pair_energy(0.0, floor=jnp.asarray(1e30))
    huge_gradient = jax.grad(
        lambda radius: _pair_energy(radius, floor=jnp.asarray(1e30))
    )(jnp.asarray(0.0))
    assert bool(jnp.isfinite(huge_floor) & jnp.isfinite(huge_gradient))
    np.testing.assert_array_equal(huge_gradient, jnp.asarray(0.0))

    empty = jnp.asarray([], dtype=jnp.float32)
    tiny_lj = SimpleNamespace(
        pair_sigma=jnp.asarray([0.3]),
        exception_sigma=empty,
        pair_epsilon=jnp.asarray([1e-30]),
        exception_epsilon=empty,
        pair_chargeprod=jnp.asarray([0.0]),
        exception_chargeprod=empty,
    )
    tiny_lj_floor = _minimum_finite_pair_floor_nm(tiny_lj, jnp.float32)
    tiny_lj_energy = _pair_energy(
        0.0,
        sigma=0.3,
        epsilon=1e-30,
        floor=tiny_lj_floor,
    )
    assert bool(jnp.isfinite(tiny_lj_energy))

    tiny_charge = SimpleNamespace(
        pair_sigma=jnp.asarray([0.0]),
        exception_sigma=empty,
        pair_epsilon=jnp.asarray([0.0]),
        exception_epsilon=empty,
        pair_chargeprod=jnp.asarray([1e-30]),
        exception_chargeprod=empty,
    )
    tiny_charge_floor = _minimum_finite_pair_floor_nm(tiny_charge, jnp.float32)
    tiny_charge_energy = _pair_energy(
        0.0,
        charge=1e-30,
        sigma=0.0,
        epsilon=0.0,
        floor=tiny_charge_floor,
    )
    assert bool(jnp.isfinite(tiny_charge_energy))


def _directed_obc(displacement, radius, scaled):
    squared = jnp.sum(displacement * displacement)
    distance = _distance_with_zero_subgradient(squared)[None, None, None]
    return Amber_OBC_Force_Field._obc_descreening_integral(
        distance,
        jnp.asarray(radius)[None, None, None],
        jnp.asarray(scaled)[None, None, None],
        jnp.zeros((1, 1, 1), dtype=bool),
    )[0, 0, 0]


def check_obc_collision_stability() -> None:
    zero = jnp.zeros((3,))
    for radius, scaled in ((0.15, 0.10), (0.15, 0.15), (0.15, 0.20)):
        value = _directed_obc(zero, radius, scaled)
        gradient = jax.grad(
            lambda displacement: _directed_obc(displacement, radius, scaled)
        )(zero)
        np.testing.assert_array_equal(value, jnp.asarray(0.0))
        np.testing.assert_array_equal(gradient, jnp.zeros((3,)))

        distances = jnp.logspace(-8, -3, 12)
        values = jax.vmap(
            lambda distance: _directed_obc(
                jnp.asarray([distance, 0.0, 0.0]), radius, scaled
            )
        )(distances)
        assert bool(jnp.isfinite(values).all())
        if scaled < radius:
            np.testing.assert_array_equal(values, jnp.zeros_like(values))


def check_live_floor_path() -> None:
    potential = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    frame = potential.reference_positions_nm[None]
    tiny_floor = jnp.asarray(1e-6, dtype=frame.dtype)
    raw = potential.forcefield(frame)
    floor_energy = potential.forcefield._energy_with_pair_distance_floor(
        frame, tiny_floor
    )
    np.testing.assert_allclose(floor_energy, raw, rtol=3e-6, atol=3e-4)
    raw_gradient = jax.grad(lambda value: potential.forcefield(value[None])[0])(
        frame[0]
    )
    floor_gradient = jax.grad(
        lambda value: potential.forcefield._energy_with_pair_distance_floor(
            value[None], tiny_floor
        )[0]
    )(frame[0])
    np.testing.assert_allclose(
        floor_gradient, raw_gradient, rtol=3e-5, atol=3e-3
    )

    regularized = potential.regularized(
        50.0, pair_distance_floor_nm=0.1
    )
    expected_reference = potential.forcefield._energy_with_pair_distance_floor(
        frame, regularized.pair_distance_floor_nm
    )[0]
    np.testing.assert_allclose(
        regularized.reference_energy_kj_mol,
        expected_reference,
        rtol=0,
        atol=2e-5,
    )
    q = potential.reference_internal()[None]
    np.testing.assert_allclose(
        regularized.physical_energy(q), potential.physical_energy(q), rtol=0, atol=0
    )


def main() -> None:
    assert not jax.config.x64_enabled
    check_public_energy_map()
    check_pair_floor()
    check_obc_collision_stability()
    check_live_floor_path()
    print("PASS float32 molecular energy/distance regularization")


if __name__ == "__main__":
    main()
