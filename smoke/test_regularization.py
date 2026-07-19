#!/usr/bin/env python
"""Molecular ``(e, r)`` regularization equations."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Molecular_Potential  # noqa: E402
from jflows_md.core.forcefield import Amber_OBC_Force_Field  # noqa: E402


def _pair_energy(distance, floor):
    x = jnp.zeros((1, 2, 3)).at[0, 1, 0].set(distance)
    return Amber_OBC_Force_Field._pair_energy_with_floor(
        x,
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        jnp.asarray([-0.2]),
        jnp.asarray([0.3]),
        jnp.asarray([0.8]),
        jnp.asarray(floor),
    )[0]


def main() -> None:
    potential = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    regularized = potential.regularized((50.0, 0.1))
    np.testing.assert_array_equal(regularized.rg_param, jnp.asarray([50.0, 0.1]))
    reference = regularized.reference_energy_kj_mol
    excess = jnp.asarray([-10.0, 0.0, 49.0, 50.0, 51.0, 5000.0])
    energy = reference + excess
    expected = reference + jnp.where(
        excess > 50.0,
        50.0 * (1.0 + jnp.log(excess / 50.0)),
        excess,
    )
    np.testing.assert_allclose(
        regularized._regularize_energy(energy), expected, rtol=2e-6, atol=2e-5
    )
    derivative = jax.grad(lambda value: regularized._regularize_energy(value))
    np.testing.assert_allclose(derivative(reference + 50.0), 1.0, atol=1e-6)

    floor = 0.2
    values = [_pair_energy(distance, floor) for distance in (0.0, 0.02, 0.10, 0.19)]
    np.testing.assert_allclose(values, values[0], rtol=0, atol=0)
    for distance in (0.0, 0.02, 0.10, 0.19):
        np.testing.assert_array_equal(
            jax.grad(lambda radius: _pair_energy(radius, floor))(distance), 0.0
        )

    q = potential.reference_internal()[None]
    expected_reference = potential.forcefield._energy_with_pair_distance_floor(
        potential.reference_positions_nm[None], regularized.rg_param[1]
    )[0]
    np.testing.assert_allclose(
        regularized.reference_energy_kj_mol,
        expected_reference,
        rtol=0,
        atol=2e-5,
    )
    np.testing.assert_array_equal(
        regularized.physical_energy(q), potential.physical_energy(q)
    )
    assert tuple(inspect.signature(Molecular_Potential.regularized).parameters) == (
        "self",
        "rg_param",
    )
    print("PASS molecular (e, r) regularization")


if __name__ == "__main__":
    main()
