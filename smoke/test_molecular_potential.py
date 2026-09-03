#!/usr/bin/env python
"""Pure-JAX energy, force, Jacobian, and API smoke tests for the five bundles."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md import Molecular_Potential  # noqa: E402
from jflows_md.core.coordinates import Internal_Coordinates  # noqa: E402
from jflows_md.core.forcefield import Amber_OBC_Force_Field  # noqa: E402
from smoke.molecular_validation import compare_stored_openmm  # noqa: E402
from jflows_md.system import Molecular_Bundle  # noqa: E402


EXPECTED = {
    "alanine_dipeptide_ff96_obc1": (60, 42, 18),
    "methane_gaff2_am1bcc_obc1": (9, 7, 2),
    "ethane_gaff2_am1bcc_obc1": (18, 13, 5),
    "propane_gaff2_am1bcc_obc1": (27, 19, 8),
    "n_butane_gaff2_am1bcc_obc1": (36, 25, 11),
}


def check_rigid_motion_quotient_jacobian(bundle: Molecular_Bundle) -> None:
    """Compare the quotient BAT factor with an independent square Jacobian."""
    spec = dict(bundle.coordinates)
    spec.update(
        order=spec["order"][:3],
        refs=spec["refs"][:3],
        dimension=3,
        euclidean_dim=3,
        periodic_dim=0,
        bond_log_offset=spec["bond_log_offset"][:2],
        bond_log_scale=spec["bond_log_scale"][:2],
        angle_logit_offset=spec["angle_logit_offset"][:1],
        angle_logit_scale=spec["angle_logit_scale"][:1],
        reference_torsions_rad=[],
        chiral_torsion_index=-1,
        chiral_torsion_sign=0,
        chirality_atoms=[-1, -1, -1, -1],
        chirality_sign=0,
        source_mean=[0.0, 0.0, 0.0],
        source_variance=[1.0, 1.0, 1.0],
    )
    coordinates = Internal_Coordinates(spec)
    q = jnp.zeros((1, 3))
    reported = coordinates.to_cartesian(q)[1][0]

    def lab_frame(value):
        translation, omega, internal = value[:3], value[3:6], value[6:]
        canonical = coordinates.to_cartesian(internal[None])[0][0]
        # At omega=0, cross(omega, x) is the tangent of the SO(3) action.
        return (canonical + jnp.cross(omega, canonical) + translation).reshape(-1)

    jacobian = jax.jacfwd(lab_frame)(jnp.zeros(9))
    _, autodiff = jnp.linalg.slogdet(jacobian)
    np.testing.assert_allclose(reported, autodiff, rtol=0, atol=1e-11)

def check_temperature_override() -> None:
    """Changing temperature rescales energy but preserves bundle mechanics."""

    cold = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
    hot = Molecular_Potential.from_bundle(
        "alanine_dipeptide_ff96_obc1", temperature_kelvin=600.0
    )
    assert cold.temperature_kelvin == 300.0
    assert hot.temperature_kelvin == 600.0
    np.testing.assert_allclose(hot.beta, 0.5 * cold.beta, rtol=0, atol=1e-15)
    q = cold.source().samples(jax.random.key(702), N=8)
    _, logdet = cold.coordinates.to_cartesian(q)
    expected = hot.beta * cold.physical_energy(q) - logdet
    np.testing.assert_allclose(hot(q), expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        hot.source().mean, cold.source().mean, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        hot.source().variance, 2.0 * cold.source().variance, rtol=0, atol=1e-15
    )
    assert hot.forcefield.n_atoms == cold.forcefield.n_atoms
    print("PASS molecular target-temperature override")


def check_empty_force_interactions() -> None:
    """Small molecules may legitimately omit one or more force families."""

    bundle = Molecular_Bundle.load("alanine_dipeptide_ff96_obc1")
    spec = dict(bundle.system)
    for index_name, value_names in (
        ("bond_idx", ("bond_length_nm", "bond_k_kj_mol_nm2")),
        ("angle_idx", ("angle_theta_rad", "angle_k_kj_mol_rad2")),
        (
            "torsion_idx",
            ("torsion_periodicity", "torsion_phase_rad", "torsion_k_kj_mol"),
        ),
        (
            "pair_idx",
            ("pair_chargeprod_e2", "pair_sigma_nm", "pair_epsilon_kj_mol"),
        ),
        (
            "exception_idx",
            (
                "exception_chargeprod_e2",
                "exception_sigma_nm",
                "exception_epsilon_kj_mol",
            ),
        ),
    ):
        spec[index_name] = []
        for name in value_names:
            spec[name] = []
    forcefield = Amber_OBC_Force_Field(spec)
    assert forcefield.bond_idx.shape == (0, 2)
    assert forcefield.angle_idx.shape == (0, 3)
    assert forcefield.torsion_idx.shape == (0, 4)
    assert forcefield.pair_idx.shape == (0, 2)
    assert forcefield.exception_idx.shape == (0, 2)
    terms = forcefield.energy_terms(jnp.asarray(bundle.validation["frames_nm"][:1]))
    for name in ("bond", "angle", "torsion", "nonbonded"):
        np.testing.assert_array_equal(terms[name], jnp.zeros((1,)))
    assert bool(jnp.isfinite(terms["gb"]).all() & jnp.isfinite(terms["total"]).all())
    frame = jnp.asarray(bundle.validation["frames_nm"][0])
    empty_gradient = jax.grad(
        lambda value: sum(
            forcefield.energy_terms(value[None])[name][0]
            for name in ("bond", "angle", "torsion", "nonbonded")
        )
    )(frame)
    np.testing.assert_array_equal(empty_gradient, jnp.zeros_like(frame))
    assert bool(jnp.isfinite(empty_gradient).all())

    print("PASS empty molecular force-interaction families")


def main() -> None:
    print(f"JAX {jax.__version__} backend={jax.default_backend()}")
    for name, (dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name)
        potential = Molecular_Potential.from_bundle(name)
        assert isinstance(potential, Potential)
        assert potential.dimension == dimension
        assert potential.domain.euclidean_dim == euclidean
        assert potential.domain.periodic_dim == periodic
        assert potential.coordinates.jacobian_measure == "rigid_motion_quotient_v1"

        frames = jnp.asarray(bundle.validation["frames_nm"])
        terms = potential.forcefield.energy_terms(frames)
        for term, expected in bundle.validation["term_energy_kj_mol"].items():
            error = float(jnp.max(jnp.abs(terms[term] - jnp.asarray(expected))))
            assert error < 1e-5, (name, term, error)
        comparison = compare_stored_openmm(potential, bundle)
        assert comparison.maximum_energy_kj_mol < 1e-5, comparison
        assert comparison.force_rmse_kj_mol_nm < 1e-4, comparison
        assert comparison.maximum_force_component_kj_mol_nm < 1e-3, comparison

        q, inverse_logdet = potential.coordinates.to_internal(frames[:1])
        reconstructed, logdet = potential.coordinates.to_cartesian(q)
        q_again = potential.coordinates.to_internal(reconstructed)[0]
        roundtrip = float(jnp.max(jnp.abs(potential.domain.displacement(q_again, q))))
        assert roundtrip < 1e-10, (name, roundtrip)
        assert float(jnp.max(jnp.abs(logdet + inverse_logdet))) < 1e-10
        expected_u = potential.beta * potential.forcefield(reconstructed) - logdet
        np.testing.assert_allclose(potential(q), expected_u, rtol=0, atol=1e-10)

        energy = jax.jit(lambda value: potential(value))(q)
        gradient = jax.jit(lambda value: potential.grad(value))(q)
        jax.block_until_ready((energy, gradient))
        assert energy.shape == (1,) and gradient.shape == (1, dimension)
        assert bool(jnp.isfinite(energy).all() & jnp.isfinite(gradient).all())

        source = potential.source()
        samples = source.samples(jax.random.key(7), N=32)
        assert samples.shape == (32, dimension)
        assert source(samples).shape == (32,)
        print(
            f"PASS potential {name}: "
            f"d={dimension}, dE={comparison.maximum_energy_kj_mol:.2e}, "
            f"force_rmse={comparison.force_rmse_kj_mol_nm:.2e}"
        )

    check_rigid_motion_quotient_jacobian(
        Molecular_Bundle.load("alanine_dipeptide_ff96_obc1")
    )
    print("PASS rigid-motion-quotient BAT Jacobian")
    check_temperature_override()
    check_empty_force_interactions()


if __name__ == "__main__":
    main()
