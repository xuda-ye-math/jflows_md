#!/usr/bin/env python
"""Pure-JAX energy, force, Jacobian, and API smoke tests for three molecules."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md import Molecular_Potential  # noqa: E402
from jflows_md.core.validation import compare_stored_openmm  # noqa: E402
from jflows_md.system import Molecular_Bundle  # noqa: E402


EXPECTED = {
    "fab_adp_ff96_obc1_v1": (60, 42, 18),
    "glycerol_gaff2_am1bcc_obc1_v1": (36, 25, 11),
    "diethanolamine_neutral_gaff2_am1bcc_obc1_v1": (48, 33, 15),
}


def main() -> None:
    print(f"JAX {jax.__version__} backend={jax.default_backend()}")
    for name, (dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name)
        potential = Molecular_Potential.from_bundle(name)
        assert isinstance(potential, Potential)
        assert potential.dimension == dimension
        assert potential.domain.euclidean_dim == euclidean
        assert potential.domain.periodic_dim == periodic

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


if __name__ == "__main__":
    main()
