#!/usr/bin/env python
"""OpenMM potential parity and native molecular sampling."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Molecular_Bundle, Molecular_Potential  # noqa: E402
from jflows_md.openmm import (  # noqa: E402
    OpenMM_Potential,
    langevin,
    parallel_tempering,
)


BUNDLES = (
    "adp_ff96_obc1",
    "glycerol_gaff2_am1bcc_obc1",
    "diethanolamine_gaff2_am1bcc_obc1",
)


def check_physical_energy() -> None:
    for name in BUNDLES:
        bundle = Molecular_Bundle.load(name)
        jax_potential = Molecular_Potential(bundle)
        openmm_potential = OpenMM_Potential(bundle)
        frames = np.asarray(bundle.validation["frames_nm"])
        expected = np.asarray(jax_potential.forcefield(jnp.asarray(frames)))
        actual = openmm_potential.physical_energy(frames, platform="Reference")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            openmm_potential(frames, platform="Reference"),
            float(jax_potential.beta) * expected,
            rtol=0,
            atol=5e-6,
        )
    print("PASS OpenMM/JAX physical-energy parity")


def check_regularization() -> None:
    bundle = Molecular_Bundle.load("glycerol_gaff2_am1bcc_obc1")
    jax_potential = Molecular_Potential(bundle).regularized((50.0, 0.1))
    openmm_potential = OpenMM_Potential(bundle).regularized((50.0, 0.1))
    frames = jnp.asarray(bundle.validation["frames_nm"])
    q = jax_potential.base.coordinates.to_internal(frames)[0]
    expected_energy = jax_potential.regularized_energy(q)
    actual_energy = openmm_potential.regularized_energy(
        np.asarray(frames), platform="Reference"
    )
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        openmm_potential.reference_energy_kj_mol,
        jax_potential.reference_energy_kj_mol,
        rtol=0,
        atol=1e-10,
    )

    floor = jax_potential.rg_param[1]

    def energy(frame):
        value = jax_potential.base.forcefield._energy_with_pair_distance_floor(
            frame[None], floor
        )[0]
        return jax_potential._regularize_energy(value)

    expected_force = -jax.vmap(jax.grad(energy))(frames)
    actual_force = openmm_potential.forces(
        np.asarray(frames), platform="Reference"
    )
    np.testing.assert_allclose(actual_force, expected_force, rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        openmm_potential.physical_energy(
            np.asarray(frames), platform="Reference"
        ),
        np.asarray(jax_potential.physical_energy(q)),
        rtol=0,
        atol=1e-5,
    )
    print("PASS OpenMM/JAX regularized energy and force parity")


def check_native_sampling() -> None:
    raw = OpenMM_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    regularized = raw.regularized((50.0, 0.1))
    for potential in (raw, regularized):
        trajectory, energies = langevin(
            potential,
            steps=4,
            sample_interval=2,
            timestep_fs=0.25,
            seed=11,
            platform="Reference",
        )
        assert trajectory.shape == (2, raw.n_atoms, 3)
        assert energies.shape == (2,)
        assert np.isfinite(trajectory).all() and np.isfinite(energies).all()

        replicas, replica_energies, acceptance = parallel_tempering(
            potential,
            (300.0, 450.0),
            rounds=2,
            steps_per_round=1,
            timestep_fs=0.25,
            seed=17,
            platform="Reference",
        )
        assert replicas.shape == (2, 2, raw.n_atoms, 3)
        assert replica_energies.shape == (2, 2)
        assert acceptance.shape == (1,)
        assert np.isfinite(replicas).all() and np.isfinite(replica_energies).all()
        assert bool(np.all((acceptance >= 0.0) & (acceptance <= 1.0)))
    print("PASS native OpenMM Langevin and parallel tempering")


def main() -> None:
    check_physical_energy()
    check_regularization()
    check_native_sampling()


if __name__ == "__main__":
    main()
