"""Bundle validation helpers shared by smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..system import Molecular_Bundle


@dataclass(frozen=True)
class Error_Summary:
    maximum_energy_kj_mol: float
    force_rmse_kj_mol_nm: float
    maximum_force_component_kj_mol_nm: float


def compare_stored_openmm(potential, bundle: Molecular_Bundle) -> Error_Summary:
    frames = jnp.asarray(bundle.validation["frames_nm"])
    expected_energy = np.asarray(bundle.validation["total_energy_kj_mol"])
    expected_force = np.asarray(bundle.validation["forces_kj_mol_nm"])
    energy = np.asarray(potential.forcefield(frames))
    force = np.asarray(-jax.vmap(jax.grad(lambda x: potential.forcefield(x[None])[0]))(frames))
    delta_force = force - expected_force
    return Error_Summary(
        maximum_energy_kj_mol=float(np.max(np.abs(energy - expected_energy))),
        force_rmse_kj_mol_nm=float(np.sqrt(np.mean(delta_force**2))),
        maximum_force_component_kj_mol_nm=float(np.max(np.abs(delta_force))),
    )
