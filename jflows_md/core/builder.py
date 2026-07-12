"""OpenMM-side construction of immutable molecular bundles.

This module is intentionally JAX-free. Run it in the ``jflows`` Conda
environment, which provides OpenMM, ParmEd, and AmberTools.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..system import sha256_file
from .zmatrix import build_zmatrix, validate_zmatrix


KB_KJ_MOL_K = 0.00831446261815324
ACE_COEFFICIENT = 28.3919551


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quantity(value, units) -> float:
    return float(value.value_in_unit(units))


def _raw_internal(
    positions: np.ndarray,
    order: Sequence[int],
    refs: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds, angles, torsions = [], [], []
    for placement in range(1, len(order)):
        atom, r1 = order[placement], refs[placement][0]
        bonds.append(np.linalg.norm(positions[atom] - positions[r1]))
    for placement in range(2, len(order)):
        atom, (r1, r2, _) = order[placement], refs[placement]
        first = positions[r2] - positions[r1]
        second = positions[atom] - positions[r1]
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    for placement in range(3, len(order)):
        atom, (r1, r2, r3) = order[placement], refs[placement]
        a, b, c, d = positions[r3], positions[r2], positions[r1], positions[atom]
        b1, b2, b3 = b - a, c - b, d - c
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        b2_hat = b2 / np.linalg.norm(b2)
        m1 = np.cross(n1, b2_hat)
        torsions.append(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
    return np.asarray(bonds), np.asarray(angles), np.asarray(torsions)


def _signed_volume(positions: np.ndarray, atoms: Sequence[int]) -> float:
    center, first, second, third = map(int, atoms)
    return float(
        np.dot(
            positions[first] - positions[center],
            np.cross(
                positions[second] - positions[center],
                positions[third] - positions[center],
            ),
        )
    )


def _formula(atomic_numbers: Sequence[int]) -> str:
    from openmm.app import element

    counts = Counter(element.Element.getByAtomicNumber(int(z)).symbol for z in atomic_numbers)
    order = [symbol for symbol in ("C", "H") if symbol in counts]
    order.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in order)


def build_obc1_system(prmtop_path: str | Path):
    from openmm import app, unit

    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        implicitSolvent=app.OBC1,
        soluteDielectric=1.0,
        solventDielectric=78.5,
        implicitSolventSaltConc=0.0 * unit.molar,
        sasaMethod="ACE",
        removeCMMotion=False,
    )
    return prmtop, system


def _minimize(topology, system, positions):
    import openmm as mm
    from openmm import unit

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, tolerance=1e-6, maxIterations=5000)
    result = context.getState(getPositions=True).getPositions(asNumpy=True)
    del context, integrator
    return result


def extract_system_spec(system) -> dict[str, Any]:
    import openmm as mm
    from openmm import unit

    forces = {force.__class__.__name__: force for force in system.getForces()}
    required = {
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "CustomGBForce",
    }
    if set(forces) != required:
        raise ValueError(f"unexpected OpenMM force set: {sorted(forces)}")
    n_atoms = system.getNumParticles()

    bond_idx, bond_length, bond_k = [], [], []
    force = forces["HarmonicBondForce"]
    for index in range(force.getNumBonds()):
        first, second, length, k = force.getBondParameters(index)
        bond_idx.append([int(first), int(second)])
        bond_length.append(_quantity(length, unit.nanometer))
        bond_k.append(_quantity(k, unit.kilojoule_per_mole / unit.nanometer**2))

    angle_idx, angle_theta, angle_k = [], [], []
    force = forces["HarmonicAngleForce"]
    for index in range(force.getNumAngles()):
        first, center, third, theta, k = force.getAngleParameters(index)
        angle_idx.append([int(first), int(center), int(third)])
        angle_theta.append(_quantity(theta, unit.radian))
        angle_k.append(_quantity(k, unit.kilojoule_per_mole / unit.radian**2))

    torsion_idx, torsion_periodicity, torsion_phase, torsion_k = [], [], [], []
    force = forces["PeriodicTorsionForce"]
    for index in range(force.getNumTorsions()):
        first, second, third, fourth, periodicity, phase, k = force.getTorsionParameters(index)
        torsion_idx.append([int(first), int(second), int(third), int(fourth)])
        torsion_periodicity.append(float(periodicity))
        torsion_phase.append(_quantity(phase, unit.radian))
        torsion_k.append(_quantity(k, unit.kilojoule_per_mole))

    force = forces["NonbondedForce"]
    charges, sigmas, epsilons = [], [], []
    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        charges.append(_quantity(charge, unit.elementary_charge))
        sigmas.append(_quantity(sigma, unit.nanometer))
        epsilons.append(_quantity(epsilon, unit.kilojoule_per_mole))
    exceptions: set[frozenset[int]] = set()
    exception_idx, exception_charge, exception_sigma, exception_epsilon = [], [], [], []
    for index in range(force.getNumExceptions()):
        first, second, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
        exceptions.add(frozenset((int(first), int(second))))
        exception_idx.append([int(first), int(second)])
        exception_charge.append(_quantity(chargeprod, unit.elementary_charge**2))
        exception_sigma.append(_quantity(sigma, unit.nanometer))
        exception_epsilon.append(_quantity(epsilon, unit.kilojoule_per_mole))

    pair_idx, pair_charge, pair_sigma, pair_epsilon = [], [], [], []
    for first in range(n_atoms):
        for second in range(first + 1, n_atoms):
            if frozenset((first, second)) in exceptions:
                continue
            pair_idx.append([first, second])
            pair_charge.append(charges[first] * charges[second])
            pair_sigma.append(0.5 * (sigmas[first] + sigmas[second]))
            pair_epsilon.append(math.sqrt(epsilons[first] * epsilons[second]))

    force = forces["CustomGBForce"]
    if [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())] != [
        "charge",
        "or",
        "sr",
    ]:
        raise ValueError("bundle builder supports only OpenMM OBC1 charge/or/sr parameters")
    gb_charge, gb_or, gb_sr = [], [], []
    for index in range(force.getNumParticles()):
        charge, offset_radius, scaled_radius = map(float, force.getParticleParameters(index))
        gb_charge.append(charge)
        gb_or.append(offset_radius)
        gb_sr.append(scaled_radius)
    if not (
        np.isfinite(gb_or).all()
        and np.isfinite(gb_sr).all()
        and np.all(np.asarray(gb_or) > 0.0)
        and np.all(np.asarray(gb_sr) > 0.0)
    ):
        raise ValueError("OBC1 radii/screens must be finite and strictly positive")
    expressions = [force.getComputedValueParameters(i)[1] for i in range(force.getNumComputedValues())]
    if not any("0.8*psi+2.909125*psi^3" in expression for expression in expressions):
        raise ValueError("CustomGBForce is not the expected OBC1 functional form")

    return {
        "schema_version": 1,
        "n_atoms": n_atoms,
        "bond_idx": bond_idx,
        "bond_length_nm": bond_length,
        "bond_k_kj_mol_nm2": bond_k,
        "angle_idx": angle_idx,
        "angle_theta_rad": angle_theta,
        "angle_k_kj_mol_rad2": angle_k,
        "torsion_idx": torsion_idx,
        "torsion_periodicity": torsion_periodicity,
        "torsion_phase_rad": torsion_phase,
        "torsion_k_kj_mol": torsion_k,
        "pair_idx": pair_idx,
        "pair_chargeprod_e2": pair_charge,
        "pair_sigma_nm": pair_sigma,
        "pair_epsilon_kj_mol": pair_epsilon,
        "exception_idx": exception_idx,
        "exception_chargeprod_e2": exception_charge,
        "exception_sigma_nm": exception_sigma,
        "exception_epsilon_kj_mol": exception_epsilon,
        "gb_charge_e": gb_charge,
        "gb_offset_radius_nm": gb_or,
        "gb_scaled_offset_radius_nm": gb_sr,
        "solute_dielectric": 1.0,
        "solvent_dielectric": 78.5,
        "ace_coefficient_kj_mol_nm2": ACE_COEFFICIENT,
    }


def build_coordinate_spec(
    system_spec: Mapping[str, Any],
    positions_nm: np.ndarray,
    bonds: Sequence[Sequence[int]],
    *,
    target: str,
) -> dict[str, Any]:
    n_atoms = int(system_spec["n_atoms"])
    if target == "adp":
        order, refs = build_zmatrix(
            bonds,
            n_atoms,
            positions=positions_nm,
            root=6,
            prefix=(6, 8, 14, 10),
            overrides={
                8: (6, -1, -1),
                14: (8, 6, -1),
                10: (8, 14, 6),
            },
        )
        chiral_index = 0
        chirality_atoms = (8, 6, 14, 10)
        diagnostic_atoms = chirality_atoms
    else:
        order, refs = build_zmatrix(bonds, n_atoms, positions=positions_nm)
        chiral_index = -1
        chirality_atoms = (-1, -1, -1, -1)
        diagnostic_atoms = {
            "glycerol": (2, 1, 3, 4),
            "diethanolamine": (3, 2, 4, 12),
        }[target]
    validate_zmatrix(order, refs, bonds)
    raw_bonds, raw_angles, raw_torsions = _raw_internal(positions_nm, order, refs)

    kbt = KB_KJ_MOL_K * 300.0
    bond_parameters = {
        frozenset(pair): (length, k)
        for pair, length, k in zip(
            system_spec["bond_idx"],
            system_spec["bond_length_nm"],
            system_spec["bond_k_kj_mol_nm2"],
            strict=True,
        )
    }
    bond_scales = []
    for placement, distance in enumerate(raw_bonds, start=1):
        pair = frozenset((order[placement], refs[placement][0]))
        _, k = bond_parameters[pair]
        bond_scales.append(float(np.clip(math.sqrt(kbt / k) / distance, 0.01, 0.15)))

    angle_parameters = {}
    for triple, k in zip(system_spec["angle_idx"], system_spec["angle_k_kj_mol_rad2"], strict=True):
        first, center, third = map(int, triple)
        angle_parameters[(center, frozenset((first, third)))] = float(k)
    angle_scales = []
    for placement, theta in enumerate(raw_angles, start=2):
        atom = order[placement]
        r1, r2, _ = refs[placement]
        k = angle_parameters.get((r1, frozenset((atom, r2))))
        sigma = 0.08 if k is None else math.sqrt(kbt / k)
        fraction = theta / math.pi
        logit_scale = sigma / (math.pi * fraction * (1.0 - fraction))
        angle_scales.append(float(np.clip(logit_scale, 0.02, 0.5)))

    chiral_sign, chirality_sign = 0, 0
    if chiral_index >= 0:
        tau = float(raw_torsions[chiral_index])
        volume = _signed_volume(positions_nm, chirality_atoms)
        chiral_sign = 1 if tau > 0 else -1
        chirality_sign = 1 if volume > 0 else -1
        if abs(tau) >= math.pi or abs(tau) < 1e-6 or abs(volume) < 1e-8:
            raise ValueError("ADP reference lies too close to a chiral chart boundary")

    euclidean_dim = n_atoms - 1 + n_atoms - 2 + int(chiral_index >= 0)
    periodic_dim = n_atoms - 3 - int(chiral_index >= 0)
    source_mean = np.zeros(euclidean_dim)
    if chiral_index >= 0:
        fraction = chiral_sign * raw_torsions[chiral_index] / math.pi
        source_mean[-1] = math.log(fraction) - math.log1p(-fraction)
    return {
        "schema_version": 1,
        "chart": "log-bond_logit-angle_BAT_v1",
        "dimension": 3 * n_atoms - 6,
        "euclidean_dim": euclidean_dim,
        "periodic_dim": periodic_dim,
        "order": list(order),
        "refs": [list(row) for row in refs],
        "bond_log_offset": np.log(raw_bonds).tolist(),
        "bond_log_scale": bond_scales,
        "angle_logit_offset": (np.log(raw_angles / math.pi) - np.log1p(-raw_angles / math.pi)).tolist(),
        "angle_logit_scale": angle_scales,
        "reference_torsions_rad": raw_torsions.tolist(),
        "chiral_torsion_index": chiral_index,
        "chiral_torsion_sign": chiral_sign,
        "chirality_atoms": list(chirality_atoms),
        "chirality_sign": chirality_sign,
        "diagnostic_chirality_atoms": list(diagnostic_atoms),
        "source_mean": source_mean.tolist(),
        "source_variance": np.ones(euclidean_dim).tolist(),
    }


def build_validation_spec(system, positions, *, seed: int = 20260711) -> dict[str, Any]:
    import openmm as mm
    from openmm import unit

    reference = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    rng = np.random.default_rng(seed)
    frames = [reference]
    for scale in (2.5e-4, 5.0e-4, 1.0e-3):
        perturbation = rng.normal(size=reference.shape)
        perturbation -= perturbation.mean(axis=0, keepdims=True)
        frames.append(reference + scale * perturbation)
    frames = np.asarray(frames)

    group_for_class = {
        "HarmonicBondForce": 0,
        "HarmonicAngleForce": 1,
        "PeriodicTorsionForce": 2,
        "NonbondedForce": 3,
        "CustomGBForce": 4,
    }
    for force in system.getForces():
        force.setForceGroup(group_for_class[force.__class__.__name__])
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    totals, forces, terms = [], [], {name: [] for name in ("bond", "angle", "torsion", "nonbonded", "gb")}
    name_for_group = {0: "bond", 1: "angle", 2: "torsion", 3: "nonbonded", 4: "gb"}
    for frame in frames:
        context.setPositions(frame * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        totals.append(_quantity(state.getPotentialEnergy(), unit.kilojoule_per_mole))
        forces.append(
            np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
            ).tolist()
        )
        for group, name in name_for_group.items():
            energy = context.getState(getEnergy=True, groups=1 << group).getPotentialEnergy()
            terms[name].append(_quantity(energy, unit.kilojoule_per_mole))
    del context, integrator
    return {
        "schema_version": 1,
        "platform": "OpenMM Reference",
        "frames_nm": frames.tolist(),
        "total_energy_kj_mol": totals,
        "forces_kj_mol_nm": forces,
        "term_energy_kj_mol": terms,
        "perturbation_seed": seed,
    }


def write_bundle(
    output: str | Path,
    *,
    name: str,
    target: str,
    prmtop_path: str | Path,
    coordinate_path: str | Path,
    model: Mapping[str, Any],
    canonical_smiles: str,
    expected_formula: str,
    expected_charge: int,
    minimize: bool,
    provenance_files: Mapping[str, str | Path] | None = None,
) -> Path:
    import openmm as mm
    from openmm import app, unit
    import parmed as pmd

    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    prmtop_source, coordinate_source = Path(prmtop_path), Path(coordinate_path)
    shutil.copy2(prmtop_source, output / "system.prmtop")
    shutil.copy2(coordinate_source, output / "system.rst7")
    structure = pmd.load_file(str(prmtop_source), xyz=str(coordinate_source))
    topology_file, system = build_obc1_system(prmtop_source)
    inpcrd = app.AmberInpcrdFile(str(coordinate_source))
    positions = inpcrd.positions
    if minimize:
        positions = _minimize(topology_file.topology, system, positions)

    system_xml = mm.XmlSerializer.serialize(system)
    (output / "system.xml").write_text(system_xml, encoding="utf-8")
    with (output / "reference.pdb").open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology_file.topology, positions, handle, keepIds=True)

    atomic_numbers = [int(atom.atomic_number) for atom in structure.atoms]
    formula = _formula(atomic_numbers)
    charge = float(sum(atom.charge for atom in structure.atoms))
    if formula != expected_formula:
        raise ValueError(f"formula mismatch for {name}: {formula} != {expected_formula}")
    if abs(charge - expected_charge) > 1e-5:
        raise ValueError(f"charge mismatch for {name}: {charge} != {expected_charge}")

    system_spec = extract_system_spec(system)
    positions_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    bonds = [[bond.atom1.idx, bond.atom2.idx] for bond in structure.bonds]
    coordinate_spec = build_coordinate_spec(
        system_spec, positions_nm, bonds, target=target
    )
    validation_spec = build_validation_spec(system, positions)
    system_spec.update(
        {
            "atomic_numbers": atomic_numbers,
            "atom_names": [atom.name for atom in structure.atoms],
            "residue_names": [atom.residue.name for atom in structure.atoms],
            "bonds": bonds,
            "formula": formula,
            "net_charge_e": charge,
        }
    )
    _json_write(output / "system.json", system_spec)
    _json_write(output / "coordinates.json", coordinate_spec)
    _json_write(output / "validation.json", validation_spec)

    if provenance_files:
        provenance = output / "provenance"
        provenance.mkdir()
        for relative, source in provenance_files.items():
            destination = provenance / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    files = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files[str(path.relative_to(output))] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "name": name,
        "target": target,
        "temperature_kelvin": 300.0,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "formal_charge": expected_charge,
        "model": dict(model),
        "openmm_version": mm.__version__,
        "system_spec": "system.json",
        "coordinate_spec": "coordinates.json",
        "validation_spec": "validation.json",
        "files": files,
    }
    _json_write(output / "manifest.json", manifest)
    return output
