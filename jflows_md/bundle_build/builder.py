"""OpenMM-side construction of minimal molecular runtime bundles.

This module is intentionally JAX-free. Ordinary runtime and training load the
checked-in pure-array bundles directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
import operator
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .zmatrix import build_zmatrix, validate_zmatrix


KB_KJ_MOL_K = 0.00831446261815324
ACE_COEFFICIENT = 28.3919551

_LEGACY_COORDINATE_OPTIONS = {
    "alanine_dipeptide": {
        "zmatrix": {
            "root": 6,
            "prefix": (6, 8, 14, 10),
            "overrides": {
                8: (6, -1, -1),
                14: (8, 6, -1),
                10: (8, 14, 6),
            },
        },
        "fixed_stereocenters": (
            {
                "label": "legacy",
                "torsion_index": 0,
                "atoms": (8, 6, 14, 10),
            },
        ),
    },
}


def _integer(value, *, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return result


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
    target: str | None = None,
    zmatrix: Mapping[str, Any] | None = None,
    fixed_stereocenters: Sequence[Mapping[str, Any]] = (),
    signed_volume_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an explicit mixed-coordinate chart.

    Each fixed stereocenter names a Z-matrix torsion and the four atoms of
    its signed-volume diagnostic. The reference geometry determines both
    allowed signs. The optional ``target`` retains the historical coordinate
    defaults for alanine dipeptide when no explicit coordinate
    option is supplied; otherwise no stereochemistry is inferred from a name.
    """

    legacy_mode = (
        target in _LEGACY_COORDINATE_OPTIONS
        and zmatrix is None
        and fixed_stereocenters == ()
        and signed_volume_diagnostics is None
    )
    if legacy_mode:
        legacy = _LEGACY_COORDINATE_OPTIONS[target]
        zmatrix = legacy.get("zmatrix")
        fixed_stereocenters = legacy.get("fixed_stereocenters", ())
        signed_volume_diagnostics = legacy.get(
            "signed_volume_diagnostics"
        )

    n_atoms = _integer(system_spec["n_atoms"], name="n_atoms")
    if zmatrix is not None and not isinstance(zmatrix, Mapping):
        raise ValueError("zmatrix must be a mapping")
    options = dict(zmatrix or {})
    unknown = set(options) - {"root", "prefix", "overrides"}
    if unknown:
        raise ValueError(f"unknown Z-matrix options: {sorted(unknown)}")
    overrides = options.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("Z-matrix overrides must be a mapping")
        overrides = {
            _integer(atom, name="Z-matrix override atom"): tuple(
                _integer(
                    reference,
                    name=f"Z-matrix override {atom!r} reference",
                )
                for reference in references
            )
            for atom, references in overrides.items()
        }
    root = options.get("root")
    if root is not None:
        root = _integer(root, name="Z-matrix root")
    prefix = tuple(
        _integer(atom, name=f"Z-matrix prefix[{position}]")
        for position, atom in enumerate(options.get("prefix", ()))
    )
    order, refs = build_zmatrix(
        bonds,
        n_atoms,
        positions=positions_nm,
        root=root,
        prefix=prefix,
        overrides=overrides,
    )
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

    if isinstance(fixed_stereocenters, (str, bytes)) or not isinstance(
        fixed_stereocenters, Sequence
    ):
        raise ValueError("fixed_stereocenters must be a sequence")
    bond_set = {frozenset(map(int, pair)) for pair in bonds}
    fixed = []
    for position, item in enumerate(fixed_stereocenters):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"fixed_stereocenters[{position}] must be a mapping"
            )
        label = str(item.get("label", f"stereocenter_{position}"))
        if not label:
            raise ValueError(
                f"fixed_stereocenters[{position}].label must be nonempty"
        )
        try:
            torsion_index = _integer(
                item["torsion_index"],
                name=f"fixed_stereocenters[{position}].torsion_index",
            )
            atoms = tuple(
                _integer(
                    atom,
                    name=f"fixed_stereocenters[{position}].atoms[{atom_position}]",
                )
                for atom_position, atom in enumerate(item["atoms"])
            )
        except KeyError as exc:
            raise ValueError(
                f"fixed_stereocenters[{position}] is missing {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fixed_stereocenters[{position}] has invalid indices"
            ) from exc
        if torsion_index < 0 or torsion_index >= n_atoms - 3:
            raise ValueError(
                f"fixed_stereocenters[{position}].torsion_index is out of range"
            )
        if len(atoms) != 4 or len(set(atoms)) != 4 or any(
            atom < 0 or atom >= n_atoms for atom in atoms
        ):
            raise ValueError(
                f"fixed_stereocenters[{position}].atoms must contain four "
                "distinct in-range indices"
            )
        placement = torsion_index + 3
        placed_atom = order[placement]
        r1, r2, r3 = refs[placement]
        if atoms[0] != r1 or set(atoms[1:]) != {r2, r3, placed_atom}:
            raise ValueError(
                f"fixed_stereocenters[{position}] is not represented by "
                "its selected Z-matrix torsion"
            )
        if any(
            frozenset((atoms[0], substituent)) not in bond_set
            for substituent in atoms[1:]
        ):
            raise ValueError(
                f"fixed_stereocenters[{position}] does not describe three "
                "substituents bonded to its center"
            )
        tau = float(raw_torsions[torsion_index])
        volume = _signed_volume(positions_nm, atoms)
        boundary_distance = min(abs(tau), math.pi - abs(tau))
        if boundary_distance < 1e-6 or abs(volume) < 1e-8:
            raise ValueError(
                f"fixed_stereocenters[{position}] reference lies too close "
                "to a stereochemical chart boundary"
            )
        configuration = item.get("configuration")
        cip_priority_atoms = item.get("cip_priority_atoms")
        if (configuration is None) != (cip_priority_atoms is None):
            raise ValueError(
                f"fixed_stereocenters[{position}] must provide both "
                "configuration and cip_priority_atoms"
            )
        configuration_metadata = {}
        if configuration is not None:
            configuration = str(configuration).upper()
            if configuration not in {"R", "S"}:
                raise ValueError(
                    f"fixed_stereocenters[{position}].configuration must be "
                    "'R' or 'S'"
                )
            try:
                cip_atoms = tuple(
                    _integer(
                        atom,
                        name=(
                            f"fixed_stereocenters[{position}]."
                            f"cip_priority_atoms[{atom_position}]"
                        ),
                    )
                    for atom_position, atom in enumerate(cip_priority_atoms)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fixed_stereocenters[{position}].cip_priority_atoms "
                    "is invalid"
                ) from exc
            if len(cip_atoms) != 4 or len(set(cip_atoms)) != 4 or any(
                atom < 0 or atom >= n_atoms for atom in cip_atoms
            ):
                raise ValueError(
                    f"fixed_stereocenters[{position}].cip_priority_atoms "
                    "must contain four distinct in-range indices"
                )
            if any(
                frozenset((atoms[0], substituent)) not in bond_set
                for substituent in cip_atoms
            ):
                raise ValueError(
                    f"fixed_stereocenters[{position}].cip_priority_atoms "
                    "must be the four substituents bonded to the center"
                )
            first, second, third, fourth = positions_nm[list(cip_atoms)]
            cip_volume = float(
                np.dot(
                    first - fourth,
                    np.cross(second - fourth, third - fourth),
                )
            )
            if abs(cip_volume) < 1e-8:
                raise ValueError(
                    f"fixed_stereocenters[{position}] reference has "
                    "degenerate CIP geometry"
                )
            observed = "R" if cip_volume < 0.0 else "S"
            if observed != configuration:
                raise ValueError(
                    f"fixed_stereocenters[{position}] reference is {observed}, "
                    f"expected {configuration}"
                )
            configuration_metadata = {
                "configuration": configuration,
                "cip_priority_atoms": list(cip_atoms),
            }
        fixed.append(
            {
                "label": label,
                "torsion_index": torsion_index,
                "torsion_sign": 1 if tau > 0 else -1,
                "atoms": list(atoms),
                "volume_sign": 1 if volume > 0 else -1,
                **configuration_metadata,
            }
        )
    labels = [item["label"] for item in fixed]
    torsion_indices = [item["torsion_index"] for item in fixed]
    center_atoms = [item["atoms"][0] for item in fixed]
    if len(set(labels)) != len(labels):
        raise ValueError("fixed stereocenter labels must be unique")
    if len(set(torsion_indices)) != len(torsion_indices):
        raise ValueError("fixed stereocenter torsion indices must be unique")
    if len(set(center_atoms)) != len(center_atoms):
        raise ValueError("fixed stereocenter center atoms must be unique")

    if signed_volume_diagnostics is None:
        diagnostics_input = tuple(
            {"label": item["label"], "atoms": item["atoms"]}
            for item in fixed
        )
    else:
        diagnostics_input = signed_volume_diagnostics
    if isinstance(diagnostics_input, (str, bytes)) or not isinstance(
        diagnostics_input, Sequence
    ):
        raise ValueError("signed_volume_diagnostics must be a sequence")
    diagnostics = []
    for position, item in enumerate(diagnostics_input):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"signed_volume_diagnostics[{position}] must be a mapping"
            )
        label = str(item.get("label", f"signed_volume_{position}"))
        try:
            atoms = tuple(
                _integer(
                    atom,
                    name=(
                        f"signed_volume_diagnostics[{position}].atoms"
                        f"[{atom_position}]"
                    ),
                )
                for atom_position, atom in enumerate(item["atoms"])
            )
        except KeyError as exc:
            raise ValueError(
                f"signed_volume_diagnostics[{position}] is missing 'atoms'"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"signed_volume_diagnostics[{position}].atoms is invalid"
            ) from exc
        if not label or len(atoms) != 4 or len(set(atoms)) != 4 or any(
            atom < 0 or atom >= n_atoms for atom in atoms
        ):
            raise ValueError(
                f"signed_volume_diagnostics[{position}] must have a nonempty "
                "label and four distinct in-range atom indices"
            )
        volume = _signed_volume(positions_nm, atoms)
        if abs(volume) < 1e-8:
            raise ValueError(
                f"signed_volume_diagnostics[{position}] reference volume is "
                "too close to zero"
            )
        diagnostics.append(
            {
                "label": label,
                "atoms": list(atoms),
                "reference_sign": 1 if volume > 0 else -1,
            }
        )
    diagnostic_labels = [item["label"] for item in diagnostics]
    if len(set(diagnostic_labels)) != len(diagnostic_labels):
        raise ValueError("signed-volume diagnostic labels must be unique")

    n_fixed = len(fixed)
    euclidean_dim = n_atoms - 1 + n_atoms - 2 + n_fixed
    periodic_dim = n_atoms - 3 - n_fixed
    source_mean = np.zeros(euclidean_dim)
    stereochemical_start = n_atoms - 1 + n_atoms - 2
    for position, item in enumerate(fixed):
        fraction = (
            item["torsion_sign"]
            * raw_torsions[item["torsion_index"]]
            / math.pi
        )
        source_mean[stereochemical_start + position] = (
            math.log(fraction) - math.log1p(-fraction)
        )
    result = {
        "schema_version": 3,
        "chart": "log-bond_logit-angle_quotient_BAT_v3",
        "jacobian_measure": "rigid_motion_quotient_v1",
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
        "fixed_stereocenters": fixed,
        "signed_volume_diagnostics": diagnostics,
        "source_mean": source_mean.tolist(),
        "source_variance": np.ones(euclidean_dim).tolist(),
    }
    if legacy_mode:
        if fixed:
            singleton = fixed[0]
            chiral_torsion_index = singleton["torsion_index"]
            chiral_torsion_sign = singleton["torsion_sign"]
            chirality_atoms = singleton["atoms"]
            chirality_sign = singleton["volume_sign"]
        else:
            chiral_torsion_index = -1
            chiral_torsion_sign = 0
            chirality_atoms = [-1, -1, -1, -1]
            chirality_sign = 0
        result.update(
            schema_version=2,
            chart="log-bond_logit-angle_quotient_BAT_v2",
            chiral_torsion_index=chiral_torsion_index,
            chiral_torsion_sign=chiral_torsion_sign,
            chirality_atoms=chirality_atoms,
            chirality_sign=chirality_sign,
            diagnostic_chirality_atoms=diagnostics[0]["atoms"],
        )
        del result["fixed_stereocenters"]
        del result["signed_volume_diagnostics"]
    return result


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


def _write_bundle(
    output: Path,
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
    zmatrix: Mapping[str, Any] | None,
    fixed_stereocenters: Sequence[Mapping[str, Any]],
    signed_volume_diagnostics: Sequence[Mapping[str, Any]] | None,
) -> Path:
    import openmm as mm
    from openmm import app, unit
    import parmed as pmd

    prmtop_source, coordinate_source = Path(prmtop_path), Path(coordinate_path)
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
        system_spec,
        positions_nm,
        bonds,
        target=target,
        zmatrix=zmatrix,
        fixed_stereocenters=fixed_stereocenters,
        signed_volume_diagnostics=signed_volume_diagnostics,
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

    manifest = {
        "schema_version": 1,
        "name": name,
        "target": target,
        "temperature_kelvin": 300.0,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "formal_charge": expected_charge,
        "model": dict(model),
        "coordinate_measure": coordinate_spec["jacobian_measure"],
        "openmm_version": mm.__version__,
        "system_spec": "system.json",
        "coordinate_spec": "coordinates.json",
        "validation_spec": "validation.json",
    }
    _json_write(output / "manifest.json", manifest)
    return output


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
    zmatrix: Mapping[str, Any] | None = None,
    fixed_stereocenters: Sequence[Mapping[str, Any]] = (),
    signed_volume_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Build completely off-path, then publish to a new destination."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}.",
        ignore_cleanup_errors=True,
    ) as temporary:
        _write_bundle(
            Path(temporary),
            name=name,
            target=target,
            prmtop_path=prmtop_path,
            coordinate_path=coordinate_path,
            model=model,
            canonical_smiles=canonical_smiles,
            expected_formula=expected_formula,
            expected_charge=expected_charge,
            minimize=minimize,
            zmatrix=zmatrix,
            fixed_stereocenters=fixed_stereocenters,
            signed_volume_diagnostics=signed_volume_diagnostics,
        )
        if output.exists():
            raise FileExistsError(output)
        Path(temporary).rename(output)
    return output
