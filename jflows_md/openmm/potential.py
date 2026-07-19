"""Bundle-backed potentials evaluated entirely by OpenMM."""

from pathlib import Path

import numpy as np
import openmm as mm
from openmm import unit

from ..potential import KB_KJ_MOL_K
from ..system import Molecular_Bundle


COULOMB = 138.9354576


def _platform(name):
    return None if name is None else mm.Platform.getPlatformByName(name)


def _context(system, platform=None):
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    selected = _platform(platform)
    context = mm.Context(system, integrator) if selected is None else mm.Context(system, integrator, selected)
    return context, integrator


def _evaluate(system, positions_nm, *, forces=False, platform=None):
    positions = np.asarray(positions_nm, dtype=float)
    single = positions.ndim == 2
    if single:
        positions = positions[None]
    context, integrator = _context(system, platform)
    energies = []
    force_values = []
    for frame in positions:
        context.setPositions(frame * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=forces)
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if forces:
            force_values.append(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
            )
    del context, integrator
    energies = np.asarray(energies)
    if not forces:
        return energies[0] if single else energies
    force_values = np.asarray(force_values)
    if single:
        return energies[0], force_values[0]
    return energies, force_values


def _pair_force(spec, pair_distance_floor_nm):
    expression = (
        f"{COULOMB}*chargeprod/reff"
        "+4*epsilon*((sigma/reff)^12-(sigma/reff)^6);"
        "reff=max(r,r_floor)"
    )
    force = mm.CustomBondForce(expression)
    for name in ("chargeprod", "sigma", "epsilon"):
        force.addPerBondParameter(name)
    force.addGlobalParameter("r_floor", pair_distance_floor_nm)
    for family in ("pair", "exception"):
        values = zip(
            spec[f"{family}_idx"],
            spec[f"{family}_chargeprod_e2"],
            spec[f"{family}_sigma_nm"],
            spec[f"{family}_epsilon_kj_mol"],
        )
        for index, chargeprod, sigma, epsilon in values:
            force.addBond(index[0], index[1], (chargeprod, sigma, epsilon))
    return force


def _floor_forces(system, spec, pair_distance_floor_nm):
    forces = []
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            forces.append(_pair_force(spec, pair_distance_floor_nm))
        else:
            forces.append(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
    return forces


def _replace_forces(system, forces):
    for index in range(system.getNumForces() - 1, -1, -1):
        system.removeForce(index)
    for force in forces:
        system.addForce(force)
    return system


def _regularized_system(bundle, rg_param):
    threshold, pair_floor = map(float, rg_param)
    raw = mm.XmlSerializer.deserialize((bundle.path / "system.xml").read_text())
    force_xml = [
        mm.XmlSerializer.serialize(force)
        for force in _floor_forces(raw, bundle.system, pair_floor)
    ]

    floor_system = mm.XmlSerializer.deserialize(
        (bundle.path / "system.xml").read_text()
    )
    _replace_forces(
        floor_system,
        [mm.XmlSerializer.deserialize(value) for value in force_xml],
    )
    reference = np.asarray(bundle.validation["frames_nm"][0])
    reference_energy = float(
        _evaluate(floor_system, reference, platform="Reference")
    )

    system = mm.XmlSerializer.deserialize((bundle.path / "system.xml").read_text())
    _replace_forces(system, [])
    names = [f"E{index}" for index in range(len(force_xml))]
    expression = (
        "Eref+min(delta,e)+e*log(max(delta/e,1));"
        f"delta={'+'.join(names)}-Eref"
    )
    force = mm.CustomCVForce(expression)
    force.addGlobalParameter("Eref", reference_energy)
    force.addGlobalParameter("e", threshold)
    for name, value in zip(names, force_xml):
        force.addCollectiveVariable(name, mm.XmlSerializer.deserialize(value))
    system.addForce(force)
    return mm.XmlSerializer.serialize(system), reference_energy


class OpenMM_Potential:
    """Cartesian reduced potential ``beta E(x)`` from a molecular bundle."""

    def __init__(self, bundle, *, temperature_kelvin=None):
        if not isinstance(bundle, Molecular_Bundle):
            bundle = Molecular_Bundle.load(bundle)
        self.bundle = bundle
        self.bundle_temperature_kelvin = float(bundle.manifest["temperature_kelvin"])
        self.temperature_kelvin = (
            self.bundle_temperature_kelvin
            if temperature_kelvin is None
            else float(temperature_kelvin)
        )
        self.beta = 1.0 / (KB_KJ_MOL_K * self.temperature_kelvin)
        self.reference_positions_nm = np.asarray(bundle.validation["frames_nm"][0])
        self.bundle_name = bundle.name
        self.bundle_path = str(bundle.path)
        self._system_xml = (bundle.path / "system.xml").read_text()

    @classmethod
    def from_bundle(
        cls,
        path_or_name: str | Path,
        *,
        verify: bool = True,
        temperature_kelvin: float | None = None,
    ):
        return cls(
            Molecular_Bundle.load(path_or_name, verify=verify),
            temperature_kelvin=temperature_kelvin,
        )

    @property
    def n_atoms(self):
        return self.bundle.n_atoms

    def create_system(self):
        """Return a fresh OpenMM System for this potential."""
        return mm.XmlSerializer.deserialize(self._system_xml)

    def physical_energy(self, positions_nm, *, platform=None):
        return _evaluate(self.create_system(), positions_nm, platform=platform)

    def forces(self, positions_nm, *, platform=None):
        return _evaluate(
            self.create_system(), positions_nm, forces=True, platform=platform
        )[1]

    def __call__(self, positions_nm, *, platform=None):
        return self.beta * self.physical_energy(positions_nm, platform=platform)

    def regularized(self, rg_param):
        return Regularized_OpenMM_Potential(self, rg_param)


class Regularized_OpenMM_Potential:
    """OpenMM realization of the molecular ``(e, r)`` surrogate."""

    def __init__(self, base, rg_param):
        self.base = base
        self.bundle = base.bundle
        self.rg_param = np.asarray(rg_param, dtype=float)
        self._system_xml, self.reference_energy_kj_mol = _regularized_system(
            self.bundle, self.rg_param
        )

    @property
    def n_atoms(self):
        return self.base.n_atoms

    @property
    def beta(self):
        return self.base.beta

    @property
    def temperature_kelvin(self):
        return self.base.temperature_kelvin

    @property
    def reference_positions_nm(self):
        return self.base.reference_positions_nm

    @property
    def energy_threshold_kj_mol(self):
        return self.rg_param[0]

    @property
    def pair_distance_floor_nm(self):
        return self.rg_param[1]

    def create_system(self):
        """Return a fresh regularized OpenMM System."""
        return mm.XmlSerializer.deserialize(self._system_xml)

    def physical_energy(self, positions_nm, *, platform=None):
        return self.base.physical_energy(positions_nm, platform=platform)

    def regularized_energy(self, positions_nm, *, platform=None):
        return _evaluate(self.create_system(), positions_nm, platform=platform)

    def forces(self, positions_nm, *, platform=None):
        return _evaluate(
            self.create_system(), positions_nm, forces=True, platform=platform
        )[1]

    def __call__(self, positions_nm, *, platform=None):
        return self.beta * self.regularized_energy(positions_nm, platform=platform)
