"""Native OpenMM Langevin and replica-exchange sampling."""

import numpy as np
import openmm as mm
from openmm import unit

from ..potential import KB_KJ_MOL_K


def _context(potential, temperature, friction, timestep, seed, platform):
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        timestep * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    if platform is None:
        context = mm.Context(potential.create_system(), integrator)
    else:
        selected = mm.Platform.getPlatformByName(platform)
        context = mm.Context(potential.create_system(), integrator, selected)
    return context, integrator


def _state(context):
    state = context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return np.asarray(positions), float(energy)


def langevin(
    potential,
    positions_nm=None,
    *,
    steps=1000,
    sample_interval=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    temperature_kelvin=None,
    seed=0,
    platform=None,
):
    """Run native OpenMM Langevin dynamics and return positions and energies."""
    temperature = (
        potential.temperature_kelvin
        if temperature_kelvin is None
        else float(temperature_kelvin)
    )
    positions = (
        potential.reference_positions_nm
        if positions_nm is None
        else np.asarray(positions_nm)
    )
    context, integrator = _context(
        potential,
        temperature,
        friction_per_ps,
        timestep_fs,
        seed,
        platform,
    )
    context.setPositions(positions * unit.nanometer)
    context.setVelocitiesToTemperature(temperature * unit.kelvin, seed + 1)
    trajectory = []
    energies = []
    completed = 0
    while completed < steps:
        advance = min(sample_interval, steps - completed)
        integrator.step(advance)
        positions, energy = _state(context)
        trajectory.append(positions)
        energies.append(energy)
        completed += advance
    del context, integrator
    return np.asarray(trajectory), np.asarray(energies)


def parallel_tempering(
    potential,
    temperatures_kelvin,
    positions_nm=None,
    *,
    rounds=100,
    steps_per_round=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    seed=0,
    platform=None,
):
    """Run native OpenMM replica exchange at fixed temperature slots."""
    temperatures = np.asarray(temperatures_kelvin, dtype=float)
    replicas = len(temperatures)
    initial = (
        potential.reference_positions_nm
        if positions_nm is None
        else np.asarray(positions_nm)
    )
    if initial.ndim == 2:
        initial = np.repeat(initial[None], replicas, axis=0)

    contexts = []
    integrators = []
    for index, temperature in enumerate(temperatures):
        context, integrator = _context(
            potential,
            temperature,
            friction_per_ps,
            timestep_fs,
            seed + index,
            platform,
        )
        context.setPositions(initial[index] * unit.nanometer)
        context.setVelocitiesToTemperature(
            temperature * unit.kelvin, seed + replicas + index
        )
        contexts.append(context)
        integrators.append(integrator)

    rng = np.random.default_rng(seed)
    attempts = np.zeros(replicas - 1, dtype=int)
    accepted = np.zeros(replicas - 1, dtype=int)
    trajectory = []
    energy_history = []
    beta = 1.0 / (KB_KJ_MOL_K * temperatures)
    for round_index in range(rounds):
        for integrator in integrators:
            integrator.step(steps_per_round)
        states = [_state(context) for context in contexts]
        positions = [state[0] for state in states]
        energies = np.asarray([state[1] for state in states])
        for left in range(round_index % 2, replicas - 1, 2):
            right = left + 1
            attempts[left] += 1
            log_acceptance = (beta[left] - beta[right]) * (
                energies[left] - energies[right]
            )
            if np.log(rng.random()) < min(0.0, log_acceptance):
                positions[left], positions[right] = positions[right], positions[left]
                energies[left], energies[right] = energies[right], energies[left]
                contexts[left].setPositions(positions[left] * unit.nanometer)
                contexts[right].setPositions(positions[right] * unit.nanometer)
                accepted[left] += 1
        trajectory.append(np.asarray(positions))
        energy_history.append(energies.copy())

    del contexts, integrators
    acceptance = np.divide(
        accepted,
        attempts,
        out=np.zeros_like(accepted, dtype=float),
        where=attempts > 0,
    )
    return np.asarray(trajectory), np.asarray(energy_history), acceptance
