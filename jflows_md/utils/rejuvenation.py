"""Wrapped MALA and HMC on ``R^p x T^q``."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from ..core.domain import Mixed_Domain
from .control import check_rejection


def wrapped_normal_relative_error_bound(
    dt: float = 1e-4, image_radius: int = 3
) -> float:
    variance = 2.0 * dt
    first = image_radius + 1
    exponent = -(((2 * first - 1) ** 2 - 1) * math.pi**2) / (2.0 * variance)
    ratio = math.exp(-4.0 * first * math.pi**2 / variance)
    return 2.0 * math.exp(exponent) / (1.0 - ratio)


def _wrapped_log_kernel(delta: Array, variance: float, image_radius: int) -> Array:
    shifts = 2.0 * jnp.pi * jnp.arange(
        -image_radius, image_radius + 1, dtype=delta.dtype
    )
    terms = -(delta[..., None] + shifts) ** 2 / (2.0 * variance)
    return jnp.sum(logsumexp(terms, axis=-1), axis=-1)


def _proposal_log_density(
    destination: Array,
    mean: Array,
    domain: Mixed_Domain,
    variance: float,
    image_radius: int,
) -> Array:
    euclidean = (
        destination[:, : domain.euclidean_dim] - mean[:, : domain.euclidean_dim]
    )
    value = -0.5 * jnp.sum(euclidean**2 / variance, axis=-1)
    if domain.periodic_dim:
        periodic = (
            destination[:, domain.euclidean_dim :]
            - mean[:, domain.euclidean_dim :]
        )
        periodic = jnp.mod(periodic + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        value = value + _wrapped_log_kernel(periodic, variance, image_radius)
    return value


def _mala_move(
    key: Array,
    samples: Array,
    energy: Array,
    force: Array,
    potential,
    domain: Mixed_Domain,
    dt: float,
    image_radius: int,
):
    """One wrapped MALA move from a state whose energy and force are known.

    Returns ``(samples, accepted, energy, force)`` at the new state, so that a
    scan can carry the value and the gradient of ``U`` instead of evaluating
    them again: one ``value_and_grad`` per step, at the proposal. This matters
    on the exact SMC levels, whose potential contains the flow.
    """
    noise_key, accept_key = jax.random.split(key)
    mean_forward = samples - dt * force
    proposal = domain.wrap(
        mean_forward
        + jnp.sqrt(2.0 * dt)
        * jax.random.normal(noise_key, samples.shape, dtype=samples.dtype)
    )
    proposal_energy, proposal_force = potential.value_and_grad(proposal)
    mean_reverse = proposal - dt * proposal_force
    log_alpha = (
        energy
        - proposal_energy
        + _proposal_log_density(
            samples, mean_reverse, domain, 2.0 * dt, image_radius
        )
        - _proposal_log_density(
            proposal, mean_forward, domain, 2.0 * dt, image_radius
        )
    )
    log_alpha = jnp.where(
        jnp.isfinite(log_alpha), jnp.minimum(log_alpha, 0.0), -jnp.inf
    )
    accepted = jnp.log(jax.random.uniform(accept_key, energy.shape)) < log_alpha
    return (
        jnp.where(accepted[:, None], proposal, samples),
        accepted,
        jnp.where(accepted, proposal_energy, energy),
        jnp.where(accepted[:, None], proposal_force, force),
    )


def mixed_mala_step(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-4,
    image_radius: int = 3,
    energy: Array | None = None,
    force: Array | None = None,
):
    """One wrapped MALA step; ``energy`` and ``force`` supply U and grad U when known."""
    if energy is None or force is None:
        energy, force = potential.value_and_grad(samples)
    samples, accepted, _, _ = _mala_move(
        key, samples, energy, force, potential, domain, dt, image_radius
    )
    return samples, accepted


@eqx.filter_jit
def _mixed_mala_chunk(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    mc_dt: float = 1e-4,
    mc_steps: int = 1,
    image_radius: int = 3,
    energy: Array | None = None,
):
    """``mc_steps`` wrapped MALA steps on one chunk, carrying U and grad U.

    ``energy`` supplies ``U(samples)`` when the caller already has it, as the
    SMC levels do from their own reweighting.
    """
    if energy is None:
        energy, force = potential.value_and_grad(samples)
    else:
        force = potential.grad(samples)

    def body(state, subkey):
        samples, energy, force = state
        samples, accepted, energy, force = _mala_move(
            subkey, samples, energy, force, potential, domain, mc_dt, image_radius
        )
        return (samples, energy, force), accepted.astype(samples.dtype).mean()

    (samples, _, _), history = jax.lax.scan(
        body, (samples, energy, force), jax.random.split(key, mc_steps)
    )
    return samples, history


def mixed_mala(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-4,
    steps: int = 1,
    image_radius: int = 3,
    chunks: int = 1,
    energy: Array | None = None,
    reject_requested=None,
):
    parts = jnp.array_split(samples, chunks, axis=0)
    energies = (
        [None] * len(parts) if energy is None
        else jnp.array_split(energy, chunks, axis=0)
    )
    keys = (key,) if chunks == 1 else jax.random.split(key, chunks)
    values, histories = [], []
    for part_key, part, part_energy in zip(keys, parts, energies):
        check_rejection(reject_requested)
        value, history = jax.block_until_ready(
            _mixed_mala_chunk(
                part_key,
                part,
                potential,
                domain,
                mc_dt=dt,
                mc_steps=steps,
                image_radius=image_radius,
                energy=part_energy,
            )
        )
        values.append(value)
        histories.append(history * part.shape[0])
    return jnp.concatenate(values), sum(histories) / samples.shape[0]


def mixed_hmc_step(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-3,
    leapfrog_steps: int = 10,
):
    """One Metropolis-adjusted HMC trajectory on the mixed domain.

    A random momentum is drawn for every coordinate, the Hamiltonian flow is
    integrated by ``leapfrog_steps`` leapfrog steps of size ``dt`` with the
    periodic coordinates wrapped after every position update (a
    volume-preserving map on the torus, so the accept/reject test is the
    Euclidean one), and the endpoint is accepted with probability
    ``min(1, exp(H(start) - H(end)))``. A non-finite trajectory is rejected.
    """
    momentum_key, accept_key = jax.random.split(key)
    momentum = jax.random.normal(momentum_key, samples.shape, dtype=samples.dtype)
    energy = potential(samples)
    kinetic = 0.5 * jnp.sum(momentum**2, axis=-1)

    # leapfrog with the half-kicks of consecutive steps combined:
    # leapfrog_steps + 1 gradient evaluations per trajectory
    velocity = momentum - 0.5 * dt * potential.grad(samples)

    def drift_and_kick(state, _):
        position, velocity = state
        position = domain.wrap(position + dt * velocity)
        velocity = velocity - dt * potential.grad(position)
        return (position, velocity), None

    (position, velocity), _ = jax.lax.scan(
        drift_and_kick, (samples, velocity), None, length=leapfrog_steps - 1
    )
    proposal = domain.wrap(position + dt * velocity)
    momentum_end = velocity - 0.5 * dt * potential.grad(proposal)
    proposal_energy = potential(proposal)
    kinetic_end = 0.5 * jnp.sum(momentum_end**2, axis=-1)
    log_alpha = energy + kinetic - proposal_energy - kinetic_end
    log_alpha = jnp.where(
        jnp.isfinite(log_alpha), jnp.minimum(log_alpha, 0.0), -jnp.inf
    )
    accepted = jnp.log(jax.random.uniform(accept_key, energy.shape)) < log_alpha
    return jnp.where(accepted[:, None], proposal, samples), accepted


@eqx.filter_jit
def _mixed_hmc_chunk(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-3,
    leapfrog_steps: int = 10,
    trajectories: int = 1,
):
    def body(state, subkey):
        state, accepted = mixed_hmc_step(
            subkey, state, potential, domain, dt=dt, leapfrog_steps=leapfrog_steps
        )
        return state, accepted.astype(state.dtype).mean()

    return jax.lax.scan(body, samples, jax.random.split(key, trajectories))


def mixed_hmc(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-3,
    leapfrog_steps: int = 10,
    trajectories: int = 1,
    chunks: int = 1,
):
    """``trajectories`` HMC trajectories per particle, chunked along the rows.

    Returns the moved particles and the mean acceptance per trajectory, the
    same contract as ``mixed_mala``.
    """
    parts = jnp.array_split(samples, chunks, axis=0)
    keys = (key,) if chunks == 1 else jax.random.split(key, chunks)
    values, histories = [], []
    for part_key, part in zip(keys, parts):
        value, history = jax.block_until_ready(
            _mixed_hmc_chunk(
                part_key, part, potential, domain,
                dt=dt, leapfrog_steps=leapfrog_steps, trajectories=trajectories,
            )
        )
        values.append(value)
        histories.append(history * part.shape[0])
    return jnp.concatenate(values), sum(histories) / samples.shape[0]
