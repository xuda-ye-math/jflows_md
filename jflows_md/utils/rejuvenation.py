"""Wrapped MALA on ``R^p x T^q``."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from ..core.domain import Mixed_Domain


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


def mixed_mala_step(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    dt: float = 1e-4,
    image_radius: int = 3,
):
    noise_key, accept_key = jax.random.split(key)
    energy = potential(samples)
    mean_forward = samples - dt * potential.grad(samples)
    proposal = domain.wrap(
        mean_forward
        + jnp.sqrt(2.0 * dt)
        * jax.random.normal(noise_key, samples.shape, dtype=samples.dtype)
    )
    proposal_energy = potential(proposal)
    mean_reverse = proposal - dt * potential.grad(proposal)
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
    accepted = jnp.log(jax.random.uniform(accept_key, energy.shape)) < log_alpha
    return jnp.where(accepted[:, None], proposal, samples), accepted


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
):
    def body(state, subkey):
        state, accepted = mixed_mala_step(
            subkey,
            state,
            potential,
            domain,
            dt=mc_dt,
            image_radius=image_radius,
        )
        return state, accepted.astype(state.dtype).mean()

    return jax.lax.scan(body, samples, jax.random.split(key, mc_steps))


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
):
    parts = jnp.array_split(samples, chunks, axis=0)
    keys = (key,) if chunks == 1 else jax.random.split(key, chunks)
    values, histories = [], []
    for part_key, part in zip(keys, parts):
        value, history = jax.block_until_ready(
            _mixed_mala_chunk(
                part_key,
                part,
                potential,
                domain,
                mc_dt=dt,
                mc_steps=steps,
                image_radius=image_radius,
            )
        )
        values.append(value)
        histories.append(history * part.shape[0])
    return jnp.concatenate(values), sum(histories) / samples.shape[0]
