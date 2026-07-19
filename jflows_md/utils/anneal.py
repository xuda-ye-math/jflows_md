"""Mixed-domain SMC and flow-proposal annealing."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential, linear_combination
from jflows.utils import compute_ESS_log, linear_weights_from_log, resample

from ..core.domain import Mixed_Domain
from .rejuvenation import mixed_mala


@eqx.filter_jit
def _bridge_log_weight(samples, source, target, delta):
    return delta * (source(samples) - target(samples))


def _bridge_weights(samples, source, target, delta, chunks):
    coefficient = jnp.asarray(delta, dtype=samples.dtype)
    return jnp.concatenate([
        jax.block_until_ready(_bridge_log_weight(part, source, target, coefficient))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])


def _potential_space_schedule(
    key: Array,
    samples: Array,
    source,
    target,
    levels,
    *,
    mc_dt: float,
    mc_steps: int,
    mc_image_radius: int,
    domain: Mixed_Domain | None,
    chunks: int,
):
    current = samples
    domain = domain if domain is not None else target.domain
    ess_values, acceptance_values = [], []
    previous = 0.0
    for index, level in enumerate(levels, start=1):
        log_weight = _bridge_weights(
            current, source, target, level - previous, chunks
        )
        ess_values.append(jax.block_until_ready(compute_ESS_log(log_weight)))
        resample_key, mala_key = jax.random.split(jax.random.fold_in(key, index))
        bridge = linear_combination([source, target], [1.0 - level, level])
        current = resample(
            resample_key,
            current,
            linear_weights_from_log(log_weight),
            N=current.shape[0],
        )
        current, acceptance = mixed_mala(
            mala_key,
            current,
            bridge,
            domain,
            dt=mc_dt,
            steps=mc_steps,
            image_radius=mc_image_radius,
            chunks=chunks,
        )
        acceptance_values.append(acceptance)
        previous = level
    return current, jnp.asarray(ess_values), jnp.stack(acceptance_values)


def sequential_monte_carlo(
    key: Array,
    samples: Array,
    source,
    target,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    mc_image_radius: int = 3,
    domain: Mixed_Domain | None = None,
    chunks: int = 1,
):
    levels = tuple(level / ladder for level in range(1, ladder + 1))
    return _potential_space_schedule(
        key,
        samples,
        source,
        target,
        levels,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        mc_image_radius=mc_image_radius,
        domain=domain,
        chunks=chunks,
    )


def potential_space_smc(
    key: Array,
    samples: Array,
    source,
    target,
    t_list,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    mc_image_radius: int = 3,
    domain: Mixed_Domain | None = None,
    chunks: int = 1,
):
    return _potential_space_schedule(
        key,
        samples,
        source,
        target,
        tuple(t_list),
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        mc_image_radius=mc_image_radius,
        domain=domain,
        chunks=chunks,
    )


@eqx.filter_jit
def _initial_flow_proposal(flow, samples, source: Potential, target: Potential):
    proposal, ladj = flow.inv_and_ladj(samples)
    return proposal, source(samples) - target(proposal) + ladj


@eqx.filter_jit
def _ais_log_weight(samples, source, target, flow, scale):
    latent, ladj = flow.call_and_ladj(samples)
    return scale * (-target(samples) + source(latent) - ladj)


def annealed_importance_sampling(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    mc_image_radius: int = 3,
    domain: Mixed_Domain | None = None,
    chunks: int = 1,
    return_initial_log_weights: bool = False,
):
    domain = domain if domain is not None else getattr(target, "domain", flow.domain)
    pushed, initial = zip(*[
        jax.block_until_ready(_initial_flow_proposal(flow, part, source, target))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    current = jnp.concatenate(pushed)
    initial_log_weight = jnp.concatenate(initial)
    for level in range(1, ladder + 1):
        if level == 1:
            log_weight = initial_log_weight / ladder
        else:
            scale = jnp.asarray(1.0 / ladder, dtype=samples.dtype)
            log_weight = jnp.concatenate([
                jax.block_until_ready(
                    _ais_log_weight(part, source, target, flow, scale)
                )
                for part in jnp.array_split(current, chunks, axis=0)
            ])
        resample_key, mala_key = jax.random.split(jax.random.fold_in(key, level))
        current = resample(
            resample_key,
            current,
            linear_weights_from_log(log_weight),
            N=current.shape[0],
        )
        current = mixed_mala(
            mala_key,
            current,
            target,
            domain,
            dt=mc_dt,
            steps=mc_steps,
            image_radius=mc_image_radius,
            chunks=chunks,
        )[0]
    if return_initial_log_weights:
        return current, initial_log_weight
    return current
