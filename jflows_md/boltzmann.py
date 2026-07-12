"""Potential-space molecular Boltzmann-generator training."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, resample

from .train import train_molecular_forward_KLX_G
from .utils import mixed_mala, sequential_monte_carlo


__all__ = ["molecular_boltzmann_forward_KLX_G"]


_DEFAULTS = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 2.0,
    "tau_smc": 0.75,
    "tau_ess": 0.4,
    "t_tol": 1e-3,
    "max_stages": 25,
    "max_retry": 8,
}


def _operation_key(base_key: Array, namespace: int, *indices: int) -> Array:
    """Derive a deterministic key in a disjoint operation namespace."""
    key = jax.random.fold_in(base_key, namespace)
    for index in indices:
        key = jax.random.fold_in(key, index)
    return key


def _finite_log_weights(log_weight: Array) -> Array:
    return jnp.where(jnp.isfinite(log_weight), log_weight, -jnp.inf)


def _ess(log_weight: Array) -> float:
    value = compute_ESS_log(_finite_log_weights(log_weight))
    return float(jnp.where(jnp.isfinite(value), value, 0.0))


def _linear_weights(log_weight: Array) -> Array:
    log_weight = _finite_log_weights(log_weight)
    finite = jnp.isfinite(log_weight)
    maximum = jnp.max(jnp.where(finite, log_weight, -jnp.inf))
    weight = jnp.where(finite, jnp.exp(log_weight - maximum), 0.0)
    return jnp.where(weight.sum() > 0, weight, jnp.ones_like(weight))


@eqx.filter_jit
def _identity_weight_chunk(samples, source, target) -> Array:
    return source(samples) - target(samples)


@eqx.filter_jit
def _inverse_chunk(flow, samples) -> Array:
    return flow.inv(samples)


@eqx.filter_jit
def _importance_weight_g_chunk(samples, source, target, flow) -> Array:
    proposal, inverse_ladj = flow.inv_and_ladj(samples)
    return -target(proposal) + source(samples) + inverse_ladj


def _chunked_identity_weights(samples, source, target, chunk: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        values.append(
            jax.block_until_ready(_identity_weight_chunk(part, source, target))
        )
    return _finite_log_weights(jnp.concatenate(values))


def _chunked_inverse(flow, samples, chunk: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        values.append(jax.block_until_ready(_inverse_chunk(flow, part)))
    return jnp.concatenate(values, axis=0)


def _flow_is_finite(flow) -> bool:
    leaves = jax.tree.leaves(flow, is_leaf=lambda value: value is None)
    arrays = [value for value in leaves if isinstance(value, Array)]
    return all(bool(jnp.all(jnp.isfinite(value))) for value in arrays)


def _importance_weights_g(samples, source, target, flow, chunk: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunk, axis=0):
        values.append(
            jax.block_until_ready(
                _importance_weight_g_chunk(part, source, target, flow)
            )
        )
    return jnp.concatenate(values, axis=0)


def molecular_boltzmann_forward_KLX_G(
    x_valid: Array,
    source,
    target,
    flow,
    *,
    n_pool: int,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    mc_step: float,
    mc_iters: int,
    coeff_lambda: float = 1.0,
    monitor=None,
    bg_param: dict | None = None,
    chunk: int = 1,
    images: int = 3,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    checkpoint: bool = False,
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain KL+X Boltzmann generator.

    Candidate bridge coefficients are selected by honest potential-space SMC
    with mixed MALA. Each incremental inverse flow is trained on those SMC
    particles; ``e_clip`` affects only that optimizer loss. Stage selection,
    importance weights, and MALA always use the unmodified target.

    MALA adjustment is mandatory in this molecular driver and therefore is not
    an experimental argument.
    """

    if not hasattr(target, "domain") or not hasattr(target, "reference_internal"):
        raise TypeError("the molecular target must expose domain and reference_internal")
    if x_valid.ndim != 2 or x_valid.shape[0] < 1:
        raise ValueError(f"x_valid must have shape [N, d], got {x_valid.shape}")
    if min(n_pool, n_batch, steps, ladder, mc_iters, chunk) < 1:
        raise ValueError("all molecular training sizes must be positive")
    if n_batch > n_pool:
        raise ValueError("n_batch cannot exceed n_pool")
    if chunk > min(n_pool, x_valid.shape[0]):
        raise ValueError("chunk cannot exceed n_pool or the x_valid sample count")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not math.isfinite(coeff_lambda) or coeff_lambda < 0:
        raise ValueError("coeff_lambda must be nonnegative and finite")
    if math.isnan(e_clip) or e_clip < 0:
        raise ValueError("e_clip must be nonnegative")
    if math.isnan(g_clip) or g_clip <= 0:
        raise ValueError("g_clip must be positive")
    parameters = dict(_DEFAULTS)
    if bg_param:
        unknown = set(bg_param) - set(parameters)
        if unknown:
            raise ValueError(f"unknown molecular bg_param keys: {sorted(unknown)}")
        parameters.update(bg_param)
    integer_controls = (parameters["max_stages"], parameters["max_retry"])
    if not (
        0 < parameters["t_safe"] <= 1
        and 0 < parameters["shrink_factor"] < 1
        and parameters["enlarge_factor"] > 1
        and 0 <= parameters["tau_smc"] <= 1
        and 0 <= parameters["tau_ess"] <= 1
        and 0 < parameters["t_tol"] <= 1
        and all(isinstance(value, int) and not isinstance(value, bool) for value in integer_controls)
        and parameters["max_stages"] >= 1
        and parameters["max_retry"] >= 1
    ):
        raise ValueError(f"invalid molecular bg_param: {parameters}")

    status = monitor.printer if monitor is not None else print
    base_key = jax.random.fold_in(jax.random.key(41), seed)
    reference = target.reference_internal()[None, :]
    domain = target.domain
    domain._validate(x_valid, "molecular BG validation samples")
    particles = domain.wrap(x_valid)
    stages: list[dict] = []
    previous_t = 0.0
    while previous_t < 1.0 and len(stages) < parameters["max_stages"]:
        stage_index = len(stages) + 1
        history = [0.0] + [record["t"] for record in stages]
        candidate_t = parameters["t_safe"] if not stages else min(
            history[-1]
            + parameters["enlarge_factor"] * (history[-1] - history[-2]),
            1.0,
        )
        if 1.0 - candidate_t < parameters["t_tol"]:
            candidate_t = 1.0
        previous = linear_combination(
            [target, source], [previous_t, 1.0 - previous_t]
        )
        selection_base = _operation_key(base_key, 1, stage_index)
        draw_key = jax.random.fold_in(selection_base, 1)
        indices = jax.random.randint(
            draw_key, (n_pool,), 0, particles.shape[0]
        )
        selection_pool = particles[indices]
        selection_accepted = False
        for selection_attempt in range(60):
            current = linear_combination(
                [target, source], [candidate_t, 1.0 - candidate_t]
            )
            smc_pool, smc_ess, smc_acceptance = sequential_monte_carlo(
                jax.random.fold_in(selection_base, selection_attempt + 2),
                selection_pool,
                previous,
                current,
                ladder=ladder,
                step=mc_step,
                iters=mc_iters,
                images=images,
                domain=domain,
                chunk=chunk,
            )
            smc_pool = jax.block_until_ready(smc_pool)
            minimum_smc_ess = float(jnp.min(smc_ess))
            status(
                f"[stage {stage_index}] [select N={n_pool}] t={candidate_t:.4f} "
                f"min SMC ESS={minimum_smc_ess:.3f} "
                f"MALA={float(jnp.mean(smc_acceptance)):.3f}"
            )
            if minimum_smc_ess <= 0.0 or minimum_smc_ess < parameters["tau_smc"]:
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] selection rejected -> shrink")
                continue
            selection_accepted = True
            break

        if not selection_accepted:
            status(
                f"[stage {stage_index}] selection failed after 60 attempts; "
                f"ladder incomplete at t={previous_t:.4f}"
            )
            break

        accepted = False
        for attempt in range(1, parameters["max_retry"] + 1):
            current = linear_combination(
                [target, source], [candidate_t, 1.0 - candidate_t]
            )
            if attempt > 1:
                smc_pool, smc_ess, smc_acceptance = sequential_monte_carlo(
                    jax.random.fold_in(
                        selection_base, 1000 + attempt
                    ),
                    selection_pool,
                    previous,
                    current,
                    ladder=ladder,
                    step=mc_step,
                    iters=mc_iters,
                    images=images,
                    domain=domain,
                    chunk=chunk,
                )
                smc_pool = jax.block_until_ready(smc_pool)
                retry_smc_ess = float(jnp.min(smc_ess))
                if retry_smc_ess < parameters["tau_smc"]:
                    candidate_t = previous_t + parameters["shrink_factor"] * (
                        candidate_t - previous_t
                    )
                    status(
                        f"[stage {stage_index}] retry SMC ESS={retry_smc_ess:.3f} "
                        f"below tau_smc={parameters['tau_smc']:.3f} -> shrink"
                    )
                    continue

            energy_origin = current(reference)[0]
            candidate, ess_history, kept_history, update_history = (
                train_molecular_forward_KLX_G(
                    smc_pool,
                    previous,
                    current,
                    flow,
                    n_batch,
                    steps,
                    lr,
                    coeff_lambda=coeff_lambda,
                    energy_origin=energy_origin,
                    e_clip=e_clip,
                    g_clip=g_clip,
                    monitor=monitor,
                    checkpoint=checkpoint,
                    seed=jax.random.key_data(
                        _operation_key(base_key, 2, stage_index, attempt)
                    )[0],
                )
            )
            candidate = jax.block_until_ready(candidate)
            if not bool(jnp.any(update_history)):
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] zero optimizer updates -> shrink")
                continue
            if not _flow_is_finite(candidate):
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] nonfinite flow rejected -> shrink")
                continue
            trained_log_weight = _finite_log_weights(
                _importance_weights_g(
                    particles, previous, current, candidate, chunk
                )
            )
            identity_log_weight = _chunked_identity_weights(
                particles, previous, current, chunk
            )
            trained_ess = _ess(trained_log_weight)
            identity_ess = _ess(identity_log_weight)
            if trained_ess >= identity_ess:
                selected_flow = candidate
                selected_log_weight = trained_log_weight
                stage_ess = trained_ess
            else:
                selected_flow = flow.zeros()
                selected_log_weight = identity_log_weight
                stage_ess = identity_ess
            improvement = stage_ess - identity_ess
            status(
                f"[stage {stage_index}] t={candidate_t:.4f} "
                f"validation ESS[N={particles.shape[0]}]="
                f"{stage_ess:.3f} (trained={trained_ess:.3f}, "
                f"identity={identity_ess:.3f}, kept={float(jnp.mean(kept_history)):.3f})"
            )
            if stage_ess <= 0.0 or stage_ess < parameters["tau_ess"]:
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] training rejected -> shrink")
                continue

            proposal = (
                _chunked_inverse(selected_flow, particles, chunk)
                if trained_ess >= identity_ess
                else particles
            )
            resample_key, mala_key = jax.random.split(
                _operation_key(base_key, 3, stage_index)
            )
            particles = resample(
                resample_key,
                proposal,
                _linear_weights(selected_log_weight),
                N=particles.shape[0],
            )
            particles, mala_acceptance = mixed_mala(
                mala_key,
                particles,
                current,
                domain,
                step=mc_step,
                iters=mc_iters,
                images=images,
                chunk=chunk,
            )
            particles = jax.block_until_ready(particles)
            stages.append(
                {
                    "t": candidate_t,
                    "ess": stage_ess,
                    "trained_ess": trained_ess,
                    "identity_ess": identity_ess,
                    "selected": "trained" if trained_ess >= identity_ess else "identity",
                    "ess_samples": particles.shape[0],
                    "flow": selected_flow,
                    "ess_history": ess_history,
                    "kept_history": kept_history,
                    "update_history": update_history,
                    "imp_history": improvement,
                    "smc_ess": smc_ess,
                    "smc_acceptance": smc_acceptance,
                    "mala_acceptance": mala_acceptance,
                    "energy_origin": energy_origin,
                }
            )
            flow = selected_flow
            previous_t = candidate_t
            accepted = True
            status(
                f"[stage {stage_index}] ACCEPTED t={candidate_t:.4f} "
                f"post-MALA={float(jnp.mean(mala_acceptance)):.3f}"
            )
            break

        if not accepted:
            status(
                f"[stage {stage_index}] gave up after {parameters['max_retry']} "
                f"attempts; ladder incomplete at t={previous_t:.4f}"
            )
            break

    complete = bool(stages and stages[-1]["t"] == 1.0)
    status(
        f"molecular_boltzmann_forward_KLX_G: "
        f"{'COMPLETE' if complete else 'INCOMPLETE'} ({len(stages)} stages)"
    )
    return particles, stages
