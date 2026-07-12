"""Packed training on molecular SMC particles."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jflows.utils import compute_ESS_log


__all__ = ["train_molecular_forward_KLX_G"]


_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _clip_global(grads, ceiling: float):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, ceiling / (norm + _EPS))
    return jax.tree.map(lambda leaf: leaf * scale, grads)


def _tree_all_finite(tree) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


@eqx.filter_jit
def train_molecular_forward_KLX_G(
    samples: Array,
    source,
    target,
    flow,
    n_batch: int,
    steps: int,
    lr: float,
    *,
    coeff_lambda: float = 1.0,
    energy_origin: Array | float = 0.0,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
) -> tuple[object, Array, Array, Array]:
    """Train an inverse flow ``G`` on potential-space SMC particles.

    ``e_clip`` is an optimizer-only relative energy screen:
    ``target(y) - energy_origin <= e_clip``. Honest SMC and stage ESS are
    computed outside this function and never see clipped target values.
    Returns the flow, per-step batch ESS, kept fraction, and a Boolean history
    recording whether each Adam update was applied.
    """

    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError(f"samples must have shape [N, d], got {samples.shape}")
    if n_batch < 1 or steps < 1:
        raise ValueError("n_batch and steps must be positive")
    if n_batch > samples.shape[0]:
        raise ValueError("n_batch cannot exceed the molecular sample pool")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not math.isfinite(coeff_lambda) or coeff_lambda < 0:
        raise ValueError("coeff_lambda must be nonnegative and finite")
    if math.isnan(e_clip) or e_clip < 0:
        raise ValueError("e_clip must be nonnegative")
    if math.isnan(g_clip) or g_clip <= 0:
        raise ValueError("g_clip must be positive")
    key = jax.random.fold_in(jax.random.key(31), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)
    origin = jnp.asarray(energy_origin)

    def body(carry, step_index):
        current, first_moment, second_moment, update_count = carry
        batch_key, perm_key = jax.random.split(jax.random.fold_in(key, step_index))
        indices = jax.random.choice(
            batch_key, samples.shape[0], (n_batch,), replace=False
        )
        y = samples[indices]
        energy = lax.stop_gradient(target(y))
        energy_keep = jnp.isfinite(energy)
        if e_clip != float("inf"):
            energy_keep = energy_keep & (energy - origin <= e_clip)
        permutation = jax.random.permutation(perm_key, n_batch)

        def loss_fn(trainable):
            candidate = eqx.combine(trainable, static)
            x, ladj = candidate.call_and_ladj(y)
            z = source(x) - energy - ladj
            keep = lax.stop_gradient(energy_keep & jnp.isfinite(z))
            safe_z = jnp.where(keep, z, 0.0)
            pair_keep = keep & keep[permutation]
            difference = jnp.abs(safe_z - safe_z[permutation])
            keep_float = keep.astype(z.dtype)
            pair_float = pair_keep.astype(z.dtype)
            kl = jnp.sum(safe_z) / jnp.maximum(jnp.sum(keep_float), 1.0)
            x_term = jnp.sum(jnp.where(pair_keep, difference, 0.0)) / jnp.maximum(
                jnp.sum(pair_float), 1.0
            )
            return kl + coeff_lambda * x_term, (z, keep)

        differentiated_loss = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (z, keep)), grads = jax.value_and_grad(
            differentiated_loss, has_aux=True
        )(current)
        count = jnp.sum(keep)
        update_applied = (
            (count > 0) & jnp.isfinite(loss) & _tree_all_finite(grads)
        )

        def apply_update(state):
            parameters, first, second, count_updates = state
            count_updates = count_updates + 1
            safe_grads = (
                _clip_global(grads, g_clip)
                if g_clip != float("inf")
                else grads
            )
            first = jax.tree.map(
                lambda moment, grad: _BETA1 * moment + (1.0 - _BETA1) * grad,
                first,
                safe_grads,
            )
            second = jax.tree.map(
                lambda moment, grad: _BETA2 * moment + (1.0 - _BETA2) * grad * grad,
                second,
                safe_grads,
            )
            first_hat = jax.tree.map(
                lambda value: value
                / (1.0 - _BETA1 ** count_updates.astype(value.dtype)),
                first,
            )
            second_hat = jax.tree.map(
                lambda value: value
                / (1.0 - _BETA2 ** count_updates.astype(value.dtype)),
                second,
            )
            parameters = jax.tree.map(
                lambda value, first_value, second_value: value
                - lr * first_value / (jnp.sqrt(second_value) + _EPS),
                parameters,
                first_hat,
                second_hat,
            )
            return parameters, first, second, count_updates

        current, first_moment, second_moment, update_count = lax.cond(
            update_applied,
            apply_update,
            lambda state: state,
            (current, first_moment, second_moment, update_count),
        )
        log_weight = jnp.where(keep, z, -jnp.inf)
        ess = lax.cond(
            count > 0,
            lambda: compute_ESS_log(log_weight),
            lambda: jnp.asarray(0.0, dtype=z.dtype),
        )
        kept_fraction = count.astype(z.dtype) / n_batch
        if monitor is not None:
            monitor.report(step_index, loss, ess)
        return (current, first_moment, second_moment, update_count), (
            ess,
            kept_fraction,
            update_applied,
        )

    indices = jnp.arange(1, steps + 1)
    (params, _, _, _), (ess, kept, updated) = lax.scan(
        body, (params, m0, v0, jnp.asarray(0, dtype=jnp.int32)), indices
    )
    return eqx.combine(params, static), ess, kept, updated
