"""Packed training on molecular SMC particles."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jflows.train import Monitor
from jflows.utils import compute_ESS_log, resample

from .core.domain import Mixed_Domain
from .utils import _mixed_mala_chunk


__all__ = [
    "Molecular_Monitor",
    "train_molecular_forward_KLX_G",
    "train_molecular_forward_KLXX_G",
]


_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8
_MAX_SNAPSHOTS = 32


class Molecular_Monitor(Monitor):
    """Report the target-pool ratio moment without calling it stage ESS.

    The packed forward trainers evaluate ``compute_ESS_log(z)`` on samples
    from the current target pool. This is a useful concentration warning, but
    it is not the proposal-side importance ESS used by the Boltzmann gate.
    """

    def _emit(self, t, loss, concentration) -> None:
        self.printer(
            f"{self.prefix}step {int(t):>5d}   "
            f"loss = {float(loss):+.4e}   "
            f"target-ratio C = {float(concentration):.4f}"
        )


def _clip_global(grads, ceiling: float):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, ceiling / (norm + _EPS))
    return jax.tree.map(lambda leaf: leaf * scale, grads)


def _tree_all_finite(tree) -> Array:
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _masked_pair_mean(values: Array, keep: Array, permutation: Array) -> Array:
    pair_keep = keep & keep[permutation]
    weights = pair_keep.astype(values.dtype)
    return jnp.sum(jnp.where(pair_keep, values, 0.0)) / jnp.maximum(
        jnp.sum(weights), 1.0
    )


def _step_learning_rate(lr: float, warmup: int, step_index: Array) -> Array:
    if warmup == 0:
        return jnp.asarray(lr)
    fraction = jnp.minimum(1.0, step_index.astype(jnp.asarray(lr).dtype) / warmup)
    return lr * fraction


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
    lr_warmup: int = 0,
    snapshot_steps: tuple[int, ...] = (),
) -> tuple:
    """Train an inverse flow ``G`` on potential-space SMC particles.

    ``e_clip`` is an optimizer-only relative energy screen:
    ``target(y) - energy_origin <= e_clip``. Honest SMC and stage ESS are
    computed outside this function and never see clipped target values.
    Returns the flow, per-step target-ratio concentration, kept fraction, and
    a Boolean history recording whether each Adam update was applied. If
    ``snapshot_steps`` is nonempty, a fifth return contains the post-update
    flows at those steps. Snapshot parameters remain inside ordinary JAX
    dataflow; training never depends on host debug callbacks.
    """

    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError(f"samples must have shape [N, d], got {samples.shape}")
    if n_batch < 1 or steps < 1:
        raise ValueError("n_batch and steps must be positive")
    if n_batch > samples.shape[0]:
        raise ValueError("n_batch cannot exceed the molecular sample pool")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if lr_warmup < 0:
        raise ValueError("lr_warmup must be nonnegative")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > steps
        for value in snapshot_steps
    ):
        raise ValueError("snapshot_steps must contain integers in [1, steps]")
    if tuple(sorted(set(snapshot_steps))) != snapshot_steps:
        raise ValueError("snapshot_steps must be strictly increasing")
    if len(snapshot_steps) > _MAX_SNAPSHOTS:
        raise ValueError(
            f"snapshot_steps is limited to {_MAX_SNAPSHOTS} sparse checkpoints"
        )
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
    snapshot_indices = jnp.asarray(snapshot_steps, dtype=jnp.int32)
    snapshot_params = jax.tree.map(
        lambda value: jnp.zeros(
            (len(snapshot_steps), *value.shape), dtype=value.dtype
        ),
        params,
    )

    def body(carry, step_index):
        current, first_moment, second_moment, update_count, snapshots = carry
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
                - _step_learning_rate(lr, lr_warmup, step_index)
                * first_value
                / (jnp.sqrt(second_value) + _EPS),
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
        if snapshot_steps:
            capture = jnp.any(snapshot_indices == step_index)

            def save_snapshot(values):
                slot = jnp.argmax(snapshot_indices == step_index)
                return jax.tree.map(
                    lambda stored, value: lax.dynamic_update_index_in_dim(
                        stored, value, slot, axis=0
                    ),
                    values,
                    current,
                )

            snapshots = lax.cond(
                capture,
                save_snapshot,
                lambda values: values,
                snapshots,
            )
        log_weight = jnp.where(keep, z, -jnp.inf)
        concentration = lax.cond(
            count > 0,
            lambda: compute_ESS_log(log_weight),
            lambda: jnp.asarray(0.0, dtype=z.dtype),
        )
        kept_fraction = count.astype(z.dtype) / n_batch
        if monitor is not None:
            monitor.report(step_index, loss, concentration)
        return (current, first_moment, second_moment, update_count, snapshots), (
            concentration,
            kept_fraction,
            update_applied,
        )

    indices = jnp.arange(1, steps + 1)
    (params, _, _, _, snapshot_params), (ratio, kept, updated) = lax.scan(
        body,
        (
            params,
            m0,
            v0,
            jnp.asarray(0, dtype=jnp.int32),
            snapshot_params,
        ),
        indices,
    )
    trained = eqx.combine(params, static)
    if snapshot_steps:
        snapshots = tuple(
            eqx.combine(
                jax.tree.map(lambda values: values[index], snapshot_params),
                static,
            )
            for index in range(len(snapshot_steps))
        )
        return trained, ratio, kept, updated, snapshots
    return trained, ratio, kept, updated


@eqx.filter_jit
def train_molecular_forward_KLXX_G(
    target_samples: Array,
    source_samples: Array,
    hat_samples: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    n_batch: int,
    steps: int,
    lr: float,
    *,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_step: float = 1e-3,
    mc_iters: int = 1,
    images: int = 3,
    energy_origin: Array | float = 0.0,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    lr_warmup: int = 0,
    snapshot_steps: tuple[int, ...] = (),
) -> tuple:
    """Train molecular ``KL + X_mu + X_mix`` on supplied stage pools.

    ``target_samples`` are honest potential-space SMC particles from the
    current bridge, ``source_samples`` are particles from the previous bridge,
    and ``hat_samples`` are a mixed-domain quench-and-temper coverage pool.
    The detached pushforward of ``source_samples`` supplies ``bar_nu``.  The
    mixture term uses ``coeff_alpha * hat_mu + coeff_beta * bar_nu`` while the
    main target batch retains the KL + ``coeff_lambda * X_mu`` objective.
    The four-value return matches :func:`train_molecular_forward_KLX_G`;
    supplying sparse ``snapshot_steps`` adds a fifth tuple of post-update
    flows. The schedule is static under JIT and is limited to 32 entries to
    bound checkpoint storage, which scales with model size.
    """

    pools = (target_samples, source_samples, hat_samples)
    if any(value.ndim != 2 or value.shape[0] < 1 for value in pools):
        raise ValueError("all molecular KLXX pools must have shape [N, d]")
    if any(value.shape[1] != target_samples.shape[1] for value in pools[1:]):
        raise ValueError("all molecular KLXX pools must share one dimension")
    domain._validate(target_samples, "molecular KLXX target samples")
    domain._validate(source_samples, "molecular KLXX source samples")
    domain._validate(hat_samples, "molecular KLXX hat samples")
    if n_batch < 2 or steps < 1:
        raise ValueError("n_batch must be at least two and steps positive")
    if n_batch > min(target_samples.shape[0], source_samples.shape[0]):
        raise ValueError("n_batch cannot exceed the source or target pool")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if lr_warmup < 0:
        raise ValueError("lr_warmup must be nonnegative")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > steps
        for value in snapshot_steps
    ):
        raise ValueError("snapshot_steps must contain integers in [1, steps]")
    if tuple(sorted(set(snapshot_steps))) != snapshot_steps:
        raise ValueError("snapshot_steps must be strictly increasing")
    if len(snapshot_steps) > _MAX_SNAPSHOTS:
        raise ValueError(
            f"snapshot_steps is limited to {_MAX_SNAPSHOTS} sparse checkpoints"
        )
    coefficients = (coeff_lambda, coeff_alpha, coeff_beta)
    if any(not math.isfinite(value) or value < 0 for value in coefficients):
        raise ValueError("KLXX coefficients must be nonnegative and finite")
    if coeff_alpha + coeff_beta <= 0:
        raise ValueError("at least one KLXX mixture coefficient must be positive")
    if mc_step <= 0 or mc_iters < 1 or images < 1:
        raise ValueError("molecular MALA controls must be positive")
    if math.isnan(e_clip) or e_clip < 0:
        raise ValueError("e_clip must be nonnegative")
    if math.isnan(g_clip) or g_clip <= 0:
        raise ValueError("g_clip must be positive")

    key = jax.random.fold_in(jax.random.key(37), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)
    origin = jnp.asarray(energy_origin)
    snapshot_indices = jnp.asarray(snapshot_steps, dtype=jnp.int32)
    snapshot_params = jax.tree.map(
        lambda value: jnp.zeros(
            (len(snapshot_steps), *value.shape), dtype=value.dtype
        ),
        params,
    )
    mixture_weight = jnp.concatenate(
        (
            jnp.full((n_batch,), coeff_alpha),
            jnp.full((n_batch,), coeff_beta),
        )
    )

    def body(carry, step_index):
        current, first_moment, second_moment, update_count, snapshots = carry
        (
            target_key,
            source_key,
            hat_key,
            hat_mala_key,
            mixture_key,
            target_perm_key,
            mixture_perm_key,
        ) = jax.random.split(jax.random.fold_in(key, step_index), 7)
        y = target_samples[
            jax.random.choice(
                target_key, target_samples.shape[0], (n_batch,), replace=False
            )
        ]
        x_source = source_samples[
            jax.random.choice(
                source_key, source_samples.shape[0], (n_batch,), replace=False
            )
        ]
        y_hat = hat_samples[
            jax.random.randint(hat_key, (n_batch,), 0, hat_samples.shape[0])
        ]
        y_hat, _ = _mixed_mala_chunk(
            hat_mala_key,
            y_hat,
            target,
            domain,
            step=mc_step,
            iters=mc_iters,
            images=images,
        )
        flow_now = eqx.combine(current, static)
        y_bar = lax.stop_gradient(domain.wrap(flow_now.inv(x_source)))
        y_mix = resample(
            mixture_key,
            jnp.concatenate((y_hat, y_bar), axis=0),
            mixture_weight,
            N=n_batch,
        )
        target_permutation = jax.random.permutation(target_perm_key, n_batch)
        mixture_permutation = jax.random.permutation(mixture_perm_key, n_batch)
        target_energy = lax.stop_gradient(target(y))
        mixture_energy = lax.stop_gradient(target(y_mix))
        target_keep = jnp.isfinite(target_energy)
        mixture_keep = jnp.isfinite(mixture_energy)
        if e_clip != float("inf"):
            target_keep = target_keep & (target_energy - origin <= e_clip)
            mixture_keep = mixture_keep & (mixture_energy - origin <= e_clip)

        def loss_fn(trainable):
            candidate = eqx.combine(trainable, static)
            latent, ladj = candidate.call_and_ladj(y)
            target_ratio = source(latent) - target_energy - ladj
            target_valid = lax.stop_gradient(
                target_keep & jnp.isfinite(target_ratio)
            )
            mix_latent, mix_ladj = candidate.call_and_ladj(y_mix)
            mixture_ratio = source(mix_latent) - mixture_energy - mix_ladj
            mixture_valid = lax.stop_gradient(
                mixture_keep & jnp.isfinite(mixture_ratio)
            )
            target_safe = jnp.where(target_valid, target_ratio, 0.0)
            mixture_safe = jnp.where(mixture_valid, mixture_ratio, 0.0)
            target_count = jnp.sum(target_valid)
            target_kl = jnp.sum(target_safe) / jnp.maximum(target_count, 1.0)
            target_x = _masked_pair_mean(
                jnp.abs(target_safe - target_safe[target_permutation]),
                target_valid,
                target_permutation,
            )
            mixture_x = _masked_pair_mean(
                jnp.abs(mixture_safe - mixture_safe[mixture_permutation]),
                mixture_valid,
                mixture_permutation,
            )
            scale = (coeff_alpha + coeff_beta) ** 2
            loss = target_kl + coeff_lambda * target_x + scale * mixture_x
            return loss, (target_ratio, target_valid, mixture_valid)

        differentiated_loss = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (ratio, target_valid, mixture_valid)), grads = jax.value_and_grad(
            differentiated_loss, has_aux=True
        )(current)
        target_count = jnp.sum(target_valid)
        mixture_count = jnp.sum(mixture_valid)
        update_applied = (
            (target_count > 0)
            & (mixture_count > 0)
            & jnp.isfinite(loss)
            & _tree_all_finite(grads)
        )

        def apply_update(state):
            parameters, first, second, count_updates = state
            count_updates = count_updates + 1
            safe_grads = _clip_global(grads, g_clip) if g_clip != float("inf") else grads
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
            step_lr = _step_learning_rate(lr, lr_warmup, step_index)
            parameters = jax.tree.map(
                lambda value, first_value, second_value: value
                - step_lr * first_value / (jnp.sqrt(second_value) + _EPS),
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
        if snapshot_steps:
            capture = jnp.any(snapshot_indices == step_index)

            def save_snapshot(values):
                slot = jnp.argmax(snapshot_indices == step_index)
                return jax.tree.map(
                    lambda stored, value: lax.dynamic_update_index_in_dim(
                        stored, value, slot, axis=0
                    ),
                    values,
                    current,
                )

            snapshots = lax.cond(
                capture,
                save_snapshot,
                lambda values: values,
                snapshots,
            )
        log_weight = jnp.where(target_valid, ratio, -jnp.inf)
        concentration = lax.cond(
            target_count > 0,
            lambda: compute_ESS_log(log_weight),
            lambda: jnp.asarray(0.0, dtype=ratio.dtype),
        )
        kept_fraction = jnp.minimum(target_count, mixture_count).astype(ratio.dtype) / n_batch
        if monitor is not None:
            monitor.report(step_index, loss, concentration)
        return (current, first_moment, second_moment, update_count, snapshots), (
            concentration,
            kept_fraction,
            update_applied,
        )

    indices = jnp.arange(1, steps + 1)
    (params, _, _, _, snapshot_params), (ratio, kept, updated) = lax.scan(
        body,
        (
            params,
            m0,
            v0,
            jnp.asarray(0, dtype=jnp.int32),
            snapshot_params,
        ),
        indices,
    )
    trained = eqx.combine(params, static)
    if snapshot_steps:
        snapshots = tuple(
            eqx.combine(
                jax.tree.map(lambda values: values[index], snapshot_params),
                static,
            )
            for index in range(len(snapshot_steps))
        )
        return trained, ratio, kept, updated, snapshots
    return trained, ratio, kept, updated
