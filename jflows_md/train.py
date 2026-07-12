"""Packed training on molecular SMC particles."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jflows.utils import compute_ESS_log, resample

from .core.checks import integer, nonnegative_real
from .core.domain import Mixed_Domain
from .utils import _mixed_mala_chunk


__all__ = [
    "train_molecular_forward_KLX_G",
    "train_molecular_forward_KLXX_G",
]


_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _clip_global(grads, ceiling: float):
    """Finite-safe global L2 clipping with an overflow-stable fallback."""

    clean = jax.tree.map(
        lambda leaf: jnp.where(
            jnp.isfinite(leaf), leaf, jnp.zeros_like(leaf)
        ),
        grads,
    )
    nonempty = [leaf for leaf in jax.tree.leaves(clean) if leaf.size > 0]
    if not nonempty:
        return clean

    square = sum(jnp.sum(jnp.square(leaf)) for leaf in nonempty)
    direct_norm = jnp.sqrt(square)
    magnitude = jnp.max(
        jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in nonempty])
    )

    def scaled_leaf(leaf):
        return jnp.where(
            leaf != 0,
            jnp.sign(leaf)
            * jnp.exp(jnp.log(jnp.abs(leaf)) - jnp.log(magnitude)),
            0.0,
        )

    scaled = jax.tree.map(scaled_leaf, clean)
    scaled_nonempty = [
        leaf for leaf in jax.tree.leaves(scaled) if leaf.size > 0
    ]
    scaled_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in scaled_nonempty)
    )
    direct_factor = jnp.minimum(1.0, ceiling / (direct_norm + _EPS))
    stable_factor = ceiling / (scaled_norm + _EPS)
    log_norm = jnp.log(magnitude) + jnp.log(scaled_norm)
    needs_stable_clip = log_norm > jnp.log(ceiling)
    return jax.tree.map(
        lambda leaf, normalized: jnp.where(
            jnp.isfinite(direct_norm),
            leaf * direct_factor,
            jnp.where(needs_stable_clip, normalized * stable_factor, leaf),
        ),
        clean,
        scaled,
    )


def _tree_all_finite(tree) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(True)
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


def _adam_step(
    params,
    first_moment,
    second_moment,
    grads,
    loss: Array,
    update_count: Array,
    step_lr: Array,
    g_clip: float,
    eligible: Array,
):
    """Build and atomically commit one finite molecular Adam update.

    A bad loss/gradient or an overflow produced while forming Adam moments or
    parameters leaves the complete optimizer state unchanged. Consequently a
    rejected step does not consume the bias-correction counter.
    """

    finite = jnp.asarray(eligible) & jnp.isfinite(loss) & _tree_all_finite(grads)
    clean = jax.tree.map(
        lambda grad: jnp.where(
            jnp.isfinite(grad), grad, jnp.zeros_like(grad)
        ),
        grads,
    )
    if g_clip != float("inf"):
        clean = _clip_global(clean, g_clip)

    candidate_count = update_count + finite.astype(update_count.dtype)
    bias_count = jnp.maximum(candidate_count, 1)
    first_new = jax.tree.map(
        lambda moment, grad: _BETA1 * moment + (1.0 - _BETA1) * grad,
        first_moment,
        clean,
    )
    second_new = jax.tree.map(
        lambda moment, grad: _BETA2 * moment + (1.0 - _BETA2) * grad * grad,
        second_moment,
        clean,
    )
    first_hat = jax.tree.map(
        lambda value: value
        / (1.0 - _BETA1 ** bias_count.astype(value.dtype)),
        first_new,
    )
    second_hat = jax.tree.map(
        lambda value: value
        / (1.0 - _BETA2 ** bias_count.astype(value.dtype)),
        second_new,
    )
    params_new = jax.tree.map(
        lambda value, first_value, second_value: value
        - step_lr * first_value / (jnp.sqrt(second_value) + _EPS),
        params,
        first_hat,
        second_hat,
    )
    commit = finite & _tree_all_finite((params_new, first_new, second_new))
    params = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), params_new, params
    )
    first_moment = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old),
        first_new,
        first_moment,
    )
    second_moment = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old),
        second_new,
        second_moment,
    )
    update_count = update_count + commit.astype(update_count.dtype)
    return params, first_moment, second_moment, update_count, commit


@eqx.filter_jit
def train_molecular_forward_KLX_G(
    target_samples: Array,
    source_samples: Array,
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
) -> tuple:
    """Train an inverse flow ``G`` on potential-space SMC particles.

    ``target_samples`` are potential-space particles approximating the stage
    target and supply the forward-KL/X loss. ``source_samples`` follow the
    stage source and are pushed through the current inverse flow before any
    correction; their full proposal-to-target importance weights supply the
    honest per-step ESS history.

    ``e_clip`` is an optimizer-only relative energy screen:
    ``target(y) - energy_origin <= e_clip``. Honest SMC and stage ESS are
    computed outside this function and never see clipped target values.
    Returns the flow, per-step batch ESS, kept fraction, and a Boolean history
    recording whether each Adam update was applied.
    """

    pools = (target_samples, source_samples)
    if any(value.ndim != 2 or value.shape[0] < 1 for value in pools):
        raise ValueError(
            "target_samples and source_samples must have shape [N, d]"
        )
    if source_samples.shape[1] != target_samples.shape[1]:
        raise ValueError(
            "target_samples and source_samples must share one dimension"
        )
    n_batch = integer("n_batch", n_batch)
    steps = integer("steps", steps)
    if n_batch > min(target_samples.shape[0], source_samples.shape[0]):
        raise ValueError(
            "n_batch cannot exceed the source or target sample pool"
        )
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    lr_warmup = integer("lr_warmup", lr_warmup, minimum=0)
    if not math.isfinite(coeff_lambda) or coeff_lambda < 0:
        raise ValueError("coeff_lambda must be nonnegative and finite")
    e_clip = nonnegative_real("e_clip", e_clip)
    g_clip = nonnegative_real("g_clip", g_clip)
    key = jax.random.fold_in(jax.random.key(31), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)
    origin = jnp.asarray(energy_origin)

    def body(carry, step_index):
        current, first_moment, second_moment, update_count = carry
        target_key, source_key, perm_key = jax.random.split(
            jax.random.fold_in(key, step_index), 3
        )
        y = target_samples[
            jax.random.choice(
                target_key,
                target_samples.shape[0],
                (n_batch,),
                replace=False,
            )
        ]
        x_source = source_samples[
            jax.random.choice(
                source_key,
                source_samples.shape[0],
                (n_batch,),
                replace=False,
            )
        ]
        flow_now = eqx.combine(current, static)
        proposal_y, proposal_ladj = flow_now.inv_and_ladj(x_source)
        proposal_y = lax.stop_gradient(proposal_y)
        proposal_log_weight = lax.stop_gradient(
            source(x_source) - target(proposal_y) + proposal_ladj
        )
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
        current, first_moment, second_moment, update_count, update_applied = (
            _adam_step(
                current,
                first_moment,
                second_moment,
                grads,
                loss,
                update_count,
                _step_learning_rate(lr, lr_warmup, step_index),
                g_clip,
                count > 0,
            )
        )
        ess = compute_ESS_log(proposal_log_weight)
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
        body,
        (
            params,
            m0,
            v0,
            jnp.asarray(0, dtype=jnp.int32),
        ),
        indices,
    )
    trained = eqx.combine(params, static)
    return trained, ess, kept, updated


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
) -> tuple:
    """Train molecular ``KL + X_mu + X_mix`` on supplied stage pools.

    ``target_samples`` are honest potential-space SMC particles from the
    current bridge, ``source_samples`` are particles from the previous bridge,
    and ``hat_samples`` are a mixed-domain quench-and-temper coverage pool.
    The detached pushforward of ``source_samples`` supplies ``bar_nu``.  The
    mixture term uses ``coeff_alpha * hat_mu + coeff_beta * bar_nu`` while the
    main target batch retains the KL + ``coeff_lambda * X_mu`` objective. The
    same pre-update source minibatch supplies the honest proposal-to-target
    importance weights returned as the per-step ESS history; optimizer-only
    ``e_clip`` never screens those weights.
    The four-value return matches :func:`train_molecular_forward_KLX_G`.
    """

    pools = (target_samples, source_samples, hat_samples)
    if any(value.ndim != 2 or value.shape[0] < 1 for value in pools):
        raise ValueError("all molecular KLXX pools must have shape [N, d]")
    if any(value.shape[1] != target_samples.shape[1] for value in pools[1:]):
        raise ValueError("all molecular KLXX pools must share one dimension")
    domain._validate(target_samples, "molecular KLXX target samples")
    domain._validate(source_samples, "molecular KLXX source samples")
    domain._validate(hat_samples, "molecular KLXX hat samples")
    n_batch = integer("n_batch", n_batch, minimum=2)
    steps = integer("steps", steps)
    if n_batch > min(target_samples.shape[0], source_samples.shape[0]):
        raise ValueError("n_batch cannot exceed the source or target pool")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    lr_warmup = integer("lr_warmup", lr_warmup, minimum=0)
    coefficients = (coeff_lambda, coeff_alpha, coeff_beta)
    if any(not math.isfinite(value) or value < 0 for value in coefficients):
        raise ValueError("KLXX coefficients must be nonnegative and finite")
    if not math.isfinite(mc_step) or mc_step <= 0:
        raise ValueError("mc_step must be positive and finite")
    mc_iters = integer("mc_iters", mc_iters, minimum=0)
    images = integer("images", images)
    e_clip = nonnegative_real("e_clip", e_clip)
    g_clip = nonnegative_real("g_clip", g_clip)

    key = jax.random.fold_in(jax.random.key(37), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)
    origin = jnp.asarray(energy_origin)
    mixture_weight = jnp.concatenate(
        (
            jnp.full((n_batch,), coeff_alpha),
            jnp.full((n_batch,), coeff_beta),
        )
    )

    def body(carry, step_index):
        current, first_moment, second_moment, update_count = carry
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
        y_bar, proposal_ladj = flow_now.inv_and_ladj(x_source)
        y_bar = lax.stop_gradient(domain.wrap(y_bar))
        proposal_log_weight = lax.stop_gradient(
            source(x_source) - target(y_bar) + proposal_ladj
        )
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
            target_z = source(latent) - target_energy - ladj
            target_valid = lax.stop_gradient(
                target_keep & jnp.isfinite(target_z)
            )
            mix_latent, mix_ladj = candidate.call_and_ladj(y_mix)
            mixture_z = source(mix_latent) - mixture_energy - mix_ladj
            mixture_valid = lax.stop_gradient(
                mixture_keep & jnp.isfinite(mixture_z)
            )
            target_safe = jnp.where(target_valid, target_z, 0.0)
            mixture_safe = jnp.where(mixture_valid, mixture_z, 0.0)
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
            return loss, (target_z, target_valid, mixture_valid)

        differentiated_loss = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (z, target_valid, mixture_valid)), grads = jax.value_and_grad(
            differentiated_loss, has_aux=True
        )(current)
        target_count = jnp.sum(target_valid)
        mixture_count = jnp.sum(mixture_valid)
        current, first_moment, second_moment, update_count, update_applied = (
            _adam_step(
                current,
                first_moment,
                second_moment,
                grads,
                loss,
                update_count,
                _step_learning_rate(lr, lr_warmup, step_index),
                g_clip,
                (target_count > 0) & (mixture_count > 0),
            )
        )
        ess = compute_ESS_log(proposal_log_weight)
        kept_fraction = jnp.minimum(target_count, mixture_count).astype(z.dtype) / n_batch
        if monitor is not None:
            monitor.report(step_index, loss, ess)
        return (current, first_moment, second_moment, update_count), (
            ess,
            kept_fraction,
            update_applied,
        )

    indices = jnp.arange(1, steps + 1)
    (params, _, _, _), (ess, kept, updated) = lax.scan(
        body,
        (
            params,
            m0,
            v0,
            jnp.asarray(0, dtype=jnp.int32),
        ),
        indices,
    )
    trained = eqx.combine(params, static)
    return trained, ess, kept, updated
