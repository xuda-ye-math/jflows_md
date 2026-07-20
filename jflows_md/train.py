"""Direct molecular KLX and KLXX trainers."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jflows.utils import compute_ESS_log, resample

from .core.domain import Mixed_Domain
from .utils.rejuvenation import _mixed_mala_chunk


__all__ = ["train_forward_KLX_G", "train_forward_KLXX_G"]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _tree_all_finite(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def _clip_global(grads, limit):
    clean = jax.tree.map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)), grads
    )
    leaves = [x for x in jax.tree.leaves(clean) if x.size]
    if not leaves:
        return clean
    square = sum(jnp.sum(jnp.square(x)) for x in leaves)
    direct = jnp.sqrt(square)
    scale = jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves]))

    def normalized(x):
        return jnp.where(
            x != 0,
            jnp.sign(x) * jnp.exp(jnp.log(jnp.abs(x)) - jnp.log(scale)),
            0.0,
        )

    scaled = jax.tree.map(normalized, clean)
    scaled_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(scaled) if x.size)
    )
    direct_factor = jnp.minimum(1.0, limit / (direct + _EPS))
    stable_factor = limit / (scaled_norm + _EPS)
    stable = jnp.log(scale) + jnp.log(scaled_norm) > jnp.log(limit)
    return jax.tree.map(
        lambda x, unit: jnp.where(
            jnp.isfinite(direct),
            x * direct_factor,
            jnp.where(stable, unit * stable_factor, x),
        ),
        clean,
        scaled,
    )


def _masked_pair_mean(values, keep, permutation):
    pair_keep = keep & keep[permutation]
    count = pair_keep.astype(values.dtype).sum()
    return jnp.where(pair_keep, values, 0.0).sum() / jnp.maximum(count, 1.0)


def _learning_rate(lr, warmup, step):
    if warmup == 0:
        return jnp.asarray(lr)
    return lr * jnp.minimum(1.0, step / warmup)


def _adam(params, first, second, grads, loss, updates, lr, g_clip, eligible):
    finite = eligible & jnp.isfinite(loss) & _tree_all_finite(grads)
    clean = jax.tree.map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)), grads
    )
    if g_clip != float("inf"):
        clean = _clip_global(clean, g_clip)
    candidate_updates = updates + finite.astype(updates.dtype)
    bias_step = jnp.maximum(candidate_updates, 1)
    first_new = jax.tree.map(
        lambda old, grad: _BETA1 * old + (1.0 - _BETA1) * grad, first, clean
    )
    second_new = jax.tree.map(
        lambda old, grad: _BETA2 * old + (1.0 - _BETA2) * grad * grad,
        second,
        clean,
    )
    first_hat = jax.tree.map(
        lambda x: x / (1.0 - _BETA1 ** bias_step.astype(x.dtype)), first_new
    )
    second_hat = jax.tree.map(
        lambda x: x / (1.0 - _BETA2 ** bias_step.astype(x.dtype)), second_new
    )
    params_new = jax.tree.map(
        lambda x, m, v: x - lr * m / (jnp.sqrt(v) + _EPS),
        params,
        first_hat,
        second_hat,
    )
    commit = finite & _tree_all_finite((params_new, first_new, second_new))
    params = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), params_new, params
    )
    first = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), first_new, first
    )
    second = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), second_new, second
    )
    return params, first, second, updates + commit.astype(updates.dtype)


@eqx.filter_jit
def train_forward_KLX_G(
    target_samples: Array,
    source_samples: Array,
    source,
    target,
    flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    coeff_lambda: float = 1.0,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    t_start: float = 0.0,
    t_end: float = 1.0,
):
    """Train one inverse molecular flow and return batch ESS history."""
    if initialize_from_identity:
        flow = flow.zeros()
    key = jax.random.fold_in(jax.random.key(31), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)

    def body(state, step):
        params, first, second, updates = state
        target_key, source_key, permutation_key = jax.random.split(
            jax.random.fold_in(key, step), 3
        )
        y = target_samples[
            jax.random.choice(
                target_key, target_samples.shape[0], (batch_size,), replace=False
            )
        ]
        x_source = source_samples[
            jax.random.choice(
                source_key, source_samples.shape[0], (batch_size,), replace=False
            )
        ]
        flow_now = eqx.combine(params, static)
        proposal, proposal_ladj = flow_now.inv_and_ladj(x_source)
        proposal_log_weight = lax.stop_gradient(
            source(x_source) - target(proposal) + proposal_ladj
        )
        energy = lax.stop_gradient(target(y))
        energy_keep = jnp.isfinite(energy)
        if u_clip != float("inf"):
            energy_keep = energy_keep & (energy <= u_clip)
        permutation = jax.random.permutation(permutation_key, batch_size)

        def loss_fn(values):
            x, ladj = eqx.combine(values, static).call_and_ladj(y)
            z = source(x) - energy - ladj
            keep = lax.stop_gradient(energy_keep & jnp.isfinite(z))
            safe = jnp.where(keep, z, 0.0)
            count = keep.astype(z.dtype).sum()
            kl = safe.sum() / jnp.maximum(count, 1.0)
            x_term = _masked_pair_mean(
                jnp.abs(safe - safe[permutation]), keep, permutation
            )
            return kl + coeff_lambda * x_term, keep

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, keep), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        params, first, second, updates = _adam(
            params,
            first,
            second,
            grads,
            loss,
            updates,
            _learning_rate(lr, lr_warmup, step),
            g_clip,
            jnp.any(keep),
        )
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        return (params, first, second, updates), ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess = lax.scan(
        body, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess


@eqx.filter_jit
def train_forward_KLXX_G(
    target_samples: Array,
    source_samples: Array,
    hat_samples: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    batch_size: int,
    train_steps: int,
    lr: float,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_dt: float = 1e-3,
    mc_steps: int = 1,
    mc_image_radius: int = 3,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    t_start: float = 0.0,
    t_end: float = 1.0,
):
    """Train molecular ``KL + X_mu + X_mix`` on fixed-shape pools."""
    if initialize_from_identity:
        flow = flow.zeros()
    key = jax.random.fold_in(jax.random.key(37), seed)
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    mixture_weight = jnp.concatenate(
        (jnp.full((batch_size,), coeff_alpha), jnp.full((batch_size,), coeff_beta))
    )

    def body(state, step):
        params, first, second, updates = state
        (
            target_key,
            source_key,
            hat_key,
            hat_mala_key,
            mixture_key,
            target_permutation_key,
            mixture_permutation_key,
        ) = jax.random.split(jax.random.fold_in(key, step), 7)
        y = target_samples[
            jax.random.choice(
                target_key, target_samples.shape[0], (batch_size,), replace=False
            )
        ]
        x_source = source_samples[
            jax.random.choice(
                source_key, source_samples.shape[0], (batch_size,), replace=False
            )
        ]
        y_hat = hat_samples[
            jax.random.randint(hat_key, (batch_size,), 0, hat_samples.shape[0])
        ]
        y_hat, _ = _mixed_mala_chunk(
            hat_mala_key,
            y_hat,
            target,
            domain,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            image_radius=mc_image_radius,
        )
        flow_now = eqx.combine(params, static)
        y_bar, proposal_ladj = flow_now.inv_and_ladj(x_source)
        y_bar = lax.stop_gradient(domain.wrap(y_bar))
        proposal_log_weight = lax.stop_gradient(
            source(x_source) - target(y_bar) + proposal_ladj
        )
        y_mix = resample(
            mixture_key,
            jnp.concatenate((y_hat, y_bar), axis=0),
            mixture_weight,
            N=batch_size,
        )
        target_permutation = jax.random.permutation(
            target_permutation_key, batch_size
        )
        mixture_permutation = jax.random.permutation(
            mixture_permutation_key, batch_size
        )
        target_energy = lax.stop_gradient(target(y))
        mixture_energy = lax.stop_gradient(target(y_mix))
        target_keep = jnp.isfinite(target_energy)
        mixture_keep = jnp.isfinite(mixture_energy)
        if u_clip != float("inf"):
            target_keep = target_keep & (target_energy <= u_clip)
            mixture_keep = mixture_keep & (mixture_energy <= u_clip)

        def loss_fn(values):
            candidate = eqx.combine(values, static)
            latent, ladj = candidate.call_and_ladj(y)
            z = source(latent) - target_energy - ladj
            mixture_latent, mixture_ladj = candidate.call_and_ladj(y_mix)
            mixture_z = source(mixture_latent) - mixture_energy - mixture_ladj
            valid = lax.stop_gradient(target_keep & jnp.isfinite(z))
            mixture_valid = lax.stop_gradient(
                mixture_keep & jnp.isfinite(mixture_z)
            )
            safe = jnp.where(valid, z, 0.0)
            mixture_safe = jnp.where(mixture_valid, mixture_z, 0.0)
            count = valid.astype(z.dtype).sum()
            loss = safe.sum() / jnp.maximum(count, 1.0)
            loss = loss + coeff_lambda * _masked_pair_mean(
                jnp.abs(safe - safe[target_permutation]),
                valid,
                target_permutation,
            )
            loss = loss + (coeff_alpha + coeff_beta) ** 2 * _masked_pair_mean(
                jnp.abs(
                    mixture_safe - mixture_safe[mixture_permutation]
                ),
                mixture_valid,
                mixture_permutation,
            )
            return loss, (valid, mixture_valid)

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (valid, mixture_valid)), grads = jax.value_and_grad(
            evaluated, has_aux=True
        )(params)
        params, first, second, updates = _adam(
            params,
            first,
            second,
            grads,
            loss,
            updates,
            _learning_rate(lr, lr_warmup, step),
            g_clip,
            jnp.any(valid) & jnp.any(mixture_valid),
        )
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        return (params, first, second, updates), ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess = lax.scan(
        body, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess
