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


def _adam(params, first, second, grads, step, lr):
    first = jax.tree.map(
        lambda old, grad: _BETA1 * old + (1.0 - _BETA1) * grad,
        first,
        grads,
    )
    second = jax.tree.map(
        lambda old, grad: _BETA2 * old + (1.0 - _BETA2) * grad * grad,
        second,
        grads,
    )
    first_hat = jax.tree.map(
        lambda value: value / (1.0 - _BETA1**step), first
    )
    second_hat = jax.tree.map(
        lambda value: value / (1.0 - _BETA2**step), second
    )
    params = jax.tree.map(
        lambda value, m, v: value - lr * m / (jnp.sqrt(v) + _EPS),
        params,
        first_hat,
        second_hat,
    )
    return params, first, second


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
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)

    def body(state, step):
        params, first, second = state
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
        permutation = jax.random.permutation(permutation_key, batch_size)

        def loss_fn(values):
            x, ladj = eqx.combine(values, static).call_and_ladj(y)
            z = source(x) - energy - ladj
            return z.mean() + coeff_lambda * jnp.abs(z - z[permutation]).mean()

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        loss, grads = jax.value_and_grad(evaluated)(params)
        params, first, second = _adam(params, first, second, grads, step, lr)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        return (params, first, second), ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _), ess = lax.scan(body, (params, first, second), steps)
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
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    mixture_weight = jnp.concatenate(
        (jnp.full((batch_size,), coeff_alpha), jnp.full((batch_size,), coeff_beta))
    )

    def body(state, step):
        params, first, second = state
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

        def loss_fn(values):
            candidate = eqx.combine(values, static)
            latent, ladj = candidate.call_and_ladj(y)
            z = source(latent) - target_energy - ladj
            mixture_latent, mixture_ladj = candidate.call_and_ladj(y_mix)
            mixture_z = source(mixture_latent) - mixture_energy - mixture_ladj
            return (
                z.mean()
                + coeff_lambda * jnp.abs(z - z[target_permutation]).mean()
                + (coeff_alpha + coeff_beta) ** 2
                * jnp.abs(
                    mixture_z - mixture_z[mixture_permutation]
                ).mean()
            )

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        loss, grads = jax.value_and_grad(evaluated)(params)
        params, first, second = _adam(params, first, second, grads, step, lr)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        return (params, first, second), ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _), ess = lax.scan(body, (params, first, second), steps)
    return eqx.combine(params, static), ess
