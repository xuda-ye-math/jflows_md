"""Direct molecular trainers: KLX, KLL1, KLXX, FAB and FABX.

The mixed-domain counterparts of ``jflows.train``: every trainer runs
``steps_total`` Adam steps in one compiled scan and regenerates its batch
inside every step. A source batch is drawn from ``x_valid`` and its target
batch is manufactured by the flow-proposal SMC ``flow_target_batch``, whose
every level rejuvenates at the target with wrapped MALA. The FAB trainers use
``flow_fab_batch`` instead: the exact two-phase SMC on to ``pi^2 / nu``, whose
intermediate levels rejuvenate at their own distribution of the path and
therefore differentiate the pushforward density through the flow.

Every trainer accepts ``reject_requested`` (a ``Manual_Reject``): its scan then
runs one compiled step at a time (``REJECT_CHECK_STEPS = 1``) and stops at the
first step boundary after the signal, returning the flow at the interruption
and a NaN-padded batch-ESS history.

``target_data`` (the forward KL family: KLX, KLL1, KLXX) replaces the SMC
target surrogate by batches drawn from a given target sample set, for
data-driven training; the reported batch ESS is then that of the pushforward
of the source batch through the current map.

Two Langevin budgets: ``mc_steps_1`` is used only on the intermediate SMC
levels; ``mc_steps_2`` is used for every other rejuvenation, that is the last
SMC level at the target, the quench-and-temper rows of the mixture batch, and
the temper of the whole quench-and-temper pool built once before the scan
(``coeff_qt > 0`` resamples the tempered pool by ``exp(-coeff_qt U)`` and
rejuvenates it again).

The X functional of KLX, KLXX and FABX is the exact batch Gini mean difference
of the log-ratio evaluated by one sort (``jflows.train._variation``); there is
no random pairing. KLL1 uses instead the centered L1 log-dispersion of LDR-L1
(``jflows.train._dispersion``), and FAB has no penalty: its loss is the mean
log-ratio over the ``pi^2 / nu`` batch, the alpha = 2 divergence surrogate.
Two screens act on every batch: the energy screen ``u_clip`` removes clashes,
and ``_screen_top`` removes the ``screen_fraction`` largest log-ratios (holes
of the pushforward density); the same fraction screens the reported batch ESS.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jflows.train import _dispersion, _masked_mean, _variation
from jflows.utils import resample

from .core.domain import Mixed_Domain
from .utils.anneal import flow_fab_batch, flow_target_batch
from .utils.quench import mixed_quench_and_temper
from .utils.rejuvenation import _mixed_mala_chunk
from .utils.screen import SCREEN_FRACTION, compute_ESS_log


__all__ = [
    "train_FABX_G",
    "train_FAB_G",
    "train_forward_KLL1_G",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
]

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


def _screen_top(values, keep, fraction):
    """Drop the ``fraction`` largest ``values`` among the kept rows (weight spikes).

    A kept row whose log-ratio z = log(pi/nu) is far above the rest is a hole of
    the pushforward density at an ordinary target point; it would dominate the
    batch. The energy screen ``u_clip`` removes the other tail (clashes).
    Exactly ``ceil(fraction * B)`` rows are dropped; ``fraction = 0`` drops none.
    """
    count = int(values.shape[0])
    k = int(-(-fraction * count // 1))
    if k == 0:
        return keep
    ordered = jnp.sort(jnp.where(keep, values, -jnp.inf))
    threshold = ordered[count - k]
    return keep & (values < threshold)


def _energy_keep(energy, u_clip):
    keep = jnp.isfinite(energy)
    if u_clip != float("inf"):
        keep = keep & (energy <= u_clip)
    return keep


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
def _scan_direct_chunk(
    x_valid: Array,
    source,
    target,
    state,
    static,
    domain: Mixed_Domain,
    key,
    *,
    step_offset,
    chunk: int,
    batch_fn,
    penalty,
    target_data,
    coeff: float,
    batch_size: int,
    steps_total: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps_1: int,
    mc_steps_2: int,
    mc_image_radius: int,
    monitor,
    checkpoint: bool,
    u_clip,
    g_clip,
    lr_warmup: int,
    screen_fraction: float,
    t_start,
    t_end,
):
    """``chunk`` compiled Adam steps of the direct trainers (KLX, KLL1, FAB).

    ``batch_fn`` manufactures the batch (``flow_target_batch`` for the forward
    KL family, ``flow_fab_batch`` for FAB) and ``penalty`` is the screened
    functional added to the mean log-ratio with weight ``coeff``
    (``_variation`` for KLX, ``_dispersion`` for KLL1, ``None`` for FAB).
    ``state`` carries the parameters and the Adam moments across chunks and
    ``step_offset`` the steps already done, so every chunk of one length is
    one compilation. With ``target_data`` the target batch is drawn from that
    array instead of being manufactured by ``batch_fn``, and the reported
    batch ESS is that of the pushforward of the source batch.
    """
    params, first, second, updates = state
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    count = x_valid.shape[0]

    def body(state, step):
        params, first, second, updates = state
        index_key, smc_key, data_key = jax.random.split(jax.random.fold_in(key, step), 3)
        x = x_valid[jax.random.choice(index_key, count, (batch_size,), replace=False)]
        if target_data is None:
            y, _, proposal_log_weight = batch_fn(
                smc_key, x, source, target, eqx.combine(params, static), domain,
                ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
                mc_steps_2=mc_steps_2, mc_image_radius=mc_image_radius,
                screen_fraction=screen_fraction,
            )
        else:
            y = target_data[
                jax.random.choice(data_key, target_data.shape[0], (batch_size,), replace=False)
            ]
            _, proposal_log_weight = _pushforward_weights(
                eqx.combine(params, static), x, source, target, domain
            )
        y = lax.stop_gradient(y)
        energy = lax.stop_gradient(target(y))
        energy_keep = _energy_keep(energy, u_clip)

        def loss_fn(values):
            latent, ladj = eqx.combine(values, static).call_and_ladj(y)
            z = source(latent) - energy - ladj
            keep = lax.stop_gradient(
                _screen_top(z, energy_keep & jnp.isfinite(z), screen_fraction)
            )
            loss = _masked_mean(z, keep)
            if penalty is not None and coeff != 0.0:
                loss = loss + coeff * penalty(z, keep)
            return loss, keep

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, keep), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        params, first, second, updates = _adam(
            params, first, second, grads, loss, updates,
            _learning_rate(lr, lr_warmup, step), g_clip, jnp.any(keep),
        )
        ess = compute_ESS_log(proposal_log_weight, screen_fraction)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        return (params, first, second, updates), ess

    steps = step_offset + jnp.arange(1, chunk + 1)
    return lax.scan(body, (params, first, second, updates), steps)


REJECT_CHECK_STEPS = 1   # a manual rejection is received within one gradient step


def _pushforward_weights(flow, x, source, target, domain):
    """The detached pushforward of a source batch and its proposal-to-target log weights."""
    proposal, ladj = flow.inv_and_ladj(x)
    proposal = lax.stop_gradient(domain.wrap(proposal))
    return proposal, lax.stop_gradient(source(x) - target(proposal) + ladj)


def _run_chunks(chunk_fn, flow, steps_total, reject_requested, **kwargs):
    """Run ``steps_total`` Adam steps and stop early on a manual rejection.

    With ``reject_requested`` the steps run as chunks of ``REJECT_CHECK_STEPS``
    and ``reject_requested.peek()`` is read between chunks without clearing
    the flag (the generator consumes it afterwards), so the rejection is
    received within one gradient step. Without it the run is one chunk. An interrupted run returns the flow at the interruption and
    the batch-ESS history padded with NaN.
    """
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    state = (
        params,
        jax.tree.map(jnp.zeros_like, params),
        jax.tree.map(jnp.zeros_like, params),
        jnp.asarray(0, dtype=jnp.int32),
    )
    chunk = steps_total if reject_requested is None else REJECT_CHECK_STEPS
    history, done = [], 0
    while done < steps_total:
        length = min(chunk, steps_total - done)
        state, ess = jax.block_until_ready(chunk_fn(
            state=state, static=static, step_offset=jnp.asarray(done, dtype=jnp.int32),
            chunk=length, steps_total=steps_total, **kwargs,
        ))
        history.append(ess)
        done += length
        if done < steps_total and reject_requested is not None and reject_requested.peek():
            break
    ess = jnp.concatenate(history)
    if done < steps_total:
        ess = jnp.concatenate([ess, jnp.full((steps_total - done,), jnp.nan, dtype=ess.dtype)])
    return eqx.combine(state[0], static), ess


def _scan_direct(
    x_valid, source, target, flow, domain, key, *, batch_fn, penalty, coeff,
    batch_size, steps_total, lr, ladder, mc_dt, mc_steps_1, mc_steps_2,
    mc_image_radius, monitor, checkpoint, u_clip, g_clip, lr_warmup,
    screen_fraction, t_start, t_end, reject_requested=None, target_data=None,
):
    """The direct trainers' scan: ``_scan_direct_chunk`` over ``_run_chunks``."""
    return _run_chunks(
        lambda **kw: _scan_direct_chunk(
            x_valid, source, target, kw.pop("state"), kw.pop("static"), domain, key, **kw,
        ),
        flow, steps_total, reject_requested,
        batch_fn=batch_fn, penalty=penalty, coeff=coeff, batch_size=batch_size,
        target_data=target_data,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
        mc_steps_2=mc_steps_2, mc_image_radius=mc_image_radius, monitor=monitor,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, t_start=t_start, t_end=t_end,
    )


def _direct_trainer(name, key_base, penalty, batch_fn, doc):
    """Build one public direct trainer around ``_scan_direct``."""

    def trainer(
        x_valid: Array,
        source,
        target,
        flow,
        domain: Mixed_Domain,
        batch_size: int,
        steps_total: int,
        lr: float,
        ladder: int,
        mc_dt: float,
        mc_steps_1: int,
        mc_steps_2: int,
        coeff_lambda: float = 1.0,
        mc_image_radius: int = 3,
        monitor=None,
        seed: int | Array = 0,
        checkpoint: bool = False,
        *,
        initialize_from_identity: bool = False,
        u_clip=float("inf"),
        g_clip=float("inf"),
        lr_warmup: int = 0,
        screen_fraction: float = SCREEN_FRACTION,
        t_start: float = 0.0,
        t_end: float = 1.0,
        reject_requested=None,
        target_data=None,
    ):
        if initialize_from_identity:
            flow = flow.zeros()
        if target_data is not None:
            target_data = jnp.asarray(target_data)
        return _scan_direct(
            x_valid, source, target, flow, domain,
            jax.random.fold_in(jax.random.key(key_base), seed),
            batch_fn=batch_fn, penalty=penalty, coeff=coeff_lambda,
            batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
            mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
            mc_image_radius=mc_image_radius, monitor=monitor,
            checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
            lr_warmup=lr_warmup, screen_fraction=screen_fraction,
            t_start=t_start, t_end=t_end, reject_requested=reject_requested,
            target_data=target_data,
        )

    trainer.__name__ = name
    trainer.__qualname__ = name
    trainer.__doc__ = doc
    return trainer


train_forward_KLX_G = _direct_trainer(
    "train_forward_KLX_G", 31, _variation, flow_target_batch,
    """Train one molecular ``KL + lambda X_pi`` map G and return flow plus batch ESS history.

    ``coeff_lambda = 0`` is the forward KL. Each step draws ``batch_size``
    source rows from ``x_valid``, manufactures the target batch through the
    current flow with ``flow_target_batch``, and minimizes the screened mean of
    ``z = source(G(y)) - target(y) - log|det J_G(y)|`` plus ``coeff_lambda``
    times its exact sorted variation.
    """,
)

train_forward_KLL1_G = _direct_trainer(
    "train_forward_KLL1_G", 41, _dispersion, flow_target_batch,
    """Train one molecular LDR-L1 map G and return flow plus batch ESS history.

    The signature and the target batch of ``train_forward_KLX_G``, with the
    centered L1 log-dispersion ``mean|z - mean z|`` in place of the pairwise
    variation: the screened mean of ``z`` plus ``coeff_lambda`` times that
    dispersion, the batch mean being the center and differentiated through.
    """,
)

def train_FAB_G(
    x_valid: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    batch_size: int,
    steps_total: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps_1: int,
    mc_steps_2: int,
    mc_image_radius: int = 3,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    screen_fraction: float = SCREEN_FRACTION,
    t_start: float = 0.0,
    t_end: float = 1.0,
    reject_requested=None,
):
    """Train one molecular map G with the FAB surrogate loss.

    Each step manufactures a batch from ``pi^2 / nu`` with ``flow_fab_batch``
    (``ladder`` levels to the target, then ``ladder`` further levels on to
    ``pi^2 / nu``) and minimizes the screened mean log-ratio over it, whose
    parameter gradient is that of the alpha = 2 divergence. There is no replay
    buffer and no penalty term. The reported batch ESS is the phase-1 proposal
    ESS, as for the forward KL family.
    """
    if initialize_from_identity:
        flow = flow.zeros()
    return _scan_direct(
        x_valid, source, target, flow, domain,
        jax.random.fold_in(jax.random.key(43), seed),
        batch_fn=flow_fab_batch, penalty=None, coeff=0.0,
        batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, monitor=monitor, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, t_start=t_start, t_end=t_end,
        reject_requested=reject_requested,
    )


@eqx.filter_jit
def _scan_klxx_chunk(
    x_valid, hat_pool, source, target, state, static, domain, key, *,
    step_offset, chunk, batch_fn, batch_size, steps_total, lr, ladder, mc_dt,
    mc_steps_1, mc_steps_2, coeff_lambda, coeff_theta, coeff_alpha,
    mc_image_radius, monitor, checkpoint, u_clip, g_clip, lr_warmup,
    screen_fraction, t_start, t_end, target_data,
):
    """``chunk`` compiled Adam steps of the mixture trainers (KLXX and FABX).

    With ``target_data`` the target batch is drawn from that array, the
    pushforward half of the mixture is the detached pushforward of the source
    batch, and the reported batch ESS is that pushforward's.

    ``batch_fn`` manufactures the target batch and its detached proposal:
    ``flow_target_batch`` for KLXX, ``flow_fab_batch`` for FABX, which passes
    ``coeff_lambda = 0`` so that only the mixture variation remains.
    """
    params, first, second, updates = state
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    count = x_valid.shape[0]
    pool_count = hat_pool.shape[0]
    mixture_weight = jnp.concatenate((
        jnp.full((batch_size,), coeff_alpha),
        jnp.full((batch_size,), 1.0 - coeff_alpha),
    ))

    def body(state, step):
        params, first, second, updates = state
        index_key, smc_key, hat_key, hat_mc_key, mixture_key, data_key = jax.random.split(
            jax.random.fold_in(key, step), 6
        )
        flow_now = eqx.combine(params, static)
        x = x_valid[jax.random.choice(index_key, count, (batch_size,), replace=False)]
        if target_data is None:
            # y_bar is the SMC proposal itself: the pushforward of the source batch
            # through the current G^{-1}, detached, so no second inverse pass.
            y, y_bar, proposal_log_weight = batch_fn(
                smc_key, x, source, target, flow_now, domain,
                ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
                mc_steps_2=mc_steps_2, mc_image_radius=mc_image_radius,
                screen_fraction=screen_fraction,
            )
        else:
            y = target_data[
                jax.random.choice(data_key, target_data.shape[0], (batch_size,), replace=False)
            ]
            y_bar, proposal_log_weight = _pushforward_weights(flow_now, x, source, target, domain)
        y = lax.stop_gradient(y)
        y_bar = lax.stop_gradient(y_bar)
        y_hat = hat_pool[jax.random.randint(hat_key, (batch_size,), 0, pool_count)]
        y_hat, _ = _mixed_mala_chunk(
            hat_mc_key, y_hat, target, domain,
            mc_dt=mc_dt, mc_steps=mc_steps_2, image_radius=mc_image_radius,
        )
        y_mix = resample(
            mixture_key, jnp.concatenate((y_hat, y_bar), axis=0), mixture_weight,
            N=batch_size,
        )
        energy = lax.stop_gradient(target(y))
        mixture_energy = lax.stop_gradient(target(y_mix))
        energy_keep = _energy_keep(energy, u_clip)
        mixture_keep = _energy_keep(mixture_energy, u_clip)

        def loss_fn(values):
            candidate = eqx.combine(values, static)
            latent, ladj = candidate.call_and_ladj(y)
            z = source(latent) - energy - ladj
            mixture_latent, mixture_ladj = candidate.call_and_ladj(y_mix)
            z_mix = source(mixture_latent) - mixture_energy - mixture_ladj
            valid = lax.stop_gradient(
                _screen_top(z, energy_keep & jnp.isfinite(z), screen_fraction)
            )
            mixture_valid = lax.stop_gradient(
                _screen_top(z_mix, mixture_keep & jnp.isfinite(z_mix), screen_fraction)
            )
            loss = _masked_mean(z, valid) + coeff_theta * _variation(
                z_mix, mixture_valid
            )
            if coeff_lambda != 0.0:
                loss = loss + coeff_lambda * _variation(z, valid)
            return loss, (valid, mixture_valid)

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, (valid, mixture_valid)), grads = jax.value_and_grad(
            evaluated, has_aux=True
        )(params)
        params, first, second, updates = _adam(
            params, first, second, grads, loss, updates,
            _learning_rate(lr, lr_warmup, step), g_clip,
            jnp.any(valid) & jnp.any(mixture_valid),
        )
        ess = compute_ESS_log(proposal_log_weight, screen_fraction)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        return (params, first, second, updates), ess

    steps = step_offset + jnp.arange(1, chunk + 1)
    return lax.scan(body, (params, first, second, updates), steps)


def _scan_klxx(
    x_valid, hat_pool, source, target, flow, domain, key, *, batch_fn,
    batch_size, steps_total, lr, ladder, mc_dt, mc_steps_1, mc_steps_2,
    coeff_lambda, coeff_theta, coeff_alpha, mc_image_radius, monitor,
    checkpoint, u_clip, g_clip, lr_warmup, screen_fraction, t_start, t_end,
    reject_requested=None, target_data=None,
):
    """The mixture trainers' scan: ``_scan_klxx_chunk`` over ``_run_chunks``."""
    return _run_chunks(
        lambda **kw: _scan_klxx_chunk(
            x_valid, hat_pool, source, target, kw.pop("state"), kw.pop("static"),
            domain, key, **kw,
        ),
        flow, steps_total, reject_requested,
        batch_fn=batch_fn, batch_size=batch_size, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        coeff_lambda=coeff_lambda, coeff_theta=coeff_theta,
        coeff_alpha=coeff_alpha, mc_image_radius=mc_image_radius,
        monitor=monitor, checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
        lr_warmup=lr_warmup, screen_fraction=screen_fraction, t_start=t_start,
        t_end=t_end, target_data=target_data,
    )


def _train_mixture(
    x_valid: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    pool_size: int,
    batch_size: int,
    steps_total: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps_1: int,
    mc_steps_2: int,
    coeff_lambda: float = 1.0,
    coeff_theta: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_qt: float = 0.0,
    mc_image_radius: int = 3,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    chunks: int = 1,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    screen_fraction: float = SCREEN_FRACTION,
    t_start: float = 0.0,
    t_end: float = 1.0,
    batch_fn,
    key_base: int,
    reject_requested=None,
    target_data=None,
):
    """Train one molecular mixture map G (KLXX or FABX) and return flow plus ESS.

    The loss is the forward KL plus ``coeff_lambda`` times the variation of the
    log-ratio over the target batch plus ``coeff_theta`` times its variation over
    the mixture batch, drawn with proportion ``coeff_alpha`` from the rejuvenated
    quench-and-temper pool and ``1 - coeff_alpha`` from the detached pushforward
    of the source batch. The pool is ``x_valid`` (``pool_size = 0``) or a random
    subset of that size, quenched and tempered once before the scan with
    ``mc_steps_2`` MALA steps; ``coeff_qt > 0`` resamples the tempered pool by
    ``exp(-coeff_qt U)`` and rejuvenates it again (see
    ``mixed_quench_and_temper``).
    """
    if initialize_from_identity:
        flow = flow.zeros()
    key = jax.random.fold_in(jax.random.key(key_base), seed)
    x_valid = jnp.asarray(x_valid)
    if pool_size == 0:
        pool = x_valid
        qt_key = jax.random.fold_in(key, 0)
    else:
        pool_key, qt_key = jax.random.split(jax.random.fold_in(key, 0))
        pool = x_valid[jax.random.randint(pool_key, (pool_size,), 0, x_valid.shape[0])]
    started = time.perf_counter()
    hat_pool, acceptance = mixed_quench_and_temper(
        qt_key, pool, target, domain,
        melt=melt, opt_alpha=opt_alpha, opt_steps=opt_steps,
        mc_dt=mc_dt, mc_steps=mc_steps_2, mc_image_radius=mc_image_radius,
        chunks=chunks, coeff_qt=coeff_qt,
        screen_fraction=screen_fraction, reject_requested=reject_requested,
    )
    hat_pool = jax.block_until_ready(hat_pool)
    if monitor is not None:
        monitor.printer(
            f"{monitor.prefix}[t: {float(t_start):.6f} -> {float(t_end):.6f}] "
            f"quench-and-temper pool: {hat_pool.shape[0]} rows, "
            f"temper acceptance = {float(jnp.mean(acceptance)):.4f}, "
            f"{time.perf_counter() - started:.1f}s"
        )
    return _scan_klxx(
        x_valid, hat_pool, source, target, flow, domain, key,
        batch_fn=batch_fn, batch_size=batch_size, steps_total=steps_total, lr=lr,
        ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        coeff_lambda=coeff_lambda,
        coeff_theta=coeff_theta, coeff_alpha=coeff_alpha,
        mc_image_radius=mc_image_radius, monitor=monitor, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, t_start=t_start, t_end=t_end,
        reject_requested=reject_requested,
        target_data=None if target_data is None else jnp.asarray(target_data),
    )


def train_forward_KLXX_G(
    x_valid: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    pool_size: int,
    batch_size: int,
    steps_total: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps_1: int,
    mc_steps_2: int,
    coeff_lambda: float = 1.0,
    coeff_theta: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_qt: float = 0.0,
    mc_image_radius: int = 3,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    chunks: int = 1,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    screen_fraction: float = SCREEN_FRACTION,
    t_start: float = 0.0,
    t_end: float = 1.0,
    reject_requested=None,
    target_data=None,
):
    """Train one molecular KLXX map G and return flow plus batch ESS history.

    The loss is the forward KL plus ``coeff_lambda`` times the variation of the
    log-ratio over the target batch plus ``coeff_theta`` times its variation over
    the mixture batch, drawn with proportion ``coeff_alpha`` from the rejuvenated
    quench-and-temper pool and ``1 - coeff_alpha`` from the detached pushforward
    of the source batch.
    """
    return _train_mixture(
        x_valid, source, target, flow, domain, pool_size, batch_size,
        steps_total, lr, ladder, melt, opt_alpha, opt_steps, mc_dt, mc_steps_1,
        mc_steps_2, coeff_lambda, coeff_theta, coeff_alpha, coeff_qt,
        mc_image_radius, monitor, seed, checkpoint, chunks=chunks,
        initialize_from_identity=initialize_from_identity, u_clip=u_clip,
        g_clip=g_clip, lr_warmup=lr_warmup, screen_fraction=screen_fraction,
        t_start=t_start, t_end=t_end, batch_fn=flow_target_batch, key_base=37,
        reject_requested=reject_requested, target_data=target_data,
    )


def train_FABX_G(
    x_valid: Array,
    source,
    target,
    flow,
    domain: Mixed_Domain,
    pool_size: int,
    batch_size: int,
    steps_total: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps_1: int,
    mc_steps_2: int,
    coeff_theta: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_qt: float = 0.0,
    mc_image_radius: int = 3,
    monitor=None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    *,
    chunks: int = 1,
    initialize_from_identity: bool = False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup: int = 0,
    screen_fraction: float = SCREEN_FRACTION,
    t_start: float = 0.0,
    t_end: float = 1.0,
    reject_requested=None,
):
    """Train one molecular FABX map G and return flow plus batch ESS history.

    KLXX with the FAB loss in place of its target term: the same
    quench-and-temper pool and the same mixture of the rejuvenated pool rows
    with the detached phase-1 proposal, and the loss is the screened mean
    log-ratio over the ``pi^2 / nu`` batch of ``flow_fab_batch`` plus
    ``coeff_theta`` times the variation over the mixture batch. There is no
    target-measure variation and no ``coeff_lambda``.
    """
    return _train_mixture(
        x_valid, source, target, flow, domain, pool_size, batch_size,
        steps_total, lr, ladder, melt, opt_alpha, opt_steps, mc_dt, mc_steps_1,
        mc_steps_2, 0.0, coeff_theta, coeff_alpha, coeff_qt, mc_image_radius,
        monitor, seed, checkpoint, chunks=chunks,
        initialize_from_identity=initialize_from_identity, u_clip=u_clip,
        g_clip=g_clip, lr_warmup=lr_warmup, screen_fraction=screen_fraction,
        t_start=t_start, t_end=t_end, batch_fn=flow_fab_batch, key_base=47,
        reject_requested=reject_requested,
    )
