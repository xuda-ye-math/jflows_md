"""Pure molecular Boltzmann generator computations.

The mixed-domain counterpart of ``jflows.boltzmann``. Each stage trains the
increment ``t_{k-1} -> t_k`` on the current population, compares the trained
map against the identity by population ESS (screened), accepts the stage when
the selected ESS reaches ``tau_valid`` and otherwise shrinks ``t_k`` towards
``t_{k-1}``, and advances the population by reweight -> resample ->
``mc_steps_2`` MALA steps under the stage target. The pushforward of the
population by the trained map is computed once and reused for the selection
weights and the advance. The stage targets are the linear interpolations
``(1 - t) U_0 + t U^{rho}`` of the source and the molecular potential at the
regularization ``rho``; there is no SMC screen.

Regularization path (``rg_param_0``, ``rg_param_1``): with
``rho_s = (1 - s) rho_0 + s rho_1`` the stage targets lie on the diagonal,
``U_{t,t} = (1 - t) U_0 + t U^{rho_t}``, and the stage ``t_{k-1} -> t_k``
trains, selects, and advances the population from ``U_{t_{k-1},t_{k-1}}`` to
``U_{t_k,t_k}`` in one arrow: the regularization tightens together with the
interpolation. ``rg_param_0 = rg_param_1`` is a fixed regularization;
``None`` uses the potential as given.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows.utils import resample

from ..train import (
    train_FABX_G,
    train_FAB_G,
    train_forward_KLL1_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)
from ..utils.rejuvenation import mixed_mala
from ..utils.screen import SCREEN_FRACTION, compute_ESS_log, linear_weights_from_log
from ..utils.control import check_rejection
from .control import Manual_Reject, Manual_Rejection


__all__ = [
    "Manual_Reject",
    "boltzmann_identity",
    "boltzmann_FABX_G",
    "boltzmann_FAB_G",
    "boltzmann_forward_KLL1_G",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLX_G_fixed",
    "boltzmann_forward_KLXX_G",
    "boltzmann_forward_KLXX_G_fixed",
    "iterate_boltzmann",
    "iterate_identity",
    "run_boltzmann",
]

_POLICY = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 2.0,
    "tau_valid": 0.6,
    "t_tol": 1e-3,
    "max_stages": 25,
    "max_retry": 8,
}


def _policy(values):
    """Complete the stage-schedule policy; an unknown key is an error."""
    result = dict(_POLICY)
    if values:
        unknown = sorted(set(values) - set(_POLICY))
        if unknown:
            raise KeyError(f"unknown bg_param keys: {unknown}")
        result.update(values)
    return result


def _operation_key(base_key, namespace, stage, attempt=0, index=0):
    key = jax.random.fold_in(base_key, namespace)
    key = jax.random.fold_in(key, stage)
    key = jax.random.fold_in(key, attempt)
    return jax.random.fold_in(key, index)


@eqx.filter_jit
def _proposal_weight_chunk(samples, source, target, flow, domain):
    proposal, ladj = flow.inv_and_ladj(samples)
    proposal = domain.wrap(proposal)
    return proposal, source(samples) - target(proposal) + ladj


@eqx.filter_jit
def _identity_weight_chunk(samples, source, target):
    return source(samples) - target(samples)


def _push_and_weights(samples, source, target, flow, domain, chunks, reject_requested=None):
    """One chunked pass: the map's pushforward of the population and its log weights."""
    proposals, weights = [], []
    for part in jnp.array_split(samples, chunks, axis=0):
        check_rejection(reject_requested)
        proposal, weight = jax.block_until_ready(
            _proposal_weight_chunk(part, source, target, flow, domain)
        )
        proposals.append(proposal)
        weights.append(weight)
    return jnp.concatenate(proposals), jnp.concatenate(weights)


def _identity_weights(samples, source, target, chunks, reject_requested=None):
    weights = []
    for part in jnp.array_split(samples, chunks, axis=0):
        check_rejection(reject_requested)
        weights.append(jax.block_until_ready(_identity_weight_chunk(part, source, target)))
    return jnp.concatenate(weights)


def _bridge(source, target, t):
    return linear_combination([target, source], [t, 1.0 - t])


def _rg(rg_param_0, rg_param_1, t):
    """``rho_t`` on the regularization path, ``None`` without one."""
    if (rg_param_0 is None) != (rg_param_1 is None):
        raise ValueError("rg_param_0 and rg_param_1 must both be given or both be None")
    if rg_param_0 is None:
        return None
    start, end = np.asarray(rg_param_0, dtype=float), np.asarray(rg_param_1, dtype=float)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("regularization parameters must be pairs (e, r)")
    return tuple(float(value) for value in start + float(t) * (end - start))


def _stage_target(target, rg):
    return target if rg is None else target.regularized(rg)


def _diagonal(source, target, rg_param_0, rg_param_1, t):
    """``U_{t,t} = (1 - t) U_0 + t U^{rho_t}`` and ``rho_t`` (``None`` without a path)."""
    rg = _rg(rg_param_0, rg_param_1, t)
    return _bridge(source, _stage_target(target, rg), t), rg


def _advance(
    key, pushforward, log_weight, target, domain, mc_dt, mc_steps_2,
    mc_image_radius, chunks, screen_fraction, reject_requested=None,
):
    """Resample the selected map's pushforward (screened weights), then MALA under the stage target."""
    resample_key, mala_key = jax.random.split(key)
    check_rejection(reject_requested)
    samples = resample(
        resample_key,
        pushforward,
        linear_weights_from_log(log_weight, screen_fraction),
        N=pushforward.shape[0],
    )
    samples, acceptance = mixed_mala(
        mala_key, samples, target, domain,
        dt=mc_dt, steps=mc_steps_2, image_radius=mc_image_radius, chunks=chunks,
        reject_requested=reject_requested,
    )
    samples = jax.block_until_ready(samples)
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise FloatingPointError(
            "post-stage samples contain infinite or undefined coordinates"
        )
    return samples, acceptance


def _stage_monitor(monitor, stage, attempt):
    if monitor is None:
        return None
    from jflows.train import Monitor

    return Monitor(
        monitor.every,
        f"{monitor.prefix}[stage {stage:03d} attempt {attempt:02d}] ",
        monitor.printer,
    )


def _next_endpoint(accepted, policy):
    if len(accepted) == 1:
        t_end = min(policy["t_safe"], 1.0)
    else:
        t_end = min(
            accepted[-1] + policy["enlarge_factor"] * (accepted[-1] - accepted[-2]),
            1.0,
        )
    if 1.0 - t_end < policy["t_tol"]:
        t_end = 1.0
    return t_end


def _train_attempt(
    objective, samples, source, target, flow, domain, *,
    pool_size, batch_size, steps_total, lr, ladder, mc_dt, mc_steps_1, mc_steps_2,
    coeff_lambda, coeff_theta, coeff_alpha, coeff_qt, melt, opt_alpha, opt_steps,
    mc_image_radius, monitor, seed, checkpoint, u_clip, g_clip, lr_warmup,
    screen_fraction, chunks, t_start, t_end, reject_requested=None,
):
    common = dict(
        initialize_from_identity=False, u_clip=u_clip, g_clip=g_clip,
        lr_warmup=lr_warmup, screen_fraction=screen_fraction,
        t_start=t_start, t_end=t_end, reject_requested=reject_requested,
    )
    if objective in ("forward_klx", "forward_kll1"):
        trainer = (
            train_forward_KLX_G if objective == "forward_klx"
            else train_forward_KLL1_G
        )
        return trainer(
            samples, source, target, flow, domain, batch_size, steps_total, lr,
            ladder, mc_dt, mc_steps_1, mc_steps_2, coeff_lambda, mc_image_radius,
            monitor, seed, checkpoint, **common,
        )
    if objective == "fab":
        return train_FAB_G(
            samples, source, target, flow, domain, batch_size, steps_total, lr,
            ladder, mc_dt, mc_steps_1, mc_steps_2, mc_image_radius, monitor,
            seed, checkpoint, **common,
        )
    if objective == "forward_klxx":
        return train_forward_KLXX_G(
            samples, source, target, flow, domain, pool_size, batch_size,
            steps_total, lr, ladder, melt, opt_alpha, opt_steps, mc_dt,
            mc_steps_1, mc_steps_2, coeff_lambda, coeff_theta, coeff_alpha,
            coeff_qt, mc_image_radius, monitor, seed, checkpoint, chunks=chunks,
            **common,
        )
    if objective == "fabx":
        return train_FABX_G(
            samples, source, target, flow, domain, pool_size, batch_size,
            steps_total, lr, ladder, melt, opt_alpha, opt_steps, mc_dt,
            mc_steps_1, mc_steps_2, coeff_theta, coeff_alpha, coeff_qt,
            mc_image_radius, monitor, seed, checkpoint, chunks=chunks, **common,
        )
    raise ValueError(f"unknown objective: {objective!r}")


def iterate_identity(
    samples,
    source,
    target,
    *,
    mc_dt,
    mc_steps_2,
    mc_image_radius=3,
    monitor=None,
    bg_param=None,
    chunks=1,
    seed=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield adaptive-staging identity-only molecular Boltzmann stages."""
    samples = jnp.asarray(samples)
    policy = _policy(bg_param)
    accepted = [float(value) for value in accepted_t]
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    domain = target.domain
    stage = start_stage

    while accepted[-1] < 1.0 and stage <= policy["max_stages"]:
        stage_started = time.perf_counter()
        t_start = accepted[-1]
        t_end = _next_endpoint(accepted, policy)
        source_bridge, rg_start = _diagonal(source, target, rg_param_0, rg_param_1, t_start)
        t_history, identity_history, status_history = [], [], []
        accepted_stage = False
        for attempt in range(1, policy["max_retry"] + 1):
            attempt_started = time.perf_counter()
            target_bridge, rg_end = _diagonal(source, target, rg_param_0, rg_param_1, t_end)
            log_weight = _identity_weights(samples, source_bridge, target_bridge, chunks)
            identity_ess = float(compute_ESS_log(log_weight, screen_fraction))
            accepted_attempt = identity_ess >= policy["tau_valid"]
            status = "accepted" if accepted_attempt else "rejected"
            t_history.append(t_end)
            identity_history.append(identity_ess)
            status_history.append(status)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] identity "
                f"ESS={identity_ess:.4f} {status.upper()}"
            )
            if accepted_attempt:
                accepted_stage = True
                break
            next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not t_start < next_t < t_end:
                break
            t_end = next_t
        if not accepted_stage:
            return
        samples, acceptance = _advance(
            _operation_key(base_key, 301, stage), samples, log_weight,
            target_bridge, domain, mc_dt, mc_steps_2, mc_image_radius, chunks,
            screen_fraction,
        )
        record = {
            "t": float(t_end),
            "t_start": float(t_start),
            "rg_start": rg_start,
            "rg_end": rg_end,
            "valid_selected_ess": float(identity_ess),
            "valid_identity_ess": float(identity_ess),
            "valid_sample_count": int(samples.shape[0]),
            "selected": "identity",
            "t_hist": jnp.asarray(t_history),
            "valid_identity_ess_hist": jnp.asarray(identity_history),
            "attempt_status_hist": tuple(status_history),
            "mala_acceptance": acceptance,
            "objective": "identity",
            "elapsed_seconds": float(time.perf_counter() - stage_started),
            "accepted_attempt_seconds": float(time.perf_counter() - attempt_started),
        }
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED in {record['elapsed_seconds']:.3f}s"
        )
        yield samples, record, None
        accepted.append(t_end)
        stage += 1


def iterate_boltzmann(
    samples,
    source,
    target,
    flow,
    *,
    objective,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    initialize_from_identity,
    coeff_lambda,
    coeff_theta,
    coeff_alpha,
    coeff_qt,
    melt,
    opt_alpha,
    opt_steps,
    monitor,
    bg_param,
    chunks,
    mc_image_radius,
    checkpoint,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    seed=0,
    t_list=None,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield each newly accepted molecular Boltzmann stage and its population.

    ``t_list=None`` selects the adaptive schedule of ``bg_param``; a sequence
    of endpoints in ``(0, 1]`` is a fixed schedule: every stage trains one
    attempt to the next endpoint above the current ``t`` and is accepted
    whatever its ESS.

    ``reject_requested`` (a ``Manual_Reject``, or any callable that returns
    and clears a flag, with ``peek``) rejects the running attempt by hand: the
    trainer stops its scan within one gradient step of the signal
    (``REJECT_CHECK_STEPS = 1``), the flag is consumed after the trainer returns, the attempt
    is recorded as ``"rejected-manual"``, and the endpoint shrinks as after a
    failed ESS gate. Off by default; ignored on a fixed schedule.
    """
    samples = jnp.asarray(samples)
    adaptive = t_list is None
    if reject_requested is not None and not adaptive:
        (monitor.printer if monitor is not None else print)(
            "manual rejection is not available on a fixed schedule; ignored"
        )
        reject_requested = None
    policy = _policy(bg_param) if adaptive else None
    schedule = None if adaptive else [float(value) for value in t_list]
    accepted = [float(value) for value in accepted_t]
    entry_flow = flow
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    domain = target.domain
    stage = start_stage

    while accepted[-1] < 1.0:
        t_start = accepted[-1]
        if adaptive:
            if stage > policy["max_stages"]:
                return
            t_end = _next_endpoint(accepted, policy)
            max_attempts = policy["max_retry"]
        else:
            remaining = [endpoint for endpoint in schedule if endpoint > t_start]
            if not remaining:
                return
            t_end = remaining[0]
            max_attempts = 1
        stage_started = time.perf_counter()
        source_bridge, rg_start = _diagonal(source, target, rg_param_0, rg_param_1, t_start)
        identity_flow = entry_flow.zeros()
        optimizer_initial = identity_flow if initialize_from_identity else entry_flow
        t_history, batch_histories = [], []
        trained_history, identity_history, status_history = [], [], []
        accepted_stage = False
        for attempt in range(1, max_attempts + 1):
            attempt_started = time.perf_counter()
            if reject_requested is not None:
                reject_requested()   # discard a request raised before this attempt
            target_bridge, rg_end = _diagonal(source, target, rg_param_0, rg_param_1, t_end)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] training"
            )
            training_key = _operation_key(base_key, 401, stage, attempt)
            trainer_seed = jax.random.key_data(training_key)[0]
            batch_ess, stage_of_rejection = None, "quench-and-temper pool"
            try:
                candidate, batch_ess = _train_attempt(
                    objective, samples, source_bridge, target_bridge, optimizer_initial,
                    domain, pool_size=pool_size, batch_size=batch_size,
                    steps_total=steps_total, lr=lr, ladder=ladder, mc_dt=mc_dt,
                    mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
                    coeff_lambda=coeff_lambda, coeff_theta=coeff_theta,
                    coeff_alpha=coeff_alpha, coeff_qt=coeff_qt, melt=melt,
                    opt_alpha=opt_alpha, opt_steps=opt_steps,
                    mc_image_radius=mc_image_radius,
                    monitor=_stage_monitor(monitor, stage, attempt), seed=trainer_seed,
                    checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
                    lr_warmup=lr_warmup, screen_fraction=screen_fraction,
                    chunks=chunks, t_start=t_start, t_end=t_end,
                    reject_requested=reject_requested,
                )
                candidate, batch_ess = jax.block_until_ready((candidate, batch_ess))
                if reject_requested is not None and reject_requested.peek():
                    raise Manual_Rejection()
                stage_of_rejection = "importance weights"
                trained_pushforward, trained_log_weight = _push_and_weights(
                    samples, source_bridge, target_bridge, candidate, domain, chunks,
                    reject_requested,
                )
                identity_log_weight = _identity_weights(
                    samples, source_bridge, target_bridge, chunks, reject_requested
                )
                trained_ess = float(compute_ESS_log(trained_log_weight, screen_fraction))
                identity_ess = float(compute_ESS_log(identity_log_weight, screen_fraction))
                if trained_ess > identity_ess:
                    selected, selected_flow, continuation_flow = "trained", candidate, candidate
                    selected_ess, selected_log_weight = trained_ess, trained_log_weight
                    selected_pushforward = trained_pushforward
                else:
                    selected, selected_flow, continuation_flow = "identity", identity_flow, identity_flow
                    selected_ess, selected_log_weight = identity_ess, identity_log_weight
                    selected_pushforward = samples
                del trained_pushforward
                accepted_attempt = not adaptive or selected_ess >= policy["tau_valid"]
                if accepted_attempt:
                    stage_of_rejection = "population advance"
                    advanced, acceptance = _advance(
                        _operation_key(base_key, 301, stage), selected_pushforward,
                        selected_log_weight, target_bridge, domain, mc_dt, mc_steps_2,
                        mc_image_radius, chunks, screen_fraction, reject_requested,
                    )
            except Manual_Rejection:
                # the flag was raised in training, the pool, the weights, or the advance
                reject_requested()
                if batch_ess is None:
                    batch_ess = jnp.full((steps_total,), jnp.nan)
                steps_done = int(jnp.sum(~jnp.isnan(batch_ess)))
                where = (
                    f"after {steps_done} steps" if 0 < steps_done < steps_total
                    else f"during the {stage_of_rejection}"
                )
                t_history.append(t_end)
                batch_histories.append(batch_ess)
                trained_history.append(float("nan"))
                identity_history.append(float("nan"))
                status_history.append("rejected-manual")
                emit(
                    f"[stage {stage:03d} attempt {attempt:02d} | "
                    f"t: {t_start:.6f} -> {t_end:.6f}] REJECTED manually {where}"
                )
                batch_ess = None
                next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
                if not t_start < next_t < t_end:
                    break
                t_end = next_t
                continue
            status = "accepted" if accepted_attempt else "rejected"
            t_history.append(t_end)
            batch_histories.append(batch_ess)
            trained_history.append(trained_ess)
            identity_history.append(identity_ess)
            status_history.append(status)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] validation "
                f"ESS={selected_ess:.4f} "
                f"(trained={trained_ess:.4f}, identity={identity_ess:.4f}) "
                f"{status.upper()}"
            )
            if accepted_attempt:
                samples = advanced
                del selected_pushforward
                accepted_stage = True
                break
            del selected_pushforward
            next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not t_start < next_t < t_end:
                break
            t_end = next_t
        if not accepted_stage:
            return
        record = {
            "t": float(t_end),
            "t_start": float(t_start),
            "rg_start": rg_start,
            "rg_end": rg_end,
            "valid_selected_ess": float(selected_ess),
            "valid_trained_ess": float(trained_ess),
            "valid_identity_ess": float(identity_ess),
            "valid_sample_count": int(samples.shape[0]),
            "initialized_from_identity": bool(initialize_from_identity),
            "selected": selected,
            "flow": selected_flow,
            "continuation_flow": continuation_flow,
            "t_hist": jnp.asarray(t_history),
            "batch_ess_hist": jnp.stack(batch_histories),
            "valid_trained_ess_hist": jnp.asarray(trained_history),
            "valid_identity_ess_hist": jnp.asarray(identity_history),
            "attempt_status_hist": tuple(status_history),
            "mala_acceptance": acceptance,
            "objective": objective,
            "elapsed_seconds": float(time.perf_counter() - stage_started),
            "accepted_attempt_seconds": float(time.perf_counter() - attempt_started),
        }
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED in {record['elapsed_seconds']:.3f}s"
        )
        yield samples, record, continuation_flow
        accepted.append(t_end)
        entry_flow = continuation_flow
        stage += 1


def _effective_seconds(stages):
    """Training time of the accepted attempts only (rejected attempts removed)."""
    return float(sum(record["accepted_attempt_seconds"] for record in stages))


def run_boltzmann(samples, source, target, flow, **controls):
    """Run the stages; return ``(samples, stages, effective_seconds)``."""
    stages = []
    current = jnp.asarray(samples)
    for current, record, _ in iterate_boltzmann(current, source, target, flow, **controls):
        stages.append(record)
    return current, stages, _effective_seconds(stages)


def boltzmann_identity(
    x_valid,
    source,
    target,
    mc_dt,
    mc_steps_2,
    *,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
):
    """Run adaptive-staging identity-only molecular Boltzmann stages."""
    stages = []
    current = jnp.asarray(x_valid)
    for current, record, _ in iterate_identity(
        current, source, target, mc_dt=mc_dt, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, monitor=monitor, bg_param=bg_param,
        chunks=chunks, seed=seed, screen_fraction=screen_fraction,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
    ):
        stages.append(record)
    return current, stages, _effective_seconds(stages)


def boltzmann_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
):
    """Run adaptive-staging molecular ``KL + lambda X_pi`` stages (``coeff_lambda = 0``: forward KL)."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klx", pool_size=0,
        batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_theta=1.0, coeff_alpha=0.5, coeff_qt=0.0,
        melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=monitor, bg_param=bg_param,
        chunks=chunks, mc_image_radius=mc_image_radius, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
        reject_requested=reject_requested,
    )


def boltzmann_forward_KLX_G_fixed(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    t_list,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    monitor=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
):
    """Run fixed-schedule molecular ``KL + lambda X_pi`` stages (``coeff_lambda = 0``: forward KL)."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klx", pool_size=0,
        batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_theta=1.0, coeff_alpha=0.5, coeff_qt=0.0,
        melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=monitor, bg_param=None,
        chunks=chunks, mc_image_radius=mc_image_radius, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1, t_list=t_list,
    )


def boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
):
    """Run adaptive-staging molecular KLXX stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klxx",
        pool_size=pool_size, batch_size=batch_size, steps_total=steps_total,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
        mc_steps_2=mc_steps_2, initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_theta=coeff_theta,
        coeff_alpha=coeff_alpha, coeff_qt=coeff_qt, melt=melt,
        opt_alpha=opt_alpha, opt_steps=opt_steps, monitor=monitor,
        bg_param=bg_param, chunks=chunks, mc_image_radius=mc_image_radius,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
        reject_requested=reject_requested,
    )


def boltzmann_forward_KLXX_G_fixed(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    t_list,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    monitor=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
):
    """Run fixed-schedule molecular KLXX stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klxx",
        pool_size=pool_size, batch_size=batch_size, steps_total=steps_total,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
        mc_steps_2=mc_steps_2, initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_theta=coeff_theta,
        coeff_alpha=coeff_alpha, coeff_qt=coeff_qt, melt=melt,
        opt_alpha=opt_alpha, opt_steps=opt_steps, monitor=monitor,
        bg_param=None, chunks=chunks, mc_image_radius=mc_image_radius,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1, t_list=t_list,
    )


def boltzmann_forward_KLL1_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
):
    """Run adaptive-staging molecular LDR-L1 stages (forward KL plus the centered L1 log-dispersion)."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_kll1", pool_size=0,
        batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_theta=1.0, coeff_alpha=0.5, coeff_qt=0.0,
        melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=monitor, bg_param=bg_param,
        chunks=chunks, mc_image_radius=mc_image_radius, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
        reject_requested=reject_requested,
    )


def boltzmann_FAB_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
):
    """Run adaptive-staging molecular FAB stages: every stage trains on ``pi_b^2 / nu`` batches from the exact two-phase SMC."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="fab", pool_size=0,
        batch_size=batch_size, steps_total=steps_total, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=1.0, coeff_theta=1.0, coeff_alpha=0.5, coeff_qt=0.0,
        melt=0.0, opt_alpha=1.0, opt_steps=0, monitor=monitor, bg_param=bg_param,
        chunks=chunks, mc_image_radius=mc_image_radius, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
        reject_requested=reject_requested,
    )


def boltzmann_FABX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    rg_param_0=None,
    rg_param_1=None,
    reject_requested=None,
):
    """Run adaptive-staging molecular FABX stages (FAB plus the mixture variation)."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="fabx",
        pool_size=pool_size, batch_size=batch_size, steps_total=steps_total,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1,
        mc_steps_2=mc_steps_2, initialize_from_identity=initialize_from_identity,
        coeff_lambda=0.0, coeff_theta=coeff_theta,
        coeff_alpha=coeff_alpha, coeff_qt=coeff_qt, melt=melt,
        opt_alpha=opt_alpha, opt_steps=opt_steps, monitor=monitor,
        bg_param=bg_param, chunks=chunks, mc_image_radius=mc_image_radius,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, lr_warmup=lr_warmup,
        screen_fraction=screen_fraction, seed=seed,
        rg_param_0=rg_param_0, rg_param_1=rg_param_1,
        reject_requested=reject_requested,
    )
