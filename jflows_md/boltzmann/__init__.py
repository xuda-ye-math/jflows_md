"""Pure molecular Boltzmann-generator computations."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, linear_weights_from_log, resample

from ..train import train_forward_KLX_G, train_forward_KLXX_G
from ..utils import mixed_mala, mixed_quench_and_temper, sequential_monte_carlo


__all__ = [
    "boltzmann_identity",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLXX_G",
    "iterate_boltzmann",
]

_POLICY = {
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 2.0,
    "tau_smc": 0.75,
    "tau_ess": 0.6,
    "t_tol": 1e-3,
    "max_stages": 25,
    "max_retry": 8,
}
_SELECTION_LIMIT = 60


def _policy(values):
    result = dict(_POLICY)
    if values:
        result.update(values)
    return result


def _operation_key(base_key, namespace, stage, attempt=0, index=0):
    key = jax.random.fold_in(base_key, namespace)
    key = jax.random.fold_in(key, stage)
    key = jax.random.fold_in(key, attempt)
    return jax.random.fold_in(key, index)


@eqx.filter_jit
def _proposal_weight_chunk(samples, source, target, flow):
    proposal, ladj = flow.inv_and_ladj(samples)
    return proposal, source(samples) - target(proposal) + ladj


@eqx.filter_jit
def _identity_weight_chunk(samples, source, target):
    return source(samples) - target(samples)


@eqx.filter_jit
def _potential_difference_chunk(samples, source, target):
    return source(samples) - target(samples)


def _proposal_weights(samples, source, target, flow, chunks):
    proposals, weights = zip(*[
        jax.block_until_ready(_proposal_weight_chunk(part, source, target, flow))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    return jnp.concatenate(proposals), jnp.concatenate(weights)


def _identity_weights(samples, source, target, chunks):
    return jnp.concatenate([
        jax.block_until_ready(_identity_weight_chunk(part, source, target))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])


def _potential_difference(samples, source, target, chunks):
    return jnp.concatenate([
        jax.block_until_ready(_potential_difference_chunk(part, source, target))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])


def _rg(rg_param_0, rg_param_1, t):
    return rg_param_0 + t * (rg_param_1 - rg_param_0)


def _bridge(source, target, t):
    return linear_combination([target, source], [t, 1.0 - t])


def _stage_monitor(monitor, stage, attempt):
    if monitor is None:
        return None
    from jflows.train import Monitor

    return Monitor(
        monitor.every,
        f"{monitor.prefix}[stage {stage:03d} attempt {attempt:02d}] ",
        monitor.printer,
    )


def _training_attempt(
    objective,
    target_samples,
    source_samples,
    source,
    target,
    flow,
    domain,
    *,
    batch_size,
    train_steps,
    lr,
    coeff_lambda,
    coeff_alpha,
    coeff_beta,
    mc_dt,
    mc_steps,
    mc_image_radius,
    monitor,
    seed,
    checkpoint,
    u_clip,
    g_clip,
    lr_warmup,
    t_start,
    t_end,
    hat_samples=None,
):
    if objective == "forward_klx":
        return train_forward_KLX_G(
            target_samples,
            source_samples,
            source,
            target,
            flow,
            batch_size,
            train_steps,
            lr,
            coeff_lambda,
            monitor,
            seed,
            checkpoint,
            u_clip=u_clip,
            g_clip=g_clip,
            lr_warmup=lr_warmup,
            t_start=t_start,
            t_end=t_end,
        )
    return train_forward_KLXX_G(
        target_samples,
        source_samples,
        hat_samples,
        source,
        target,
        flow,
        domain,
        batch_size,
        train_steps,
        lr,
        coeff_lambda,
        coeff_alpha,
        coeff_beta,
        mc_dt,
        mc_steps,
        mc_image_radius,
        monitor,
        seed,
        checkpoint,
        u_clip=u_clip,
        g_clip=g_clip,
        lr_warmup=lr_warmup,
        t_start=t_start,
        t_end=t_end,
    )


def iterate_identity(
    samples,
    source,
    target,
    *,
    ladder,
    mc_dt,
    mc_steps,
    rg_param_0,
    rg_param_1,
    monitor,
    bg_param,
    chunks,
    mc_image_radius,
    seed=0,
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield post-sharpen identity-only molecular Boltzmann stages."""
    samples = jnp.asarray(samples)
    policy = _policy(bg_param)
    accepted = [float(value) for value in accepted_t]
    rg_param_0 = jnp.asarray(rg_param_0)
    rg_param_1 = jnp.asarray(rg_param_1)
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    domain = target.domain
    stage = start_stage

    while accepted[-1] < 1.0 and stage <= policy["max_stages"]:
        stage_started = time.perf_counter()
        t_start = accepted[-1]
        if len(accepted) == 1:
            t_end = min(policy["t_safe"], 1.0)
        else:
            t_end = min(
                accepted[-1]
                + policy["enlarge_factor"] * (accepted[-1] - accepted[-2]),
                1.0,
            )
        if 1.0 - t_end < policy["t_tol"]:
            t_end = 1.0

        rg_start = _rg(rg_param_0, rg_param_1, t_start)
        target_start = target.regularized(rg_start)
        source_bridge = _bridge(source, target_start, t_start)
        selection_history = []
        t_history = []
        identity_history = []
        sharpen_history = []
        status_history = []
        accepted_stage = False
        selection_index = 0
        for attempt in range(1, policy["max_retry"] + 1):
            selected_endpoint = False
            for _ in range(_SELECTION_LIMIT):
                if not t_start < t_end:
                    break
                selection_index += 1
                target_soft = _bridge(source, target_start, t_end)
                _, smc_ess, smc_acceptance = sequential_monte_carlo(
                    _operation_key(
                        base_key, 201, stage, attempt, selection_index
                    ),
                    samples,
                    source_bridge,
                    target_soft,
                    ladder=ladder,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    domain=domain,
                    chunks=chunks,
                )
                smc_ess, smc_acceptance = jax.block_until_ready(
                    (smc_ess, smc_acceptance)
                )
                minimum_smc_ess = float(jnp.min(smc_ess))
                selected_endpoint = minimum_smc_ess >= policy["tau_smc"]
                selection_history.append({
                    "index": selection_index,
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "smc_ess": np.asarray(smc_ess).tolist(),
                    "minimum_smc_ess": minimum_smc_ess,
                    "decision": "accepted" if selected_endpoint else "shrink",
                    "validation_sample_count": int(samples.shape[0]),
                })
                emit(
                    f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
                    f"selection min SMC ESS={minimum_smc_ess:.4f}"
                )
                if selected_endpoint:
                    break
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not selected_endpoint:
                return

            target_soft = _bridge(source, target_start, t_end)
            identity_log_weight = _identity_weights(
                samples, source_bridge, target_soft, chunks
            )
            identity_ess = float(compute_ESS_log(identity_log_weight))
            t_history.append(t_end)
            identity_history.append(identity_ess)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] identity "
                f"ESS={identity_ess:.4f}"
            )
            if not identity_ess >= policy["tau_ess"]:
                sharpen_history.append(float("nan"))
                status_history.append("rejected")
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
                continue

            soft_resample_key, soft_mala_key = jax.random.split(
                _operation_key(base_key, 501, stage, attempt)
            )
            soft_samples = resample(
                soft_resample_key,
                samples,
                linear_weights_from_log(identity_log_weight),
                N=samples.shape[0],
            )
            soft_samples, mala_acceptance = mixed_mala(
                soft_mala_key,
                soft_samples,
                target_soft,
                domain,
                dt=mc_dt,
                steps=mc_steps,
                image_radius=mc_image_radius,
                chunks=chunks,
            )

            rg_end = _rg(rg_param_0, rg_param_1, t_end)
            target_end = target.regularized(rg_end)
            target_sharp = _bridge(source, target_end, t_end)
            sharpen_log_weight = _potential_difference(
                soft_samples, target_soft, target_sharp, chunks
            )
            sharpen_ess = float(compute_ESS_log(sharpen_log_weight))
            sharpen_history.append(sharpen_ess)
            if not sharpen_ess >= policy["tau_ess"]:
                status_history.append("rejected")
                emit(
                    f"[stage {stage:03d} attempt {attempt:02d} | "
                    f"t: {t_start:.6f} -> {t_end:.6f}] sharpening "
                    f"ESS={sharpen_ess:.4f} REJECTED"
                )
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
                continue

            sharpen_resample_key, sharpen_mala_key = jax.random.split(
                _operation_key(base_key, 601, stage, attempt)
            )
            samples = resample(
                sharpen_resample_key,
                soft_samples,
                linear_weights_from_log(sharpen_log_weight),
                N=soft_samples.shape[0],
            )
            samples, sharpen_mala_acceptance = mixed_mala(
                sharpen_mala_key,
                samples,
                target_sharp,
                domain,
                dt=mc_dt,
                steps=mc_steps,
                image_radius=mc_image_radius,
                chunks=chunks,
            )
            samples = jax.block_until_ready(samples)
            if not bool(jnp.all(jnp.isfinite(samples))):
                raise FloatingPointError(
                    "post-stage samples contain nonfinite coordinates"
                )
            status_history.append("accepted")
            accepted_stage = True
            break

        if not accepted_stage:
            return
        record = {
            "t": float(t_end),
            "t_start": float(t_start),
            "rg_start": tuple(float(value) for value in rg_start),
            "rg_end": tuple(float(value) for value in rg_end),
            "population_rg": tuple(float(value) for value in rg_end),
            "valid_selected_ess": float(identity_ess),
            "valid_identity_ess": float(identity_ess),
            "valid_sample_count": int(samples.shape[0]),
            "selected": "identity",
            "t_hist": jnp.asarray(t_history),
            "valid_identity_ess_hist": jnp.asarray(identity_history),
            "sharpen_ess_hist": jnp.asarray(sharpen_history),
            "attempt_status_hist": tuple(status_history),
            "selection_history": tuple(selection_history),
            "smc_ess": smc_ess,
            "smc_acceptance": smc_acceptance,
            "mala_acceptance": mala_acceptance,
            "sharpen_ess": sharpen_ess,
            "sharpen_mala_acceptance": sharpen_mala_acceptance,
            "objective": "identity",
            "elapsed_seconds": float(time.perf_counter() - stage_started),
        }
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED sharpening ESS={sharpen_ess:.4f}"
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
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    rg_param_0,
    rg_param_1,
    initialize_from_identity,
    coeff_lambda,
    coeff_alpha,
    coeff_beta,
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
    seed=0,
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield post-sharpen populations and complete accepted-stage records."""
    policy = _policy(bg_param)
    accepted = [float(value) for value in accepted_t]
    rg_param_0 = jnp.asarray(rg_param_0)
    rg_param_1 = jnp.asarray(rg_param_1)
    entry_flow = flow
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    domain = target.domain
    stage = start_stage

    while accepted[-1] < 1.0 and stage <= policy["max_stages"]:
        stage_started = time.perf_counter()
        t_start = accepted[-1]
        if len(accepted) == 1:
            t_end = min(policy["t_safe"], 1.0)
        else:
            t_end = min(
                accepted[-1]
                + policy["enlarge_factor"] * (accepted[-1] - accepted[-2]),
                1.0,
            )
        if 1.0 - t_end < policy["t_tol"]:
            t_end = 1.0

        rg_start = _rg(rg_param_0, rg_param_1, t_start)
        target_start = target.regularized(rg_start)
        source_bridge = _bridge(source, target_start, t_start)
        if pool_size == 0:
            selection_pool = samples
        else:
            indices = jax.random.randint(
                _operation_key(base_key, 101, stage),
                (pool_size,),
                0,
                samples.shape[0],
            )
            selection_pool = samples[indices]

        identity_flow = entry_flow.zeros()
        optimizer_initial = identity_flow if initialize_from_identity else entry_flow
        selection_history = []
        t_history = []
        batch_histories = []
        trained_history = []
        identity_history = []
        sharpen_history = []
        status_history = []
        accepted_stage = False
        selection_index = 0
        for attempt in range(1, policy["max_retry"] + 1):
            selected_endpoint = False
            for _ in range(_SELECTION_LIMIT):
                if not t_start < t_end:
                    break
                selection_index += 1
                target_soft = _bridge(source, target_start, t_end)
                smc_pool, smc_ess, smc_acceptance = sequential_monte_carlo(
                    _operation_key(
                        base_key, 201, stage, attempt, selection_index
                    ),
                    selection_pool,
                    source_bridge,
                    target_soft,
                    ladder=ladder,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    domain=domain,
                    chunks=chunks,
                )
                smc_pool, smc_ess, smc_acceptance = jax.block_until_ready(
                    (smc_pool, smc_ess, smc_acceptance)
                )
                minimum_smc_ess = float(jnp.min(smc_ess))
                selected_endpoint = minimum_smc_ess >= policy["tau_smc"]
                selection_history.append({
                    "index": selection_index,
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "smc_ess": np.asarray(smc_ess).tolist(),
                    "minimum_smc_ess": minimum_smc_ess,
                    "decision": "accepted" if selected_endpoint else "shrink",
                    "validation_sample_count": int(selection_pool.shape[0]),
                })
                emit(
                    f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
                    f"selection min SMC ESS={minimum_smc_ess:.4f}"
                )
                if selected_endpoint:
                    break
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not selected_endpoint:
                return

            target_soft = _bridge(source, target_start, t_end)
            hat_samples = None
            hat_acceptance = None
            if objective == "forward_klxx":
                hat_seed, hat_key = jax.random.split(
                    _operation_key(base_key, 301, stage, attempt)
                )
                hat_initial = source.samples(hat_seed, selection_pool.shape[0])
                hat_samples, hat_acceptance = mixed_quench_and_temper(
                    hat_key,
                    hat_initial,
                    target_soft,
                    domain,
                    melt=melt,
                    opt_alpha=opt_alpha,
                    opt_steps=opt_steps,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    chunks=chunks,
                )
                hat_samples = jax.block_until_ready(hat_samples)

            training_key = _operation_key(base_key, 401, stage, attempt)
            trainer_seed = jax.random.key_data(training_key)[0]
            candidate, batch_ess = _training_attempt(
                objective,
                smc_pool,
                selection_pool,
                source_bridge,
                target_soft,
                optimizer_initial,
                domain,
                batch_size=batch_size,
                train_steps=train_steps,
                lr=lr,
                coeff_lambda=coeff_lambda,
                coeff_alpha=coeff_alpha,
                coeff_beta=coeff_beta,
                mc_dt=mc_dt,
                mc_steps=mc_steps,
                mc_image_radius=mc_image_radius,
                monitor=_stage_monitor(monitor, stage, attempt),
                seed=trainer_seed,
                checkpoint=checkpoint,
                u_clip=u_clip,
                g_clip=g_clip,
                lr_warmup=lr_warmup,
                t_start=t_start,
                t_end=t_end,
                hat_samples=hat_samples,
            )
            candidate, batch_ess = jax.block_until_ready((candidate, batch_ess))
            trained_proposal, trained_log_weight = _proposal_weights(
                samples, source_bridge, target_soft, candidate, chunks
            )
            identity_log_weight = _identity_weights(
                samples, source_bridge, target_soft, chunks
            )
            trained_ess = float(compute_ESS_log(trained_log_weight))
            identity_ess = float(compute_ESS_log(identity_log_weight))
            if trained_ess > identity_ess:
                selected = "trained"
                selected_flow = candidate
                continuation_flow = candidate
                proposal = trained_proposal
                selected_log_weight = trained_log_weight
                selected_ess = trained_ess
            else:
                selected = "identity"
                selected_flow = identity_flow
                continuation_flow = identity_flow
                proposal = samples
                selected_log_weight = identity_log_weight
                selected_ess = identity_ess
            t_history.append(t_end)
            batch_histories.append(batch_ess)
            trained_history.append(trained_ess)
            identity_history.append(identity_ess)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] validation "
                f"ESS={selected_ess:.4f} "
                f"(trained={trained_ess:.4f}, identity={identity_ess:.4f})"
            )
            if not selected_ess >= policy["tau_ess"]:
                sharpen_history.append(float("nan"))
                status_history.append("rejected")
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
                continue

            soft_resample_key, soft_mala_key = jax.random.split(
                _operation_key(base_key, 501, stage, attempt)
            )
            soft_samples = resample(
                soft_resample_key,
                proposal,
                linear_weights_from_log(selected_log_weight),
                N=samples.shape[0],
            )
            soft_samples, mala_acceptance = mixed_mala(
                soft_mala_key,
                soft_samples,
                target_soft,
                domain,
                dt=mc_dt,
                steps=mc_steps,
                image_radius=mc_image_radius,
                chunks=chunks,
            )

            rg_end = _rg(rg_param_0, rg_param_1, t_end)
            target_end = target.regularized(rg_end)
            target_sharp = _bridge(source, target_end, t_end)
            sharpen_log_weight = _potential_difference(
                soft_samples, target_soft, target_sharp, chunks
            )
            sharpen_ess = float(compute_ESS_log(sharpen_log_weight))
            sharpen_history.append(sharpen_ess)
            if not sharpen_ess >= policy["tau_ess"]:
                status_history.append("rejected")
                emit(
                    f"[stage {stage:03d} attempt {attempt:02d} | "
                    f"t: {t_start:.6f} -> {t_end:.6f}] sharpening "
                    f"ESS={sharpen_ess:.4f} REJECTED"
                )
                t_end = t_start + policy["shrink_factor"] * (t_end - t_start)
                continue

            sharpen_resample_key, sharpen_mala_key = jax.random.split(
                _operation_key(base_key, 601, stage, attempt)
            )
            samples = resample(
                sharpen_resample_key,
                soft_samples,
                linear_weights_from_log(sharpen_log_weight),
                N=soft_samples.shape[0],
            )
            samples, sharpen_mala_acceptance = mixed_mala(
                sharpen_mala_key,
                samples,
                target_sharp,
                domain,
                dt=mc_dt,
                steps=mc_steps,
                image_radius=mc_image_radius,
                chunks=chunks,
            )
            samples = jax.block_until_ready(samples)
            if not bool(jnp.all(jnp.isfinite(samples))):
                raise FloatingPointError(
                    "post-stage samples contain nonfinite coordinates"
                )
            status_history.append("accepted")
            accepted_stage = True
            break

        if not accepted_stage:
            return
        record = {
            "t": float(t_end),
            "t_start": float(t_start),
            "rg_start": tuple(float(value) for value in rg_start),
            "rg_end": tuple(float(value) for value in rg_end),
            "flow_rg": tuple(float(value) for value in rg_start),
            "population_rg": tuple(float(value) for value in rg_end),
            "flow_endpoint": "pre_sharpen",
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
            "sharpen_ess_hist": jnp.asarray(sharpen_history),
            "attempt_status_hist": tuple(status_history),
            "selection_history": tuple(selection_history),
            "smc_ess": smc_ess,
            "smc_acceptance": smc_acceptance,
            "mala_acceptance": mala_acceptance,
            "hat_mala_acceptance": hat_acceptance,
            "sharpen_ess": sharpen_ess,
            "sharpen_mala_acceptance": sharpen_mala_acceptance,
            "objective": objective,
            "elapsed_seconds": float(time.perf_counter() - stage_started),
        }
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED sharpening ESS={sharpen_ess:.4f}"
        )
        yield samples, record, continuation_flow
        accepted.append(t_end)
        entry_flow = continuation_flow
        stage += 1


def run_boltzmann(samples, source, target, flow, **controls):
    stages = []
    current = jnp.asarray(samples)
    for current, record, _ in iterate_boltzmann(
        current, source, target, flow, **controls
    ):
        stages.append(record)
    return current, stages


def boltzmann_identity(
    x_valid,
    source,
    target,
    ladder,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
):
    """Run adaptive-staging identity-only molecular Boltzmann stages."""
    stages = []
    current = jnp.asarray(x_valid)
    for current, record, _ in iterate_identity(
        current,
        source,
        target,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        rg_param_0=rg_param_0,
        rg_param_1=rg_param_1,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        mc_image_radius=mc_image_radius,
        seed=seed,
    ):
        stages.append(record)
    return current, stages


def boltzmann_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
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
):
    return run_boltzmann(
        x_valid,
        source,
        target,
        flow,
        objective="forward_klx",
        pool_size=pool_size,
        batch_size=batch_size,
        train_steps=train_steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        rg_param_0=rg_param_0,
        rg_param_1=rg_param_1,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda,
        coeff_alpha=0.5,
        coeff_beta=0.5,
        melt=0.0,
        opt_alpha=1.0,
        opt_steps=0,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        mc_image_radius=mc_image_radius,
        checkpoint=checkpoint,
        u_clip=u_clip,
        g_clip=g_clip,
        lr_warmup=lr_warmup,
        seed=seed,
    )


def boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
):
    return run_boltzmann(
        x_valid,
        source,
        target,
        flow,
        objective="forward_klxx",
        pool_size=pool_size,
        batch_size=batch_size,
        train_steps=train_steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        rg_param_0=rg_param_0,
        rg_param_1=rg_param_1,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda,
        coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta,
        melt=melt,
        opt_alpha=opt_alpha,
        opt_steps=opt_steps,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        mc_image_radius=mc_image_radius,
        checkpoint=checkpoint,
        u_clip=u_clip,
        g_clip=g_clip,
        lr_warmup=lr_warmup,
        seed=seed,
    )
