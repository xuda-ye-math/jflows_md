"""Potential-space molecular Boltzmann-generator training."""

from __future__ import annotations

import math
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import linear_combination
from jflows.utils import compute_ESS_log, resample

from .train import (
    _MAX_SNAPSHOTS,
    train_molecular_forward_KLX_G,
    train_molecular_forward_KLXX_G,
)
from .utils import mixed_mala, mixed_quench_and_temper, sequential_monte_carlo


__all__ = [
    "molecular_boltzmann_forward_KLX_G",
    "molecular_boltzmann_forward_KLXX_G",
]


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


def _integer(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return result


def _screen(name: str, value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar, got {value!r}") from exc
    if math.isnan(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _bg_parameters(bg_param: dict | None) -> dict:
    """Merge and validate molecular adaptive-ladder controls."""

    parameters = dict(_DEFAULTS)
    if bg_param:
        unknown = set(bg_param) - set(parameters)
        if unknown:
            raise ValueError(f"unknown molecular bg_param keys: {sorted(unknown)}")
        parameters.update(bg_param)

    real_names = (
        "t_safe",
        "shrink_factor",
        "enlarge_factor",
        "tau_smc",
        "tau_ess",
        "t_tol",
    )
    try:
        values = {name: float(parameters[name]) for name in real_names}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"molecular bg_param values must be real: {parameters!r}"
        ) from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"molecular bg_param values must be finite: {parameters!r}")
    if not (
        0.0 < values["t_safe"] <= 1.0
        and 0.0 < values["shrink_factor"] < 1.0
        and values["enlarge_factor"] > 1.0
        and 0.0 <= values["tau_smc"] <= 1.0
        and 0.0 <= values["tau_ess"] <= 1.0
        and 0.0 < values["t_tol"] <= 1.0
    ):
        raise ValueError(f"invalid molecular bg_param: {parameters!r}")
    if values["t_safe"] < 1.0 and min(
        values["t_safe"] * (1.0 + values["enlarge_factor"]), 1.0
    ) <= values["t_safe"]:
        raise ValueError(
            "molecular bg_param cannot advance the ladder in floating-point "
            "arithmetic"
        )
    for name in ("max_stages", "max_retry"):
        parameters[name] = _integer(name, parameters[name])
    parameters.update(values)
    return parameters


def _operation_key(base_key: Array, namespace: int, *indices: int) -> Array:
    """Derive a deterministic key in a disjoint operation namespace."""
    key = jax.random.fold_in(base_key, namespace)
    for index in indices:
        key = jax.random.fold_in(key, index)
    return key


def _ess(log_weight: Array) -> float:
    value = compute_ESS_log(log_weight)
    return float(jnp.where(jnp.isfinite(value), value, 0.0))


def _linear_weights(log_weight: Array) -> Array:
    log_weight = jnp.asarray(log_weight)
    if not jnp.issubdtype(log_weight.dtype, jnp.inexact):
        log_weight = log_weight.astype(jnp.result_type(float))
    has_nan = jnp.any(jnp.isnan(log_weight))
    positive_infinity = jnp.isposinf(log_weight)
    has_positive_infinity = jnp.any(positive_infinity)
    finite = jnp.isfinite(log_weight)
    has_finite = jnp.any(finite)
    maximum = jnp.max(jnp.where(finite, log_weight, -jnp.inf))
    regular = jnp.where(finite, jnp.exp(log_weight - maximum), 0.0)
    weight = jnp.where(
        has_positive_infinity,
        positive_infinity.astype(log_weight.dtype),
        regular,
    )
    valid = ~has_nan & (has_positive_infinity | has_finite)
    weight = jnp.where(valid, weight, jnp.zeros_like(weight))
    return jnp.where(valid & (weight.sum() > 0), weight, jnp.ones_like(weight))


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
    return jnp.concatenate(values)


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
    lr_warmup: int = 0,
    selection_steps: tuple[int, ...] = (),
    _objective: str = "klx",
    _coeff_alpha: float = 0.5,
    _coeff_beta: float = 0.5,
    _melt: float = 0.0,
    _opt_step: float = 1e-2,
    _opt_iters: int = 100,
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain KL+X Boltzmann generator.

    Candidate bridge coefficients are selected by honest potential-space SMC
    with mixed MALA. Each incremental inverse flow is trained on those SMC
    particles; ``e_clip`` affects only that optimizer loss. Stage selection,
    importance weights, and MALA always use the unmodified target. With sparse
    ``selection_steps``, exact identity (step -1), the pre-update warm start
    (step 0), every requested post-update checkpoint, and the final flow are
    compared by proposal-side ESS on all validation particles; the best one
    then faces the unchanged ``tau_ess`` gate. An empty schedule preserves the
    original final-versus-identity gate. At most 32 snapshots may be requested
    because storage scales with checkpoint count times model size.

    ``checkpoint`` controls backward-pass rematerialization and is unrelated
    to ``selection_steps``. MALA adjustment is mandatory in this molecular
    driver and therefore is not an experimental argument.
    """

    if not hasattr(target, "domain") or not hasattr(target, "reference_internal"):
        raise TypeError("the molecular target must expose domain and reference_internal")
    if x_valid.ndim != 2 or x_valid.shape[0] < 1:
        raise ValueError(f"x_valid must have shape [N, d], got {x_valid.shape}")
    n_pool = _integer("n_pool", n_pool)
    n_batch = _integer("n_batch", n_batch)
    steps = _integer("steps", steps)
    ladder = _integer("ladder", ladder)
    mc_iters = _integer("mc_iters", mc_iters)
    chunk = _integer("chunk", chunk)
    images = _integer("images", images)
    if n_batch > n_pool:
        raise ValueError("n_batch cannot exceed n_pool")
    if chunk > min(n_pool, x_valid.shape[0]):
        raise ValueError("chunk cannot exceed n_pool or the x_valid sample count")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not math.isfinite(mc_step) or mc_step <= 0:
        raise ValueError("mc_step must be positive and finite")
    lr_warmup = _integer("lr_warmup", lr_warmup, minimum=0)
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > steps
        for value in selection_steps
    ):
        raise ValueError("selection_steps must contain integers in [1, steps]")
    if tuple(sorted(set(selection_steps))) != selection_steps:
        raise ValueError("selection_steps must be strictly increasing")
    if len(selection_steps) > _MAX_SNAPSHOTS:
        raise ValueError(
            f"selection_steps is limited to {_MAX_SNAPSHOTS} sparse checkpoints"
        )
    if _objective not in ("klx", "klxx"):
        raise ValueError("unknown molecular objective")
    if not math.isfinite(coeff_lambda) or coeff_lambda < 0:
        raise ValueError("coeff_lambda must be nonnegative and finite")
    if _objective == "klxx":
        if any(
            not math.isfinite(value) or value < 0
            for value in (_coeff_alpha, _coeff_beta, _melt)
        ):
            raise ValueError("molecular KLXX coefficients and melt must be nonnegative")
        if _coeff_alpha + _coeff_beta <= 0:
            raise ValueError("molecular KLXX mixture weights cannot both be zero")
        if not math.isfinite(_opt_step) or _opt_step <= 0:
            raise ValueError("molecular KLXX QT controls must be positive")
        _opt_iters = _integer("opt_iters", _opt_iters)
    e_clip = _screen("e_clip", e_clip)
    g_clip = _screen("g_clip", g_clip)
    parameters = _bg_parameters(bg_param)

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
        if not (previous_t < candidate_t <= 1.0):
            status(
                f"[stage {stage_index}] candidate t={candidate_t!r} does not "
                f"advance past t={previous_t!r}; ladder incomplete"
            )
            break
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
            if not candidate_t > previous_t:
                status(
                    f"[stage {stage_index}] selection shrink made no "
                    "floating-point progress"
                )
                break
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
            if not candidate_t > previous_t:
                status(
                    f"[stage {stage_index}] retry shrink made no "
                    "floating-point progress"
                )
                break
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
            trainer_seed = jax.random.key_data(
                _operation_key(base_key, 2, stage_index, attempt)
            )[0]
            # Preserve the original warm-start scheme: each attempt starts
            # from the last accepted flow.  The zeroed architecture is a
            # separate identity fallback used only by the stage ESS gate.
            attempt_flow = flow
            identity_flow = flow.zeros()
            hat_acceptance = None
            if _objective == "klxx":
                qt_seed, qt_key = jax.random.split(
                    _operation_key(base_key, 4, stage_index, attempt)
                )
                qt_initial = source.samples(qt_seed, n_pool)
                hat_pool, hat_acceptance = mixed_quench_and_temper(
                    qt_key,
                    qt_initial,
                    current,
                    domain,
                    melt=_melt,
                    opt_step=_opt_step,
                    opt_iters=_opt_iters,
                    mc_step=mc_step,
                    mc_iters=mc_iters,
                    images=images,
                    chunk=chunk,
                )
                hat_pool = jax.block_until_ready(hat_pool)
                status(
                    f"[stage {stage_index}] KLXX coverage pool ready "
                    f"(N={n_pool}, MALA={float(jnp.mean(hat_acceptance)):.3f})"
                )
                training_result = train_molecular_forward_KLXX_G(
                    smc_pool,
                    selection_pool,
                    hat_pool,
                    previous,
                    current,
                    attempt_flow,
                    domain,
                    n_batch,
                    steps,
                    lr,
                    coeff_lambda=coeff_lambda,
                    coeff_alpha=_coeff_alpha,
                    coeff_beta=_coeff_beta,
                    mc_step=mc_step,
                    mc_iters=mc_iters,
                    images=images,
                    energy_origin=energy_origin,
                    e_clip=e_clip,
                    g_clip=g_clip,
                    monitor=monitor,
                    checkpoint=checkpoint,
                    lr_warmup=lr_warmup,
                    snapshot_steps=selection_steps,
                    seed=trainer_seed,
                )
            else:
                training_result = train_molecular_forward_KLX_G(
                    smc_pool,
                    previous,
                    current,
                    attempt_flow,
                    n_batch,
                    steps,
                    lr,
                    coeff_lambda=coeff_lambda,
                    energy_origin=energy_origin,
                    e_clip=e_clip,
                    g_clip=g_clip,
                    monitor=monitor,
                    checkpoint=checkpoint,
                    lr_warmup=lr_warmup,
                    snapshot_steps=selection_steps,
                    seed=trainer_seed,
                )
            if selection_steps:
                (
                    candidate,
                    ratio_history,
                    kept_history,
                    update_history,
                    snapshot_flows,
                ) = training_result
            else:
                candidate, ratio_history, kept_history, update_history = (
                    training_result
                )
                snapshot_flows = ()
            candidate = jax.block_until_ready(candidate)
            if not bool(jnp.any(update_history)):
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] zero optimizer updates -> shrink")
                continue
            candidate_is_finite = _flow_is_finite(candidate)
            if not candidate_is_finite and not selection_steps:
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] nonfinite flow rejected -> shrink")
                continue
            identity_log_weight = _chunked_identity_weights(
                particles, previous, current, chunk
            )
            identity_ess = _ess(identity_log_weight)
            checkpoint_flows = [(-1, "identity", identity_flow)]
            if selection_steps:
                checkpoint_flows.append((0, "warm_start", attempt_flow))
            checkpoint_flows.extend(
                (
                    step_value,
                    "final" if step_value == steps else "checkpoint",
                    checkpoint_flow,
                )
                for step_value, checkpoint_flow in zip(
                    selection_steps, snapshot_flows
                )
            )
            if not selection_steps or selection_steps[-1] != steps:
                checkpoint_flows.append((steps, "final", candidate))

            checkpoint_records = []
            for step_value, label, checkpoint_flow in checkpoint_flows:
                if label == "identity":
                    checkpoint_log_weight = identity_log_weight
                elif not _flow_is_finite(checkpoint_flow):
                    checkpoint_log_weight = jnp.full_like(
                        identity_log_weight, -jnp.inf
                    )
                else:
                    checkpoint_log_weight = _importance_weights_g(
                        particles,
                        previous,
                        current,
                        checkpoint_flow,
                        chunk,
                    )
                checkpoint_ess = _ess(checkpoint_log_weight)
                status(
                    f"[stage {stage_index}] [attempt {attempt}] "
                    f"checkpoint {label} step={step_value} "
                    f"proposal ESS[N={particles.shape[0]}]={checkpoint_ess:.6f}"
                )
                checkpoint_records.append(
                    {
                        "step": step_value,
                        "label": label,
                        "ess": checkpoint_ess,
                        "flow": checkpoint_flow,
                        "log_weight": checkpoint_log_weight,
                    }
                )

            # On an exact ESS tie, retain the original behavior: a trained
            # flow wins over identity, and the later trained checkpoint wins.
            selected_record = max(
                checkpoint_records,
                key=lambda record: (
                    record["ess"],
                    record["step"] > 0,
                    record["step"],
                ),
            )
            trained_record = max(
                (
                    record
                    for record in checkpoint_records
                    if record["label"] != "identity"
                ),
                key=lambda record: (record["ess"], record["step"]),
            )
            final_record = next(
                record
                for record in reversed(checkpoint_records)
                if record["label"] == "final"
            )
            selected_step = selected_record["step"]
            selected_label = selected_record["label"]
            selected_flow = selected_record["flow"]
            selected_log_weight = selected_record["log_weight"]
            stage_ess = selected_record["ess"]
            trained_ess = trained_record["ess"]
            final_ess = final_record["ess"]
            improvement = stage_ess - identity_ess
            status(
                f"[stage {stage_index}] t={candidate_t:.4f} "
                f"validation ESS[N={particles.shape[0]}]="
                f"{stage_ess:.3f} (selected={selected_label}, "
                f"step={selected_step}, "
                f"trained_best={trained_ess:.3f}, final={final_ess:.3f}, "
                f"identity={identity_ess:.3f}, "
                f"kept={float(jnp.mean(kept_history)):.3f})"
            )
            if stage_ess <= 0.0 or stage_ess < parameters["tau_ess"]:
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] training rejected -> shrink")
                continue

            proposal = (
                _chunked_inverse(selected_flow, particles, chunk)
                if selected_label != "identity"
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
                    "final_ess": final_ess,
                    "identity_ess": identity_ess,
                    "selected": (
                        "identity" if selected_label == "identity" else "trained"
                    ),
                    "selected_checkpoint": selected_label,
                    "selected_step": selected_step,
                    "checkpoint_steps": jnp.asarray(
                        [record["step"] for record in checkpoint_records]
                    ),
                    "checkpoint_ess": jnp.asarray(
                        [record["ess"] for record in checkpoint_records]
                    ),
                    "checkpoint_labels": tuple(
                        record["label"] for record in checkpoint_records
                    ),
                    "ess_samples": particles.shape[0],
                    "flow": selected_flow,
                    "ratio_history": ratio_history,
                    "ess_history": ratio_history,
                    "kept_history": kept_history,
                    "update_history": update_history,
                    "imp_history": improvement,
                    "smc_ess": smc_ess,
                    "smc_acceptance": smc_acceptance,
                    "mala_acceptance": mala_acceptance,
                    "hat_mala_acceptance": hat_acceptance,
                    "objective": _objective,
                    "energy_origin": energy_origin,
                }
            )
            flow = selected_flow
            previous_t = candidate_t
            accepted = True
            status(
                f"[stage {stage_index}] ACCEPTED t={candidate_t:.4f} "
                f"checkpoint={selected_label} step={selected_step} "
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
        f"molecular_boltzmann_forward_{_objective.upper()}_G: "
        f"{'COMPLETE' if complete else 'INCOMPLETE'} ({len(stages)} stages)"
    )
    return particles, stages


def molecular_boltzmann_forward_KLXX_G(
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
    melt: float,
    opt_step: float,
    opt_iters: int,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    monitor=None,
    bg_param: dict | None = None,
    chunk: int = 1,
    images: int = 3,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    checkpoint: bool = False,
    lr_warmup: int = 0,
    selection_steps: tuple[int, ...] = (),
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain ``KL + X_mu + X_mix`` generator.

    Sparse ``selection_steps`` request post-update flow snapshots. The stage
    gate compares exact identity (reported at step -1), the pre-update warm
    start (step 0), requested snapshots, and the final flow on full validation
    proposal ESS. Stage records expose the selected label/step and all tested
    checkpoint labels, steps, and ESS values. ``checkpoint`` is the separate
    backward-pass rematerialization switch.
    """

    return molecular_boltzmann_forward_KLX_G(
        x_valid,
        source,
        target,
        flow,
        n_pool=n_pool,
        n_batch=n_batch,
        steps=steps,
        lr=lr,
        ladder=ladder,
        mc_step=mc_step,
        mc_iters=mc_iters,
        coeff_lambda=coeff_lambda,
        monitor=monitor,
        bg_param=bg_param,
        chunk=chunk,
        images=images,
        e_clip=e_clip,
        g_clip=g_clip,
        seed=seed,
        checkpoint=checkpoint,
        lr_warmup=lr_warmup,
        selection_steps=selection_steps,
        _objective="klxx",
        _coeff_alpha=coeff_alpha,
        _coeff_beta=coeff_beta,
        _melt=melt,
        _opt_step=opt_step,
        _opt_iters=opt_iters,
    )
