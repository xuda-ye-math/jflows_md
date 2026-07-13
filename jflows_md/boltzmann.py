"""Potential-space molecular Boltzmann-generator training."""

from __future__ import annotations

import math
import json
from pathlib import Path
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jflows.potential import linear_combination
from jflows.utils import (
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
    resample,
)

from .core.checks import integer, nonnegative_real
from .artifacts import save_mixed_flow
from .train import (
    train_forward_KLX_G,
    train_forward_KLXX_G,
)
from .utils import mixed_mala, mixed_quench_and_temper, sequential_monte_carlo


__all__ = [
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLXX_G",
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
        and values["enlarge_factor"] > 0.0
        and 0.0 <= values["tau_smc"] <= 1.0
        and 0.0 <= values["tau_ess"] <= 1.0
        and 0.0 <= values["t_tol"] <= 1.0
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
        parameters[name] = integer(name, parameters[name])
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
    weight = linear_weights_from_log(log_weight)
    return jnp.where(weight.sum() > 0, weight, jnp.ones_like(weight))


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_attempt(
    flow_root: Path | None,
    *,
    stage: int,
    attempt: int,
    t: float,
    status: str,
    candidate,
    valid_trained_ess: float,
    valid_identity_ess: float,
    valid_selected_ess: float,
    selected: str,
    batch_ess: Array,
    kept_fraction: Array,
    update_applied: Array,
) -> str | None:
    """Persist one trained attempt and update the discoverable run manifest."""

    if flow_root is None:
        return None
    relative_dir = Path(f"stage_{stage:04d}") / f"attempt_{attempt:04d}"
    stage_dir = flow_root / relative_dir
    payload = stage_dir / "trained_flow.eqx"
    save_mixed_flow(payload, candidate)
    monitor_path = stage_dir / "monitor.npz"
    _atomic_npz(
        monitor_path,
        batch_ess_hist=np.asarray(batch_ess),
        kept_fraction_hist=np.asarray(kept_fraction),
        update_applied_hist=np.asarray(update_applied),
    )
    manifest_path = flow_root / "attempts.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "attempts": []}
    record = {
        "stage": stage,
        "attempt": attempt,
        "t": t,
        "status": status,
        "valid_selected_ess": valid_selected_ess,
        "valid_trained_ess": valid_trained_ess,
        "valid_identity_ess": valid_identity_ess,
        "selected": selected,
        "trained_flow_path": payload.relative_to(flow_root).as_posix(),
        "monitor_path": monitor_path.relative_to(flow_root).as_posix(),
    }
    _atomic_json(stage_dir / "attempt.json", record)
    manifest["attempts"].append(record)
    _atomic_json(manifest_path, manifest)
    return record["trained_flow_path"]


def _record_selected(
    flow_root: Path | None, *, stage: int, selected_flow
) -> str | None:
    if flow_root is None:
        return None
    payload = flow_root / f"stage_{stage:04d}" / "selected_flow.eqx"
    save_mixed_flow(payload, selected_flow)
    manifest_path = flow_root / "attempts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in reversed(manifest["attempts"]):
        if record["stage"] == stage and record["status"] == "accepted":
            record["selected_flow_path"] = payload.relative_to(flow_root).as_posix()
            break
    _atomic_json(manifest_path, manifest)
    return payload.relative_to(flow_root).as_posix()


def _record_run_state(
    flow_root: Path | None,
    *,
    complete: bool,
    accepted_stage_count: int,
    final_t: float,
    terminal_reason: str | None = None,
    terminal_stage: int | None = None,
    terminal_t: float | None = None,
) -> None:
    """Make both complete and terminal-incomplete controllers discoverable."""

    if flow_root is None:
        return
    manifest_path = flow_root / "attempts.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "attempts": []}
    run_state = {
        "status": "complete" if complete else "incomplete",
        "accepted_stage_count": accepted_stage_count,
        "final_t": final_t,
    }
    if terminal_reason is not None:
        run_state.update({
            "terminal_reason": terminal_reason,
            "terminal_stage": terminal_stage,
            "terminal_t": terminal_t,
        })
    manifest["run_state"] = run_state
    _atomic_json(manifest_path, manifest)


@eqx.filter_jit
def _identity_weight_chunk(samples, source, target) -> Array:
    return source(samples) - target(samples)


@eqx.filter_jit
def _inverse_chunk(flow, samples) -> Array:
    return flow.inv(samples)


@eqx.filter_jit
def _importance_weight_g_chunk(samples, source, target, flow) -> Array:
    return importance_weights_log(samples, source, target, flow, "G")


def _chunked_identity_weights(samples, source, target, chunks: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunks, axis=0):
        values.append(
            jax.block_until_ready(_identity_weight_chunk(part, source, target))
        )
    return jnp.concatenate(values)


def _chunked_inverse(flow, samples, chunks: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunks, axis=0):
        values.append(jax.block_until_ready(_inverse_chunk(flow, part)))
    return jnp.concatenate(values, axis=0)


def _flow_is_finite(flow) -> bool:
    leaves = jax.tree.leaves(flow, is_leaf=lambda value: value is None)
    arrays = [value for value in leaves if isinstance(value, Array)]
    return all(bool(jnp.all(jnp.isfinite(value))) for value in arrays)


def _importance_weights_g(samples, source, target, flow, chunks: int) -> Array:
    values = []
    for part in jnp.array_split(samples, chunks, axis=0):
        values.append(
            jax.block_until_ready(
                _importance_weight_g_chunk(part, source, target, flow)
            )
        )
    return jnp.concatenate(values, axis=0)


def _boltzmann_forward_G(
    x_valid: Array,
    source,
    target,
    flow,
    *,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    monitor=None,
    bg_param: dict | None = None,
    chunks: int = 1,
    mc_image_radius: int = 3,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    checkpoint: bool = False,
    lr_warmup: int = 0,
    flow_dir: str | Path | None = None,
    _objective: str = "klx",
    _coeff_alpha: float = 0.5,
    _coeff_beta: float = 0.5,
    _melt: float = 0.0,
    _opt_alpha: float = 1e-2,
    _opt_steps: int = 100,
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain KL+X Boltzmann generator.

    Candidate bridge coefficients are selected by honest potential-space SMC
    with mixed MALA. Each incremental inverse flow is trained on those SMC
    particles; ``e_clip`` affects only that optimizer loss. Stage selection,
    importance weights, and MALA always use the unmodified target. The final
    trained flow and exact identity are compared by proposal-side ESS on all
    validation particles; the better map then faces the unchanged ``tau_ess``
    gate. ``checkpoint`` controls backward-pass rematerialization. MALA
    adjustment is mandatory and therefore is not an experimental argument.
    """

    if not hasattr(target, "domain") or not hasattr(target, "reference_internal"):
        raise TypeError("the molecular target must expose domain and reference_internal")
    if x_valid.ndim != 2 or x_valid.shape[0] < 1:
        raise ValueError(f"x_valid must have shape [N, d], got {x_valid.shape}")
    pool_size = integer("pool_size", pool_size)
    batch_size = integer("batch_size", batch_size)
    train_steps = integer("train_steps", train_steps)
    ladder = integer("ladder", ladder)
    mc_steps = integer("mc_steps", mc_steps, minimum=0)
    chunks = integer("chunks", chunks)
    mc_image_radius = integer("mc_image_radius", mc_image_radius)
    if batch_size > pool_size:
        raise ValueError("batch_size cannot exceed pool_size")
    if chunks > min(pool_size, x_valid.shape[0]):
        raise ValueError("chunks cannot exceed pool_size or the x_valid sample count")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not math.isfinite(mc_dt) or mc_dt <= 0:
        raise ValueError("mc_dt must be positive and finite")
    lr_warmup = integer("lr_warmup", lr_warmup, minimum=0)
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
        if not math.isfinite(_opt_alpha) or _opt_alpha <= 0:
            raise ValueError("molecular KLXX QT controls must be positive")
        _opt_steps = integer("opt_steps", _opt_steps, minimum=0)
    e_clip = nonnegative_real("e_clip", e_clip)
    g_clip = nonnegative_real("g_clip", g_clip)
    parameters = _bg_parameters(bg_param)
    flow_root = None if flow_dir is None else Path(flow_dir).expanduser().resolve()
    if flow_root is not None:
        if flow_root.exists() and any(flow_root.iterdir()):
            raise FileExistsError(
                f"flow_dir must be empty for a new molecular BG run: {flow_root}"
            )
        flow_root.mkdir(parents=True, exist_ok=True)

    status = monitor.printer if monitor is not None else print
    base_key = jax.random.fold_in(jax.random.key(41), seed)
    reference = target.reference_internal()[None, :]
    domain = target.domain
    domain._validate(x_valid, "molecular BG validation samples")
    particles = domain.wrap(x_valid)
    stages: list[dict] = []
    previous_t = 0.0
    terminal_reason = None
    terminal_stage = None
    terminal_t = None
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
            terminal_reason = "nonadvancing_candidate"
            terminal_stage = stage_index
            terminal_t = candidate_t
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
            draw_key, (pool_size,), 0, particles.shape[0]
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
                mc_dt=mc_dt,
                mc_steps=mc_steps,
                mc_image_radius=mc_image_radius,
                domain=domain,
                chunks=chunks,
            )
            smc_pool = jax.block_until_ready(smc_pool)
            minimum_smc_ess = float(jnp.min(smc_ess))
            status(
                f"[stage {stage_index}] [select N={pool_size}] t={candidate_t:.4f} "
                f"min SMC ESS={minimum_smc_ess:.3f} "
                f"MALA={float(jnp.mean(smc_acceptance)):.3f}"
            )
            if minimum_smc_ess < parameters["tau_smc"]:
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] selection rejected -> shrink")
                continue
            selection_accepted = True
            break

        if not selection_accepted:
            terminal_reason = "smc_selection_failed"
            terminal_stage = stage_index
            terminal_t = candidate_t
            status(
                f"[stage {stage_index}] selection failed after 60 attempts; "
                f"ladder incomplete at t={previous_t:.4f}"
            )
            break

        accepted = False
        t_hist = []
        batch_ess_hist = []
        kept_fraction_hist = []
        update_applied_hist = []
        valid_trained_ess_hist = []
        valid_identity_ess_hist = []
        attempt_status_hist = []
        trained_flow_path_hist = []
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
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    domain=domain,
                    chunks=chunks,
                )
                smc_pool = jax.block_until_ready(smc_pool)
                retry_smc_ess = float(jnp.min(smc_ess))
                status(
                    f"[stage {stage_index}] retry SMC ESS={retry_smc_ess:.3f} "
                    "(diagnostic; final validation ESS controls retry)"
                )

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
                qt_initial = source.samples(qt_seed, pool_size)
                hat_pool, hat_acceptance = mixed_quench_and_temper(
                    qt_key,
                    qt_initial,
                    current,
                    domain,
                    melt=_melt,
                    opt_alpha=_opt_alpha,
                    opt_steps=_opt_steps,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    chunks=chunks,
                )
                hat_pool = jax.block_until_ready(hat_pool)
                status(
                    f"[stage {stage_index}] KLXX coverage pool ready "
                    f"(N={pool_size}, MALA={float(jnp.mean(hat_acceptance)):.3f})"
                )
                training_result = train_forward_KLXX_G(
                    smc_pool,
                    selection_pool,
                    hat_pool,
                    previous,
                    current,
                    attempt_flow,
                    domain,
                    batch_size,
                    train_steps,
                    lr,
                    coeff_lambda=coeff_lambda,
                    coeff_alpha=_coeff_alpha,
                    coeff_beta=_coeff_beta,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_image_radius=mc_image_radius,
                    energy_origin=energy_origin,
                    e_clip=e_clip,
                    g_clip=g_clip,
                    monitor=monitor,
                    checkpoint=checkpoint,
                    lr_warmup=lr_warmup,
                    seed=trainer_seed,
                )
            else:
                training_result = train_forward_KLX_G(
                    smc_pool,
                    selection_pool,
                    previous,
                    current,
                    attempt_flow,
                    batch_size,
                    train_steps,
                    lr,
                    coeff_lambda=coeff_lambda,
                    energy_origin=energy_origin,
                    e_clip=e_clip,
                    g_clip=g_clip,
                    monitor=monitor,
                    checkpoint=checkpoint,
                    lr_warmup=lr_warmup,
                    seed=trainer_seed,
                )
            candidate, ess_history, kept_history, update_history = training_result
            candidate = jax.block_until_ready(candidate)
            if not bool(jnp.any(update_history)):
                status(
                    f"[stage {stage_index}] zero optimizer updates; "
                    "continuing to final validation ESS"
                )
            if not _flow_is_finite(candidate):
                status(
                    f"[stage {stage_index}] nonfinite final trained flow; "
                    "excluding it from the final validation ESS comparison"
                )
            identity_log_weight = _chunked_identity_weights(
                particles, previous, current, chunks
            )
            identity_ess = _ess(identity_log_weight)
            candidate_finite = _flow_is_finite(candidate)
            if candidate_finite:
                trained_log_weight = _importance_weights_g(
                    particles, previous, current, candidate, chunks
                )
                trained_ess = _ess(trained_log_weight)
            else:
                trained_log_weight = jnp.full_like(
                    identity_log_weight, -jnp.inf
                )
                trained_ess = 0.0
            # Preserve generic jflows tie semantics: the trained endpoint wins
            # an exact ESS tie with identity.
            if candidate_finite and trained_ess >= identity_ess:
                selected_label = "trained"
                selected_flow = candidate
                selected_log_weight = trained_log_weight
                stage_ess = trained_ess
            else:
                selected_label = "identity"
                selected_flow = identity_flow
                selected_log_weight = identity_log_weight
                stage_ess = identity_ess
            improvement = stage_ess - identity_ess
            status(
                f"[stage {stage_index}] t={candidate_t:.4f} "
                f"validation ESS[N={particles.shape[0]}]="
                f"{stage_ess:.3f} (selected={selected_label}, "
                f"trained={trained_ess:.3f}, "
                f"identity={identity_ess:.3f}, "
                f"kept={float(jnp.mean(kept_history)):.3f})"
            )
            attempt_status = (
                "accepted"
                if stage_ess >= parameters["tau_ess"]
                else "rejected"
            )
            t_hist.append(candidate_t)
            batch_ess_hist.append(ess_history)
            kept_fraction_hist.append(kept_history)
            update_applied_hist.append(update_history)
            valid_trained_ess_hist.append(trained_ess)
            valid_identity_ess_hist.append(identity_ess)
            attempt_status_hist.append(attempt_status)
            trained_flow_path_hist.append(
                _record_attempt(
                    flow_root,
                    stage=stage_index,
                    attempt=attempt,
                    t=candidate_t,
                    status=attempt_status,
                    candidate=candidate,
                    valid_selected_ess=stage_ess,
                    valid_trained_ess=trained_ess,
                    valid_identity_ess=identity_ess,
                    selected=selected_label,
                    batch_ess=ess_history,
                    kept_fraction=kept_history,
                    update_applied=update_history,
                )
            )
            if attempt_status == "rejected":
                candidate_t = previous_t + parameters["shrink_factor"] * (
                    candidate_t - previous_t
                )
                status(f"[stage {stage_index}] training rejected -> shrink")
                continue

            proposal = (
                _chunked_inverse(selected_flow, particles, chunks)
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
                dt=mc_dt,
                steps=mc_steps,
                image_radius=mc_image_radius,
                chunks=chunks,
            )
            particles = jax.block_until_ready(particles)
            selected_flow_path = _record_selected(
                flow_root, stage=stage_index, selected_flow=selected_flow
            )
            stages.append(
                {
                    "t": candidate_t,
                    "valid_selected_ess": stage_ess,
                    "valid_trained_ess": trained_ess,
                    "valid_identity_ess": identity_ess,
                    "valid_sample_count": particles.shape[0],
                    "selected": selected_label,
                    "flow": selected_flow,
                    "t_hist": jnp.asarray(t_hist),
                    "batch_ess_hist": jnp.stack(batch_ess_hist),
                    "valid_trained_ess_hist": jnp.asarray(
                        valid_trained_ess_hist
                    ),
                    "valid_identity_ess_hist": jnp.asarray(
                        valid_identity_ess_hist
                    ),
                    "attempt_status_hist": tuple(attempt_status_hist),
                    "trained_flow_path_hist": tuple(trained_flow_path_hist),
                    "selected_flow_path": selected_flow_path,
                    "kept_fraction_hist": jnp.stack(kept_fraction_hist),
                    "update_applied_hist": jnp.stack(update_applied_hist),
                    # Compatibility aliases retained for existing drivers.
                    "ess": stage_ess,
                    "trained_ess": trained_ess,
                    "identity_ess": identity_ess,
                    "ess_samples": particles.shape[0],
                    "ess_history": ess_history,
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
                f"selected={selected_label} "
                f"post-MALA={float(jnp.mean(mala_acceptance)):.3f}"
            )
            break

        if not accepted:
            terminal_reason = "training_attempts_failed"
            terminal_stage = stage_index
            terminal_t = t_hist[-1] if t_hist else candidate_t
            status(
                f"[stage {stage_index}] gave up after {len(t_hist)} attempts; "
                f"last attempted t={terminal_t:.4f}; ladder incomplete at "
                f"t={previous_t:.4f}"
            )
            break

    complete = bool(stages and stages[-1]["t"] == 1.0)
    if not complete and terminal_reason is None:
        terminal_reason = "max_stages"
        terminal_stage = len(stages) + 1
        terminal_t = previous_t
    _record_run_state(
        flow_root,
        complete=complete,
        accepted_stage_count=len(stages),
        final_t=previous_t,
        terminal_reason=terminal_reason,
        terminal_stage=terminal_stage,
        terminal_t=terminal_t,
    )
    status(
        f"boltzmann_forward_{_objective.upper()}_G: "
        f"{'COMPLETE' if complete else 'INCOMPLETE'} ({len(stages)} stages)"
    )
    return particles, stages


def boltzmann_forward_KLX_G(
    x_valid: Array,
    source,
    target,
    flow,
    *,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    monitor=None,
    bg_param: dict | None = None,
    chunks: int = 1,
    mc_image_radius: int = 3,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    checkpoint: bool = False,
    lr_warmup: int = 0,
    flow_dir: str | Path | None = None,
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain KL+X Boltzmann generator.

    Candidate levels are selected by potential-space SMC. The trained flow
    and exact identity are then compared using proposal ESS on the full
    validation set; their maximum is the sole post-training acceptance gate.
    ``e_clip`` affects only the optimizer loss, and ``checkpoint`` controls
    backward-pass rematerialization. Passing ``flow_dir`` saves every trained
    attempt and the selected stage flow.
    """

    return _boltzmann_forward_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=pool_size,
        batch_size=batch_size,
        train_steps=train_steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        coeff_lambda=coeff_lambda,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        mc_image_radius=mc_image_radius,
        e_clip=e_clip,
        g_clip=g_clip,
        seed=seed,
        checkpoint=checkpoint,
        lr_warmup=lr_warmup,
        flow_dir=flow_dir,
    )


def boltzmann_forward_KLXX_G(
    x_valid: Array,
    source,
    target,
    flow,
    *,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    monitor=None,
    bg_param: dict | None = None,
    chunks: int = 1,
    mc_image_radius: int = 3,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    checkpoint: bool = False,
    lr_warmup: int = 0,
    flow_dir: str | Path | None = None,
) -> tuple[Array, list[dict]]:
    """Adaptive mixed-domain ``KL + X_mu + X_mix`` generator.

    The stage gate compares only the final trained flow and exact identity on
    full-validation proposal ESS. ``checkpoint`` is the independent
    backward-pass rematerialization switch.
    """

    return _boltzmann_forward_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=pool_size,
        batch_size=batch_size,
        train_steps=train_steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        coeff_lambda=coeff_lambda,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        mc_image_radius=mc_image_radius,
        e_clip=e_clip,
        g_clip=g_clip,
        seed=seed,
        checkpoint=checkpoint,
        lr_warmup=lr_warmup,
        flow_dir=flow_dir,
        _objective="klxx",
        _coeff_alpha=coeff_alpha,
        _coeff_beta=coeff_beta,
        _melt=melt,
        _opt_alpha=opt_alpha,
        _opt_steps=opt_steps,
    )


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
    flow_dir: str | Path | None = None,
) -> tuple[Array, list[dict]]:
    """Compatibility alias for :func:`boltzmann_forward_KLX_G`."""

    return boltzmann_forward_KLX_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=n_pool,
        batch_size=n_batch,
        train_steps=steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_step,
        mc_steps=mc_iters,
        coeff_lambda=coeff_lambda,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunk,
        mc_image_radius=images,
        e_clip=e_clip,
        g_clip=g_clip,
        seed=seed,
        checkpoint=checkpoint,
        lr_warmup=lr_warmup,
        flow_dir=flow_dir,
    )


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
    flow_dir: str | Path | None = None,
) -> tuple[Array, list[dict]]:
    """Compatibility alias for :func:`boltzmann_forward_KLXX_G`."""

    return boltzmann_forward_KLXX_G(
        x_valid,
        source,
        target,
        flow,
        pool_size=n_pool,
        batch_size=n_batch,
        train_steps=steps,
        lr=lr,
        ladder=ladder,
        mc_dt=mc_step,
        mc_steps=mc_iters,
        melt=melt,
        opt_alpha=opt_step,
        opt_steps=opt_iters,
        coeff_lambda=coeff_lambda,
        coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunk,
        mc_image_radius=images,
        e_clip=e_clip,
        g_clip=g_clip,
        seed=seed,
        checkpoint=checkpoint,
        lr_warmup=lr_warmup,
        flow_dir=flow_dir,
    )
