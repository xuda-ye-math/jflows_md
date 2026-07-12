"""Sampling utilities for molecular flows on ``R^p x T^q``.

This flat module mirrors :mod:`jflows.utils` at the smaller scale needed by
``jflows_md``: mixed-domain MALA is the rejuvenation kernel, while SMC and AIS
are the annealing controllers built on it. Molecular Boltzmann generators
import these utilities rather than maintaining separate sampler modules.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from jflows.potential import Potential, linear_combination
from jflows.utils import compute_ESS_log, lbfgs, resample

from .core.domain import Mixed_Domain


__all__ = [
    "ais",
    "annealed_importance_sampling",
    "mixed_mala",
    "mixed_mala_step",
    "mixed_quench_and_temper",
    "potential_space_smc",
    "sequential_monte_carlo",
    "smc",
    "wrapped_normal_relative_error_bound",
]


_WRAPPED_RELATIVE_TOLERANCE = 1e-12


def wrapped_normal_relative_error_bound(step: float, images: int) -> float:
    """Certified relative tail bound for the truncated wrapped proposal.

    The displacement is first reduced to ``[-pi, pi)``. For proposal variance
    ``2 * step``, this bounds the omitted image mass relative to the retained
    wrapped-normal sum uniformly over that interval.
    """

    if step <= 0 or images < 1:
        raise ValueError("step must be positive and images at least one")
    variance = 2.0 * float(step)
    first_omitted = images + 1
    exponent = -(
        ((2 * first_omitted - 1) ** 2 - 1) * math.pi**2
    ) / (2.0 * variance)
    ratio = math.exp(-4.0 * first_omitted * math.pi**2 / variance)
    return 2.0 * math.exp(exponent) / (1.0 - ratio)


def _wrapped_log_kernel(delta: Array, variance: float, images: int) -> Array:
    shifts = 2.0 * jnp.pi * jnp.arange(-images, images + 1, dtype=delta.dtype)
    terms = -(delta[..., None] + shifts) ** 2 / (2.0 * variance)
    return jnp.sum(logsumexp(terms, axis=-1), axis=-1)


def _proposal_log_density(
    destination: Array,
    mean: Array,
    domain: Mixed_Domain,
    variance: float,
    images: int,
) -> Array:
    euclidean_delta = (
        destination[:, : domain.euclidean_dim] - mean[:, : domain.euclidean_dim]
    )
    value = -0.5 * jnp.sum(euclidean_delta**2 / variance, axis=-1)
    if domain.periodic_dim:
        periodic_delta = (
            destination[:, domain.euclidean_dim :]
            - mean[:, domain.euclidean_dim :]
        )
        # Center the truncated image sum on the shortest torus displacement.
        # The drifted proposal mean itself need not lie in the principal box.
        periodic_delta = jnp.mod(periodic_delta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        value = value + _wrapped_log_kernel(periodic_delta, variance, images)
    return value


def mixed_mala_step(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float,
    images: int = 3,
) -> tuple[Array, Array]:
    """Take one Metropolis-adjusted Langevin step on ``R^p x T^q``.

    Returns the updated samples and a Boolean acceptance value per chain.
    The wrapped-normal proposal density is evaluated symmetrically on every
    periodic coordinate, so the Metropolis correction remains valid at the
    torus seam.
    """

    error_bound = wrapped_normal_relative_error_bound(step, images)
    if error_bound > _WRAPPED_RELATIVE_TOLERANCE:
        raise ValueError(
            "wrapped-normal image truncation is not certified: "
            f"relative bound {error_bound:.3e} exceeds "
            f"{_WRAPPED_RELATIVE_TOLERANCE:.1e}; increase images or reduce step"
        )
    noise_key, accept_key = jax.random.split(key)
    energy = potential(samples)
    gradient = potential.grad(samples)
    mean_forward = samples - step * gradient
    proposal = domain.wrap(
        mean_forward
        + jnp.sqrt(2.0 * step)
        * jax.random.normal(noise_key, samples.shape, dtype=samples.dtype)
    )
    proposal_energy = potential(proposal)
    proposal_gradient = potential.grad(proposal)
    mean_reverse = proposal - step * proposal_gradient
    log_forward = _proposal_log_density(
        proposal, mean_forward, domain, 2.0 * step, images
    )
    log_reverse = _proposal_log_density(
        samples, mean_reverse, domain, 2.0 * step, images
    )
    log_alpha = energy - proposal_energy + log_reverse - log_forward
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
    accepted = jnp.log(jax.random.uniform(accept_key, energy.shape)) < jnp.minimum(
        log_alpha, 0.0
    )
    result = jnp.where(accepted[:, None], proposal, samples)
    return result, accepted


@eqx.filter_jit
def _mixed_mala_chunk(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float = 1e-4,
    iters: int = 1,
    images: int = 3,
) -> tuple[Array, Array]:
    """Compile one fixed-shape MALA chunk; orchestration stays eager."""

    keys = jax.random.split(key, iters)

    def body(state, subkey):
        updated, accepted = mixed_mala_step(
            subkey, state, potential, domain, step=step, images=images
        )
        return updated, accepted.astype(updated.dtype).mean()

    return jax.lax.scan(body, samples, keys)


def _validate_rows_and_chunk(samples: Array, chunk: int) -> None:
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError(f"samples must have shape [N, d] with N >= 1, got {samples.shape}")
    if chunk < 1 or chunk > samples.shape[0]:
        raise ValueError("chunk must lie in [1, sample count]")


def mixed_mala(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float = 1e-4,
    iters: int = 1,
    images: int = 3,
    chunk: int = 1,
) -> tuple[Array, Array]:
    """Run mixed-domain MALA with eager, memory-bounding chunk control.

    ``chunk`` is the number of row partitions, matching :mod:`jflows`.
    Each partition calls the same small compiled scan and is synchronized
    before the next one is submitted. Consequently increasing ``chunk``
    reduces the physical batch held by the compiled molecular graph instead
    of duplicating that graph inside one giant JIT trace.

    This controller is intentionally not itself jittable. Use
    :func:`mixed_mala_step` or the private fixed-shape kernel when composing a
    custom compiled calculation.
    """

    _validate_rows_and_chunk(samples, chunk)
    domain._validate(samples, "mixed MALA samples")
    if iters < 1:
        raise ValueError("iters must be positive")
    # Fail before the first (potentially expensive) compilation.
    error_bound = wrapped_normal_relative_error_bound(step, images)
    if error_bound > _WRAPPED_RELATIVE_TOLERANCE:
        raise ValueError(
            "wrapped-normal image truncation is not certified: "
            f"relative bound {error_bound:.3e} exceeds "
            f"{_WRAPPED_RELATIVE_TOLERANCE:.1e}; increase images or reduce step"
        )

    parts = jnp.array_split(samples, chunk, axis=0)
    # Preserve the original unchunked stream exactly. The multi-chunk path
    # retains its established split-per-part convention.
    keys = (key,) if chunk == 1 else jax.random.split(key, len(parts))
    outputs, acceptances, sizes = [], [], []
    for part_key, part in zip(keys, parts):
        output, acceptance = _mixed_mala_chunk(
            part_key,
            part,
            potential,
            domain,
            step=step,
            iters=iters,
            images=images,
        )
        output, acceptance = jax.block_until_ready((output, acceptance))
        outputs.append(output)
        acceptances.append(acceptance)
        sizes.append(part.shape[0])
    total = sum(sizes)
    weighted_acceptance = sum(
        acceptance * (size / total)
        for acceptance, size in zip(acceptances, sizes)
    )
    return jnp.concatenate(outputs, axis=0), weighted_acceptance


@eqx.filter_jit
def _mixed_lbfgs_chunk(
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    step: float,
    iters: int,
) -> Array:
    """Quench one fixed-shape mixed-domain chunk."""

    quenched = lbfgs(
        samples,
        potential,
        step=step,
        iters=iters,
        armijo=True,
        chunk=1,
    )
    return domain.wrap(quenched)


def mixed_quench_and_temper(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    melt: float = 0.0,
    opt_step: float = 1e-2,
    opt_iters: int = 100,
    mc_step: float = 1e-3,
    mc_iters: int = 100,
    images: int = 3,
    chunk: int = 1,
) -> tuple[Array, Array]:
    """Construct a mixed-domain quench-and-temper coverage pool.

    The Euclidean block receives Gaussian melt noise.  When ``melt`` is
    positive the periodic block is refreshed from its maximum-entropy uniform
    law; with ``melt=0`` the supplied torsions are retained.  L-BFGS follows
    the periodic extension of the potential and every chunk is wrapped before
    exact mixed-domain MALA tempering.
    """

    _validate_rows_and_chunk(samples, chunk)
    domain._validate(samples, "mixed quench-and-temper samples")
    if not math.isfinite(melt) or melt < 0:
        raise ValueError("melt must be nonnegative and finite")
    if not math.isfinite(opt_step) or opt_step <= 0 or opt_iters < 1:
        raise ValueError("opt_step and opt_iters must be positive")
    if mc_iters < 1:
        raise ValueError("mc_iters must be positive")
    # Validate the wrapped proposal before compiling either expensive kernel.
    error_bound = wrapped_normal_relative_error_bound(mc_step, images)
    if error_bound > _WRAPPED_RELATIVE_TOLERANCE:
        raise ValueError(
            "wrapped-normal image truncation is not certified: "
            f"relative bound {error_bound:.3e} exceeds "
            f"{_WRAPPED_RELATIVE_TOLERANCE:.1e}"
        )

    melt_key, periodic_key, temper_key = jax.random.split(key, 3)
    current = samples
    if melt > 0:
        euclidean = current[:, : domain.euclidean_dim]
        euclidean = euclidean + melt * jax.random.normal(
            melt_key, euclidean.shape, dtype=current.dtype
        )
        periodic = jax.random.uniform(
            periodic_key,
            (current.shape[0], domain.periodic_dim),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=current.dtype,
        )
        current = jnp.concatenate((euclidean, periodic), axis=-1)

    quenched = []
    for part in jnp.array_split(current, chunk, axis=0):
        value = _mixed_lbfgs_chunk(
            part,
            potential,
            domain,
            step=opt_step,
            iters=opt_iters,
        )
        quenched.append(jax.block_until_ready(value))
    current = jnp.concatenate(quenched, axis=0)
    return mixed_mala(
        temper_key,
        current,
        potential,
        domain,
        step=mc_step,
        iters=mc_iters,
        images=images,
        chunk=chunk,
    )


@eqx.filter_jit
def _bridge_log_weight_chunk(
    samples: Array,
    source,
    target,
    delta: Array,
) -> Array:
    return delta * (source(samples) - target(samples))


def _chunked_bridge_log_weights(
    samples: Array,
    source,
    target,
    delta: float,
    chunk: int,
) -> Array:
    pieces = []
    coefficient = jnp.asarray(delta, dtype=samples.dtype)
    for part in jnp.array_split(samples, chunk, axis=0):
        value = _bridge_log_weight_chunk(part, source, target, coefficient)
        pieces.append(jax.block_until_ready(value))
    value = jnp.concatenate(pieces, axis=0)
    return jnp.where(jnp.isfinite(value), value, -jnp.inf)


def _linear_weights(log_weight: Array) -> tuple[Array, bool]:
    finite = jnp.isfinite(log_weight)
    maximum = jnp.max(jnp.where(finite, log_weight, -jnp.inf))
    weight = jnp.where(finite, jnp.exp(log_weight - maximum), 0.0)
    weight = jax.block_until_ready(weight)
    has_weight = bool(jnp.sum(weight) > 0)
    return (weight if has_weight else jnp.ones_like(weight)), has_weight


def _potential_space_schedule(
    key: Array,
    samples: Array,
    source,
    target,
    levels: tuple[float, ...],
    *,
    step: float,
    iters: int,
    images: int,
    domain: Mixed_Domain | None,
    chunk: int,
) -> tuple[Array, Array, Array]:
    _validate_rows_and_chunk(samples, chunk)
    if iters < 1:
        raise ValueError("iters must be positive")
    current = samples
    mala_domain = domain if domain is not None else getattr(target, "domain", None)
    if mala_domain is None:
        raise ValueError("domain is required when the target potential has no domain")
    mala_domain._validate(samples, "SMC samples")
    ess_values, acceptance_values = [], []
    previous_value = 0.0
    for level_index, value in enumerate(levels, start=1):
        log_weight = _chunked_bridge_log_weights(
            current, source, target, value - previous_value, chunk
        )
        ess = compute_ESS_log(log_weight)
        ess = jax.block_until_ready(jnp.where(jnp.isfinite(ess), ess, 0.0))
        ess_values.append(ess)
        safe_weight, has_weight = _linear_weights(log_weight)
        resample_key, mala_key = jax.random.split(
            jax.random.fold_in(key, level_index)
        )
        bridge = linear_combination([source, target], [1.0 - value, value])
        if has_weight:
            current = resample(
                resample_key, current, safe_weight, N=current.shape[0]
            )
            current, acceptance = mixed_mala(
                mala_key,
                current,
                bridge,
                mala_domain,
                step=step,
                iters=iters,
                images=images,
                chunk=chunk,
            )
        else:
            acceptance = jnp.zeros((iters,), dtype=current.dtype)
        acceptance_values.append(acceptance)
        previous_value = value
    return current, jnp.asarray(ess_values), jnp.stack(acceptance_values)


def sequential_monte_carlo(
    key: Array,
    samples: Array,
    source,
    target,
    ladder: int = 1,
    step: float = 1e-3,
    iters: int = 100,
    images: int = 3,
    domain: Mixed_Domain | None = None,
    chunk: int = 1,
) -> tuple[Array, Array, Array]:
    """Reweight, resample, and rejuvenate over a uniform potential bridge.

    ``ladder``, ``step``, ``iters``, and ``chunk`` match the corresponding
    :mod:`jflows` controls. Returns final particles, normalized incremental
    ESS per ladder level, and molecular MALA acceptance per level/iteration.
    The level and chunk loops remain eager so molecular compilation scales
    with one physical chunk rather than ``ladder * chunk`` copies.
    """

    if ladder < 1:
        raise ValueError("ladder must be at least one")
    levels = tuple(level / ladder for level in range(1, ladder + 1))
    return _potential_space_schedule(
        key,
        samples,
        source,
        target,
        levels,
        step=step,
        iters=iters,
        images=images,
        domain=domain,
        chunk=chunk,
    )


def potential_space_smc(
    key: Array,
    samples: Array,
    source,
    target,
    t_list,
    step: float = 1e-3,
    iters: int = 100,
    images: int = 3,
    domain: Mixed_Domain | None = None,
    chunk: int = 1,
) -> tuple[Array, Array, Array]:
    """Compatibility SMC interface for an explicit, nonuniform schedule.

    ``t_list`` contains the positive bridge coefficients to visit, in
    strictly increasing order. It may end below one for a deliberately
    partial bridge; use :func:`sequential_monte_carlo` for a uniform complete
    bridge controlled by ``ladder``.
    """

    levels = tuple(float(value) for value in t_list)
    if not levels:
        raise ValueError("t_list must contain at least one bridge level")
    if any(not 0.0 < value <= 1.0 for value in levels) or any(
        right <= left for left, right in zip(levels, levels[1:])
    ):
        raise ValueError("t_list must be strictly increasing in (0, 1]")
    return _potential_space_schedule(
        key,
        samples,
        source,
        target,
        levels,
        step=step,
        iters=iters,
        images=images,
        domain=domain,
        chunk=chunk,
    )


@eqx.filter_jit
def _flow_inverse_chunk(flow, samples: Array) -> Array:
    return flow.inv(samples)


@eqx.filter_jit
def _ais_log_weight_chunk(
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    scale: Array,
) -> Array:
    latent, ladj_g = flow.call_and_ladj(samples)
    return scale * (-target(samples) + source(latent) - ladj_g)


def _chunked_ais_log_weights(
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    scale: float,
    chunk: int,
) -> Array:
    values = []
    coefficient = jnp.asarray(scale, dtype=samples.dtype)
    for part in jnp.array_split(samples, chunk, axis=0):
        value = _ais_log_weight_chunk(
            part, source, target, flow, coefficient
        )
        values.append(jax.block_until_ready(value))
    value = jnp.concatenate(values, axis=0)
    return jnp.where(jnp.isfinite(value), value, -jnp.inf)


def annealed_importance_sampling(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    ladder: int = 1,
    step: float = 1e-3,
    iters: int = 100,
    images: int = 3,
    domain: Mixed_Domain | None = None,
    chunk: int = 1,
) -> Array:
    """Flow-proposal AIS for a molecular inverse flow ``G``.

    ``flow`` always acts as ``G`` (target to source), so this companion API has
    no direction string. Source samples are first pushed by ``G^-1``.
    Reweighting follows the nominal geometric path from that pushforward
    proposal toward ``target`` over ``ladder`` levels. Every rejuvenation uses
    mixed MALA at the final target. This keeps MCMC score-free in the flow
    rather than differentiating a flow-dependent intermediate.

    The core controls ``ladder``, ``step``, ``iters``, and ``chunk`` have the
    same names and meanings as in :func:`jflows.utils.annealed_importance_sampling`
    and :func:`sequential_monte_carlo`. Unlike potential-space SMC, this
    target-rejuvenated training sampler is deliberately biased.
    """

    if ladder < 1 or iters < 1:
        raise ValueError("ladder must be at least one")
    _validate_rows_and_chunk(samples, chunk)
    mala_domain = domain
    if mala_domain is None:
        mala_domain = getattr(target, "domain", getattr(flow, "domain", None))
    if mala_domain is None:
        raise ValueError("domain is required when neither target nor flow exposes it")
    mala_domain._validate(samples, "AIS source samples")

    pushed = []
    for part in jnp.array_split(samples, chunk, axis=0):
        pushed.append(jax.block_until_ready(_flow_inverse_chunk(flow, part)))
    y = jnp.concatenate(pushed, axis=0)
    for level in range(1, ladder + 1):
        log_weight = _chunked_ais_log_weights(
            y, source, target, flow, 1.0 / ladder, chunk
        )
        safe_weight, has_weight = _linear_weights(log_weight)
        resample_key, mala_key = jax.random.split(
            jax.random.fold_in(key, level)
        )
        if has_weight:
            y = resample(resample_key, y, safe_weight, N=y.shape[0])
            y = mixed_mala(
                mala_key,
                y,
                target,
                mala_domain,
                step=step,
                iters=iters,
                images=images,
                chunk=chunk,
            )[0]
        else:
            return jnp.full_like(y, jnp.nan)
    return y


# jflows-compatible short names plus the earlier companion compatibility name.
smc = sequential_monte_carlo
ais = annealed_importance_sampling
