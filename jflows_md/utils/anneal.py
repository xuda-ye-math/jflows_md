"""Flow-proposal sequential Monte Carlo on the mixed domain.

``sequential_monte_carlo`` (alias ``smc``) manufactures target samples from
source samples through a trained flow, the molecular counterpart of
``jflows.utils.sequential_monte_carlo``: the source particles are pushed
through the flow, and each of the ``ladder`` levels reweights them by the
1/M-th power of the proposal-to-target weight, resamples (screened weights),
and rejuvenates at the target with wrapped MALA, ``mc_steps_1`` steps on the
intermediate levels ``1 .. M-1`` and ``mc_steps_2`` steps on the last. Every
level therefore rejuvenates at ``pi``: the intermediate levels are a target
surrogate, not exact levels of the geometric path, and no level differentiates
the flow. This is the SMC of the forward KL family.

``sequential_monte_carlo_fab`` (alias ``smc_fab``) is the exact two-phase SMC
of FAB. Phase 1 repeats the levels above; phase 2 continues with ``ladder``
further levels from ``pi`` to ``pi^2 / nu`` along
``rho_k = pi (pi / nu)^(k/M)``, and rejuvenates each level at its own
``rho_k``, whose potential contains the pushforward density of the flow. The
flow is always the inverse map ``G`` (target -> source), as everywhere in
``jflows_md``.

``flow_target_batch`` and ``flow_fab_batch`` are the pure, single-chunk forms
used inside the trainers' compiled scans; ``sequential_monte_carlo`` and
``sequential_monte_carlo_fab`` are their eager, chunked forms for populations.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.potential import Potential, potential_from
from jflows.utils import resample_index

from ..core.domain import Mixed_Domain
from .rejuvenation import _mixed_mala_chunk, mixed_mala
from .screen import SCREEN_FRACTION, linear_weights_from_log


__all__ = [
    "flow_fab_batch",
    "flow_target_batch",
    "sequential_monte_carlo",
    "sequential_monte_carlo_fab",
    "smc",
    "smc_fab",
]


def _proposal(flow, samples, source, target, domain):
    """Pushforward of the source rows, its full log weights, and the target energy."""
    proposal, ladj = flow.inv_and_ladj(samples)
    proposal = domain.wrap(proposal)
    energy = target(proposal)
    return proposal, source(samples) - energy + ladj, energy


def _weights_and_energy(flow, samples, source, target):
    """Full ``log(pi / nu)``, the target energy, and ``log nu`` at moved particles.

    Through ``G``: with ``latent = G(y)`` and ``ladj = log|det J_G(y)|`` the
    pushforward density is ``log nu(y) = -source(latent) + ladj``, so the
    proposal-to-target log weight is ``log w = -target(y) - log nu(y)``. One
    flow pass yields all three.
    """
    latent, ladj = flow.call_and_ladj(samples)
    energy = target(samples)
    log_nu = -source(latent) + ladj
    return -energy - log_nu, energy, log_nu


def _path_potential(source, target, flow, domain, s: float):
    """Potential of the path distribution ``pi (pi / nu)^(-s)``, through the flow.

    ``U_s(y) = (1 - s) target(y) - s log nu(y)``: ``s = 1 - k/M`` gives the
    intermediate ``mu_k = nu^(1 - k/M) pi^(k/M)``, ``s = -k/M`` gives the FAB
    path ``rho_k = pi (pi / nu)^(k/M)``, ``s = 0`` is ``pi`` and ``s = -1`` is
    ``pi^2 / nu``. The pushforward density enters through the flow, so the
    Langevin drift is differentiated through it.
    """

    def energy(y):
        latent, ladj = flow.call_and_ladj(y)
        return (1.0 - s) * target(y) - s * (-source(latent) + ladj)

    return potential_from(energy)


def flow_target_batch(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    domain: Mixed_Domain,
    *,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    mc_image_radius: int = 3,
    screen_fraction: float = SCREEN_FRACTION,
):
    """Pure single-chunk SMC for one batch (traceable inside a scan).

    Every level rejuvenates at the target with wrapped MALA, ``mc_steps_1``
    steps on the intermediate levels and ``mc_steps_2`` on the last; the
    target energy of each level's reweighting is carried through the
    resampling into that MALA run. Returns
    ``(samples, proposal, proposal_log_weights)``: the target batch, the
    pushforward the levels started from, and the full proposal-to-target log
    weights on that pushforward (the batch ESS diagnostic).
    """
    proposal, proposal_log_weight, energy = _proposal(
        flow, samples, source, target, domain
    )
    current, log_weight = proposal, proposal_log_weight
    for level in range(1, ladder + 1):
        if level > 1:
            log_weight, energy, _ = _weights_and_energy(
                flow, current, source, target
            )
        resample_key, move_key = jax.random.split(jax.random.fold_in(key, level))
        index = resample_index(
            resample_key,
            current,
            linear_weights_from_log(log_weight / ladder, screen_fraction),
            N=current.shape[0],
        )
        current, energy = current[index], energy[index]
        current, _ = _mixed_mala_chunk(
            move_key, current, target, domain, mc_dt=mc_dt,
            mc_steps=mc_steps_1 if level < ladder else mc_steps_2,
            image_radius=mc_image_radius, energy=energy,
        )
    return current, proposal, proposal_log_weight


def flow_fab_batch(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    domain: Mixed_Domain,
    *,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    mc_image_radius: int = 3,
    screen_fraction: float = SCREEN_FRACTION,
):
    """Pure single-chunk two-phase SMC on to ``pi^2 / nu`` (traceable inside a scan).

    Phase 1 is ``flow_target_batch``. Phase 2 runs ``ladder`` further levels
    along ``rho_k = pi (pi / nu)^(k/M)``, each reweighting by ``log(pi / nu)``
    divided by ``M``, resampling, and rejuvenating at its own ``rho_k`` with
    ``mc_steps_1`` wrapped MALA steps (``mc_steps_2`` on the last level, whose
    distribution is ``pi^2 / nu``). Returns
    ``(samples, proposal, proposal_log_weights)`` with the phase-2 particles
    and phase 1's proposal and log weights.
    """
    key_1, key_2 = jax.random.split(key)
    current, proposal, proposal_log_weight = flow_target_batch(
        key_1, samples, source, target, flow, domain, ladder=ladder, mc_dt=mc_dt,
        mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, screen_fraction=screen_fraction,
    )
    for level in range(1, ladder + 1):
        log_weight, energy, log_nu = _weights_and_energy(
            flow, current, source, target
        )
        scale = -level / ladder
        resample_key, move_key = jax.random.split(jax.random.fold_in(key_2, level))
        index = resample_index(
            resample_key,
            current,
            linear_weights_from_log(log_weight / ladder, screen_fraction),
            N=current.shape[0],
        )
        level_energy = (1.0 - scale) * energy - scale * log_nu
        current, level_energy = current[index], level_energy[index]
        current, _ = _mixed_mala_chunk(
            move_key, current, _path_potential(source, target, flow, domain, scale),
            domain, mc_dt=mc_dt,
            mc_steps=mc_steps_1 if level < ladder else mc_steps_2,
            image_radius=mc_image_radius, energy=level_energy,
        )
    return current, proposal, proposal_log_weight


_proposal_chunk = eqx.filter_jit(_proposal)
_refreshed_chunk = eqx.filter_jit(_weights_and_energy)


def _eager_levels(
    key, current, energy, log_weight, source, target, flow, domain, *,
    ladder, mc_dt, mc_steps_1, mc_steps_2, mc_image_radius, chunks,
    screen_fraction, scale_of_level=None,
):
    """Run ``ladder`` chunked SMC levels; ``scale_of_level`` selects the path.

    ``None`` rejuvenates every level at the target (the forward KL surrogate);
    a callable ``level -> s`` rejuvenates at the exact path distribution
    ``U_s`` of ``_path_potential`` (the FAB phase).
    """
    for level in range(1, ladder + 1):
        if log_weight is None:
            refreshed = [
                jax.block_until_ready(_refreshed_chunk(flow, part, source, target))
                for part in jnp.array_split(current, chunks, axis=0)
            ]
            log_weight = jnp.concatenate([value[0] for value in refreshed])
            energy = jnp.concatenate([value[1] for value in refreshed])
            log_nu = jnp.concatenate([value[2] for value in refreshed])
        else:
            log_nu = None
        resample_key, move_key = jax.random.split(jax.random.fold_in(key, level))
        index = resample_index(
            resample_key,
            current,
            linear_weights_from_log(log_weight / ladder, screen_fraction),
            N=current.shape[0],
        )
        if scale_of_level is None:
            potential, level_energy = target, energy
        else:
            scale = scale_of_level(level)
            potential = _path_potential(source, target, flow, domain, scale)
            level_energy = (1.0 - scale) * energy - scale * log_nu
        current, level_energy = current[index], level_energy[index]
        current, _ = mixed_mala(
            move_key, current, potential, domain, dt=mc_dt,
            steps=mc_steps_1 if level < ladder else mc_steps_2,
            image_radius=mc_image_radius, chunks=chunks, energy=level_energy,
        )
        log_weight = None
    return current


def sequential_monte_carlo(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    mc_image_radius: int = 3,
    domain: Mixed_Domain | None = None,
    chunks: int = 1,
    screen_fraction: float = SCREEN_FRACTION,
):
    """Eager, chunked flow-proposal SMC for a population.

    Same levels and kernels as ``flow_target_batch``; the pushforward, the
    weights, and the rejuvenations run chunk by chunk. Returns
    ``(samples, proposal, proposal_log_weights)``.
    """
    domain = domain if domain is not None else getattr(target, "domain", flow.domain)
    pushed, initial, energies = zip(*[
        jax.block_until_ready(_proposal_chunk(flow, part, source, target, domain))
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    proposal = jnp.concatenate(pushed)
    proposal_log_weight = jnp.concatenate(initial)
    current = _eager_levels(
        key, proposal, jnp.concatenate(energies), proposal_log_weight,
        source, target, flow, domain, ladder=ladder, mc_dt=mc_dt,
        mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, chunks=chunks,
        screen_fraction=screen_fraction,
    )
    return current, proposal, proposal_log_weight


def sequential_monte_carlo_fab(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    mc_image_radius: int = 3,
    domain: Mixed_Domain | None = None,
    chunks: int = 1,
    screen_fraction: float = SCREEN_FRACTION,
):
    """Eager, chunked two-phase SMC on to ``pi^2 / nu`` for a population.

    Same levels and kernels as ``flow_fab_batch``, chunk by chunk. Returns
    ``(samples, proposal, proposal_log_weights)`` with the phase-2 particles
    and phase 1's proposal and log weights.
    """
    domain = domain if domain is not None else getattr(target, "domain", flow.domain)
    key_1, key_2 = jax.random.split(key)
    current, proposal, proposal_log_weight = sequential_monte_carlo(
        key_1, samples, source, target, flow, ladder=ladder, mc_dt=mc_dt,
        mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, domain=domain, chunks=chunks,
        screen_fraction=screen_fraction,
    )
    current = _eager_levels(
        key_2, current, None, None, source, target, flow, domain,
        ladder=ladder, mc_dt=mc_dt, mc_steps_1=mc_steps_1, mc_steps_2=mc_steps_2,
        mc_image_radius=mc_image_radius, chunks=chunks,
        screen_fraction=screen_fraction,
        scale_of_level=lambda level: -level / ladder,
    )
    return current, proposal, proposal_log_weight


smc = sequential_monte_carlo
smc_fab = sequential_monte_carlo_fab
