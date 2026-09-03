"""Mixed-domain quench and temper.

Melt (Gaussian scatter of the Euclidean block, uniform redraw of the periodic
block), L-BFGS quench into the basins of ``potential``, MALA temper. With
``coeff_qt > 0`` a weighted resampling and a second rejuvenation follow the
temper: each tempered particle ``y`` carries the weight
``exp(-coeff_qt * potential(y))``, the pool is resampled with the screened
weights and rejuvenated again with ``mc_steps`` MALA steps under
``potential``. A tempered particle left at high energy, such as a clash the
quench carried away along the singular core, is removed by the resampling;
``coeff_qt = 0`` (default) returns the tempered pool.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.utils import lbfgs, resample

from ..core.domain import Mixed_Domain
from .control import check_rejection
from .rejuvenation import mixed_mala
from .screen import SCREEN_FRACTION, linear_weights_from_log


@eqx.filter_jit
def _mixed_lbfgs_chunk(
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    opt_alpha: float,
    opt_steps: int,
) -> Array:
    return domain.wrap(
        lbfgs(
            samples,
            potential,
            alpha=opt_alpha,
            steps=opt_steps,
            armijo=True,
            chunks=1,
        )
    )


@eqx.filter_jit
def _energy_log_weight(samples, potential, coeff_qt):
    return -coeff_qt * potential(samples)


def mixed_quench_and_temper(
    key: Array,
    samples: Array,
    potential,
    domain: Mixed_Domain,
    *,
    melt: float = 0.0,
    opt_alpha: float = 1e-2,
    opt_steps: int = 100,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    mc_image_radius: int = 3,
    chunks: int = 1,
    coeff_qt: float = 0.0,
    screen_fraction: float = SCREEN_FRACTION,
    reject_requested=None,
):
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
    parts = []
    for part in jnp.array_split(current, chunks, axis=0):
        check_rejection(reject_requested)
        parts.append(jax.block_until_ready(_mixed_lbfgs_chunk(
            part, potential, domain, opt_alpha=opt_alpha, opt_steps=opt_steps,
        )))
    quenched = jnp.concatenate(parts)
    tempered, acceptance = mixed_mala(
        temper_key,
        quenched,
        potential,
        domain,
        dt=mc_dt,
        steps=mc_steps,
        image_radius=mc_image_radius,
        chunks=chunks,
        reject_requested=reject_requested,
    )
    if coeff_qt == 0:
        return tempered, acceptance
    resample_key, rejuvenate_key = jax.random.split(jax.random.fold_in(key, 1))
    log_weight = jnp.concatenate([
        jax.block_until_ready(_energy_log_weight(part, potential, coeff_qt))
        for part in jnp.array_split(tempered, chunks, axis=0)
    ])
    tempered = resample(
        resample_key,
        tempered,
        linear_weights_from_log(log_weight, screen_fraction),
        N=tempered.shape[0],
    )
    return mixed_mala(
        rejuvenate_key,
        tempered,
        potential,
        domain,
        dt=mc_dt,
        steps=mc_steps,
        image_radius=mc_image_radius,
        chunks=chunks,
        reject_requested=reject_requested,
    )
