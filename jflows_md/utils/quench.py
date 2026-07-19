"""Mixed-domain quench and temper."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from jflows.utils import lbfgs

from ..core.domain import Mixed_Domain
from .rejuvenation import mixed_mala


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
    quenched = [
        jax.block_until_ready(
            _mixed_lbfgs_chunk(
                part,
                potential,
                domain,
                opt_alpha=opt_alpha,
                opt_steps=opt_steps,
            )
        )
        for part in jnp.array_split(current, chunks, axis=0)
    ]
    return mixed_mala(
        temper_key,
        jnp.concatenate(quenched),
        potential,
        domain,
        dt=mc_dt,
        steps=mc_steps,
        image_radius=mc_image_radius,
        chunks=chunks,
    )
