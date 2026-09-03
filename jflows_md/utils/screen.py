"""Importance-weight screen carried by every ESS and every resampling weight in jflows_md.

``compute_ESS_log`` and ``linear_weights_from_log`` here are the ``jflows`` reductions
applied to screened log weights: infinite or NaN entries and the ``fraction`` largest
ones get weight zero. Every module of ``jflows_md`` imports these two names from this
module, so a log weight is screened exactly once, at the point where it is reduced to
an ESS or to resampling weights, and never at the point where it is created. One
default, ``SCREEN_FRACTION``, serves training and inference alike; every caller
passes the fraction it was given.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jflows.utils import compute_ESS_log as _ess_log
from jflows.utils import linear_weights_from_log as _linear_from_log

SCREEN_FRACTION = 1e-4   # default fraction of the largest log weights removed at the top


def screen_log_weight(log_weight: Array, fraction: float = SCREEN_FRACTION) -> Array:
    """Infinite or NaN log weights and the ``fraction`` largest ones get weight zero.

    A log weight far above the rest marks a hole of the pushforward density at
    an ordinary target point; kept, it would dominate an ESS and be copied into
    most of a resampled population. Exactly ``ceil(fraction * N)`` entries are
    removed at the top; ``fraction = 0`` removes only the infinite or NaN ones.
    """
    finite = jnp.isfinite(log_weight)
    clean = jnp.where(finite, log_weight, -jnp.inf)
    count = int(log_weight.shape[0])
    k = int(-(-fraction * count // 1))
    if k == 0:
        return clean
    top = jnp.argsort(clean)[count - k:]          # exactly k entries, ties included only once
    return clean.at[top].set(-jnp.inf)


def compute_ESS_log(log_weight: Array, fraction: float = SCREEN_FRACTION) -> Array:
    """ESS of the screened log weights (as a fraction of the count)."""
    return _ess_log(screen_log_weight(log_weight, fraction))


def linear_weights_from_log(
    log_weight: Array, fraction: float = SCREEN_FRACTION
) -> Array:
    """Normalized linear weights of the screened log weights, for resampling."""
    return _linear_from_log(screen_log_weight(log_weight, fraction))
