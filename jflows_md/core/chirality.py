"""Coordinate-independent stereochemical diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def signed_volume(
    x: Array,
    center: int,
    first: int,
    second: int,
    third: int,
) -> Array:
    """Signed neighbor volume for Cartesian arrays ``[..., atoms, 3]``."""

    c = x[..., center, :]
    a = x[..., first, :] - c
    b = x[..., second, :] - c
    d = x[..., third, :] - c
    return jnp.sum(a * jnp.cross(b, d), axis=-1)


def support_sign(volume: Array, expected_sign: int, *, tolerance: float = 0.0) -> Array:
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be -1 or +1")
    return expected_sign * volume > tolerance
