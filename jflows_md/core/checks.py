"""Shared scalar validation for the molecular companion internals."""

from __future__ import annotations

import math
import operator

import numpy as np


def boolean(name: str, value) -> bool:
    """Return a host Boolean while rejecting numeric and traced substitutes."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a Python or NumPy boolean, got {value!r}")


def integer(name: str, value, minimum: int = 1) -> int:
    """Return an integer-like value after enforcing a lower bound."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return result


def nonnegative_real(name: str, value) -> float:
    """Return a nonnegative real scalar, allowing positive infinity."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar, got {value!r}") from exc
    if math.isnan(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def positive_real(name: str, value) -> float:
    """Return a positive finite real scalar."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result
