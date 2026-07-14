#!/usr/bin/env python
"""Eager API tests for molecular flow initialization controls."""

from __future__ import annotations

import inspect
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jflows_md.boltzmann as boltzmann  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
import jflows_md.train as training  # noqa: E402


class Probe_Flow(eqx.Module):
    shift: jax.Array

    def zeros(self):
        return Probe_Flow(jnp.zeros_like(self.shift))


def _result(flow, train_steps):
    return (
        flow,
        jnp.ones((train_steps,)),
        jnp.ones((train_steps,)),
        jnp.zeros((train_steps,), dtype=bool),
    )


def check_klx_wrapper() -> None:
    samples = jnp.zeros((4, 1))
    flow = Probe_Flow(jnp.asarray([3.0]))
    seen = []
    original = training._train_forward_KLX_G

    def kernel(*args, **kwargs):
        del kwargs
        seen.append(args[4])
        return _result(args[4], args[6])

    training._train_forward_KLX_G = kernel
    try:
        omitted = training.train_forward_KLX_G(
            samples, samples, None, None, flow, 2, 1, 1e-3
        )[0]
        explicit = training.train_forward_KLX_G(
            samples,
            samples,
            None,
            None,
            flow,
            2,
            1,
            1e-3,
            initialize_from_identity=False,
        )[0]
        zeroed = training.train_forward_KLX_G(
            samples,
            samples,
            None,
            None,
            flow,
            2,
            1,
            1e-3,
            initialize_from_identity=np.bool_(True),
        )[0]
        assert eqx.tree_equal(omitted, explicit)
        assert eqx.tree_equal(omitted, flow)
        np.testing.assert_array_equal(zeroed.shift, jnp.zeros((1,)))
        np.testing.assert_array_equal(flow.shift, jnp.asarray([3.0]))
        assert len(seen) == 3
        for invalid in (1, 0, None, "true", jnp.asarray(True)):
            try:
                training.train_forward_KLX_G(
                    samples,
                    samples,
                    None,
                    None,
                    flow,
                    2,
                    1,
                    1e-3,
                    initialize_from_identity=invalid,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid KLX initialization flag: {invalid!r}")
        assert len(seen) == 3
    finally:
        training._train_forward_KLX_G = original


def check_klxx_wrapper() -> None:
    samples = jnp.zeros((4, 1))
    domain = Mixed_Domain(1, 0)
    flow = Probe_Flow(jnp.asarray([4.0]))
    seen = []
    original = training._train_forward_KLXX_G

    def kernel(*args, **kwargs):
        del kwargs
        seen.append(args[5])
        return _result(args[5], args[8])

    training._train_forward_KLXX_G = kernel
    try:
        omitted = training.train_forward_KLXX_G(
            samples,
            samples,
            samples,
            None,
            None,
            flow,
            domain,
            2,
            1,
            1e-3,
        )[0]
        explicit = training.train_forward_KLXX_G(
            samples,
            samples,
            samples,
            None,
            None,
            flow,
            domain,
            2,
            1,
            1e-3,
            initialize_from_identity=False,
        )[0]
        zeroed = training.train_forward_KLXX_G(
            samples,
            samples,
            samples,
            None,
            None,
            flow,
            domain,
            2,
            1,
            1e-3,
            initialize_from_identity=True,
        )[0]
        assert eqx.tree_equal(omitted, explicit)
        assert eqx.tree_equal(omitted, flow)
        np.testing.assert_array_equal(zeroed.shift, jnp.zeros((1,)))
        np.testing.assert_array_equal(flow.shift, jnp.asarray([4.0]))
        assert len(seen) == 3
        for invalid in (1, jnp.asarray(False)):
            try:
                training.train_forward_KLXX_G(
                    samples,
                    samples,
                    samples,
                    None,
                    None,
                    flow,
                    domain,
                    2,
                    1,
                    1e-3,
                    initialize_from_identity=invalid,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid KLXX initialization flag: {invalid!r}")
        assert len(seen) == 3
    finally:
        training._train_forward_KLXX_G = original


def check_signatures() -> None:
    for function in (
        training.train_forward_KLX_G,
        training.train_forward_KLXX_G,
    ):
        parameter = inspect.signature(function).parameters[
            "initialize_from_identity"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False
    for function in (
        boltzmann.boltzmann_forward_KLX_G,
        boltzmann.boltzmann_forward_KLXX_G,
    ):
        parameter = inspect.signature(function).parameters[
            "initialize_from_identity"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is True


def main() -> None:
    check_klx_wrapper()
    check_klxx_wrapper()
    check_signatures()
    print("PASS molecular identity-initialization API")


if __name__ == "__main__":
    main()
