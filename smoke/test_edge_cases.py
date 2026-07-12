#!/usr/bin/env python
"""Numerical and validation regressions specific to the molecular companion."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md.boltzmann import (  # noqa: E402
    _bg_parameters,
    _ess as _bg_ess,
    _linear_weights as _bg_linear_weights,
    molecular_boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.flow import Mixed_NSF  # noqa: E402
from jflows_md.source import Molecular_Source  # noqa: E402
from jflows_md.train import (  # noqa: E402
    _adam_step,
    _clip_global,
    train_molecular_forward_KLXX_G,
)
from jflows_md.utils import (  # noqa: E402
    _linear_weights,
    mixed_mala,
    potential_space_smc,
    wrapped_normal_relative_error_bound,
)


class Toy_Potential(Potential):
    domain: Mixed_Domain

    def __init__(self, domain: Mixed_Domain):
        self.domain = domain

    def __call__(self, samples):
        return 0.5 * jnp.sum(samples[:, : self.domain.euclidean_dim] ** 2, axis=-1)

    def reference_internal(self):
        return jnp.zeros((self.domain.dimension,))


def raises(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    # The molecular trainer must share jflows' finite-safe clipping contract.
    clipped = _clip_global({"g": jnp.asarray([jnp.inf, 2.0])}, 1.0)["g"]
    assert bool(jnp.isfinite(clipped).all())
    huge = _clip_global({"g": jnp.full((4,), 1e38)}, 1.0)["g"]
    assert bool(jnp.isfinite(huge).all())
    assert bool(jnp.allclose(jnp.linalg.vector_norm(huge), 1.0, rtol=2e-5))
    below_ceiling = _clip_global({"g": jnp.asarray([1e20])}, 1e30)["g"]
    assert bool(jnp.array_equal(below_ceiling, jnp.asarray([1e20])))
    zeroed = _clip_global({"g": jnp.asarray([3.0])}, 0.0)["g"]
    assert bool(jnp.array_equal(zeroed, jnp.asarray([0.0])))

    # Rejected Adam steps retain moments and the actual-update counter.
    params = {"p": jnp.asarray([0.0])}
    first = {"p": jnp.asarray([0.0])}
    second = {"p": jnp.asarray([0.0])}
    count = jnp.asarray(0, dtype=jnp.int32)
    skipped = _adam_step(
        params,
        first,
        second,
        {"p": jnp.asarray([jnp.nan])},
        jnp.asarray(jnp.nan),
        count,
        jnp.asarray(0.1),
        float("inf"),
        jnp.asarray(True),
    )
    after_skip = _adam_step(
        skipped[0],
        skipped[1],
        skipped[2],
        {"p": jnp.asarray([1.0])},
        jnp.asarray(1.0),
        skipped[3],
        jnp.asarray(0.1),
        float("inf"),
        jnp.asarray(True),
    )
    clean_first = _adam_step(
        params,
        first,
        second,
        {"p": jnp.asarray([1.0])},
        jnp.asarray(1.0),
        count,
        jnp.asarray(0.1),
        float("inf"),
        jnp.asarray(True),
    )
    assert not bool(skipped[4]) and int(skipped[3]) == 0
    assert bool(jnp.array_equal(after_skip[0]["p"], clean_first[0]["p"]))
    assert int(after_skip[3]) == int(clean_first[3]) == 1

    overflowed = _adam_step(
        params,
        first,
        second,
        {"p": jnp.asarray([1e30])},
        jnp.asarray(1.0),
        count,
        jnp.asarray(0.1),
        float("inf"),
        jnp.asarray(True),
    )
    assert not bool(overflowed[4]) and int(overflowed[3]) == 0
    assert bool(jnp.array_equal(overflowed[0]["p"], params["p"]))

    # Local log-weight conversion mirrors the safe jflows edge semantics.
    weights, valid = _linear_weights(
        jnp.asarray([jnp.inf, 0.0, jnp.inf, -1.0])
    )
    assert valid and bool(
        jnp.array_equal(weights, jnp.asarray([1.0, 0.0, 1.0, 0.0]))
    )
    fallback, valid = _linear_weights(jnp.asarray([jnp.nan, 0.0]))
    assert not valid and bool(jnp.array_equal(fallback, jnp.ones(2)))
    fallback, valid = _linear_weights(jnp.full((3,), -jnp.inf))
    assert not valid and bool(jnp.array_equal(fallback, jnp.ones(3)))
    assert _bg_ess(jnp.asarray([jnp.nan, 0.0, 0.0])) == 0.0
    assert _bg_ess(jnp.asarray([jnp.inf, 0.0, jnp.inf, -1.0])) == 0.5
    assert bool(
        jnp.array_equal(
            _bg_linear_weights(jnp.asarray([jnp.nan, 0.0])), jnp.ones(2)
        )
    )

    # Invalid constructor and MCMC controls fail before expensive tracing.
    raises(lambda: Mixed_Domain(1.5, 1))
    raises(lambda: Mixed_Domain(True, 1))
    domain = Mixed_Domain(2, 1)
    raises(lambda: Molecular_Source(domain, mean=[jnp.nan, 0.0]))
    raises(lambda: Molecular_Source(domain, variance=[1.0, jnp.inf]))
    source = Molecular_Source(domain)
    raises(lambda: source.samples(jax.random.key(1), N=1.5))
    raises(lambda: source.samples(jax.random.key(1), N=True))
    raises(lambda: Mixed_NSF(jax.random.key(2), domain, bins=4.0))
    raises(lambda: Mixed_NSF(jax.random.key(2), domain, transforms=True))
    raises(
        lambda: Mixed_NSF(
            jax.random.key(2), domain, euclidean_bound=float("nan")
        )
    )
    raises(lambda: wrapped_normal_relative_error_bound(float("nan"), 3))
    raises(lambda: wrapped_normal_relative_error_bound(1e-3, 2.5))

    samples = source.samples(jax.random.key(3), N=4)
    target = Toy_Potential(domain)
    raises(
        lambda: mixed_mala(
            jax.random.key(4), samples, target, domain, step=float("nan")
        )
    )
    raises(
        lambda: mixed_mala(
            jax.random.key(4), samples, target, domain, iters=1.5
        )
    )
    raises(
        lambda: potential_space_smc(
            jax.random.key(5),
            samples,
            source,
            target,
            t_list=(0.5, float("nan")),
            iters=1,
            domain=domain,
        )
    )
    raises(lambda: _bg_parameters({"tau_ess": float("nan")}))
    raises(lambda: _bg_parameters({"enlarge_factor": 1.0}))
    raises(lambda: _bg_parameters({"max_retry": True}))

    flow = Mixed_NSF(
        jax.random.key(6),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    for alpha, beta in (
        (-0.1, 0.5),
        (0.5, -0.1),
        (float("nan"), 0.5),
        (0.5, float("nan")),
        (float("inf"), 0.5),
        (0.5, float("inf")),
    ):
        raises(
            lambda alpha=alpha, beta=beta: train_molecular_forward_KLXX_G(
                samples,
                samples,
                samples,
                source,
                target,
                flow,
                domain,
                n_batch=2,
                steps=1,
                lr=1e-3,
                coeff_alpha=alpha,
                coeff_beta=beta,
                mc_iters=0,
            )
        )
        raises(
            lambda alpha=alpha, beta=beta: molecular_boltzmann_forward_KLXX_G(
                samples,
                source,
                target,
                flow,
                n_pool=2,
                n_batch=2,
                steps=1,
                lr=1e-3,
                ladder=1,
                mc_step=1e-3,
                mc_iters=0,
                melt=0.0,
                opt_step=1e-2,
                opt_iters=0,
                coeff_alpha=alpha,
                coeff_beta=beta,
            )
        )

    print("PASS molecular finite-safety and pre-compilation validation edges")


if __name__ == "__main__":
    main()
