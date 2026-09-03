#!/usr/bin/env python
"""Tiny flow-proposal SMC, HMC, quench-and-temper, and screen smoke test."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows.potential import Potential  # noqa: E402
from jflows_md import Mixed_NSF, Molecular_Source  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.utils import (  # noqa: E402
    compute_ESS_log,
    flow_target_batch,
    linear_weights_from_log,
    mixed_hmc,
    flow_fab_batch,
    mixed_quench_and_temper,
    screen_log_weight,
    sequential_monte_carlo,
    sequential_monte_carlo_fab,
)


class Toy_Target(Potential):
    domain: Mixed_Domain

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, value):
        return 0.5 * jnp.sum((value[:, :2] - 0.3) ** 2, axis=-1) - 0.2 * jnp.cos(value[:, 2])


def main() -> None:
    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain)
    target = Toy_Target(domain)
    flow = Mixed_NSF(jax.random.key(1), domain, bins=4, transforms=1, hidden_features=(8,)).zeros()
    x = source.samples(jax.random.key(2), 64)

    moved, acceptance = mixed_hmc(jax.random.key(3), x, target, domain, dt=0.1, leapfrog_steps=4, chunks=2)
    assert moved.shape == x.shape and bool(jnp.isfinite(moved).all())
    assert bool(jnp.all(jnp.abs(moved[:, 2]) <= jnp.pi))
    assert acceptance.shape == (1,) and 0.0 < float(acceptance[0]) <= 1.0
    assert float(jnp.abs(moved - x).mean()) > 0.0

    y, proposal, log_w = flow_target_batch(
        jax.random.key(4), x, source, target, flow, domain, ladder=2, mc_dt=0.05,
        mc_steps_1=3, mc_steps_2=3,
    )
    assert y.shape == proposal.shape == x.shape and log_w.shape == (64,)
    assert bool(jnp.allclose(proposal, x, atol=1e-6))       # identity flow: proposal is the input
    assert bool(jnp.isfinite(y).all())
    # with no rejuvenation the levels reduce to reweight + resample: every output
    # row is a row of the proposal, and the moved run differs from it
    y_none, _, _ = flow_target_batch(
        jax.random.key(4), x, source, target, flow, domain, ladder=2, mc_dt=0.05,
        mc_steps_1=0, mc_steps_2=0,
    )
    nearest = jnp.min(jnp.abs(y_none[:, None, :] - proposal[None, :, :]).sum(-1), axis=1)
    assert bool(jnp.all(nearest < 1e-6))
    assert float(jnp.abs(y - y_none).mean()) > 0.0

    # the exact two-phase FAB SMC: same phase-1 proposal and weights, further particles
    y_fab, proposal_fab, log_w_fab = flow_fab_batch(
        jax.random.key(4), x, source, target, flow, domain, ladder=2, mc_dt=0.05,
        mc_steps_1=3, mc_steps_2=3,
    )
    assert bool(jnp.allclose(proposal_fab, proposal, atol=1e-6))
    assert bool(jnp.allclose(log_w_fab, log_w, atol=1e-5))
    assert bool(jnp.isfinite(y_fab).all()) and not bool(jnp.allclose(y_fab, y, atol=1e-6))
    y_fab_pop, _, log_w_fab_pop = sequential_monte_carlo_fab(
        jax.random.key(4), x, source, target, flow, ladder=2, mc_dt=0.05,
        mc_steps_1=3, mc_steps_2=3, domain=domain, chunks=1,
    )
    assert bool(jnp.allclose(log_w_fab_pop, log_w_fab, atol=1e-5))
    assert bool(jnp.allclose(y_fab_pop, y_fab, atol=1e-4))

    y_pop, proposal_pop, log_w_pop = sequential_monte_carlo(
        jax.random.key(5), x, source, target, flow, ladder=2, mc_dt=0.05,
        mc_steps_1=3, mc_steps_2=3, chunks=2,
    )
    assert y_pop.shape == x.shape and bool(jnp.allclose(proposal_pop, x, atol=1e-6))
    assert bool(jnp.allclose(log_w_pop, log_w, atol=1e-5))

    screened = screen_log_weight(jnp.asarray([0.0, 1.0, jnp.inf, 5.0]), 0.25)
    untouched = screen_log_weight(jnp.asarray([0.0, 1.0, jnp.inf, 5.0]), 0.0)
    assert bool(jnp.array_equal(jnp.isinf(untouched), jnp.asarray([False, False, True, False])))
    assert bool(jnp.array_equal(jnp.isinf(screened), jnp.asarray([False, False, True, True])))
    ess_loose = float(compute_ESS_log(log_w, 1e-4))
    ess_tight = float(compute_ESS_log(log_w, 0.25))
    assert 0.0 < ess_loose <= 1.0 and 0.0 < ess_tight <= 1.0
    weights = linear_weights_from_log(log_w, 0.25)
    assert int(jnp.sum(weights == 0.0)) >= 16

    pool, _ = mixed_quench_and_temper(
        jax.random.key(6), x, target, domain, melt=0.5, opt_alpha=0.1, opt_steps=2,
        mc_dt=0.05, mc_steps=2, chunks=2, coeff_qt=0.5,
    )
    assert pool.shape == x.shape and bool(jnp.isfinite(pool).all())
    plain, _ = mixed_quench_and_temper(
        jax.random.key(6), x, target, domain, melt=0.5, opt_alpha=0.1, opt_steps=2,
        mc_dt=0.05, mc_steps=2, chunks=2, coeff_qt=0.0,
    )
    assert float(jnp.mean(target(pool))) < float(jnp.mean(target(plain))), (
        "the energy-weighted resampling must lower the mean pool energy"
    )
    print("PASS flow-proposal SMC, FAB SMC, screen, and coeff_qt")


if __name__ == "__main__":
    main()
