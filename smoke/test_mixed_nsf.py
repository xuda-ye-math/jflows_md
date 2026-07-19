#!/usr/bin/env python
"""Mixed_NSF reconstruction, Jacobian, and NCSF seam-regression tests."""

from __future__ import annotations

import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jflows.flow import Flow  # noqa: E402
from jflows_md import Mixed_NSF  # noqa: E402
from jflows_md.core.domain import Mixed_Domain  # noqa: E402


def main() -> None:
    domain = Mixed_Domain(3, 2)
    x = domain.wrap(jax.random.normal(jax.random.key(1), (32, domain.dimension)))

    identity = Mixed_NSF(
        jax.random.key(2),
        domain,
        bins=8,
        transforms=4,
        hidden_features=(32, 32),
    ).zeros()
    identity_y, identity_ladj = identity.call_and_ladj(x)
    assert float(jnp.max(jnp.abs(domain.displacement(identity_y, x)))) < 1e-10
    assert float(jnp.max(jnp.abs(identity_ladj))) < 1e-10

    variable_width = Mixed_NSF(
        jax.random.key(200),
        domain,
        bins=4,
        transforms=2,
        hidden_features=(7, 11),
    )
    for coupling in variable_width.couplings:
        assert tuple(
            layer.out_features for layer in coupling.network.layers[:-1]
        ) == (7, 11)

    # The opt-in balanced schedule must split both nontrivial domain blocks in
    # every complementary pair. In particular, two torsions can then directly
    # condition one another rather than always being transformed together.
    balanced_domain = Mixed_Domain(7, 2)
    balanced = Mixed_NSF(
        jax.random.key(1400),
        balanced_domain,
        bins=8,
        transforms=4,
        hidden_features=(16, 16),
        mask_strategy="balanced",
    )
    torsions = set(range(7, 9))
    for first, second in zip(
        balanced.couplings[::2], balanced.couplings[1::2], strict=True
    ):
        first_condition = set(first.condition_indices)
        second_condition = set(second.condition_indices)
        assert first_condition.isdisjoint(second_condition)
        assert first_condition | second_condition == set(range(9))
        assert first_condition & torsions and second_condition & torsions
        assert first_condition & set(range(7))
        assert second_condition & set(range(7))

    flow = Mixed_NSF(
        jax.random.key(3),
        domain,
        bins=8,
        transforms=4,
        hidden_features=(32, 32),
    )
    assert isinstance(flow, Flow)
    y, ladj = jax.jit(lambda value: flow.call_and_ladj(value))(x)
    reconstructed, inverse_ladj = jax.jit(
        lambda value: flow.inv_and_ladj(value)
    )(y)
    assert float(jnp.max(jnp.abs(domain.displacement(reconstructed, x)))) < 1e-8
    assert float(jnp.max(jnp.abs(ladj + inverse_ladj))) < 1e-8
    assert bool(jnp.all((y[:, 3:] >= -jnp.pi) & (y[:, 3:] < jnp.pi)))

    advanced_y, advanced_ladj = flow.t().call_and_ladj(x)
    advanced_x, advanced_inverse_ladj = flow.t().inv.call_and_ladj(advanced_y)
    assert float(jnp.max(jnp.abs(domain.displacement(advanced_y, y)))) < 1e-9
    assert float(jnp.max(jnp.abs(advanced_ladj - ladj))) < 1e-9
    assert float(jnp.max(jnp.abs(domain.displacement(advanced_x, x)))) < 1e-8
    assert float(jnp.max(jnp.abs(advanced_ladj + advanced_inverse_ladj))) < 1e-8

    jacobian = jax.jacfwd(lambda value: flow(value[None])[0])(x[0])
    _, autodiff_logdet = jnp.linalg.slogdet(jacobian)
    assert float(jnp.abs(autodiff_logdet - ladj[0])) < 1e-8

    # Regression gates inherited from corrected jflows circular splines:
    # periodic conditioner embeddings, representative wrapping, and the
    # learned derivative at the exact torus seam.
    shift = jnp.asarray([0.0, 0.0, 0.0, 2.0 * jnp.pi, -2.0 * jnp.pi])
    shifted_y, shifted_ladj = flow.call_and_ladj(x + shift)
    assert float(jnp.max(jnp.abs(domain.displacement(shifted_y, y)))) < 1e-9
    assert float(jnp.max(jnp.abs(shifted_ladj - ladj))) < 1e-9
    shifted_x, shifted_inverse_ladj = flow.inv_and_ladj(y + shift)
    assert float(jnp.max(jnp.abs(domain.displacement(shifted_x, reconstructed)))) < 1e-9
    assert float(jnp.max(jnp.abs(shifted_inverse_ladj - inverse_ladj))) < 1e-9

    eps = 1e-7
    seam = jnp.repeat(x[:1], 2, axis=0)
    seam = seam.at[0, 3].set(jnp.pi - eps)
    seam = seam.at[1, 3].set(-jnp.pi + eps)
    seam_y, seam_ladj = flow.call_and_ladj(seam)
    assert float(jnp.max(jnp.abs(domain.displacement(seam_y[:1], seam_y[1:])))) < 1e-4
    assert float(jnp.abs(seam_ladj[0] - seam_ladj[1])) < 1e-4

    exact_seam = jnp.repeat(x[:1], 4, axis=0)
    exact_seam = exact_seam.at[:, 3].set(
        jnp.asarray([-3.0 * jnp.pi, -jnp.pi, jnp.pi, 3.0 * jnp.pi])
    )
    exact_y, exact_ladj = flow.call_and_ladj(exact_seam)
    assert float(
        jnp.max(jnp.abs(domain.displacement(exact_y, exact_y[:1].repeat(4, axis=0))))
    ) < 1e-10
    assert float(jnp.max(jnp.abs(exact_ladj - exact_ladj[0]))) < 1e-10
    assert float(jnp.abs(exact_ladj[0] - seam_ladj[1])) < 1e-4
    exact_x, exact_inverse_ladj = flow.inv_and_ladj(exact_y)
    assert float(
        jnp.max(jnp.abs(domain.displacement(exact_x, domain.wrap(exact_seam))))
    ) < 1e-8
    assert float(jnp.max(jnp.abs(exact_ladj + exact_inverse_ladj))) < 1e-8

    gradients = eqx.filter_grad(
        lambda candidate: (
            jnp.mean(candidate.call_and_ladj(x)[0] ** 2)
            + jnp.mean(candidate.call_and_ladj(x)[1] ** 2)
        )
    )(flow)
    leaves = jax.tree.leaves(eqx.filter(gradients, eqx.is_array))
    assert leaves and all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)

    for edge_index, edge_domain in enumerate((Mixed_Domain(3, 0), Mixed_Domain(0, 3))):
        edge_x = edge_domain.wrap(
            jax.random.normal(jax.random.key(20 + edge_index), (8, 3))
        )
        edge_flow = Mixed_NSF(
            jax.random.key(30 + edge_index),
            edge_domain,
            bins=4,
            transforms=2,
            hidden_features=(8, 8),
        )
        edge_y, edge_ladj = edge_flow.call_and_ladj(edge_x)
        edge_reconstructed, edge_inverse_ladj = edge_flow.inv_and_ladj(edge_y)
        assert float(
            jnp.max(jnp.abs(edge_domain.displacement(edge_reconstructed, edge_x)))
        ) < 1e-8
        assert float(jnp.max(jnp.abs(edge_ladj + edge_inverse_ladj))) < 1e-8

    print(
        "PASS Mixed_NSF: identity, roundtrip, ladj, autodiff, "
        "Flow.t, period representatives, seam, edge domains, and gradients"
    )


if __name__ == "__main__":
    main()
