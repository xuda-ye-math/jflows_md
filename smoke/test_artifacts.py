#!/usr/bin/env python
"""Template-based molecular artifact round trips."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Mixed_NSF, Molecular_Potential  # noqa: E402
from jflows_md.artifacts import (  # noqa: E402
    load_flow,
    load_history,
    load_samples,
    save_flow,
    save_history,
    save_samples,
)


def main() -> None:
    target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
    flow = Mixed_NSF(
        jax.random.key(80),
        target.domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    samples = target.source().samples(jax.random.key(81), 4)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        save_flow(root / "flow.eqx", flow)
        loaded = load_flow(root / "flow.eqx", flow)
        assert eqx.tree_equal(flow, loaded)
        save_samples(root / "samples.npy", samples)
        np.testing.assert_array_equal(load_samples(root / "samples.npy"), samples)
        save_history(root / "history.npz", ess=np.asarray([0.5, 0.75]))
        np.testing.assert_array_equal(
            load_history(root / "history.npz")["ess"], [0.5, 0.75]
        )
    print("PASS template-based molecular artifacts")


if __name__ == "__main__":
    main()
