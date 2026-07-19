#!/usr/bin/env python
"""Public molecular API and v0.5 layout checks."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jflows_md import Molecular_Potential, Molecular_Source  # noqa: E402
from jflows_md.boltzmann import (  # noqa: E402
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.core.domain import Mixed_Domain  # noqa: E402
from jflows_md.train import train_forward_KLX_G, train_forward_KLXX_G  # noqa: E402


def main() -> None:
    for trainer in (train_forward_KLX_G, train_forward_KLXX_G):
        parameters = inspect.signature(trainer).parameters
        assert parameters["initialize_from_identity"].default is False
        assert "e_clip" not in parameters
        assert "g_clip" not in parameters
        assert "lr_warmup" not in parameters
        assert "t_start" in parameters and "t_end" in parameters
    for generator in (boltzmann_forward_KLX_G, boltzmann_forward_KLXX_G):
        parameters = inspect.signature(generator).parameters
        assert parameters["initialize_from_identity"].default is True
        assert parameters["rg_param_0"].default is inspect.Parameter.empty
        assert parameters["rg_param_1"].default is inspect.Parameter.empty
        assert "flow_dir" not in parameters
        assert "resume" not in parameters
        assert "run_dir" not in parameters

    domain = Mixed_Domain(2, 1)
    source = Molecular_Source(domain, mean=[0, 0], variance=[1, 1])
    samples = source.samples(jax.random.key(20), 4)
    assert samples.shape == (4, 3)
    assert samples.dtype == jnp.float32
    assert source(samples).shape == (4,)
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    assert target.source().samples(jax.random.key(21), 1).dtype == jnp.float32
    print("PASS modern minimal molecular API")


if __name__ == "__main__":
    main()
