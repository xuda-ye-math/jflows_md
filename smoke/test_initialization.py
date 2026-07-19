#!/usr/bin/env python
"""Identity-initialization interface checks."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jflows_md.boltzmann import (  # noqa: E402
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
from jflows_md.train import train_forward_KLX_G, train_forward_KLXX_G  # noqa: E402


def main() -> None:
    for function in (train_forward_KLX_G, train_forward_KLXX_G):
        parameter = inspect.signature(function).parameters["initialize_from_identity"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is False
    for function in (boltzmann_forward_KLX_G, boltzmann_forward_KLXX_G):
        parameter = inspect.signature(function).parameters["initialize_from_identity"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is True
    print("PASS molecular identity-initialization API")


if __name__ == "__main__":
    main()
