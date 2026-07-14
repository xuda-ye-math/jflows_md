"""Molecular companions for the local :mod:`jflows` package.

The package keeps import-time dependencies light so bundle construction can run
in an OpenMM/AmberTools environment while training runs in a JAX environment.
Public JAX objects are imported lazily through ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "Mixed_Identity",
    "Mixed_NSF",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "annealed_importance_sampling",
    "available_bundles",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLXX_G",
    "load_mixed_flow",
    "mixed_flow_metadata",
    "mixed_mala",
    "mixed_quench_and_temper",
    "potential_space_smc",
    "sequential_monte_carlo",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
    "save_mixed_flow",
]


_EXPORTS = {
    "Mixed_Identity": (".flow", "Mixed_Identity"),
    "Mixed_NSF": (".flow", "Mixed_NSF"),
    "Molecular_Bundle": (".system", "Molecular_Bundle"),
    "Molecular_Potential": (".potential", "Molecular_Potential"),
    "Molecular_Source": (".source", "Molecular_Source"),
    "annealed_importance_sampling": (
        ".utils",
        "annealed_importance_sampling",
    ),
    "available_bundles": (".system", "available_bundles"),
    "boltzmann_forward_KLX_G": (".boltzmann", "boltzmann_forward_KLX_G"),
    "boltzmann_forward_KLXX_G": (".boltzmann", "boltzmann_forward_KLXX_G"),
    "load_mixed_flow": (".artifacts", "load_mixed_flow"),
    "mixed_flow_metadata": (".artifacts", "mixed_flow_metadata"),
    "mixed_mala": (".utils", "mixed_mala"),
    "mixed_quench_and_temper": (".utils", "mixed_quench_and_temper"),
    "potential_space_smc": (".utils", "potential_space_smc"),
    "sequential_monte_carlo": (".utils", "sequential_monte_carlo"),
    "train_forward_KLX_G": (".train", "train_forward_KLX_G"),
    "train_forward_KLXX_G": (".train", "train_forward_KLXX_G"),
    "save_mixed_flow": (".artifacts", "save_mixed_flow"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
