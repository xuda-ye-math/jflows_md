"""Mixed-domain molecular companions for :mod:`jflows`."""

from __future__ import annotations

import os
import subprocess
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    distributions as _distributions,
    version as _package_version,
)
from shutil import which


__all__ = [
    "__version__",
    "backend",
    "Mixed_Identity",
    "Mixed_NSF",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "available_bundles",
    "Manual_Reject",
    "boltzmann_identity",
    "boltzmann_FABX_G",
    "boltzmann_FAB_G",
    "boltzmann_forward_KLL1_G",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLX_G_fixed",
    "boltzmann_forward_KLXX_G",
    "boltzmann_forward_KLXX_G_fixed",
    "mixed_hmc",
    "mixed_mala",
    "mixed_quench_and_temper",
    "run_inference",
    "sequential_monte_carlo",
    "sequential_monte_carlo_fab",
    "train_FABX_G",
    "train_FAB_G",
    "train_forward_KLL1_G",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
]


def backend():
    """Print the JAX and OpenMM accelerator configuration."""
    from jflows import backend as jax_backend

    jax_backend()

    try:
        openmm_version = _package_version("openmm")
    except _PackageNotFoundError:
        print("OpenMM: not installed")
        print("OpenMM GPU backend: unavailable")
        return

    print(f"OpenMM {openmm_version}")
    installed = {
        item.metadata["Name"].lower().replace("_", "-").replace(".", "-"): item.version
        for item in _distributions()
        if item.metadata.get("Name")
    }
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
    cuda_available = cuda_visible and which("nvidia-smi") and subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    hip_visible = os.environ.get("ROCR_VISIBLE_DEVICES") not in ("", "-1")
    hip_available = hip_visible and which("rocm-smi") and subprocess.run(
        ["rocm-smi", "--showproductname"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

    gpu = []
    for name, plugin_version in sorted(installed.items()):
        if name.startswith("openmm-cuda-"):
            label = f"CUDA {name.removeprefix('openmm-cuda-')} ({plugin_version})"
            available = cuda_available
        elif name.startswith("openmm-hip-"):
            label = f"HIP {name.removeprefix('openmm-hip-')} ({plugin_version})"
            available = hip_available
        else:
            continue
        if plugin_version != openmm_version:
            state = f"version mismatch with OpenMM {openmm_version}"
        elif available:
            state = "available"
        else:
            state = "installed, accelerator unavailable"
        gpu.append(f"{label} — {state}")

    print(f"OpenMM GPU backend: {'; '.join(gpu) if gpu else 'unavailable'}")


_EXPORTS = {
    "Mixed_Identity": (".flow", "Mixed_Identity"),
    "Mixed_NSF": (".flow", "Mixed_NSF"),
    "Molecular_Bundle": (".system", "Molecular_Bundle"),
    "Molecular_Potential": (".potential", "Molecular_Potential"),
    "Molecular_Source": (".source", "Molecular_Source"),
    "available_bundles": (".system", "available_bundles"),
    "Manual_Reject": (".boltzmann.control", "Manual_Reject"),
    "boltzmann_identity": (".boltzmann", "boltzmann_identity"),
    "boltzmann_FABX_G": (".boltzmann", "boltzmann_FABX_G"),
    "boltzmann_FAB_G": (".boltzmann", "boltzmann_FAB_G"),
    "boltzmann_forward_KLL1_G": (".boltzmann", "boltzmann_forward_KLL1_G"),
    "boltzmann_forward_KLX_G": (".boltzmann", "boltzmann_forward_KLX_G"),
    "boltzmann_forward_KLX_G_fixed": (".boltzmann", "boltzmann_forward_KLX_G_fixed"),
    "boltzmann_forward_KLXX_G": (".boltzmann", "boltzmann_forward_KLXX_G"),
    "boltzmann_forward_KLXX_G_fixed": (".boltzmann", "boltzmann_forward_KLXX_G_fixed"),
    "mixed_hmc": (".utils", "mixed_hmc"),
    "mixed_mala": (".utils", "mixed_mala"),
    "mixed_quench_and_temper": (".utils", "mixed_quench_and_temper"),
    "run_inference": (".boltzmann.inference", "run_inference"),
    "sequential_monte_carlo": (".utils", "sequential_monte_carlo"),
    "sequential_monte_carlo_fab": (".utils", "sequential_monte_carlo_fab"),
    "train_FABX_G": (".train", "train_FABX_G"),
    "train_FAB_G": (".train", "train_FAB_G"),
    "train_forward_KLL1_G": (".train", "train_forward_KLL1_G"),
    "train_forward_KLX_G": (".train", "train_forward_KLX_G"),
    "train_forward_KLXX_G": (".train", "train_forward_KLXX_G"),
    "__version__": (".version", "__version__"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
