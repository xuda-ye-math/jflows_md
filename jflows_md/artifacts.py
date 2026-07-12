"""Versioned loading helpers for molecular training artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import equinox as eqx
import jax
import numpy as np

from .flow import Mixed_NSF
from .potential import Molecular_Potential
from .system import sha256_file


__all__ = ["load_mixed_flow_stages", "package_source_sha256"]


def package_source_sha256(package: str | ModuleType) -> str:
    """Hash every Python source file in an imported package tree."""
    module = importlib.import_module(package) if isinstance(package, str) else package
    root = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _scalar(metadata, name: str):
    if name not in metadata:
        raise ValueError(f"saved run is missing metadata field {name!r}")
    return metadata[name].item()


def load_mixed_flow_stages(
    run_directory: str | Path,
    *,
    target: Molecular_Potential | None = None,
    verify: bool = True,
) -> tuple[Molecular_Potential, tuple[Mixed_NSF, ...]]:
    """Reconstruct serialized ``Mixed_NSF`` stages from a completed run.

    The run directory must contain ``data.npz``, ``flows.eqx``, and the
    completion manifest written by the molecular driver. Source and file
    hashes are checked by default so an incompatible code tree cannot be used
    silently.
    """
    path = Path(run_directory).expanduser().resolve()
    marker_path = path / "COMPLETE.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"completed-run marker not found: {marker_path}")
    with marker_path.open(encoding="utf-8") as handle:
        marker = json.load(handle)
    if marker.get("schema_version") != 1:
        raise ValueError(
            f"unsupported molecular run schema: {marker.get('schema_version')}"
        )
    data_path = path / "data.npz"
    flow_path = path / "flows.eqx"
    status_path = path / "train_status.log"
    if verify:
        for name, artifact in (
            ("data_sha256", data_path),
            ("flows_sha256", flow_path),
            ("status_sha256", status_path),
        ):
            expected = marker.get(name)
            actual = sha256_file(artifact)
            if expected != actual:
                raise ValueError(
                    f"saved-run hash mismatch for {artifact}: {actual} != {expected}"
                )

    with np.load(data_path, allow_pickle=False) as metadata:
        bundle = str(_scalar(metadata, "bundle"))
        manifest_sha256 = str(_scalar(metadata, "manifest_sha256"))
        stage_count = int(_scalar(metadata, "stage_count"))
        bins = int(_scalar(metadata, "bins"))
        transforms = int(_scalar(metadata, "transforms"))
        hidden_features = tuple(int(value) for value in metadata["hidden_features"])
        slope = float(_scalar(metadata, "slope"))
        euclidean_bound = float(_scalar(metadata, "nsf_lim"))
        flow_key_data = np.asarray(metadata["flow_key_data"], dtype=np.uint32)
        saved_jflows_hash = str(_scalar(metadata, "jflows_source_sha256"))
        saved_md_hash = str(_scalar(metadata, "jflows_md_source_sha256"))

    if target is None:
        target = Molecular_Potential.from_bundle(bundle)
    if target.manifest_sha256 != manifest_sha256:
        raise ValueError(
            "saved flow and molecular target use different bundle manifests"
        )
    if verify:
        current_hashes = {
            "jflows": package_source_sha256("jflows"),
            "jflows_md": package_source_sha256("jflows_md"),
        }
        saved_hashes = {
            "jflows": saved_jflows_hash,
            "jflows_md": saved_md_hash,
        }
        if current_hashes != saved_hashes:
            raise ValueError(
                "saved flow source hashes do not match the imported packages: "
                f"saved={saved_hashes}, current={current_hashes}"
            )

    template = Mixed_NSF(
        jax.random.wrap_key_data(flow_key_data),
        target.domain,
        bins=bins,
        transforms=transforms,
        euclidean_bound=euclidean_bound,
        hidden_features=hidden_features,
        slope=slope,
    ).zeros()
    skeleton = tuple(template for _ in range(stage_count))
    flows = eqx.tree_deserialise_leaves(flow_path, skeleton)
    return target, flows
