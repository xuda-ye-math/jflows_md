"""Versioned loading helpers for molecular training artifacts."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .flow import Mixed_NSF
from .core.domain import Mixed_Domain
from .potential import Molecular_Potential
from .system import (
    _artifact_bundle_name,
    _manifest_matches_bundle_name,
    sha256_file,
)


__all__ = [
    "load_mixed_flow",
    "load_mixed_flow_stages",
    "mixed_flow_metadata",
    "package_source_sha256",
    "save_mixed_flow",
]


_ACTIVATIONS = {
    "jax.nn.silu": jax.nn.silu,
    "jax.nn.tanh": jax.nn.tanh,
}

# Exact package digest at public commit deac775, immediately before the
# reviewed bundle-name migration. It is accepted only together with the
# corresponding frozen artifact name and manifest hash.
_MIGRATED_SCHEMA2_SOURCE_SHA256 = {
    "11c26e119d9f7d19bbe04444da56c3577ed32abad0477749406c75d35a0a618c"
}


def _activation_id(activation) -> str:
    for identifier, candidate in _ACTIVATIONS.items():
        if activation is candidate:
            return identifier
    raise ValueError(
        "Mixed_NSF artifact activation is not registered; supported values are "
        f"{sorted(_ACTIVATIONS)}"
    )


def mixed_flow_metadata(flow: Mixed_NSF) -> dict[str, object]:
    """Return architecture metadata required for exact artifact reconstruction."""
    if not isinstance(flow, Mixed_NSF) or not flow.couplings:
        raise TypeError("mixed_flow_metadata requires a nonempty Mixed_NSF")
    activation_ids = {
        _activation_id(coupling.network.activation) for coupling in flow.couplings
    }
    if len(activation_ids) != 1:
        raise ValueError("Mixed_NSF couplings use inconsistent activations")
    dtypes = {
        np.dtype(str(leaf.dtype)).name
        for leaf in jax.tree.leaves(flow)
        if eqx.is_inexact_array(leaf)
    }
    if len(dtypes) != 1:
        raise ValueError(f"Mixed_NSF parameters use inconsistent dtypes: {sorted(dtypes)}")
    condition_mask = np.zeros((flow.transforms, flow.domain.dimension), dtype=bool)
    for index, coupling in enumerate(flow.couplings):
        condition_mask[index, list(coupling.condition_indices)] = True
    first = flow.couplings[0]
    hidden_features = tuple(
        int(layer.out_features) for layer in first.network.layers[:-1]
    )
    slopes = {float(coupling.slope) for coupling in flow.couplings}
    if len(slopes) != 1:
        raise ValueError("Mixed_NSF couplings use inconsistent slopes")
    return {
        "artifact_schema": 1,
        "flow_type": "Mixed_NSF",
        "euclidean_dim": flow.domain.euclidean_dim,
        "periodic_dim": flow.domain.periodic_dim,
        "bins": flow.bins,
        "transforms": flow.transforms,
        "euclidean_bound": flow.euclidean_bound,
        "hidden_features": hidden_features,
        "slope": next(iter(slopes)),
        "activation_id": next(iter(activation_ids)),
        "parameter_dtype": next(iter(dtypes)),
        "mask_strategy": flow.mask_strategy,
        "condition_mask": condition_mask,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _flow_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def save_mixed_flow(path: str | Path, flow: Mixed_NSF) -> Path:
    """Atomically save one reloadable ``Mixed_NSF`` and its architecture."""

    if not isinstance(flow, Mixed_NSF):
        raise TypeError("save_mixed_flow requires a Mixed_NSF")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _flow_metadata_path(destination)
    token = uuid4().hex
    payload_tmp = destination.with_name(f".{destination.name}.{token}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    try:
        eqx.tree_serialise_leaves(payload_tmp, flow)
        metadata = {
            name: _jsonable(value)
            for name, value in mixed_flow_metadata(flow).items()
        }
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload_tmp.replace(destination)
        metadata_tmp.replace(metadata_path)
    finally:
        payload_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return destination


def load_mixed_flow(path: str | Path) -> Mixed_NSF:
    """Load one flow written by :func:`save_mixed_flow` without a template."""

    source = Path(path).expanduser().resolve()
    metadata_path = _flow_metadata_path(source)
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"mixed-flow payload or metadata is missing: {source}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_schema") != 1 or metadata.get("flow_type") != "Mixed_NSF":
        raise ValueError("unsupported mixed-flow artifact metadata")
    activation_id = metadata.get("activation_id")
    if activation_id not in _ACTIVATIONS:
        raise ValueError(f"unsupported Mixed_NSF activation {activation_id!r}")
    parameter_dtype = metadata.get("parameter_dtype")
    if parameter_dtype not in ("float32", "float64"):
        raise ValueError(f"unsupported Mixed_NSF dtype {parameter_dtype!r}")
    if parameter_dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("loading a float64 Mixed_NSF requires JAX x64")
    domain = Mixed_Domain(
        int(metadata["euclidean_dim"]), int(metadata["periodic_dim"])
    )
    template = Mixed_NSF(
        jax.random.key(0),
        domain,
        bins=int(metadata["bins"]),
        transforms=int(metadata["transforms"]),
        euclidean_bound=float(metadata["euclidean_bound"]),
        hidden_features=tuple(int(value) for value in metadata["hidden_features"]),
        slope=float(metadata["slope"]),
        activation=_ACTIVATIONS[activation_id],
        mask_strategy=str(metadata["mask_strategy"]),
        condition_mask=np.asarray(metadata["condition_mask"], dtype=bool),
    ).zeros()
    dtype = jnp.dtype(parameter_dtype)
    template = jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
        template,
    )
    return eqx.tree_deserialise_leaves(source, template)


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
    silently. Schema-1 metadata remains readable only with an explicit
    ``target=`` from its original source tree; built-in target discovery is a
    schema-2 contract.
    """
    path = Path(run_directory).expanduser().resolve()
    marker_path = path / "COMPLETE.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"completed-run marker not found: {marker_path}")
    with marker_path.open(encoding="utf-8") as handle:
        marker = json.load(handle)
    schema_value = marker.get("schema_version")
    if (
        isinstance(schema_value, bool)
        or not isinstance(schema_value, int)
        or schema_value not in (1, 2)
    ):
        raise ValueError(
            f"unsupported molecular run schema: {schema_value!r}"
        )
    schema_version = schema_value
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
        if "schema_version" in metadata:
            data_schema_value = _scalar(metadata, "schema_version")
            if (
                isinstance(data_schema_value, (bool, np.bool_))
                or not isinstance(data_schema_value, (int, np.integer))
            ):
                raise ValueError(
                    f"saved data has invalid schema_version: {data_schema_value!r}"
                )
            data_schema_version = int(data_schema_value)
        elif schema_version == 1:
            data_schema_version = 1
        else:
            raise ValueError("schema-2 saved data is missing schema_version")
        if data_schema_version != schema_version:
            raise ValueError(
                "completion marker and saved data disagree on schema version: "
                f"{schema_version!r} != {data_schema_version!r}"
            )
        bundle = str(_scalar(metadata, "bundle"))
        manifest_sha256 = str(_scalar(metadata, "manifest_sha256"))
        stage_count_value = _scalar(metadata, "stage_count")
        if (
            isinstance(stage_count_value, (bool, np.bool_))
            or not isinstance(stage_count_value, (int, np.integer))
            or int(stage_count_value) < 1
        ):
            raise ValueError(
                f"saved run has invalid positive stage_count: {stage_count_value!r}"
            )
        stage_count = int(stage_count_value)
        bins = int(_scalar(metadata, "bins"))
        transforms = int(_scalar(metadata, "transforms"))
        hidden_features = tuple(int(value) for value in metadata["hidden_features"])
        slope = float(_scalar(metadata, "slope"))
        euclidean_bound = float(_scalar(metadata, "nsf_lim"))
        flow_key_data = np.asarray(metadata["flow_key_data"], dtype=np.uint32)
        saved_jflows_hash = str(_scalar(metadata, "jflows_source_sha256"))
        saved_md_hash = str(_scalar(metadata, "jflows_md_source_sha256"))
        if schema_version == 1:
            activation_id = "jax.nn.silu"
            parameter_dtype = "float32"
            mask_strategy = "random"
            condition_mask = None
            saved_jax_version = None
            saved_equinox_version = None
        else:
            activation_id = str(_scalar(metadata, "activation_id"))
            parameter_dtype = str(_scalar(metadata, "parameter_dtype"))
            # Schema-2 predates selectable mask schedules. Missing metadata
            # therefore denotes the historical random schedule exactly.
            mask_strategy = (
                str(_scalar(metadata, "mask_strategy"))
                if "mask_strategy" in metadata
                else "random"
            )
            condition_mask = (
                np.asarray(metadata["condition_mask"], dtype=bool)
                if "condition_mask" in metadata
                else None
            )
            saved_jax_version = str(_scalar(metadata, "jax_version"))
            saved_equinox_version = str(_scalar(metadata, "equinox_version"))

    if target is None and schema_version == 1:
        raise ValueError(
            "schema-1 artifacts require an explicit target from their original "
            "bundle/source revision"
        )
    if target is None:
        bundle_reference: str | Path = _artifact_bundle_name(bundle)
        saved_path = Path(bundle).expanduser()
        if not saved_path.is_absolute():
            local_path = (path / saved_path).resolve()
            if local_path.is_dir():
                bundle_reference = local_path
        target = Molecular_Potential.from_bundle(bundle_reference)
    manifest_matches = _manifest_matches_bundle_name(
        bundle, manifest_sha256, target.bundle_name, target.manifest_sha256
    )
    if not manifest_matches:
        raise ValueError(
            "saved flow and molecular target use different bundle manifests"
        )
    migrated_name_manifest = manifest_sha256 != target.manifest_sha256
    if verify:
        current_hashes = {
            "jflows": package_source_sha256("jflows"),
            "jflows_md": package_source_sha256("jflows_md"),
        }
        saved_hashes = {
            "jflows": saved_jflows_hash,
            "jflows_md": saved_md_hash,
        }
        reviewed_name_migration = (
            schema_version == 2
            and migrated_name_manifest
            and saved_hashes["jflows"] == current_hashes["jflows"]
            and saved_hashes["jflows_md"] in _MIGRATED_SCHEMA2_SOURCE_SHA256
        )
        if current_hashes != saved_hashes and not reviewed_name_migration:
            raise ValueError(
                "saved flow source hashes do not match the imported packages: "
                f"saved={saved_hashes}, current={current_hashes}"
            )
        if schema_version == 2:
            saved_versions = {
                "jax": saved_jax_version,
                "equinox": saved_equinox_version,
            }
            current_versions = {"jax": jax.__version__, "equinox": eqx.__version__}
            if saved_versions != current_versions:
                raise ValueError(
                    "saved flow dependency versions do not match the runtime: "
                    f"saved={saved_versions}, current={current_versions}"
                )

    if activation_id not in _ACTIVATIONS:
        raise ValueError(
            f"unsupported saved Mixed_NSF activation {activation_id!r}; "
            f"supported values are {sorted(_ACTIVATIONS)}"
        )
    if parameter_dtype not in ("float32", "float64"):
        raise ValueError(f"unsupported saved Mixed_NSF dtype {parameter_dtype!r}")
    if parameter_dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("loading a float64 Mixed_NSF requires JAX x64 to be enabled")
    if mask_strategy not in ("random", "balanced"):
        raise ValueError(
            f"unsupported saved Mixed_NSF mask strategy {mask_strategy!r}"
        )

    template = Mixed_NSF(
        jax.random.wrap_key_data(flow_key_data),
        target.domain,
        bins=bins,
        transforms=transforms,
        euclidean_bound=euclidean_bound,
        hidden_features=hidden_features,
        slope=slope,
        activation=_ACTIVATIONS[activation_id],
        mask_strategy=mask_strategy,
        condition_mask=condition_mask,
    ).zeros()
    dtype = jnp.dtype(parameter_dtype)
    template = jax.tree.map(
        lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
        template,
    )
    skeleton = tuple(template for _ in range(stage_count))
    flows = eqx.tree_deserialise_leaves(flow_path, skeleton)
    encoded = io.BytesIO()
    eqx.tree_serialise_leaves(encoded, flows)
    if encoded.getvalue() != flow_path.read_bytes():
        raise ValueError(
            "serialized Mixed_NSF payload does not match the declared stage_count"
        )
    return target, flows
