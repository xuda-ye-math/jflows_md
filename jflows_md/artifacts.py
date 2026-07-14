"""Template-free persistence for current ``Mixed_NSF`` flows."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .core.domain import Mixed_Domain
from .flow import Mixed_NSF


__all__ = ["load_mixed_flow", "mixed_flow_metadata", "save_mixed_flow"]


_ACTIVATIONS = {
    "jax.nn.silu": jax.nn.silu,
    "jax.nn.tanh": jax.nn.tanh,
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
    """Return architecture metadata required for exact reconstruction."""
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
    """Load one flow written by :func:`save_mixed_flow`."""
    source = Path(path).expanduser().resolve()
    metadata_path = _flow_metadata_path(source)
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"mixed-flow payload or metadata is missing: {source}")
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
