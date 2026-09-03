"""Write complete molecular Boltzmann stages."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import partial
import hashlib
import json
import marshal
import os
from pathlib import Path
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
from uuid import uuid4

import equinox as eqx
import numpy as np


__all__ = ["FORMAT", "create", "finish", "stage"]

FORMAT = "jflows-md-stage-resume-3"

_HISTORY_NAMES = (
    "t_hist",
    "batch_ess_hist",
    "valid_trained_ess_hist",
    "valid_identity_ess_hist",
    "mala_acceptance",
)
_IDENTITY_HISTORY_NAMES = (
    "t_hist",
    "valid_identity_ess_hist",
    "mala_acceptance",
)
_METADATA_NAMES = (
    "t",
    "t_start",
    "rg_start",
    "rg_end",
    "valid_selected_ess",
    "valid_trained_ess",
    "valid_identity_ess",
    "valid_sample_count",
    "initialized_from_identity",
    "selected",
    "attempt_status_hist",
    "objective",
    "elapsed_seconds",
    "accepted_attempt_seconds",
)
_IDENTITY_METADATA_NAMES = (
    "t",
    "t_start",
    "rg_start",
    "rg_end",
    "valid_selected_ess",
    "valid_identity_ess",
    "valid_sample_count",
    "selected",
    "attempt_status_hist",
    "objective",
    "elapsed_seconds",
    "accepted_attempt_seconds",
)


def _value(value):
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if eqx.is_array(value):
        return np.asarray(value).tolist()
    return value


def jsonfile(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _array(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _history(path: Path, record: dict) -> None:
    names = (
        _IDENTITY_HISTORY_NAMES
        if record["objective"] == "identity"
        else _HISTORY_NAMES
    )
    values = {name: record[name] for name in names}
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _flow(path: Path, flow) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    eqx.tree_serialise_leaves(temporary, flow)
    os.replace(temporary, path)


def _closure_values(closure):
    if closure is None:
        return None
    values = []
    for cell in closure:
        try:
            values.append(cell.cell_contents)
        except ValueError:
            values.append({"empty_cell": True})
    return tuple(values)


def _name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _object_state(value):
    state = {}
    if hasattr(value, "__dict__"):
        state["dict"] = vars(value)
    slots = {}
    for cls in reversed(type(value).__mro__):
        names = cls.__dict__.get("__slots__", ())
        names = (names,) if isinstance(names, str) else names
        for name in names:
            if name in ("__dict__", "__weakref__"):
                continue
            attribute = name
            if name.startswith("__") and not name.endswith("__"):
                attribute = f"_{cls.__name__.lstrip('_')}{name}"
            key = f"{cls.__module__}.{cls.__qualname__}.{name}"
            try:
                slots[key] = getattr(value, attribute)
            except AttributeError:
                slots[key] = {"unset_slot": True}
    if slots:
        state["slots"] = slots
    return state or None


def _class_state(value):
    state = {}
    excluded = {
        "__dict__",
        "__doc__",
        "__module__",
        "__slots__",
        "__weakref__",
    }
    for cls in reversed(type(value).__mro__):
        if cls is object or cls.__module__ == "builtins":
            continue
        namespace = {
            name: item
            for name, item in vars(cls).items()
            if name not in excluded
            and not isinstance(item, (GetSetDescriptorType, MemberDescriptorType))
        }
        if namespace:
            state[f"{cls.__module__}.{cls.__qualname__}"] = namespace
    return state or None


def _flow_template(value, static=False, memo=None):
    memo = {} if memo is None else memo
    if eqx.is_array(value):
        result = {"array": [*value.shape], "dtype": str(value.dtype)}
        if static:
            result["value"] = hashlib.sha256(
                np.asarray(value).tobytes()
            ).hexdigest()
        return result
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "type": _name(value),
            "fields": {
                field.name: _flow_template(
                    getattr(value, field.name),
                    static or field.metadata.get("static", False),
                    memo,
                )
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "mapping": _name(value),
            "items": [
                [
                    _flow_template(key, True, memo),
                    _flow_template(item, static, memo),
                ]
                for key, item in sorted(value.items(), key=lambda item: repr(item[0]))
            ],
        }
    if isinstance(value, tuple):
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "tuple": [_flow_template(item, static, memo) for item in value],
        }
    if isinstance(value, list):
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "list": [_flow_template(item, static, memo) for item in value],
        }
    if isinstance(value, partial):
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "partial": _flow_template(value.func, True, memo),
            "args": _flow_template(value.args, True, memo),
            "keywords": _flow_template(value.keywords, True, memo),
        }
    if isinstance(value, ModuleType):
        return {"module": value.__name__}
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    if callable(value):
        receiver = getattr(value, "__self__", None)
        function = getattr(value, "__func__", value)
        callable_instance = (
            receiver is None
            and function is value
            and not isinstance(
                value,
                (FunctionType, BuiltinFunctionType, BuiltinMethodType),
            )
        )
        value_jax_wrapper = type(value).__module__.split(".", 1)[0] in {
            "jax",
            "jaxlib",
        }
        if callable_instance and not value_jax_wrapper:
            receiver = value
            function = type(value).__call__
        code = getattr(function, "__code__", None)
        closure = getattr(function, "__closure__", None)
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        globals_ = getattr(function, "__globals__", {})
        jax_wrapper = value_jax_wrapper or (
            type(function).__module__.split(".", 1)[0] in {"jax", "jaxlib"}
        )
        function_state = (
            None
            if jax_wrapper
            else _object_state(function)
        )
        instance_state = (
            _object_state(value)
            if callable_instance and not jax_wrapper
            else None
        )
        receiver_class_state = (
            _class_state(receiver)
            if receiver is not None and not jax_wrapper
            else None
        )
        if (
            callable_instance
            and not jax_wrapper
            and code is None
            and instance_state is None
        ):
            raise TypeError(f"unsupported static flow value: {_name(value)}")
        referenced_globals = {
            name: _flow_template(globals_[name], True, memo)
            for name in (() if code is None else code.co_names)
            if name in globals_ and globals_[name] is not function
        }
        return {
            "identity": reference,
            "callable": (
                f"{getattr(function, '__module__', type(value).__module__)}."
                f"{getattr(function, '__qualname__', type(value).__qualname__)}"
            ),
            "code": (
                None
                if code is None
                else hashlib.sha256(marshal.dumps(code)).hexdigest()
            ),
            "defaults": _flow_template(
                getattr(function, "__defaults__", None), True, memo
            ),
            "kwdefaults": _flow_template(
                getattr(function, "__kwdefaults__", None), True, memo
            ),
            "closure": _flow_template(
                _closure_values(closure), True, memo
            ),
            "globals": referenced_globals,
            "wrapped": _flow_template(
                getattr(function, "__wrapped__", None), True, memo
            ),
            "receiver": _flow_template(
                receiver, True, memo
            ),
            "state": _flow_template(
                function_state, True, memo
            ),
            "instance_state": _flow_template(
                instance_state, True, memo
            ),
            "receiver_class_state": _flow_template(
                receiver_class_state, True, memo
            ),
        }
    if value is None:
        return {"none": True}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, str):
        return {"str": value}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, np.dtype):
        return {"dtype": str(value)}
    if isinstance(value, np.generic):
        return _flow_template(value.item(), static, memo)
    object_state = _object_state(value)
    if object_state is not None:
        identity = id(value)
        if identity in memo:
            return {"reference": memo[identity]}
        reference = memo[identity] = len(memo)
        return {
            "identity": reference,
            "object": _name(value),
            "state": _flow_template(object_state, True, memo),
        }
    raise TypeError(f"unsupported static flow value: {_name(value)}")


def create(run_dir, problem_id, config, samples, flow) -> dict:
    config = _value(config)
    run = {
        "format": FORMAT,
        "problem_id": problem_id,
        "status": "running",
        "config": config,
        "initial_samples_path": "initial_samples.npy",
        "stages": [],
    }
    if flow is not None:
        run["flow_template"] = _flow_template(flow)
        run["initial_flow_path"] = "initial_flow.eqx"
    json.dumps(_value(run))
    root = Path(run_dir).expanduser().resolve()
    if root.is_dir() and any(root.iterdir()):
        raise FileExistsError(f"run directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "stages").mkdir(exist_ok=True)
    _array(root / "initial_samples.npy", samples)
    if flow is not None:
        _flow(root / "initial_flow.eqx", flow)
    jsonfile(root / "run.json", run)
    return run


def stage(run_dir, run: dict, record: dict, samples) -> dict:
    root = Path(run_dir).expanduser().resolve()
    identity = record.get("objective") == "identity"
    has_flow = "flow_template" in run
    if identity and has_flow:
        raise ValueError("identity stages cannot be added to a flow run")
    if not identity and not has_flow:
        raise ValueError("trained stages require a stored flow template")
    if identity:
        required = set(_IDENTITY_METADATA_NAMES) | set(_IDENTITY_HISTORY_NAMES)
    else:
        required = set(_METADATA_NAMES) | set(_HISTORY_NAMES) | {
            "flow",
            "continuation_flow",
        }
    missing = required - set(record)
    if missing:
        raise ValueError(f"incomplete stage record: {sorted(missing)}")
    previous = run["stages"][-1]["t"] if run["stages"] else 0.0
    if record["t_start"] != previous or not previous < record["t"] <= 1.0:
        raise ValueError("invalid accepted-stage schedule")
    if identity and (
        record["selected"] != "identity"
        or record["valid_sample_count"] != samples.shape[0]
    ):
        raise ValueError("invalid identity stage metadata")
    if not identity:
        if (
            record["selected"] not in ("trained", "identity")
            or record["objective"] not in ("forward_klx", "forward_kll1", "fab", "forward_klxx", "fabx")
            or record["valid_sample_count"] != samples.shape[0]
        ):
            raise ValueError("invalid stage metadata")
        if (
            _flow_template(record["flow"]) != run["flow_template"]
            or _flow_template(record["continuation_flow"])
            != run["flow_template"]
        ):
            raise ValueError("stage flow does not match the run template")
    number = len(run["stages"]) + 1
    relative = Path("stages") / f"stage_{number:06d}"
    directory = root / relative
    stages = root / "stages"
    if (
        stages.is_symlink()
        or not stages.is_dir()
        or stages.resolve() != stages
        or directory.is_symlink()
    ):
        raise ValueError("unsafe stage directory")
    directory.mkdir(exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError("unsafe stage directory")
    population = relative / "samples.npy"
    history = relative / "history.npz"
    _array(root / population, samples)
    _history(root / history, record)
    names = _IDENTITY_METADATA_NAMES if identity else _METADATA_NAMES
    metadata = {name: record[name] for name in names}
    metadata.update({
        "stage": number,
        "validation_samples_path": str(population),
        "history_path": str(history),
    })
    if not identity:
        selected = relative / "selected.eqx"
        continuation = relative / "continuation.eqx"
        _flow(root / selected, record["flow"])
        _flow(root / continuation, record["continuation_flow"])
        metadata.update({
            "selected_flow_path": str(selected),
            "continuation_flow_path": str(continuation),
        })
    jsonfile(directory / "stage.json", metadata)
    run["stages"].append({
        "stage": number,
        "t": record["t"],
        "path": str(relative),
    })
    jsonfile(root / "run.json", run)
    saved = dict(record)
    saved["validation_samples_path"] = str(population)
    if not identity:
        saved["selected_flow_path"] = str(selected)
        saved["continuation_flow_path"] = str(continuation)
    return saved


def finish(run_dir, run: dict, status: str) -> None:
    run["status"] = status
    jsonfile(Path(run_dir).expanduser().resolve() / "run.json", run)
