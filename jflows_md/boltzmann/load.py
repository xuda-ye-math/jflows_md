"""Load and resume complete molecular Boltzmann stages."""

import json
import shutil
from pathlib import Path

import equinox as eqx
import numpy as np

from .write import (
    _HISTORY_NAMES,
    _IDENTITY_HISTORY_NAMES,
    _IDENTITY_METADATA_NAMES,
    _METADATA_NAMES,
    _check_regularization,
    _flow_template,
    _value,
    create,
    finish,
    stage,
)


__all__ = [
    "fork",
    "fork_run",
    "inspect_run",
    "load",
    "load_stage_flow",
    "load_training_history",
    "load_validation_samples",
    "manifest",
    "run",
    "validate",
    "validate_run",
]


def manifest(run_dir) -> dict:
    with (Path(run_dir).expanduser().resolve() / "run.json").open(
        encoding="utf-8"
    ) as stream:
        return json.load(stream)


def _path(root: Path, relative) -> Path:
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"absolute run path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"run path escapes root: {relative}")
    return path


def _stage(root: Path, item: dict) -> dict:
    path = _path(root, Path(item["path"]) / "stage.json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate(run_dir) -> dict:
    root = Path(run_dir).expanduser().resolve()
    record = manifest(root)
    run_fields = {
        "format",
        "problem_id",
        "status",
        "config",
        "initial_samples_path",
        "stages",
    }
    missing = run_fields - set(record)
    if missing:
        raise ValueError(f"incomplete run manifest: {sorted(missing)}")
    if record["format"] != "jflows-md-stage-resume-1":
        raise ValueError(f"unknown run format: {record['format']}")
    if record["status"] not in ("running", "exhausted", "complete"):
        raise ValueError(f"unknown run status: {record['status']}")
    has_flow = "flow_template" in record or "initial_flow_path" in record
    if has_flow and not {"flow_template", "initial_flow_path"} <= set(record):
        raise ValueError("incomplete run flow metadata")
    required = [record["initial_samples_path"]]
    if has_flow:
        required.append(record["initial_flow_path"])
    previous = 0.0
    for number, item in enumerate(record["stages"], start=1):
        if not {"stage", "t", "path"} <= set(item):
            raise ValueError("incomplete stage manifest item")
        saved = _stage(root, item)
        if item["stage"] != number or saved["stage"] != number:
            raise ValueError("nonconsecutive stage manifest")
        if item["t"] != saved["t"]:
            raise ValueError("stage endpoint disagrees with run manifest")
        if saved["t_start"] != previous or not previous < saved["t"] <= 1.0:
            raise ValueError("invalid accepted-stage schedule")
        identity = saved.get("objective") == "identity"
        names = _IDENTITY_METADATA_NAMES if identity else _METADATA_NAMES
        stage_fields = set(names) | {
            "stage",
            "validation_samples_path",
            "history_path",
        }
        if not identity:
            stage_fields.update({
                "selected_flow_path",
                "continuation_flow_path",
            })
        missing = stage_fields - set(saved)
        if missing:
            raise ValueError(f"incomplete stage metadata: {sorted(missing)}")
        if identity:
            if (
                has_flow
                or saved["selected"] != "identity"
                or not np.allclose(saved["population_rg"], saved["rg_end"])
            ):
                raise ValueError("invalid identity stage metadata")
        elif (
            not has_flow
            or saved["flow_endpoint"] != "pre_sharpen"
            or not np.allclose(saved["flow_rg"], saved["rg_start"])
            or not np.allclose(saved["population_rg"], saved["rg_end"])
            or saved["selected"] not in ("trained", "identity")
            or saved["objective"] not in ("forward_klx", "forward_klxx")
        ):
            raise ValueError("invalid sharpening stage metadata")
        _check_regularization(record["config"], saved)
        history = _path(root, saved["history_path"])
        if not history.is_file():
            raise FileNotFoundError(history)
        with np.load(history, allow_pickle=False) as data:
            history_names = (
                _IDENTITY_HISTORY_NAMES if identity else _HISTORY_NAMES
            )
            missing = set(history_names) - set(data.files)
            has_hat = "hat_mala_acceptance" in data.files
        if missing:
            raise ValueError(f"incomplete stage history: {sorted(missing)}")
        if has_hat != (saved["objective"] == "forward_klxx"):
            raise ValueError("history does not match the stage objective")
        previous = saved["t"]
        required.extend([saved["validation_samples_path"], saved["history_path"]])
        if not identity:
            required.extend([
                saved["selected_flow_path"],
                saved["continuation_flow_path"],
            ])
    for relative in required:
        path = _path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
    if record["status"] == "complete" and previous != 1.0:
        raise ValueError("complete run does not end at t=1")
    return record


def load(run_dir, template=None):
    root = Path(run_dir).expanduser().resolve()
    run_record = validate(root)
    continuation = None
    has_flow = "flow_template" in run_record
    if has_flow:
        if _flow_template(template) != run_record["flow_template"]:
            raise ValueError("flow template does not match the saved run")
        continuation = eqx.tree_deserialise_leaves(
            _path(root, run_record["initial_flow_path"]), template
        )
    samples = np.load(
        _path(root, run_record["initial_samples_path"]), allow_pickle=False
    )
    records = []
    for item in run_record["stages"]:
        saved = _stage(root, item)
        with np.load(
            _path(root, saved["history_path"]), allow_pickle=False
        ) as data:
            record = {
                name: data[name].copy()
                for name in data.files
            }
        record.update({
            "t": float(saved["t"]),
            "t_start": float(saved["t_start"]),
            "rg_start": tuple(saved["rg_start"]),
            "rg_end": tuple(saved["rg_end"]),
            "population_rg": tuple(saved["population_rg"]),
            "valid_selected_ess": float(saved["valid_selected_ess"]),
            "valid_identity_ess": float(saved["valid_identity_ess"]),
            "valid_sample_count": int(saved["valid_sample_count"]),
            "selected": saved["selected"],
            "attempt_status_hist": tuple(saved["attempt_status_hist"]),
            "selection_history": tuple(saved["selection_history"]),
            "sharpen_ess": float(saved["sharpen_ess"]),
            "objective": saved["objective"],
            "elapsed_seconds": float(saved["elapsed_seconds"]),
            "validation_samples_path": saved["validation_samples_path"],
        })
        if saved["objective"] != "identity":
            record.update({
                "flow_rg": tuple(saved["flow_rg"]),
                "flow_endpoint": saved["flow_endpoint"],
                "valid_trained_ess": float(saved["valid_trained_ess"]),
                "initialized_from_identity": bool(
                    saved["initialized_from_identity"]
                ),
                "selected_flow_path": saved["selected_flow_path"],
                "continuation_flow_path": saved["continuation_flow_path"],
            })
            record["hat_mala_acceptance"] = record.get(
                "hat_mala_acceptance"
            )
            record["flow"] = eqx.tree_deserialise_leaves(
                _path(root, saved["selected_flow_path"]), template
            )
            continuation = eqx.tree_deserialise_leaves(
                _path(root, saved["continuation_flow_path"]), template
            )
            record["continuation_flow"] = continuation
        records.append(record)
        samples = np.load(
            _path(root, saved["validation_samples_path"]), allow_pickle=False
        )
    return samples, continuation, records


def fork(run_dir, destination, problem_id=None) -> dict:
    source = Path(run_dir).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("fork destination must lie outside the source run")
    record = validate(source)
    shutil.copytree(source, destination)
    record["problem_id"] = problem_id or record["problem_id"]
    record["status"] = "running"
    from .write import jsonfile

    jsonfile(destination / "run.json", record)
    return record


inspect_run = manifest
validate_run = validate
fork_run = fork


def _stage_item(record, stage):
    if not isinstance(stage, int) or not 1 <= stage <= len(record["stages"]):
        raise IndexError(f"invalid one-based stage: {stage}")
    return record["stages"][stage - 1]


def load_stage_flow(run_dir, stage: int, role: str, template):
    root = Path(run_dir).expanduser().resolve()
    record = validate(root)
    if "flow_template" not in record:
        raise ValueError("run has no stored flow")
    if _flow_template(template) != record["flow_template"]:
        raise ValueError("flow template does not match the saved run")
    if role not in ("selected", "continuation"):
        raise ValueError(f"unknown flow role: {role}")
    saved = _stage(root, _stage_item(record, stage))
    return eqx.tree_deserialise_leaves(
        _path(root, saved[f"{role}_flow_path"]), template
    )


def load_validation_samples(run_dir, stage=None, *, mmap_mode=None):
    root = Path(run_dir).expanduser().resolve()
    record = validate(root)
    if stage is None:
        relative = record["initial_samples_path"]
    else:
        relative = _stage(
            root, _stage_item(record, stage)
        )["validation_samples_path"]
    return np.load(
        _path(root, relative), mmap_mode=mmap_mode, allow_pickle=False
    )


def load_training_history(run_dir, stage: int) -> dict:
    root = Path(run_dir).expanduser().resolve()
    record = validate(root)
    saved = _stage(root, _stage_item(record, stage))
    with np.load(
        _path(root, saved["history_path"]), allow_pickle=False
    ) as data:
        result = {name: data[name].copy() for name in data.files}
    result["attempt_status_hist"] = tuple(saved["attempt_status_hist"])
    result["selection_history"] = tuple(saved["selection_history"])
    return result


def run(
    run_dir,
    problem_id,
    config,
    samples,
    flow,
    iterate,
    *,
    resume=False,
):
    if resume:
        saved = manifest(run_dir)
        if saved["problem_id"] != problem_id:
            raise ValueError("problem_id does not match the saved run")
        if saved["config"] != _value(config):
            raise ValueError("config does not match the saved run")
        samples, flow, records = load(run_dir, flow)
    else:
        records = []
        saved = create(run_dir, problem_id, config, samples, flow)
    accepted = (0.0, *(record["t"] for record in records))
    for samples, record, flow in iterate(
        samples, flow, accepted, len(records) + 1
    ):
        records.append(stage(run_dir, saved, record, samples))
        accepted = (*accepted, record["t"])
    finish(run_dir, saved, "complete" if accepted[-1] == 1.0 else "exhausted")
    return samples, records
