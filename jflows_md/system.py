"""Minimal molecular-bundle loading and structural verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parent.parent / "bundles"

_BUNDLE_FILES = {
    "coordinates.json",
    "manifest.json",
    "reference.pdb",
    "system.json",
    "system.xml",
    "validation.json",
}

_MANIFEST_FIELDS = {
    "canonical_smiles",
    "coordinate_measure",
    "coordinate_spec",
    "formal_charge",
    "formula",
    "model",
    "name",
    "openmm_version",
    "schema_version",
    "system_spec",
    "target",
    "temperature_kelvin",
    "validation_spec",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _safe_relative_file(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError(f"bundle field {field!r} must name a relative file")
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or relative != path.as_posix()
    ):
        raise ValueError(f"unsafe bundle file path: {relative!r}")
    target = root / path
    if not target.is_file():
        raise FileNotFoundError(f"bundle file is missing: {target}")
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError(f"bundle file escapes its root: {relative!r}")
    return target


def resolve_bundle(path_or_name: str | Path) -> Path:
    candidate = Path(path_or_name).expanduser()
    if not candidate.exists():
        candidate = BUNDLE_ROOT / candidate
    if candidate.is_file():
        if candidate.name != "manifest.json":
            raise ValueError(f"bundle file must be manifest.json, got {candidate}")
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"molecular bundle not found: {path_or_name}")
    if not (candidate / "manifest.json").is_file():
        raise FileNotFoundError(f"missing manifest.json in {candidate}")
    return candidate.resolve()


@dataclass(frozen=True)
class Molecular_Bundle:
    """Host-side representation of one minimal molecular runtime bundle."""

    path: Path
    manifest: dict[str, Any]
    system: dict[str, Any]
    coordinates: dict[str, Any]
    validation: dict[str, Any]

    @classmethod
    def load(cls, path_or_name: str | Path, *, verify: bool = True) -> "Molecular_Bundle":
        path = resolve_bundle(path_or_name)
        manifest = _read_json(path / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"unsupported molecular bundle schema: {manifest.get('schema_version')}"
            )
        fields = set(manifest)
        if fields != _MANIFEST_FIELDS:
            missing = sorted(_MANIFEST_FIELDS - fields)
            extra = sorted(fields - _MANIFEST_FIELDS)
            raise ValueError(
                f"invalid molecular bundle manifest fields: missing={missing}, extra={extra}"
            )
        if not isinstance(manifest["model"], dict):
            raise ValueError("bundle model must be a JSON object")
        if verify:
            cls._verify_structure(path, manifest)
        system = _read_json(
            _safe_relative_file(path, manifest.get("system_spec"), "system_spec")
        )
        coordinates = _read_json(
            _safe_relative_file(
                path, manifest.get("coordinate_spec"), "coordinate_spec"
            )
        )
        validation = _read_json(
            _safe_relative_file(
                path, manifest.get("validation_spec"), "validation_spec"
            )
        )
        if coordinates.get("schema_version") != 2:
            raise ValueError("CoordinateSpec schema must be 2")
        coordinate_measure = coordinates.get("jacobian_measure")
        manifest_measure = manifest["coordinate_measure"]
        if (
            coordinate_measure != "rigid_motion_quotient_v1"
            or manifest_measure != coordinate_measure
        ):
            raise ValueError(
                "bundle manifest and CoordinateSpec disagree on coordinate measure: "
                f"{manifest_measure!r} != {coordinate_measure!r}"
            )
        return cls(
            path=path,
            manifest=manifest,
            system=system,
            coordinates=coordinates,
            validation=validation,
        )

    @staticmethod
    def _verify_structure(path: Path, manifest: dict[str, Any]) -> None:
        children = {candidate.name for candidate in path.iterdir()}
        if children != _BUNDLE_FILES:
            missing = sorted(_BUNDLE_FILES - children)
            extra = sorted(children - _BUNDLE_FILES)
            raise ValueError(
                f"invalid molecular bundle contents: missing={missing}, extra={extra}"
            )
        for field in ("system_spec", "coordinate_spec", "validation_spec"):
            _safe_relative_file(path, manifest.get(field), field)
        for name in ("reference.pdb", "system.xml"):
            _safe_relative_file(path, name, name)
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"bundle contains a symlink: {candidate}")
            if not candidate.is_file() and not candidate.is_dir():
                raise ValueError(f"bundle contains a special filesystem node: {candidate}")

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def dimension(self) -> int:
        return int(self.coordinates["dimension"])

    @property
    def n_atoms(self) -> int:
        return int(self.system["n_atoms"])


def available_bundles(root: str | Path | None = None) -> tuple[str, ...]:
    base = BUNDLE_ROOT if root is None else Path(root)
    if not base.is_dir():
        return ()
    return tuple(sorted(p.name for p in base.iterdir() if (p / "manifest.json").is_file()))
