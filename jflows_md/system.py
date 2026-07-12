"""Immutable molecular-bundle loading and integrity verification.

This module intentionally has no JAX or OpenMM dependency. A bundle contains a
complete Cartesian SystemSpec, CoordinateSpec, validation frames, and the
provenance files needed to reconstruct the target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parent.parent / "bundles"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


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
    """Host-side representation of a verified molecular target bundle."""

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
        if verify:
            cls._verify_files(path, manifest)
        return cls(
            path=path,
            manifest=manifest,
            system=_read_json(path / manifest["system_spec"]),
            coordinates=_read_json(path / manifest["coordinate_spec"]),
            validation=_read_json(path / manifest["validation_spec"]),
        )

    @staticmethod
    def _verify_files(path: Path, manifest: dict[str, Any]) -> None:
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("bundle manifest has no file hashes")
        for relative, expected in files.items():
            target = path / relative
            if not target.is_file():
                raise FileNotFoundError(f"bundle file is missing: {target}")
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(
                    f"bundle hash mismatch for {target}: {actual} != {expected}"
                )

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
