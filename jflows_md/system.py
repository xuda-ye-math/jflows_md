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

# These digests make the three built-in names immutable scientific targets,
# rather than merely self-consistent directories whose manifests could be
# rewritten after changing a seed or Hamiltonian. External bundle paths remain
# supported and are validated by their own complete hash closure.
_FROZEN_BUNDLE_MANIFEST_SHA256 = {
    "adp_ff96_obc1": (
        "c21fcdc3a74ae68ff50e1e23ac30080963debcacbec2183e8fc32cc9ffcd16e4"
    ),
    "glycerol_gaff2_am1bcc_obc1": (
        "d6469e1cf0c43bf0fe1b0d0c63d0e2b57167ea4e0a16b90157958f85bf8e6e3d"
    ),
    "diethanolamine_gaff2_am1bcc_obc1": (
        "178a508261d2d37269ea892f78a1bd52b07d2de36b0bf1606911989b9115a029"
    ),
}

# Public names were shortened after the coordinate-schema-2 targets were
# frozen. Keep old artifact metadata readable without retaining duplicate
# bundle directories or advertising the retired names.
_LEGACY_BUNDLE_ALIASES = {
    "fab_adp_ff96_obc1_v2": "adp_ff96_obc1",
    "glycerol_gaff2_am1bcc_obc1_v2": "glycerol_gaff2_am1bcc_obc1",
    "diethanolamine_neutral_gaff2_am1bcc_obc1_v2": (
        "diethanolamine_gaff2_am1bcc_obc1"
    ),
}
_LEGACY_BUNDLE_MANIFEST_SHA256 = {
    "fab_adp_ff96_obc1_v2": (
        "8555616ebf85b83c47635eae2a3a8ecb4d03e8622e80257dd1d4d5ecad2155f1"
    ),
    "glycerol_gaff2_am1bcc_obc1_v2": (
        "c60f520ef8d4146a0ffb4ff8875d56b90abebd47630d842780e7b3d71911c736"
    ),
    "diethanolamine_neutral_gaff2_am1bcc_obc1_v2": (
        "2af005fdb95bf060f5973fddf6825eb7c8d7fe3cb2d05a2203ef7deb5cdbc367"
    ),
}


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
        canonical = _LEGACY_BUNDLE_ALIASES.get(str(path_or_name), candidate)
        candidate = BUNDLE_ROOT / canonical
    if candidate.is_file():
        if candidate.name != "manifest.json":
            raise ValueError(f"bundle file must be manifest.json, got {candidate}")
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"molecular bundle not found: {path_or_name}")
    if not (candidate / "manifest.json").is_file():
        raise FileNotFoundError(f"missing manifest.json in {candidate}")
    return candidate.resolve()


def _manifest_matches_bundle_name(
    saved_name: str, saved_sha256: str, current_name: str, current_sha256: str
) -> bool:
    """Accept an exact current manifest or its reviewed public-name migration."""
    if saved_sha256 == current_sha256:
        return True
    canonical = _LEGACY_BUNDLE_ALIASES.get(saved_name)
    return (
        canonical == current_name
        and _LEGACY_BUNDLE_MANIFEST_SHA256.get(saved_name) == saved_sha256
        and _FROZEN_BUNDLE_MANIFEST_SHA256.get(current_name) == current_sha256
    )


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
        if verify and path.parent == BUNDLE_ROOT.resolve():
            expected_manifest = _FROZEN_BUNDLE_MANIFEST_SHA256.get(path.name)
            if expected_manifest is not None:
                actual_manifest = sha256_file(path / "manifest.json")
                if actual_manifest != expected_manifest:
                    raise ValueError(
                        "built-in molecular target manifest changed without a new "
                        f"bundle version: {actual_manifest} != {expected_manifest}"
                    )
        manifest = _read_json(path / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"unsupported molecular bundle schema: {manifest.get('schema_version')}"
            )
        if verify:
            cls._verify_files(path, manifest)
        system = _read_json(path / manifest["system_spec"])
        coordinates = _read_json(path / manifest["coordinate_spec"])
        validation = _read_json(path / manifest["validation_spec"])
        coordinate_schema = int(coordinates.get("schema_version", 1))
        coordinate_measure = str(
            coordinates.get("jacobian_measure", "canonical_gauge_slice_v1")
        )
        manifest_measure = manifest.get("coordinate_measure")
        if coordinate_schema >= 2 and manifest_measure != coordinate_measure:
            raise ValueError(
                "bundle manifest and CoordinateSpec disagree on coordinate measure: "
                f"{manifest_measure!r} != {coordinate_measure!r}"
            )
        if manifest_measure is not None and str(manifest_measure) != coordinate_measure:
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
    def _verify_files(path: Path, manifest: dict[str, Any]) -> None:
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("bundle manifest has no file hashes")

        root = path.resolve()
        for field in ("system_spec", "coordinate_spec", "validation_spec"):
            relative = manifest.get(field)
            if not isinstance(relative, str) or relative not in files:
                raise ValueError(
                    f"bundle manifest field {field!r} is not hash-bound: {relative!r}"
                )

        listed: set[str] = set()
        for relative, expected in files.items():
            if not isinstance(relative, str):
                raise ValueError("bundle file paths must be strings")
            relative_path = Path(relative)
            normalized = relative_path.as_posix()
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in relative
                or relative != normalized
            ):
                raise ValueError(f"unsafe bundle file path: {relative!r}")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"invalid SHA-256 digest for bundle file {relative!r}")
            try:
                int(expected, 16)
            except ValueError as error:
                raise ValueError(
                    f"invalid SHA-256 digest for bundle file {relative!r}"
                ) from error

            target = root / relative_path
            if not target.is_file():
                raise FileNotFoundError(f"bundle file is missing: {target}")
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError(f"bundle file escapes its bundle root: {relative!r}")
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(
                    f"bundle hash mismatch for {target}: {actual} != {expected}"
                )
            listed.add(normalized)

        present = set()
        root_manifest = root / "manifest.json"
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"bundle contains a symlink: {candidate}")
            if candidate.is_file():
                if candidate != root_manifest:
                    present.add(candidate.relative_to(root).as_posix())
            elif not candidate.is_dir():
                raise ValueError(f"bundle contains a special filesystem node: {candidate}")
        unhashed = present - listed
        if unhashed:
            raise ValueError(f"bundle contains unhashed files: {sorted(unhashed)}")

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
