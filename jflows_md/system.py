"""Molecular bundle loading."""

from dataclasses import dataclass
import json
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent.parent / "bundles"
_FILES = {
    "coordinates.json",
    "manifest.json",
    "reference.pdb",
    "system.json",
    "system.xml",
    "validation.json",
}


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def resolve_bundle(path_or_name: str | Path) -> Path:
    path = Path(path_or_name).expanduser()
    if not path.exists():
        path = BUNDLE_ROOT / path
    if path.is_file():
        path = path.parent
    return path.resolve()


@dataclass(frozen=True)
class Molecular_Bundle:
    path: Path
    manifest: dict
    system: dict
    coordinates: dict
    validation: dict

    @classmethod
    def load(
        cls, path_or_name: str | Path, *, verify: bool = True
    ) -> "Molecular_Bundle":
        path = resolve_bundle(path_or_name)
        manifest = _json(path / "manifest.json")
        if verify and {item.name for item in path.iterdir()} != _FILES:
            raise ValueError(f"invalid molecular bundle contents: {path}")
        return cls(
            path,
            manifest,
            _json(path / "system.json"),
            _json(path / "coordinates.json"),
            _json(path / "validation.json"),
        )

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def dimension(self) -> int:
        return self.coordinates["dimension"]

    @property
    def n_atoms(self) -> int:
        return self.system["n_atoms"]


def available_bundles(root: str | Path | None = None) -> tuple[str, ...]:
    base = BUNDLE_ROOT if root is None else Path(root)
    if not base.is_dir():
        return ()
    return tuple(
        sorted(path.name for path in base.iterdir() if (path / "manifest.json").is_file())
    )
