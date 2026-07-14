#!/usr/bin/env python
"""Host-only checks for the minimal molecular bundle contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import jflows_md
from jflows_md import Molecular_Bundle, available_bundles


EXPECTED = {
    "adp_ff96_obc1": ("C6H12N2O2", 22, 60, 42, 18),
    "glycerol_gaff2_am1bcc_obc1": ("C3H8O3", 14, 36, 25, 11),
    "diethanolamine_gaff2_am1bcc_obc1": ("C4H11NO2", 18, 48, 33, 15),
}

EXPECTED_FILES = {
    "coordinates.json",
    "manifest.json",
    "reference.pdb",
    "system.json",
    "system.xml",
    "validation.json",
}

EXPECTED_PUBLIC = {
    "Mixed_Identity",
    "Mixed_NSF",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "annealed_importance_sampling",
    "available_bundles",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLXX_G",
    "load_mixed_flow",
    "mixed_flow_metadata",
    "mixed_mala",
    "mixed_quench_and_temper",
    "potential_space_smc",
    "save_mixed_flow",
    "sequential_monte_carlo",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
}


def _edit_json(path: Path, edit) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _expect_rejected(source: Path, label: str, mutate) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        shutil.copytree(source, bundle)
        mutate(bundle, root)
        try:
            Molecular_Bundle.load(bundle)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            print(f"PASS bundle loader rejects {label}")
            return
        raise AssertionError(f"bundle loader accepted {label}")


def _check_rejections(source: Path) -> None:
    _expect_rejected(
        source,
        "a missing runtime file",
        lambda bundle, root: (bundle / "system.xml").unlink(),
    )
    _expect_rejected(
        source,
        "an extra runtime file",
        lambda bundle, root: (bundle / "extra.txt").write_text("extra"),
    )
    _expect_rejected(
        source,
        "an extra directory",
        lambda bundle, root: (bundle / "extra").mkdir(),
    )

    def parent_escape(bundle: Path, root: Path) -> None:
        (root / "outside.json").write_text("{}", encoding="utf-8")
        _edit_json(
            bundle / "manifest.json",
            lambda value: value.__setitem__("system_spec", "../outside.json"),
        )

    _expect_rejected(source, "a parent-path reference", parent_escape)

    def symlink_escape(bundle: Path, root: Path) -> None:
        outside = root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (bundle / "system.json").unlink()
        (bundle / "system.json").symlink_to(outside)

    _expect_rejected(source, "a symbolic-link reference", symlink_escape)

    def special_node(bundle: Path, root: Path) -> None:
        os.mkfifo(bundle / "pipe")

    _expect_rejected(source, "a special filesystem node", special_node)
    _expect_rejected(
        source,
        "an unknown manifest field",
        lambda bundle, root: _edit_json(
            bundle / "manifest.json",
            lambda value: value.__setitem__("unknown", True),
        ),
    )
    _expect_rejected(
        source,
        "a missing manifest field",
        lambda bundle, root: _edit_json(
            bundle / "manifest.json",
            lambda value: value.pop("formula"),
        ),
    )
    _expect_rejected(
        source,
        "a non-current coordinate schema",
        lambda bundle, root: _edit_json(
            bundle / "coordinates.json",
            lambda value: value.__setitem__("schema_version", 1),
        ),
    )
    _expect_rejected(
        source,
        "an inconsistent coordinate measure",
        lambda bundle, root: _edit_json(
            bundle / "manifest.json",
            lambda value: value.__setitem__("coordinate_measure", "other"),
        ),
    )


def main() -> None:
    assert set(jflows_md.__all__) == EXPECTED_PUBLIC
    for private_name in (
        "Amber_OBC_Force_Field",
        "Internal_Coordinates",
        "Mixed_Domain",
    ):
        assert not hasattr(jflows_md, private_name), private_name

    assert set(available_bundles()) == set(EXPECTED)
    for name, (formula, atoms, dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name)
        assert bundle.name == name
        assert bundle.manifest["formula"] == formula
        assert bundle.n_atoms == atoms
        assert bundle.dimension == dimension
        assert bundle.coordinates["euclidean_dim"] == euclidean
        assert bundle.coordinates["periodic_dim"] == periodic
        assert bundle.coordinates["schema_version"] == 2
        assert bundle.coordinates["jacobian_measure"] == "rigid_motion_quotient_v1"
        assert bundle.manifest["coordinate_measure"] == "rigid_motion_quotient_v1"
        assert {path.name for path in bundle.path.iterdir()} == EXPECTED_FILES
        assert all(path.is_file() and not path.is_symlink() for path in bundle.path.iterdir())
        print(f"PASS minimal bundle {name}: {dimension}D")

    _check_rejections(Molecular_Bundle.load("adp_ff96_obc1").path)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert available_bundles(root) == ()
        shutil.copytree(
            Molecular_Bundle.load("adp_ff96_obc1").path,
            root / "one",
        )
        assert available_bundles(root) == ("one",)

    print("PASS public API and current six-file bundle format")


if __name__ == "__main__":
    main()
