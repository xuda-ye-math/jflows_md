#!/usr/bin/env python
"""Host-only integrity and metadata smoke test for all molecular bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import jflows_md  # noqa: E402
from jflows_md import Molecular_Bundle, available_bundles  # noqa: E402
from jflows_md.core.builder import normalize_amber_transcript  # noqa: E402
from jflows_md.system import sha256_file  # noqa: E402


EXPECTED = {
    "adp_ff96_obc1": ("C6H12N2O2", 22, 60, 42, 18),
    "glycerol_gaff2_am1bcc_obc1": ("C3H8O3", 14, 36, 25, 11),
    "diethanolamine_gaff2_am1bcc_obc1": ("C4H11NO2", 18, 48, 33, 15),
}

EXPECTED_LINEAGE = {
    "adp_ff96_obc1": {
        "manifest": "c21fcdc3a74ae68ff50e1e23ac30080963debcacbec2183e8fc32cc9ffcd16e4",
        "source_bundle": "fab_adp_ff96_obc1_v1",
        "source_manifest": "26f71aa4fb696e3f9641db1e32253ca3093063fb2efa48d46fefe0eb3e636744",
    },
    "glycerol_gaff2_am1bcc_obc1": {
        "manifest": "d6469e1cf0c43bf0fe1b0d0c63d0e2b57167ea4e0a16b90157958f85bf8e6e3d",
        "source_bundle": "glycerol_gaff2_am1bcc_obc1_v1",
        "source_manifest": "3a7fb6c95dba2fc20abf71ddde58c5dd5b10bc562dc0ea32f8969470f70afe95",
    },
    "diethanolamine_gaff2_am1bcc_obc1": {
        "manifest": "178a508261d2d37269ea892f78a1bd52b07d2de36b0bf1606911989b9115a029",
        "source_bundle": "diethanolamine_neutral_gaff2_am1bcc_obc1_v1",
        "source_manifest": "617049504f0f8e4834a986a69c294deb2ee558d197976f22786b19d13a6191e8",
    },
}


def expect_rejected(source: Path, label: str, mutate) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        shutil.copytree(source, bundle)
        mutate(bundle, root)
        try:
            Molecular_Bundle.load(bundle, verify=True)
        except (FileNotFoundError, ValueError):
            print(f"PASS bundle verifier rejects {label}")
        else:
            raise AssertionError(f"bundle verifier accepted {label}")


def check_verifier_closure(source: Path) -> None:
    def edit_manifest(bundle: Path, edit) -> None:
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        edit(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    expect_rejected(
        source,
        "an unbound runtime spec",
        lambda bundle, root: edit_manifest(
            bundle,
            lambda manifest: manifest["files"].pop(manifest["coordinate_spec"]),
        ),
    )

    def escape(bundle: Path, root: Path) -> None:
        outside = root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        edit_manifest(
            bundle,
            lambda manifest: (
                manifest.__setitem__("system_spec", "../outside.json"),
                manifest["files"].__setitem__("../outside.json", sha256_file(outside)),
            ),
        )

    expect_rejected(source, "a parent-path escape", escape)

    def symlink_escape(bundle: Path, root: Path) -> None:
        outside = root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (bundle / "escape.json").symlink_to(outside)
        edit_manifest(
            bundle,
            lambda manifest: (
                manifest.__setitem__("system_spec", "escape.json"),
                manifest["files"].__setitem__("escape.json", sha256_file(outside)),
            ),
        )

    expect_rejected(source, "a symlink escape", symlink_escape)
    expect_rejected(
        source,
        "an unhashed extra file",
        lambda bundle, root: (bundle / "extra.txt").write_text("extra", encoding="utf-8"),
    )

    def nested_manifest(bundle: Path, root: Path) -> None:
        nested = bundle / "nested"
        nested.mkdir()
        (nested / "manifest.json").write_text("{}", encoding="utf-8")

    expect_rejected(source, "an unhashed nested manifest", nested_manifest)

    def directory_symlink(bundle: Path, root: Path) -> None:
        outside = root / "outside"
        outside.mkdir()
        (outside / "payload.txt").write_text("outside", encoding="utf-8")
        (bundle / "linked-directory").symlink_to(outside, target_is_directory=True)

    expect_rejected(source, "an untracked directory symlink", directory_symlink)

    def special_node(bundle: Path, root: Path) -> None:
        os.mkfifo(bundle / "untracked.pipe")

    expect_rejected(source, "an untracked special filesystem node", special_node)
    expect_rejected(
        source,
        "a missing listed file",
        lambda bundle, root: (bundle / "reference.pdb").unlink(),
    )
    expect_rejected(
        source,
        "a corrupted listed file",
        lambda bundle, root: (bundle / "coordinates.json").write_text("{}", encoding="utf-8"),
    )
    expect_rejected(
        source,
        "a malformed digest",
        lambda bundle, root: edit_manifest(
            bundle,
            lambda manifest: manifest["files"].__setitem__("coordinates.json", "not-a-hash"),
        ),
    )
    expect_rejected(
        source,
        "contradictory coordinate-measure metadata",
        lambda bundle, root: edit_manifest(
            bundle,
            lambda manifest: manifest.__setitem__(
                "coordinate_measure", "canonical_gauge_slice_v1"
            ),
        ),
    )


def main() -> None:
    assert normalize_amber_transcript(
        "Running: /opt/amber/bin/antechamber\n\n", Path("/opt/amber")
    ) == "Running: <AMBERHOME>/bin/antechamber\n"
    expected_public = {
        "Mixed_Identity",
        "Mixed_NSF",
        "Molecular_Bundle",
        "Molecular_Potential",
        "Molecular_Source",
        "annealed_importance_sampling",
        "available_bundles",
        "load_mixed_flow_stages",
        "mixed_flow_metadata",
        "mixed_mala",
        "molecular_boltzmann_forward_KLX_G",
        "potential_space_smc",
        "package_source_sha256",
        "sequential_monte_carlo",
        "train_molecular_forward_KLX_G",
    }
    assert set(jflows_md.__all__) == expected_public
    for private_name in ("Amber_OBC_Force_Field", "Internal_Coordinates", "Mixed_Domain"):
        assert not hasattr(jflows_md, private_name), private_name

    names = set(available_bundles())
    assert names == set(EXPECTED), (names, set(EXPECTED))
    legacy_aliases = {
        "fab_adp_ff96_obc1_v2": "adp_ff96_obc1",
        "glycerol_gaff2_am1bcc_obc1_v2": "glycerol_gaff2_am1bcc_obc1",
        "diethanolamine_neutral_gaff2_am1bcc_obc1_v2": (
            "diethanolamine_gaff2_am1bcc_obc1"
        ),
    }
    for legacy, current in legacy_aliases.items():
        assert Molecular_Bundle.load(legacy).name == current
    for name, (formula, atoms, dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name, verify=True)
        assert bundle.system["formula"] == formula
        assert bundle.n_atoms == atoms
        assert bundle.dimension == dimension
        assert bundle.coordinates["euclidean_dim"] == euclidean
        assert bundle.coordinates["periodic_dim"] == periodic
        assert bundle.manifest["model"]["implicit_solvent"] == "OBC1 (igb=2)"
        assert bundle.manifest["coordinate_measure"] == "rigid_motion_quotient_v1"
        assert bundle.coordinates["jacobian_measure"] == "rigid_motion_quotient_v1"
        upgrade = json.loads(
            (bundle.path / "provenance/coordinate_measure_upgrade.json").read_text()
        )
        assert upgrade["source_bundle"] == EXPECTED_LINEAGE[name]["source_bundle"]
        assert (
            upgrade["source_manifest_sha256"]
            == EXPECTED_LINEAGE[name]["source_manifest"]
        )
        assert (
            sha256_file(bundle.path / "manifest.json")
            == EXPECTED_LINEAGE[name]["manifest"]
        )
        assert min(bundle.system["gb_offset_radius_nm"]) > 0
        assert min(bundle.system["gb_scaled_offset_radius_nm"]) > 0
        assert len(bundle.validation["frames_nm"]) == 4
        assert len(bundle.validation["forces_kj_mol_nm"]) == 4
        if name == "adp_ff96_obc1":
            assert (
                sha256_file(bundle.path / "system.prmtop")
                == "2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e"
            )
            assert bundle.coordinates["chiral_torsion_sign"] == -1
            assert bundle.coordinates["chirality_sign"] == 1
        else:
            assert bundle.coordinates["chiral_torsion_index"] == -1
            for relative in (
                "provenance/antechamber.stdout",
                "provenance/leap.transcript",
                "provenance/tleap.stdout",
            ):
                text = (bundle.path / relative).read_text()
                assert "<AMBERHOME>" in text
                assert "/home/" not in text and "conda" not in text.lower()
                assert text.endswith("\n") and not text.endswith("\n\n")
            leap_text = (bundle.path / "provenance/tleap.stdout").read_text()
            assert "Exiting LEaP: Errors = 0; Warnings = 0; Notes = 0." in leap_text
        print(f"PASS bundle {name}")

    check_verifier_closure(
        Molecular_Bundle.load("glycerol_gaff2_am1bcc_obc1").path
    )

    # A self-consistent rewrite of a built-in seed is still a different target
    # and must require a new versioned name.
    import jflows_md.system as bundle_system

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "bundles"
        root.mkdir()
        name = "glycerol_gaff2_am1bcc_obc1"
        shutil.copytree(Molecular_Bundle.load(name).path, root / name)
        seed = root / name / "provenance/input.mol2"
        seed.write_text(seed.read_text() + "\n", encoding="utf-8")
        manifest_path = root / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["provenance/input.mol2"] = sha256_file(seed)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        original_root = bundle_system.BUNDLE_ROOT
        bundle_system.BUNDLE_ROOT = root
        try:
            try:
                Molecular_Bundle.load(name, verify=True)
            except ValueError as error:
                assert "changed without a new bundle version" in str(error)
            else:
                raise AssertionError("rewritten built-in target was accepted")
        finally:
            bundle_system.BUNDLE_ROOT = original_root
    print("PASS built-in bundle names are manifest-frozen")


if __name__ == "__main__":
    main()
