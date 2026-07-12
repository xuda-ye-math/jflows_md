#!/usr/bin/env python
"""Rebuild the three frozen molecular bundles in the OpenMM/AmberTools env.

Each bundle contains its own immutable seed artifact: exact PRMTOP/RST7 files
for FAB ADP and the explicit-H input MOL2 for each GAFF2 small molecule. The
builder stages those inputs in a temporary candidate, then accepts only an
exact match to the frozen manifest. A changed seed, toolchain, or output must
use a new bundle version, and no external assets directory is required.

Run from the repository root with

    python bundles/build_molecular_bundles.py

Install both editable packages as described in the root README first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


MOLECULAR_ROOT = Path(__file__).resolve().parents[1]

from jflows_md.core.builder import write_bundle  # noqa: E402
from jflows_md.system import (  # noqa: E402
    Molecular_Bundle,
    _FROZEN_BUNDLE_MANIFEST_SHA256,
    sha256_file,
)


def bundle_seed(current_name: str) -> Path:
    """Return a verified seed carried by an exactly frozen built-in bundle."""
    current = MOLECULAR_ROOT / "bundles" / current_name
    if not current.is_dir():
        raise FileNotFoundError(f"bundle seed is missing: {current}")
    expected = _FROZEN_BUNDLE_MANIFEST_SHA256.get(current_name)
    if expected is None:
        raise ValueError(f"bundle has no frozen-manifest gate: {current_name}")
    actual = sha256_file(current / "manifest.json")
    if actual != expected:
        raise ValueError(
            "bundle seed changed without a new bundle version: "
            f"{actual} != {expected}"
        )
    return Molecular_Bundle.load(current, verify=True).path


def verify_frozen_rebuild(candidate: Path, bundle_name: str) -> Path:
    """Accept an exact candidate; any scientific change requires a new name."""
    Molecular_Bundle.load(candidate, verify=True)
    expected = _FROZEN_BUNDLE_MANIFEST_SHA256[bundle_name]
    actual = sha256_file(candidate / "manifest.json")
    if actual != expected:
        raise ValueError(
            f"rebuild of {bundle_name} changed its frozen payload; create a new "
            f"bundle version ({actual} != {expected})"
        )
    # The checked-in bundle is already byte-identical. Leaving it in place
    # avoids a destructive replacement while still exercising the full build.
    return bundle_seed(bundle_name)


def run(command: list[str], cwd: Path, log_name: str) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    (cwd / log_name).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def amber_data_hashes(prefix: Path) -> dict[str, str]:
    paths = {
        "gaff2.dat": prefix / "dat/leap/parm/gaff2.dat",
        "BCCPARM.DAT": prefix / "dat/antechamber/BCCPARM.DAT",
        "ATOMTYPE_GFF2.DEF": prefix / "dat/antechamber/ATOMTYPE_GFF2.DEF",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def ambertools_version(prefix: Path) -> str:
    records = sorted((prefix / "conda-meta").glob("ambertools-*.json"))
    if not records:
        raise FileNotFoundError("cannot identify the installed AmberTools package")
    return str(json.loads(records[-1].read_text())["version"])


def parameterize_small_molecule(
    *,
    name: str,
    source_mol2: Path,
    bundle_name: str,
    canonical_smiles: str,
    formula: str,
    diagnostic: str,
) -> Path:
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    if not prefix or not (prefix / "bin/antechamber").is_file():
        raise RuntimeError("run this builder inside the AmberTools-enabled jflows environment")
    frozen_record = json.loads(
        (source_mol2.parent / "build_record.json").read_text(encoding="utf-8")
    )
    hashes = amber_data_hashes(prefix)
    version = ambertools_version(prefix)
    expected_toolchain = {
        "ambertools_version": frozen_record["ambertools_version"],
        "amber_data_sha256": frozen_record["amber_data_sha256"],
    }
    actual_toolchain = {
        "ambertools_version": version,
        "amber_data_sha256": hashes,
    }
    if actual_toolchain != expected_toolchain:
        raise ValueError(
            "AmberTools installation differs from the frozen bundle toolchain; "
            f"a changed build requires a new bundle version: "
            f"{actual_toolchain} != {expected_toolchain}"
        )
    with tempfile.TemporaryDirectory(prefix=f"jflows_md_{name}_") as temporary:
        work = Path(temporary)
        shutil.copy2(source_mol2, work / "input.mol2")
        coordinate_upgrade = source_mol2.parent / "coordinate_measure_upgrade.json"
        if coordinate_upgrade.is_file():
            shutil.copy2(coordinate_upgrade, work / coordinate_upgrade.name)
        run(
            [
                str(prefix / "bin/antechamber"),
                "-i",
                "input.mol2",
                "-fi",
                "mol2",
                "-o",
                "gaff2.mol2",
                "-fo",
                "mol2",
                "-at",
                "gaff2",
                "-c",
                "bcc",
                "-nc",
                "0",
                "-m",
                "1",
                "-s",
                "2",
                "-pf",
                "y",
            ],
            work,
            "antechamber.stdout",
        )
        run(
            [
                str(prefix / "bin/parmchk2"),
                "-i",
                "gaff2.mol2",
                "-f",
                "mol2",
                "-o",
                "gaff2.frcmod",
                "-s",
                "gaff2",
            ],
            work,
            "parmchk2.stdout",
        )
        leap_input = """source leaprc.gaff2
set default PBRadii mbondi2
loadamberparams gaff2.frcmod
MOL = loadmol2 gaff2.mol2
check MOL
saveamberparm MOL molecule.prmtop molecule.rst7
savepdb MOL molecule.pdb
quit
"""
        (work / "leap.in").write_text(leap_input, encoding="utf-8")
        run([str(prefix / "bin/tleap"), "-f", "leap.in"], work, "tleap.stdout")
        # LEaP writes the wall-clock time into the otherwise deterministic
        # prmtop. Normalize only that nonphysical header so bundle hashes are
        # stable across identical rebuilds.
        prmtop_path = work / "molecule.prmtop"
        prmtop_text = prmtop_path.read_text(encoding="utf-8")
        prmtop_text = re.sub(
            r"^(%VERSION\s+VERSION_STAMP\s*=\s*V0001\.000\s+DATE\s*=\s*).*$",
            r"\g<1>01/01/70  00:00:00",
            prmtop_text,
            count=1,
            flags=re.MULTILINE,
        )
        prmtop_path.write_text(prmtop_text, encoding="utf-8")
        logs = "\n".join(
            (work / filename).read_text(errors="replace")
            for filename in ("antechamber.stdout", "parmchk2.stdout", "tleap.stdout")
        )
        if "Exiting LEaP: Errors = 0" not in logs or "FATAL" in logs or "ATTN" in logs:
            raise RuntimeError(f"{name} parameterization did not pass its log gate")
        # Preserve scientific logs while normalizing wall-clock-only fields so
        # identical parameterizations produce identical bundle hashes.
        leap_log = work / "leap.log"
        leap_log.write_text(
            re.sub(
                r"^log started:.*$",
                "log started: <normalized>",
                leap_log.read_text(),
                count=1,
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        sqm_output = work / "sqm.out"
        sqm_output.write_text(
            re.sub(r"=\s+[0-9]+\.[0-9]+ seconds", "= <timing> seconds", sqm_output.read_text()),
            encoding="utf-8",
        )
        build_record = {
            "name": name,
            "canonical_smiles": canonical_smiles,
            "formal_charge": 0,
            "multiplicity": 1,
            "ambertools_version": version,
            "amber_data_sha256": hashes,
            "pipeline": [
                "antechamber -at gaff2 -c bcc -nc 0 -m 1",
                "parmchk2 -s gaff2",
                "tleap leaprc.gaff2 with PBRadii mbondi2",
            ],
            "diagnostic_support": diagnostic,
        }
        (work / "build_record.json").write_text(
            json.dumps(build_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = {
            "input.mol2": work / "input.mol2",
            "gaff2.mol2": work / "gaff2.mol2",
            "gaff2.frcmod": work / "gaff2.frcmod",
            "leap.in": work / "leap.in",
            # Avoid globally ignored *.log/*.out suffixes: these transcripts
            # are hash-bound bundle inputs, not disposable runtime logs.
            "leap.transcript": work / "leap.log",
            "antechamber.stdout": work / "antechamber.stdout",
            "parmchk2.stdout": work / "parmchk2.stdout",
            "tleap.stdout": work / "tleap.stdout",
            "sqm.in": work / "sqm.in",
            "sqm.transcript": work / "sqm.out",
            "build_record.json": work / "build_record.json",
        }
        if (work / "coordinate_measure_upgrade.json").is_file():
            provenance["coordinate_measure_upgrade.json"] = (
                work / "coordinate_measure_upgrade.json"
            )
        candidate = write_bundle(
            work / "candidate_bundle",
            name=bundle_name,
            target=name,
            prmtop_path=work / "molecule.prmtop",
            coordinate_path=work / "molecule.rst7",
            model={
                "force_field": "GAFF2",
                "charges": "AM1-BCC",
                "implicit_solvent": "OBC1 (igb=2)",
                "radii": "mbondi2",
                "sasa": "ACE",
                "solute_dielectric": 1.0,
                "solvent_dielectric": 78.5,
                "salt_molar": 0.0,
                "nonbonded_method": "NoCutoff",
                "constraints": None,
                "ambertools_version": version,
                "amber_data_sha256": hashes,
            },
            canonical_smiles=canonical_smiles,
            expected_formula=formula,
            expected_charge=0,
            minimize=True,
            provenance_files=provenance,
        )
        return verify_frozen_rebuild(candidate, bundle_name)


def build_adp() -> Path:
    name = "fab_adp_ff96_obc1_v2"
    seed = bundle_seed(name)
    # Build from staged bundle-contained seeds without modifying the source.
    with tempfile.TemporaryDirectory(prefix="jflows_md_adp_") as temporary:
        work = Path(temporary)
        prmtop = work / "system.prmtop"
        coordinate = work / "system.rst7"
        coordinate_upgrade = work / "coordinate_measure_upgrade.json"
        shutil.copy2(seed / "system.prmtop", prmtop)
        shutil.copy2(seed / "system.rst7", coordinate)
        shutil.copy2(
            seed / "provenance/coordinate_measure_upgrade.json",
            coordinate_upgrade,
        )
        expected = {
            prmtop: "2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e",
            coordinate: "b8a151fd35b909de52b7f50b09f4a0c9fc5221d0b423c6a9ec3133bf867c4954",
        }
        for path, digest in expected.items():
            if sha256_file(path) != digest:
                raise ValueError(f"canonical FAB artifact hash mismatch: {path}")
        candidate = write_bundle(
            work / "candidate_bundle",
            name=name,
            target="adp",
            prmtop_path=prmtop,
            coordinate_path=coordinate,
            model={
                "force_field": "Amber ff96",
                "implicit_solvent": "OBC1 (igb=2)",
                "radii": "mbondi2",
                "sasa": "ACE",
                "solute_dielectric": 1.0,
                "solvent_dielectric": 78.5,
                "salt_molar": 0.0,
                "nonbonded_method": "NoCutoff",
                "constraints": None,
                "openmmtools_commit": "e7e5847a677a9ecc83fcb284d0f120960416353f",
            },
            canonical_smiles="CC(=O)N[C@@H](C)C(=O)NC",
            expected_formula="C6H12N2O2",
            expected_charge=0,
            minimize=False,
            provenance_files={
                "coordinate_measure_upgrade.json": coordinate_upgrade,
            },
        )
        return verify_frozen_rebuild(candidate, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("adp", "glycerol", "diethanolamine"))
    args = parser.parse_args()
    outputs = []
    if args.only in (None, "adp"):
        outputs.append(build_adp())
    if args.only in (None, "glycerol"):
        name = "glycerol_gaff2_am1bcc_obc1_v2"
        outputs.append(
            parameterize_small_molecule(
                name="glycerol",
                source_mol2=(
                    bundle_seed(name)
                    / "provenance/input.mol2"
                ),
                bundle_name=name,
                canonical_smiles="OCC(O)CO",
                formula="C3H8O3",
                diagnostic="full support; both central-carbon determinant signs",
            )
        )
    if args.only in (None, "diethanolamine"):
        name = "diethanolamine_neutral_gaff2_am1bcc_obc1_v2"
        outputs.append(
            parameterize_small_molecule(
                name="diethanolamine",
                source_mol2=(
                    bundle_seed(name)
                    / "provenance/input.mol2"
                ),
                bundle_name=name,
                canonical_smiles="OCCNCCO",
                formula="C4H11NO2",
                diagnostic="neutral microstate; both nitrogen-pyramid signs",
            )
        )
    for output in outputs:
        print(f"verified frozen rebuild {output}")


if __name__ == "__main__":
    main()
