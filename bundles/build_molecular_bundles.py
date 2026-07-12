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
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


MOLECULAR_ROOT = Path(__file__).resolve().parents[1]

from jflows_md.core.builder import normalize_amber_transcript, write_bundle  # noqa: E402
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


def ambertools_prefix() -> Path:
    """Locate an optional AmberTools installation through AMBERHOME or PATH."""
    amberhome = os.environ.get("AMBERHOME")
    if amberhome:
        candidates = [Path(amberhome).expanduser().resolve()]
    else:
        executables = {
            name: shutil.which(name) for name in ("antechamber", "parmchk2", "tleap")
        }
        if any(path is None for path in executables.values()):
            raise RuntimeError(
                "small-molecule rebuilding requires an optional AmberTools "
                "installation exposed through AMBERHOME or PATH"
            )
        candidates = [
            Path(path).resolve().parent.parent
            for path in executables.values()
            if path is not None
        ]
    prefix = candidates[0]
    if any(candidate != prefix for candidate in candidates):
        raise RuntimeError("AmberTools executables resolve to different prefixes")
    missing = [
        name
        for name in ("antechamber", "parmchk2", "tleap")
        if not (prefix / "bin" / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"AmberTools prefix {prefix} is missing executables: {missing}"
        )
    return prefix


def ambertools_unofficial_version(prefix: Path) -> str:
    """Return the active optional wheel version for a newly versioned bundle."""
    if prefix != Path(sys.prefix).resolve():
        raise RuntimeError(
            "new-bundle mode requires ambertools-unofficial in the active "
            "Python environment"
        )
    try:
        return metadata.version("ambertools-unofficial")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "new-bundle mode requires the optional jflows_md[bundles] extra"
        ) from error


def parameterize_small_molecule(
    *,
    name: str,
    source_mol2: Path,
    bundle_name: str,
    canonical_smiles: str,
    formula: str,
    diagnostic: str,
    output: Path | None = None,
) -> Path:
    prefix = ambertools_prefix()
    frozen_record = json.loads(
        (source_mol2.parent / "build_record.json").read_text(encoding="utf-8")
    )
    hashes = amber_data_hashes(prefix)
    if output is None:
        expected_hashes = frozen_record["amber_data_sha256"]
        if hashes != expected_hashes:
            raise ValueError(
                "AmberTools data files differ from the frozen bundle toolchain; "
                f"a changed build requires --name and --output: {hashes} != "
                f"{expected_hashes}"
            )
        # The data hashes and exact candidate-manifest gate are authoritative.
        # Preserve the recorded release label rather than depending on a
        # specific environment manager's package metadata.
        version = str(frozen_record["ambertools_version"])
    else:
        version = ambertools_unofficial_version(prefix)
    with tempfile.TemporaryDirectory(prefix=f"jflows_md_{name}_") as temporary:
        work = Path(temporary)
        shutil.copy2(source_mol2, work / "input.mol2")
        coordinate_upgrade = source_mol2.parent / "coordinate_measure_upgrade.json"
        if output is None and coordinate_upgrade.is_file():
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
        # Provenance must not expose a workstation path or environment-manager
        # layout. Preserve commands and data-file identities while replacing
        # the installation prefix by a portable marker.
        for filename in (
            "antechamber.stdout",
            "parmchk2.stdout",
            "tleap.stdout",
            "leap.log",
            "sqm.in",
            "sqm.out",
        ):
            transcript = work / filename
            if transcript.is_file():
                transcript.write_text(
                    normalize_amber_transcript(
                        transcript.read_text(errors="replace"), prefix
                    ),
                    encoding="utf-8",
                )
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
        if output is None:
            return verify_frozen_rebuild(candidate, bundle_name)
        destination = output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"new bundle output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidate, destination)
        return Molecular_Bundle.load(destination, verify=True).path


def build_adp() -> Path:
    name = "adp_ff96_obc1"
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
    parser.add_argument(
        "--name",
        help="new small-molecule bundle name; requires --only and --output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new small-molecule bundle directory; requires --name",
    )
    args = parser.parse_args()
    if (args.name is None) != (args.output is None):
        parser.error("--name and --output must be supplied together")
    if args.name is not None:
        if args.only not in ("glycerol", "diethanolamine"):
            parser.error("new-bundle mode requires --only glycerol or diethanolamine")
        if args.name in _FROZEN_BUNDLE_MANIFEST_SHA256:
            parser.error("new-bundle mode cannot overwrite a frozen built-in name")
    outputs = []
    if args.only in (None, "adp"):
        outputs.append(build_adp())
    if args.only in (None, "glycerol"):
        seed_name = "glycerol_gaff2_am1bcc_obc1"
        name = args.name if args.only == "glycerol" and args.name else seed_name
        outputs.append(
            parameterize_small_molecule(
                name="glycerol",
                source_mol2=(
                    bundle_seed(seed_name)
                    / "provenance/input.mol2"
                ),
                bundle_name=name,
                canonical_smiles="OCC(O)CO",
                formula="C3H8O3",
                diagnostic="full support; both central-carbon determinant signs",
                output=args.output if args.only == "glycerol" else None,
            )
        )
    if args.only in (None, "diethanolamine"):
        seed_name = "diethanolamine_gaff2_am1bcc_obc1"
        name = args.name if args.only == "diethanolamine" and args.name else seed_name
        outputs.append(
            parameterize_small_molecule(
                name="diethanolamine",
                source_mol2=(
                    bundle_seed(seed_name)
                    / "provenance/input.mol2"
                ),
                bundle_name=name,
                canonical_smiles="OCCNCCO",
                formula="C4H11NO2",
                diagnostic="neutral microstate; both nitrogen-pyramid signs",
                output=args.output if args.only == "diethanolamine" else None,
            )
        )
    for output in outputs:
        print(f"bundle ready {output}")


if __name__ == "__main__":
    main()
