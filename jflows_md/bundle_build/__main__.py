#!/usr/bin/env python
"""Build one minimal molecular runtime bundle from Amber topology inputs.

The output contains only ``manifest.json``, ``system.json``,
``coordinates.json``, ``validation.json``, ``system.xml``, and
``reference.pdb``. Construction transcripts and copied topology files are not
runtime inputs and are not placed in the bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import write_bundle


PRESETS = {
    "alanine_dipeptide": {
        "name": "alanine_dipeptide_ff96_obc1",
        "canonical_smiles": "CC(=O)N[C@@H](C)C(=O)NC",
        "formula": "C6H12N2O2",
        "minimize": False,
        "coordinates": {
            "zmatrix": {
                "root": 6,
                "prefix": (6, 8, 14, 10),
                "overrides": {
                    8: (6, -1, -1),
                    14: (8, 6, -1),
                    10: (8, 14, 6),
                },
            },
            "fixed_stereocenters": (
                {
                    "label": "alanine_ca_L",
                    "torsion_index": 0,
                    "atoms": (8, 6, 14, 10),
                    "configuration": "S",
                    "cip_priority_atoms": (6, 14, 10, 9),
                },
            ),
        },
        "model": {
            "force_field": "Amber ff96",
            "implicit_solvent": "OBC1 (igb=2)",
            "radii": "mbondi2",
            "sasa": "ACE",
            "solute_dielectric": 1.0,
            "solvent_dielectric": 78.5,
            "salt_molar": 0.0,
            "nonbonded_method": "NoCutoff",
            "constraints": None,
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=tuple(PRESETS))
    parser.add_argument("prmtop", type=Path)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    preset = PRESETS[args.target]
    output = write_bundle(
        args.output,
        name=args.name or preset["name"],
        target=args.target,
        prmtop_path=args.prmtop,
        coordinate_path=args.coordinates,
        model=preset["model"],
        canonical_smiles=preset["canonical_smiles"],
        expected_formula=preset["formula"],
        expected_charge=0,
        minimize=preset["minimize"],
        **preset["coordinates"],
    )
    print(f"bundle ready {output}")


if __name__ == "__main__":
    main()
