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
    "adp": {
        "name": "adp_ff96_obc1",
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
    "glycerol": {
        "name": "glycerol_gaff2_am1bcc_obc1",
        "canonical_smiles": "OCC(O)CO",
        "formula": "C3H8O3",
        "minimize": True,
        "coordinates": {
            "signed_volume_diagnostics": (
                {
                    "label": "central_carbon_orientation",
                    "atoms": (2, 1, 3, 4),
                },
            ),
        },
        "model": {
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
        },
    },
    "diethanolamine": {
        "name": "diethanolamine_gaff2_am1bcc_obc1",
        "canonical_smiles": "OCCNCCO",
        "formula": "C4H11NO2",
        "minimize": True,
        "coordinates": {
            "signed_volume_diagnostics": (
                {
                    "label": "amine_orientation",
                    "atoms": (3, 2, 4, 12),
                },
            ),
        },
        "model": {
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
        },
    },
    "nma": {
        "name": "nma_ff96_obc1",
        "canonical_smiles": "CC(=O)NC",
        "formula": "C3H7NO",
        "minimize": False,
        "coordinates": {
            "fixed_stereocenters": (),
        },
        "model": {
            "ambertools_version": "26.0.0",
            "force_field": "Amber ff96",
            "implicit_solvent": "OBC1 (igb=2)",
            "radii": "mbondi2",
            "sasa": "ACE",
            "solute_dielectric": 1.0,
            "solvent_dielectric": 78.5,
            "salt_molar": 0.0,
            "nonbonded_method": "NoCutoff",
            "constraints": None,
            "parameter_provenance": "ff96 ACE/NME residue templates",
            "structure_source": "AmberTools tleap sequence { ACE NME }",
        },
    },
    "s_2_butanol": {
        "name": "s_2_butanol_gaff2_am1bcc_obc1",
        "canonical_smiles": "CC[C@H](C)O",
        "formula": "C4H10O",
        "minimize": True,
        "coordinates": {
            "fixed_stereocenters": (
                {
                    "label": "butanol_C2_S",
                    "torsion_index": 0,
                    "atoms": (1, 2, 0, 3),
                    "configuration": "S",
                    "cip_priority_atoms": (0, 2, 3, 5),
                },
            ),
        },
        "model": {
            "ambertools_version": "26.0.0",
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
            "parameter_provenance": (
                "parmchk2 GAFF2 produced no supplemental terms"
            ),
            "structure_source": "PubChem CID 444683 3D conformer",
        },
    },
    "rr_2_3_butanediol": {
        "name": "rr_2_3_butanediol_gaff2_am1bcc_obc1",
        "canonical_smiles": "C[C@H]([C@@H](C)O)O",
        "formula": "C4H10O2",
        "minimize": True,
        "coordinates": {
            "zmatrix": {"overrides": {7: (3, 1, 5)}},
            "fixed_stereocenters": (
                {
                    "label": "butanediol_C2_R",
                    "torsion_index": 0,
                    "atoms": (2, 3, 0, 4),
                    "configuration": "R",
                    "cip_priority_atoms": (0, 3, 4, 6),
                },
                {
                    "label": "butanediol_C3_R",
                    "torsion_index": 5,
                    "atoms": (3, 1, 5, 7),
                    "configuration": "R",
                    "cip_priority_atoms": (1, 2, 5, 7),
                },
            ),
        },
        "model": {
            "ambertools_version": "26.0.0",
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
            "parameter_provenance": (
                "parmchk2 GAFF2 produced no supplemental terms"
            ),
            "structure_source": "PubChem CID 225936 3D conformer",
        },
    },
    "cyclohexane": {
        "name": "cyclohexane_gaff2_am1bcc_obc1",
        "canonical_smiles": "C1CCCCC1",
        "formula": "C6H12",
        "minimize": True,
        "coordinates": {
            "fixed_stereocenters": (),
        },
        "model": {
            "ambertools_version": "26.0.0",
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
            "parameter_provenance": (
                "parmchk2 GAFF2 c6 terms transferred from c3 at penalty 0.0"
            ),
            "structure_source": "PubChem CID 8078 3D conformer",
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
