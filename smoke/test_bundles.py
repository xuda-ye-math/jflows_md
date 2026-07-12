#!/usr/bin/env python
"""Host-only integrity and metadata smoke test for all molecular bundles."""

from __future__ import annotations




import jflows_md  # noqa: E402
from jflows_md import Molecular_Bundle, available_bundles  # noqa: E402
from jflows_md.system import sha256_file  # noqa: E402


EXPECTED = {
    "fab_adp_ff96_obc1_v1": ("C6H12N2O2", 22, 60, 42, 18),
    "glycerol_gaff2_am1bcc_obc1_v1": ("C3H8O3", 14, 36, 25, 11),
    "diethanolamine_neutral_gaff2_am1bcc_obc1_v1": ("C4H11NO2", 18, 48, 33, 15),
}


def main() -> None:
    expected_public = {
        "Mixed_Identity",
        "Mixed_NSF",
        "Molecular_Bundle",
        "Molecular_Potential",
        "Molecular_Source",
        "annealed_importance_sampling",
        "available_bundles",
        "load_mixed_flow_stages",
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
    for name, (formula, atoms, dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name, verify=True)
        assert bundle.system["formula"] == formula
        assert bundle.n_atoms == atoms
        assert bundle.dimension == dimension
        assert bundle.coordinates["euclidean_dim"] == euclidean
        assert bundle.coordinates["periodic_dim"] == periodic
        assert bundle.manifest["model"]["implicit_solvent"] == "OBC1 (igb=2)"
        assert min(bundle.system["gb_offset_radius_nm"]) > 0
        assert min(bundle.system["gb_scaled_offset_radius_nm"]) > 0
        assert len(bundle.validation["frames_nm"]) == 4
        assert len(bundle.validation["forces_kj_mol_nm"]) == 4
        if name == "fab_adp_ff96_obc1_v1":
            assert (
                sha256_file(bundle.path / "system.prmtop")
                == "2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e"
            )
            assert bundle.coordinates["chiral_torsion_sign"] == -1
            assert bundle.coordinates["chirality_sign"] == 1
        else:
            assert bundle.coordinates["chiral_torsion_index"] == -1
            text = (bundle.path / "provenance/tleap.stdout").read_text()
            assert "Exiting LEaP: Errors = 0; Warnings = 0; Notes = 0." in text
        print(f"PASS bundle {name}")


if __name__ == "__main__":
    main()
