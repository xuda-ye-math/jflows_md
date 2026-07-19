#!/usr/bin/env python
"""Host-only checks for the minimal molecular bundle contract."""

import tempfile
from pathlib import Path
import shutil

import jflows_md
from jflows_md import Molecular_Bundle, available_bundles
from jflows_md.bundle_build import write_bundle


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
    "__version__",
    "Mixed_Identity",
    "Mixed_NSF",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "annealed_importance_sampling",
    "available_bundles",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLXX_G",
    "mixed_mala",
    "mixed_quench_and_temper",
    "potential_space_smc",
    "sequential_monte_carlo",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
}


def main() -> None:
    assert jflows_md.__version__ == "0.5.0"
    assert set(jflows_md.__all__) == EXPECTED_PUBLIC
    assert set(available_bundles()) == set(EXPECTED)
    for name, (formula, atoms, dimension, euclidean, periodic) in EXPECTED.items():
        bundle = Molecular_Bundle.load(name)
        assert bundle.name == name
        assert bundle.manifest["formula"] == formula
        assert bundle.n_atoms == atoms
        assert bundle.dimension == dimension
        assert bundle.coordinates["euclidean_dim"] == euclidean
        assert bundle.coordinates["periodic_dim"] == periodic
        assert {path.name for path in bundle.path.iterdir()} == EXPECTED_FILES
        print(f"PASS minimal bundle {name}: {dimension}D")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert available_bundles(root) == ()
        assert available_bundles(root / "missing") == ()
        shutil.copytree(Molecular_Bundle.load("adp_ff96_obc1").path, root / "one")
        assert available_bundles(root) == ("one",)
        occupied = root / "occupied"
        occupied.mkdir()
        marker = occupied / "keep.txt"
        marker.write_text("keep\n", encoding="utf-8")
        try:
            write_bundle(
                occupied,
                name="unused",
                target="unused",
                prmtop_path="missing.prmtop",
                coordinate_path="missing.rst7",
                model={},
                canonical_smiles="",
                expected_formula="",
                expected_charge=0,
                minimize=False,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("bundle builder overwrote an existing directory")
        assert marker.read_text(encoding="utf-8") == "keep\n"
    print("PASS minimal molecular bundle loading")


if __name__ == "__main__":
    main()
