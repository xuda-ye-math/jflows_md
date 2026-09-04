#!/usr/bin/env python
"""Host-only checks for the minimal molecular bundle contract."""

import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import jflows_md
from jflows_md import Molecular_Bundle, available_bundles
from jflows_md.bundle_build import write_bundle


EXPECTED = {
    "alanine_dipeptide_ff96_obc1": ("C6H12N2O2", 22, 60, 42, 18),
    "methane_gaff2_am1bcc_obc1": ("CH4", 5, 9, 7, 2),
    "ethane_gaff2_am1bcc_obc1": ("C2H6", 8, 18, 13, 5),
    "propane_gaff2_am1bcc_obc1": ("C3H8", 11, 27, 19, 8),
    "n_butane_gaff2_am1bcc_obc1": ("C4H10", 14, 36, 25, 11),
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
    "backend",
    "Manual_Reject",
    "Mixed_Identity",
    "Mixed_NSF",
    "Molecular_Bundle",
    "Molecular_Potential",
    "Molecular_Source",
    "available_bundles",
    "boltzmann_identity",
    "boltzmann_FABX_G",
    "boltzmann_FAB_G",
    "boltzmann_forward_KLL1_G",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLX_G_fixed",
    "boltzmann_forward_KLXX_G",
    "boltzmann_forward_KLXX_G_fixed",
    "mixed_hmc",
    "mixed_mala",
    "mixed_quench_and_temper",
    "run_inference",
    "sequential_monte_carlo",
    "sequential_monte_carlo_fab",
    "train_FABX_G",
    "train_FAB_G",
    "train_forward_KLL1_G",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
}


def main() -> None:
    assert jflows_md.__version__ == "0.6.1"
    assert set(jflows_md.__all__) == EXPECTED_PUBLIC
    output = io.StringIO()
    with redirect_stdout(output):
        result = jflows_md.backend()
    report = output.getvalue()
    assert result is None
    assert "JAX " in report
    assert "Equinox " in report
    assert "Selected backend:" in report
    assert "Available backends:" in report
    assert "OpenMM" in report
    assert "OpenMM GPU backend:" in report
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
        shutil.copytree(Molecular_Bundle.load("alanine_dipeptide_ff96_obc1").path, root / "one")
        assert available_bundles(root) == ("one",)
        selected = Molecular_Bundle.load("one", root=root)
        assert selected.name == "alanine_dipeptide_ff96_obc1"
        assert selected.path == (root / "one").resolve()
        selected_target = jflows_md.Molecular_Potential.from_bundle(
            "one", root=root
        )
        assert Path(selected_target.bundle_path) == selected.path
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
