#!/usr/bin/env python
"""Legacy, zero-, one-, and multi-stereocenter coordinate smoke tests."""

from __future__ import annotations

from dataclasses import replace
import os


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import Mixed_NSF, Molecular_Potential  # noqa: E402
from jflows_md.bundle_build.__main__ import PRESETS  # noqa: E402
from jflows_md.bundle_build.builder import build_coordinate_spec  # noqa: E402
from jflows_md.core.coordinates import (  # noqa: E402
    Internal_Coordinates,
    signed_volume,
)
from jflows_md.system import Molecular_Bundle  # noqa: E402


LEGACY_PRESETS = ("adp", "glycerol", "diethanolamine")
CANDIDATE_PRESETS = {
    "nma": (0, ()),
    "s_2_butanol": (1, ("S",)),
    "rr_2_3_butanediol": (2, ("R", "R")),
    "cyclohexane": (0, ()),
}


def _preset_spec(target: str) -> tuple[Molecular_Bundle, dict]:
    preset = PRESETS[target]
    bundle = Molecular_Bundle.load(preset["name"])
    spec = build_coordinate_spec(
        bundle.system,
        np.asarray(bundle.validation["frames_nm"][0]),
        bundle.system["bonds"],
        **preset["coordinates"],
    )
    return bundle, spec


def check_schema2_compatibility() -> None:
    """New schema-3 presets reproduce every legacy schema-2 chart exactly."""

    for seed, target in enumerate(LEGACY_PRESETS):
        bundle, spec = _preset_spec(target)
        legacy = Molecular_Potential(bundle)
        explicit = Molecular_Potential(replace(bundle, coordinates=spec))
        legacy_source = legacy.source()
        explicit_source = explicit.source()
        np.testing.assert_array_equal(explicit_source.mean, legacy_source.mean)
        np.testing.assert_array_equal(
            explicit_source.variance, legacy_source.variance
        )
        key = jax.random.key(810 + seed)
        q = legacy_source.samples(key, N=32)
        np.testing.assert_array_equal(explicit_source.samples(key, N=32), q)
        legacy_x, legacy_logdet = legacy.coordinates.to_cartesian(q)
        explicit_x, explicit_logdet = explicit.coordinates.to_cartesian(q)
        np.testing.assert_array_equal(explicit_x, legacy_x)
        np.testing.assert_array_equal(explicit_logdet, legacy_logdet)
        np.testing.assert_array_equal(explicit(q), legacy(q))
        np.testing.assert_array_equal(
            explicit.physical_energy(q), legacy.physical_energy(q)
        )
        assert explicit.domain == legacy.domain

        expected_fixed = 1 if target == "adp" else 0
        assert explicit.coordinates.n_fixed_stereocenters == expected_fixed
        assert legacy.coordinates.n_fixed_stereocenters == expected_fixed
    print("PASS schema-2 molecular-coordinate compatibility")


def check_legacy_builder_target_routing() -> None:
    """Historical target-only builder calls reproduce schema-2 specs exactly."""

    for target_name in LEGACY_PRESETS:
        preset = PRESETS[target_name]
        bundle = Molecular_Bundle.load(preset["name"])
        rebuilt = build_coordinate_spec(
            bundle.system,
            np.asarray(bundle.validation["frames_nm"][0]),
            bundle.system["bonds"],
            target=target_name,
        )
        assert rebuilt == bundle.coordinates

    adp = Molecular_Bundle.load(PRESETS["adp"]["name"])
    explicit_generic = build_coordinate_spec(
        adp.system,
        np.asarray(adp.validation["frames_nm"][0]),
        adp.system["bonds"],
        target="adp",
        signed_volume_diagnostics=(),
    )
    assert explicit_generic["schema_version"] == 3
    assert explicit_generic["fixed_stereocenters"] == []
    print("PASS legacy target-only bundle-builder routing")


def check_candidate_molecules() -> None:
    """Exercise the four finite candidate targets on their actual bundles."""

    for seed, (target_name, (fixed_count, configurations)) in enumerate(
        CANDIDATE_PRESETS.items()
    ):
        bundle, rebuilt_spec = _preset_spec(target_name)
        target = Molecular_Potential(bundle)
        rebuilt = Molecular_Potential(replace(bundle, coordinates=rebuilt_spec))
        assert bundle.manifest["model"] == PRESETS[target_name]["model"]
        assert (
            bundle.manifest["canonical_smiles"]
            == PRESETS[target_name]["canonical_smiles"]
        )
        assert rebuilt_spec == bundle.coordinates
        assert target.coordinates.n_fixed_stereocenters == fixed_count
        assert tuple(
            item["configuration"]
            for item in bundle.coordinates["fixed_stereocenters"]
        ) == configurations

        q = target.source().samples(jax.random.key(840 + seed), N=64)
        x, logdet = target.coordinates.to_cartesian(q)
        q_back, inverse_logdet = target.coordinates.to_internal(x)
        np.testing.assert_allclose(
            target.domain.displacement(q_back, q), 0.0, rtol=0, atol=3e-13
        )
        np.testing.assert_allclose(
            logdet + inverse_logdet, 0.0, rtol=0, atol=5e-13
        )
        np.testing.assert_array_equal(rebuilt(q), target(q))
        assert bool(jnp.all(target.support_mask(x)))
        for fixed in bundle.coordinates["fixed_stereocenters"]:
            volume = signed_volume(x, *fixed["atoms"])
            assert bool(jnp.all(fixed["volume_sign"] * volume > 0.0))
        mirrored = x.at[..., 0].multiply(-1.0)
        if fixed_count:
            assert not bool(jnp.any(target.support_mask(mirrored)))
        else:
            assert bool(jnp.all(target.support_mask(mirrored)))

        energy = jax.jit(lambda value: target(value))(q[:2])
        gradient = jax.jit(lambda value: target.grad(value))(q[:2])
        assert bool(jnp.isfinite(energy).all() & jnp.isfinite(gradient).all())
        flow = Mixed_NSF(
            jax.random.key(850 + seed),
            target.domain,
            bins=4,
            transforms=2,
            euclidean_bound=6.0,
            hidden_features=(16, 16),
            mask_strategy="balanced",
        ).zeros()
        mapped, forward_ladj = flow.call_and_ladj(q[:8])
        recovered, inverse_ladj = flow.inv_and_ladj(mapped)
        np.testing.assert_allclose(
            target.domain.displacement(recovered, q[:8]),
            0.0,
            rtol=0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            forward_ladj + inverse_ladj, 0.0, rtol=0, atol=2e-12
        )
        print(
            f"PASS candidate {target_name}: "
            f"R^{target.domain.euclidean_dim} x "
            f"T^{target.domain.periodic_dim}, fixed={fixed_count}"
        )

    nma = Molecular_Potential.from_bundle("nma_ff96_obc1")
    trans = nma.reference_internal()[None]
    # The ACE-C-N-C torsion is ordinary and periodic: both trans and a
    # pi-shifted cis representative are in the chart and remain energy-finite.
    cis = nma.domain.wrap(
        trans.at[:, nma.domain.euclidean_dim + 5].add(jnp.pi)
    )
    assert bool(jnp.isfinite(nma(jnp.concatenate((trans, cis)))).all())
    assert bool(jnp.all(nma.support_mask(nma.cartesian(cis))))

    cyclohexane = Molecular_Potential.from_bundle(
        "cyclohexane_gaff2_am1bcc_obc1"
    )
    cyclo_bundle = Molecular_Bundle.load(
        "cyclohexane_gaff2_am1bcc_obc1"
    )
    tree_bonds = {
        frozenset((cyclo_bundle.coordinates["order"][placement], refs[0]))
        for placement, refs in enumerate(cyclo_bundle.coordinates["refs"])
        if placement > 0
    }
    system_bonds = {
        frozenset(map(int, pair)) for pair in cyclo_bundle.system["bonds"]
    }
    closure_bonds = system_bonds - tree_bonds
    assert len(closure_bonds) == 1
    closure = tuple(closure_bonds.pop())
    assert all(
        cyclo_bundle.system["atomic_numbers"][atom] == 6 for atom in closure
    )
    reference = jnp.asarray(cyclo_bundle.validation["frames_nm"][:1])
    reference_q = cyclohexane.coordinates.to_internal(reference)[0]
    reconstructed = cyclohexane.cartesian(reference_q)
    expected_distance = jnp.linalg.norm(
        reference[:, closure[0]] - reference[:, closure[1]], axis=-1
    )
    reconstructed_distance = jnp.linalg.norm(
        reconstructed[:, closure[0]] - reconstructed[:, closure[1]], axis=-1
    )
    np.testing.assert_allclose(
        reconstructed_distance, expected_distance, rtol=0, atol=2e-13
    )
    mirrored_chair = reference.at[..., 2].multiply(-1.0)
    assert bool(jnp.all(cyclohexane.support_mask(mirrored_chair)))
    mirrored_q = cyclohexane.coordinates.to_internal(mirrored_chair)[0]
    assert bool(jnp.isfinite(cyclohexane(mirrored_q)).all())
    print("PASS NMA cis/trans and cyclohexane ring-support policies")

    rr_bundle = Molecular_Bundle.load(
        "rr_2_3_butanediol_gaff2_am1bcc_obc1"
    )
    rr = Molecular_Potential(rr_bundle)
    rr_q = rr.source().samples(jax.random.key(860), N=32)
    rr_x = rr.cartesian(rr_q)
    first, second = rr_bundle.coordinates["fixed_stereocenters"]

    def swap(positions, first_atom, second_atom):
        first_position = positions[:, first_atom, :]
        second_position = positions[:, second_atom, :]
        return positions.at[:, first_atom, :].set(second_position).at[
            :, second_atom, :
        ].set(first_position)

    first_violated = swap(rr_x, 0, 4)
    first_check = first["volume_sign"] * signed_volume(
        first_violated, *first["atoms"]
    )
    second_check = second["volume_sign"] * signed_volume(
        first_violated, *second["atoms"]
    )
    assert bool(jnp.all(first_check < 0.0) & jnp.all(second_check > 0.0))
    assert not bool(jnp.any(rr.support_mask(first_violated)))

    second_violated = swap(rr_x, 1, 5)
    first_check = first["volume_sign"] * signed_volume(
        second_violated, *first["atoms"]
    )
    second_check = second["volume_sign"] * signed_volume(
        second_violated, *second["atoms"]
    )
    assert bool(jnp.all(first_check > 0.0) & jnp.all(second_check < 0.0))
    assert not bool(jnp.any(rr.support_mask(second_violated)))
    print("PASS independent (2R,3R)-2,3-butanediol support conjunction")


def check_zero_and_one_center_support() -> None:
    """Diagnostics alone do not restrict support; one fixed center does."""

    glycerol_bundle, glycerol_spec = _preset_spec("glycerol")
    generic_spec = build_coordinate_spec(
        glycerol_bundle.system,
        np.asarray(glycerol_bundle.validation["frames_nm"][0]),
        glycerol_bundle.system["bonds"],
    )
    assert generic_spec["fixed_stereocenters"] == []
    assert generic_spec["signed_volume_diagnostics"] == []
    glycerol = Molecular_Potential(
        replace(glycerol_bundle, coordinates=glycerol_spec)
    )
    assert glycerol_spec["fixed_stereocenters"] == []
    assert len(glycerol_spec["signed_volume_diagnostics"]) == 1
    reference = glycerol.cartesian(glycerol.reference_internal()[None])
    mirrored = reference.at[..., 0].multiply(-1.0)
    assert bool(glycerol.support_mask(reference)[0])
    assert bool(glycerol.support_mask(mirrored)[0])
    generic = Internal_Coordinates(generic_spec)
    np.testing.assert_array_equal(
        generic.to_cartesian(glycerol.reference_internal()[None])[0], reference
    )

    adp_bundle, adp_spec = _preset_spec("adp")
    adp = Molecular_Potential(replace(adp_bundle, coordinates=adp_spec))
    fixed = adp_spec["fixed_stereocenters"][0]
    assert fixed["label"] == "alanine_ca_L"
    assert adp.coordinates.stereocenter_torsion_indices == (0,)
    q = adp.source().samples(jax.random.key(820), N=128)
    x = adp.cartesian(q)
    volume = signed_volume(x, *fixed["atoms"])
    assert bool(jnp.all(fixed["volume_sign"] * volume > 0.0))
    assert bool(jnp.all(adp.support_mask(x)))
    assert not bool(jnp.any(adp.support_mask(x.at[..., 0].multiply(-1.0))))
    print("PASS zero- and one-stereocenter support")


def check_multiple_fixed_centers() -> None:
    """Two explicit half-charts compose without changing force-field energy."""

    bundle = Molecular_Bundle.load("glycerol_gaff2_am1bcc_obc1")
    spec = build_coordinate_spec(
        bundle.system,
        np.asarray(bundle.validation["frames_nm"][0]),
        bundle.system["bonds"],
        zmatrix={"overrides": {12: (4, 5, 11)}},
        fixed_stereocenters=(
            {
                "label": "first_tetrahedral_center",
                "torsion_index": 0,
                "atoms": (1, 0, 2, 7),
            },
            {
                "label": "second_tetrahedral_center",
                "torsion_index": 9,
                "atoms": (4, 11, 5, 12),
            },
        ),
    )
    target = Molecular_Potential(replace(bundle, coordinates=spec))
    legacy = Molecular_Potential(bundle)
    assert spec["schema_version"] == 3
    assert target.coordinates.n_fixed_stereocenters == 2
    assert target.coordinates.stereocenter_torsion_indices == (0, 9)
    assert target.domain.dimension == legacy.domain.dimension == 36
    assert target.domain.euclidean_dim == legacy.domain.euclidean_dim + 2
    assert target.domain.periodic_dim == legacy.domain.periodic_dim - 2

    q = target.source().samples(jax.random.key(830), N=128)
    x, logdet = target.coordinates.to_cartesian(q)
    q_back, inverse_logdet = target.coordinates.to_internal(x)
    np.testing.assert_allclose(
        target.domain.displacement(q_back, q), 0.0, rtol=0, atol=3e-13
    )
    np.testing.assert_allclose(
        logdet + inverse_logdet, 0.0, rtol=0, atol=5e-13
    )
    for fixed in spec["fixed_stereocenters"]:
        volume = signed_volume(x, *fixed["atoms"])
        assert bool(jnp.all(fixed["volume_sign"] * volume > 0.0))
    assert bool(jnp.all(target.support_mask(x)))

    fixed_first, fixed_second = spec["fixed_stereocenters"]

    def swap_substituents(positions, fixed):
        first, second = fixed["atoms"][1:3]
        first_position = positions[:, first, :]
        second_position = positions[:, second, :]
        return positions.at[:, first, :].set(second_position).at[
            :, second, :
        ].set(first_position)

    first_violated = swap_substituents(x, fixed_first)
    first_volume = signed_volume(first_violated, *fixed_first["atoms"])
    second_volume = signed_volume(first_violated, *fixed_second["atoms"])
    assert bool(jnp.all(fixed_first["volume_sign"] * first_volume < 0.0))
    assert bool(jnp.all(fixed_second["volume_sign"] * second_volume > 0.0))
    assert not bool(jnp.any(target.support_mask(first_violated)))

    second_violated = swap_substituents(x, fixed_second)
    first_volume = signed_volume(second_violated, *fixed_first["atoms"])
    second_volume = signed_volume(second_violated, *fixed_second["atoms"])
    assert bool(jnp.all(fixed_first["volume_sign"] * first_volume > 0.0))
    assert bool(jnp.all(fixed_second["volume_sign"] * second_volume < 0.0))
    assert not bool(jnp.any(target.support_mask(second_violated)))

    mirrored = x.at[..., 0].multiply(-1.0)
    assert not bool(jnp.any(target.support_mask(mirrored)))
    np.testing.assert_array_equal(target.forcefield(x), legacy.forcefield(x))
    np.testing.assert_allclose(
        target.forcefield(mirrored), target.forcefield(x), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        target(q), target.beta * target.forcefield(x) - logdet, rtol=0, atol=1e-11
    )
    energy = jax.jit(lambda value: target(value))(q[:2])
    gradient = jax.jit(lambda value: target.grad(value))(q[:2])
    assert bool(jnp.isfinite(energy).all() & jnp.isfinite(gradient).all())

    flow = Mixed_NSF(
        jax.random.key(831),
        target.domain,
        bins=4,
        transforms=2,
        euclidean_bound=6.0,
        hidden_features=(16, 16),
        mask_strategy="balanced",
    ).zeros()
    mapped, forward_ladj = flow.call_and_ladj(q[:8])
    recovered, inverse_ladj = flow.inv_and_ladj(mapped)
    np.testing.assert_allclose(
        target.domain.displacement(recovered, q[:8]), 0.0, rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        forward_ladj + inverse_ladj, 0.0, rtol=0, atol=2e-12
    )
    print("PASS multiple fixed stereocenters preserve the molecular potential")


def check_invalid_stereochemistry_rejected() -> None:
    bundle, spec = _preset_spec("adp")
    mixed = dict(spec)
    mixed.update(bundle.coordinates)
    mixed["fixed_stereocenters"] = spec["fixed_stereocenters"]
    try:
        Internal_Coordinates(mixed)
    except ValueError as exc:
        assert "cannot mix" in str(exc)
    else:
        raise AssertionError("mixed singleton/list stereochemistry was accepted")

    duplicate = [dict(spec["fixed_stereocenters"][0])] * 2
    try:
        build_coordinate_spec(
            bundle.system,
            np.asarray(bundle.validation["frames_nm"][0]),
            bundle.system["bonds"],
            zmatrix=PRESETS["adp"]["coordinates"]["zmatrix"],
            fixed_stereocenters=duplicate,
        )
    except ValueError as exc:
        assert "labels must be unique" in str(exc)
    else:
        raise AssertionError("duplicate fixed stereocenters were accepted")

    fractional = dict(spec["fixed_stereocenters"][0])
    fractional["torsion_index"] = 0.7
    try:
        build_coordinate_spec(
            bundle.system,
            np.asarray(bundle.validation["frames_nm"][0]),
            bundle.system["bonds"],
            zmatrix=PRESETS["adp"]["coordinates"]["zmatrix"],
            fixed_stereocenters=(fractional,),
        )
    except ValueError as exc:
        assert "invalid indices" in str(exc)
    else:
        raise AssertionError("fractional stereocenter index was accepted")

    butanol_bundle, _ = _preset_spec("s_2_butanol")
    wrong_configuration = dict(
        PRESETS["s_2_butanol"]["coordinates"]["fixed_stereocenters"][0]
    )
    wrong_configuration["configuration"] = "R"
    try:
        build_coordinate_spec(
            butanol_bundle.system,
            np.asarray(butanol_bundle.validation["frames_nm"][0]),
            butanol_bundle.system["bonds"],
            fixed_stereocenters=(wrong_configuration,),
        )
    except ValueError as exc:
        assert "reference is S, expected R" in str(exc)
    else:
        raise AssertionError("incorrect requested CIP configuration was accepted")
    print("PASS invalid stereochemical specifications are rejected")


def main() -> None:
    check_schema2_compatibility()
    check_legacy_builder_target_routing()
    check_candidate_molecules()
    check_zero_and_one_center_support()
    check_multiple_fixed_centers()
    check_invalid_stereochemistry_rejected()


if __name__ == "__main__":
    main()
