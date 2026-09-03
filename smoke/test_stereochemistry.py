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


LEGACY_PRESETS = ("alanine_dipeptide",)


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

        expected_fixed = 1
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

    adp = Molecular_Bundle.load(PRESETS["alanine_dipeptide"]["name"])
    explicit_generic = build_coordinate_spec(
        adp.system,
        np.asarray(adp.validation["frames_nm"][0]),
        adp.system["bonds"],
        target="alanine_dipeptide",
        signed_volume_diagnostics=(),
    )
    assert explicit_generic["schema_version"] == 3
    assert explicit_generic["fixed_stereocenters"] == []
    print("PASS legacy target-only bundle-builder routing")


def check_one_center_support() -> None:
    """One fixed center restricts the support to the L configuration."""

    adp_bundle, adp_spec = _preset_spec("alanine_dipeptide")
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
    print("PASS one-stereocenter support")


def check_invalid_stereochemistry_rejected() -> None:
    bundle, spec = _preset_spec("alanine_dipeptide")
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
            zmatrix=PRESETS["alanine_dipeptide"]["coordinates"]["zmatrix"],
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
            zmatrix=PRESETS["alanine_dipeptide"]["coordinates"]["zmatrix"],
            fixed_stereocenters=(fractional,),
        )
    except ValueError as exc:
        assert "invalid indices" in str(exc)
    else:
        raise AssertionError("fractional stereocenter index was accepted")

    print("PASS invalid stereochemical specifications are rejected")


def main() -> None:
    check_schema2_compatibility()
    check_legacy_builder_target_routing()
    check_one_center_support()
    check_invalid_stereochemistry_rejected()


if __name__ == "__main__":
    main()
