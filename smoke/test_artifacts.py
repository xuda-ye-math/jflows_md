#!/usr/bin/env python
"""Round-trip and integrity checks for versioned molecular flow artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import (  # noqa: E402
    Mixed_NSF,
    Molecular_Potential,
    load_mixed_flow,
    load_mixed_flow_stages,
    mixed_flow_metadata,
    package_source_sha256,
    save_mixed_flow,
)
from jflows_md.system import sha256_file  # noqa: E402


def write_run(
    run: Path,
    target: Molecular_Potential,
    flow: Mixed_NSF,
    flow_key,
    *,
    schema_version: int,
    overrides: dict | None = None,
) -> tuple[Path, Path]:
    run.mkdir()
    data_path = run / "data.npz"
    flow_path = run / "flows.eqx"
    metadata = {
        "schema_version": schema_version,
        "bundle": target.bundle_name,
        "manifest_sha256": target.manifest_sha256,
        "stage_count": 1,
        "bins": 4,
        "transforms": 2,
        "hidden_features": np.asarray((8, 8)),
        "slope": 1e-3,
        "nsf_lim": 5.0,
        "flow_key_data": np.asarray(jax.random.key_data(flow_key)),
        "jflows_source_sha256": package_source_sha256("jflows"),
        "jflows_md_source_sha256": package_source_sha256("jflows_md"),
    }
    if schema_version == 2:
        metadata.update(mixed_flow_metadata(flow))
        metadata.update(jax_version=jax.__version__, equinox_version=eqx.__version__)
    if overrides:
        metadata.update(overrides)
    np.savez_compressed(data_path, **metadata)
    eqx.tree_serialise_leaves(flow_path, [flow])
    status_path = run / "train_status.log"
    status_path.write_text("DONE\n", encoding="utf-8")
    marker = {
        "schema_version": schema_version,
        "data_sha256": sha256_file(data_path),
        "flows_sha256": sha256_file(flow_path),
        "status_sha256": sha256_file(status_path),
    }
    (run / "COMPLETE.json").write_text(json.dumps(marker), encoding="utf-8")
    return data_path, flow_path


def expect_rejected(run: Path, text: str) -> None:
    try:
        load_mixed_flow_stages(run)
    except (KeyError, TypeError, ValueError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"invalid artifact was accepted: {run}")


def refresh_flow_hash(run: Path) -> None:
    marker_path = run / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["flows_sha256"] = sha256_file(run / "flows.eqx")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def main() -> None:
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    flow_key = jax.random.key(80)
    default_flow = Mixed_NSF(
        flow_key,
        target.domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        standalone = root / "standalone.eqx"
        save_mixed_flow(standalone, default_flow)
        standalone_loaded = load_mixed_flow(standalone)
        assert eqx.tree_equal(default_flow, standalone_loaded)

        schema1 = root / "schema1"
        schema1_data, _ = write_run(
            schema1, target, default_flow, flow_key, schema_version=1
        )
        loaded_target, loaded_flows = load_mixed_flow_stages(schema1, target=target)
        assert loaded_target.manifest_sha256 == target.manifest_sha256
        assert len(loaded_flows) == 1
        assert eqx.tree_equal(default_flow, loaded_flows[0])
        expect_rejected(schema1, "explicit target")
        with schema1_data.open("ab") as handle:
            handle.write(b"corruption")
        expect_rejected(schema1, "hash mismatch")

        tanh_flow = Mixed_NSF(
            flow_key,
            target.domain,
            bins=4,
            transforms=2,
            hidden_features=(8, 8),
            activation=jax.nn.tanh,
        )
        current = root / "current"
        write_run(current, target, tanh_flow, flow_key, schema_version=2)
        _, loaded = load_mixed_flow_stages(current)
        assert eqx.tree_equal(tanh_flow, loaded[0])
        samples = target.source().samples(jax.random.key(81), 4)
        expected_y, expected_ladj = tanh_flow.call_and_ladj(samples)
        actual_y, actual_ladj = loaded[0].call_and_ladj(samples)
        np.testing.assert_allclose(actual_y, expected_y, rtol=0, atol=0)
        np.testing.assert_allclose(actual_ladj, expected_ladj, rtol=0, atol=0)
        expected_x, expected_inverse_ladj = tanh_flow.inv_and_ladj(expected_y)
        actual_x, actual_inverse_ladj = loaded[0].inv_and_ladj(expected_y)
        np.testing.assert_allclose(actual_x, expected_x, rtol=0, atol=0)
        np.testing.assert_allclose(
            actual_inverse_ladj, expected_inverse_ladj, rtol=0, atol=0
        )

        balanced_flow = Mixed_NSF(
            flow_key,
            target.domain,
            bins=4,
            transforms=2,
            hidden_features=(8, 8),
            mask_strategy="balanced",
        )
        balanced = root / "balanced"
        write_run(balanced, target, balanced_flow, flow_key, schema_version=2)
        _, loaded_balanced = load_mixed_flow_stages(balanced)
        assert loaded_balanced[0].mask_strategy == "balanced"
        assert eqx.tree_equal(balanced_flow, loaded_balanced[0])

        explicit_mask = np.zeros(
            (2, target.domain.dimension), dtype=bool
        )
        explicit_mask[0, ::2] = True
        explicit_mask[1, 1::2] = True
        assert not np.array_equal(
            explicit_mask,
            mixed_flow_metadata(default_flow)["condition_mask"],
        )
        explicit_flow = Mixed_NSF(
            flow_key,
            target.domain,
            bins=4,
            transforms=2,
            hidden_features=(8, 8),
            condition_mask=explicit_mask,
        )
        explicit = root / "explicit_mask"
        write_run(explicit, target, explicit_flow, flow_key, schema_version=2)
        _, loaded_explicit = load_mixed_flow_stages(explicit)
        assert eqx.tree_equal(explicit_flow, loaded_explicit[0])
        explicit_y, explicit_ladj = explicit_flow.call_and_ladj(samples)
        loaded_y, loaded_ladj = loaded_explicit[0].call_and_ladj(samples)
        np.testing.assert_allclose(loaded_y, explicit_y, rtol=0, atol=0)
        np.testing.assert_allclose(loaded_ladj, explicit_ladj, rtol=0, atol=0)

        local_bundle = root / "local_bundle"
        shutil.copytree(target.bundle_path, local_bundle)
        relative = root / "relative"
        write_run(
            relative,
            target,
            balanced_flow,
            flow_key,
            schema_version=2,
            overrides={"bundle": np.asarray("../local_bundle")},
        )
        relative_target, relative_flows = load_mixed_flow_stages(relative)
        assert Path(relative_target.bundle_path) == local_bundle.resolve()
        assert eqx.tree_equal(balanced_flow, relative_flows[0])

        historical_random = root / "historical_random"
        write_run(
            historical_random,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"mask_strategy": np.asarray("random")},
        )
        historical_data = dict(
            np.load(historical_random / "data.npz", allow_pickle=False)
        )
        historical_data.pop("mask_strategy")
        historical_data.pop("condition_mask")
        np.savez_compressed(historical_random / "data.npz", **historical_data)
        marker = json.loads((historical_random / "COMPLETE.json").read_text())
        marker["data_sha256"] = sha256_file(historical_random / "data.npz")
        (historical_random / "COMPLETE.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
        _, loaded_historical_random = load_mixed_flow_stages(historical_random)
        assert loaded_historical_random[0].mask_strategy == "random"
        assert eqx.tree_equal(tanh_flow, loaded_historical_random[0])

        migrated_name = root / "migrated_name"
        write_run(
            migrated_name,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={
                "bundle": "glycerol_gaff2_am1bcc_obc1_v2",
                "manifest_sha256": (
                    "c60f520ef8d4146a0ffb4ff8875d56b90abebd47630d842780e7b3d71911c736"
                ),
                "jflows_md_source_sha256": (
                    "11c26e119d9f7d19bbe04444da56c3577ed32abad0477749406c75d35a0a618c"
                ),
            },
        )
        migrated_target, migrated_flows = load_mixed_flow_stages(migrated_name)
        assert migrated_target.bundle_name == "glycerol_gaff2_am1bcc_obc1"
        assert eqx.tree_equal(tanh_flow, migrated_flows[0])

        forged_migration = root / "forged_migration"
        write_run(
            forged_migration,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={
                "bundle": "glycerol_gaff2_am1bcc_obc1_v2",
                "manifest_sha256": (
                    "c60f520ef8d4146a0ffb4ff8875d56b90abebd47630d842780e7b3d71911c736"
                ),
                "jflows_md_source_sha256": "0" * 64,
            },
        )
        expect_rejected(forged_migration, "source hashes do not match")

        invalid_mask = np.asarray(
            mixed_flow_metadata(tanh_flow)["condition_mask"]
        ).copy()
        invalid_mask[0] = True
        invalid = root / "invalid_mask"
        write_run(
            invalid,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"condition_mask": invalid_mask},
        )
        expect_rejected(invalid, "condition_mask")

        unknown = root / "unknown"
        write_run(
            unknown,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"activation_id": "unknown.activation"},
        )
        expect_rejected(unknown, "unsupported saved Mixed_NSF activation")

        invalid_dtype = root / "invalid_dtype"
        write_run(
            invalid_dtype,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"parameter_dtype": "float16"},
        )
        expect_rejected(invalid_dtype, "unsupported saved Mixed_NSF dtype")

        downgraded = root / "downgraded"
        write_run(downgraded, target, tanh_flow, flow_key, schema_version=2)
        marker_path = downgraded / "COMPLETE.json"
        marker = json.loads(marker_path.read_text())
        marker["schema_version"] = 1
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        expect_rejected(downgraded, "disagree on schema version")

        for label, invalid_schema in (("boolean_schema", True), ("float_schema", 1.0)):
            invalid = root / label
            write_run(invalid, target, default_flow, flow_key, schema_version=1)
            marker_path = invalid / "COMPLETE.json"
            marker = json.loads(marker_path.read_text())
            marker["schema_version"] = invalid_schema
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            expect_rejected(invalid, "unsupported molecular run schema")

        zero_stages = root / "zero_stages"
        write_run(
            zero_stages,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"stage_count": 0},
        )
        expect_rejected(zero_stages, "positive stage_count")

        fractional_stages = root / "fractional_stages"
        write_run(
            fractional_stages,
            target,
            tanh_flow,
            flow_key,
            schema_version=2,
            overrides={"stage_count": 1.5},
        )
        expect_rejected(fractional_stages, "positive stage_count")

        trailing_flow = root / "trailing_flow"
        _, trailing_path = write_run(
            trailing_flow, target, tanh_flow, flow_key, schema_version=2
        )
        eqx.tree_serialise_leaves(trailing_path, [tanh_flow, tanh_flow])
        refresh_flow_hash(trailing_flow)
        expect_rejected(trailing_flow, "declared stage_count")

        assert jnp.asarray(samples).dtype == jnp.float32

    print("PASS molecular artifacts: schema-1 contract plus schema-2 architecture/counts")


if __name__ == "__main__":
    main()
