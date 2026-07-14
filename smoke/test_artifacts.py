#!/usr/bin/env python
"""Round-trip checks for the current mixed-flow artifact format."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import (  # noqa: E402
    Mixed_NSF,
    Molecular_Potential,
    load_mixed_flow,
    mixed_flow_metadata,
    save_mixed_flow,
)


def _expect_value_error(path: Path, field: str, value) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    try:
        load_mixed_flow(path)
    except (TypeError, ValueError):
        return
    raise AssertionError(f"invalid mixed-flow metadata was accepted: {field}")


def main() -> None:
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
    key = jax.random.key(80)
    flow = Mixed_NSF(
        key,
        target.domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
        activation=jax.nn.tanh,
        mask_strategy="balanced",
    ).zeros()
    metadata = mixed_flow_metadata(flow)
    assert metadata["artifact_schema"] == 1
    assert metadata["flow_type"] == "Mixed_NSF"
    assert metadata["parameter_dtype"] == "float32"
    assert metadata["activation_id"] == "jax.nn.tanh"
    assert metadata["mask_strategy"] == "balanced"
    assert np.asarray(metadata["condition_mask"]).shape == (
        flow.transforms,
        flow.domain.dimension,
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "flow.eqx"
        assert save_mixed_flow(path, flow) == path.resolve()
        loaded = load_mixed_flow(path)
        assert eqx.tree_equal(flow, loaded)

        samples = target.source().samples(jax.random.key(81), N=4)
        expected_y, expected_ladj = flow.call_and_ladj(samples)
        actual_y, actual_ladj = loaded.call_and_ladj(samples)
        np.testing.assert_array_equal(actual_y, expected_y)
        np.testing.assert_array_equal(actual_ladj, expected_ladj)
        expected_x, expected_inverse_ladj = flow.inv_and_ladj(expected_y)
        actual_x, actual_inverse_ladj = loaded.inv_and_ladj(expected_y)
        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_inverse_ladj, expected_inverse_ladj)

        for field, value in (
            ("artifact_schema", 2),
            ("flow_type", "other"),
            ("activation_id", "jax.nn.relu"),
            ("parameter_dtype", "float16"),
        ):
            invalid = root / f"invalid_{field}.eqx"
            shutil.copy2(path, invalid)
            shutil.copy2(
                path.with_suffix(path.suffix + ".json"),
                invalid.with_suffix(invalid.suffix + ".json"),
            )
            _expect_value_error(invalid, field, value)

        missing_metadata = root / "missing_metadata.eqx"
        shutil.copy2(path, missing_metadata)
        try:
            load_mixed_flow(missing_metadata)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("flow without architecture metadata was accepted")

    try:
        save_mixed_flow("not-used.eqx", object())
    except TypeError:
        pass
    else:
        raise AssertionError("save_mixed_flow accepted a non-Mixed_NSF object")

    unsupported = Mixed_NSF(
        key,
        target.domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
        activation=jax.nn.relu,
    )
    try:
        mixed_flow_metadata(unsupported)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered activation was accepted")

    print("PASS current mixed-flow artifact save/load contract")


if __name__ == "__main__":
    main()
