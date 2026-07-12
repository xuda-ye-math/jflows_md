#!/usr/bin/env python
"""Round-trip and integrity checks for versioned molecular flow artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from jflows_md import (  # noqa: E402
    Mixed_NSF,
    Molecular_Potential,
    load_mixed_flow_stages,
    package_source_sha256,
)
from jflows_md.system import sha256_file  # noqa: E402


def main() -> None:
    target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1_v1")
    flow_key = jax.random.key(80)
    flow = Mixed_NSF(
        flow_key,
        target.domain,
        bins=4,
        transforms=2,
        hidden_features=(8, 8),
    ).zeros()
    jflows_hash = package_source_sha256("jflows")
    md_hash = package_source_sha256("jflows_md")

    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "run"
        run.mkdir()
        data_path = run / "data.npz"
        flow_path = run / "flows.eqx"
        np.savez_compressed(
            data_path,
            bundle=target.bundle_name,
            manifest_sha256=target.manifest_sha256,
            stage_count=1,
            bins=4,
            transforms=2,
            hidden_features=np.asarray((8, 8)),
            slope=1e-3,
            nsf_lim=5.0,
            flow_key_data=np.asarray(jax.random.key_data(flow_key)),
            jflows_source_sha256=jflows_hash,
            jflows_md_source_sha256=md_hash,
        )
        eqx.tree_serialise_leaves(flow_path, [flow])
        status_path = run / "train_status.log"
        status_path.write_text("DONE\n", encoding="utf-8")
        marker = {
            "schema_version": 1,
            "data_sha256": sha256_file(data_path),
            "flows_sha256": sha256_file(flow_path),
            "status_sha256": sha256_file(status_path),
        }
        (run / "COMPLETE.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )

        loaded_target, loaded_flows = load_mixed_flow_stages(run)
        assert loaded_target.manifest_sha256 == target.manifest_sha256
        assert len(loaded_flows) == 1
        assert eqx.tree_equal(flow, loaded_flows[0])

        with data_path.open("ab") as handle:
            handle.write(b"corruption")
        try:
            load_mixed_flow_stages(run)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("artifact corruption was not detected")

    print("PASS molecular artifact round-trip, source hashes, and integrity gate")


if __name__ == "__main__":
    main()
