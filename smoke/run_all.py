#!/usr/bin/env python
"""Run every local molecular smoke test in a fresh Python subprocess."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
TESTS = (
    "test_bundles.py",
    "test_api_consistency.py",
    "test_artifacts.py",
    "test_jflows_compatibility.py",
    "test_mixed_nsf.py",
    "test_jflows_md_chunking.py",
    "test_float32_training.py",
    "test_mixed_training.py",
    "test_molecular_potential.py",
    "test_support_and_utils.py",
    "test_float32_glycerol_compile.py",
)


def main() -> None:
    for test in TESTS:
        print(f"=== {test} ===", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / test)], check=True, timeout=300
        )
    print("ALL jflows_md SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
