#!/usr/bin/env python
"""Opt-in cold/warm JAX compilation benchmark for ``jflows``/``jflows_md``.

The controller starts one fresh Python subprocess per cell and gives every
worker a fresh persistent-compilation-cache directory.  Consequently
``cold_compile_first_ms`` includes tracing, lowering, backend compilation, and
the first synchronized execution without reusing an executable from another
cell.  ``warm_mean_ms`` measures synchronized calls to that executable.

The bounded default grid covers:

* a local :class:`jflows.flow.NCSF` map;
* a :class:`jflows_md.Mixed_NSF` map on glycerol's ``R^25 x T^11`` domain;
* one complete public :func:`jflows.train.train_forward_KLX_G` step, including
  source-particle construction and its AIS path, a biased, score-free,
  final-target-rejuvenated sampling surrogate;
* one complete public
  :func:`jflows_md.train.train_molecular_forward_KLX_G` step, including
  construction of its supplied mixed-domain particle set;
* glycerol's real :class:`jflows_md.Molecular_Potential` energy;
* the same energy together with its batched gradient; and
* one small, compiled mixed-MALA chunk (two iterations, no SMC/training).

The generic trainer's AIS path is intentionally a biased, score-free,
final-target-rejuvenated surrogate and is distinct from classical SMC: its
nominal incremental weights are followed by
rejuvenation at the final target at every level, while classical SMC uses the
matching intermediate potential at each level.  No classical SMC workload is
timed here.

This script is deliberately not imported by ``run_all.py``.  Run it manually
only when compile latency is being investigated; use ``--quick`` for one cell
per workload and ``--timeout`` to cap each fresh worker.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DEFAULT_CSV = SCRIPT.with_name("compile_benchmark.csv")
RESULT_PREFIX = "JFLOWS_COMPILE_RESULT="

# These grids are intentionally much smaller than training configurations.
# Flow batches expose shape sensitivity without constructing validation/pool
# sets; molecular batches are kept tiny because force/gradient graphs dominate
# compile cost independently of a realistic training population.
FLOW_BATCHES: tuple[int, ...] = (32, 128)
MOLECULAR_BATCHES: tuple[int, ...] = (1, 4)
FLOW_MODELS: tuple[tuple[str, tuple[int, ...], int], ...] = (
    ("small", (32, 32), 2),
    ("medium", (64, 64), 4),
)
FLOW_DIMENSION = 36
FLOW_BINS = 8
GLYCEROL_BUNDLE = "glycerol_gaff2_am1bcc_obc1"
GLYCEROL_EUCLIDEAN = 25
GLYCEROL_PERIODIC = 11
MALA_ITERS = 2
MALA_STEP = 1e-8
MALA_IMAGES = 3
TRAIN_VALID = 16
TRAIN_BATCH = 8
TRAIN_STEPS = 1
TRAIN_LADDER = 2
TRAIN_MC_STEP = 1e-3
TRAIN_MC_ITERS = 1

WORKLOADS = (
    "jflows_ncsf",
    "jflows_md_mixed_nsf",
    "jflows_train_forward_klx_g",
    "jflows_md_train_molecular_forward_klx_g",
    "glycerol_energy",
    "glycerol_energy_grad",
    "glycerol_mixed_mala_chunk",
)


@dataclass(frozen=True)
class Cell:
    workload: str
    batch: int
    model_size: str
    dimension: int
    hidden_features: tuple[int, ...] = ()
    transforms: int = 0
    bins: int = 0
    data_samples: int = 0
    trainer_steps: int = 0
    ladder: int = 0
    mc_iters: int = 0

    @property
    def label(self) -> str:
        return f"{self.workload}[{self.model_size},batch={self.batch}]"


CSV_FIELDS = (
    "status",
    "workload",
    "model_size",
    "dimension",
    "batch",
    "hidden_features",
    "transforms",
    "bins",
    "data_samples",
    "trainer_steps",
    "ladder",
    "mc_iters",
    "array_elements",
    "backend",
    "device",
    "jax_version",
    "dtype",
    "peak_host_rss_mb",
    "peak_gpu_in_use_mb",
    "peak_gpu_reserved_mb",
    "cold_compile_first_ms",
    "warm_mean_ms",
    "warm_std_ms",
    "warm_min_ms",
    "warm_repeats",
    "worker_wall_s",
    "timeout_s",
    "error",
)


def _cells(selected: Sequence[str], quick: bool) -> list[Cell]:
    cells: list[Cell] = []
    if "jflows_ncsf" in selected:
        for size, hidden, transforms in FLOW_MODELS:
            for batch in FLOW_BATCHES:
                cells.append(
                    Cell(
                        "jflows_ncsf",
                        batch,
                        f"d{FLOW_DIMENSION}_{size}",
                        FLOW_DIMENSION,
                        hidden,
                        transforms,
                        FLOW_BINS,
                    )
                )
    if "jflows_md_mixed_nsf" in selected:
        for size, hidden, transforms in FLOW_MODELS:
            for batch in FLOW_BATCHES:
                cells.append(
                    Cell(
                        "jflows_md_mixed_nsf",
                        batch,
                        f"R{GLYCEROL_EUCLIDEAN}xT{GLYCEROL_PERIODIC}_{size}",
                        GLYCEROL_EUCLIDEAN + GLYCEROL_PERIODIC,
                        hidden,
                        transforms,
                        FLOW_BINS,
                    )
                )
    if "jflows_train_forward_klx_g" in selected:
        for size, hidden, transforms in FLOW_MODELS:
            cells.append(
                Cell(
                    "jflows_train_forward_klx_g",
                    TRAIN_BATCH,
                    f"d{FLOW_DIMENSION}_{size}",
                    FLOW_DIMENSION,
                    hidden,
                    transforms,
                    FLOW_BINS,
                    TRAIN_VALID,
                    TRAIN_STEPS,
                    TRAIN_LADDER,
                    TRAIN_MC_ITERS,
                )
            )
    if "jflows_md_train_molecular_forward_klx_g" in selected:
        for size, hidden, transforms in FLOW_MODELS:
            cells.append(
                Cell(
                    "jflows_md_train_molecular_forward_klx_g",
                    TRAIN_BATCH,
                    f"R{GLYCEROL_EUCLIDEAN}xT{GLYCEROL_PERIODIC}_{size}",
                    GLYCEROL_EUCLIDEAN + GLYCEROL_PERIODIC,
                    hidden,
                    transforms,
                    FLOW_BINS,
                    TRAIN_VALID,
                    TRAIN_STEPS,
                    0,
                    0,
                )
            )
    for workload in ("glycerol_energy", "glycerol_energy_grad"):
        if workload in selected:
            for batch in MOLECULAR_BATCHES:
                cells.append(
                    Cell(workload, batch, "glycerol_36d", 36)
                )
    if "glycerol_mixed_mala_chunk" in selected:
        for batch in MOLECULAR_BATCHES:
            cells.append(
                Cell(
                    "glycerol_mixed_mala_chunk",
                    batch,
                    f"glycerol_36d_iters{MALA_ITERS}",
                    36,
                )
            )
    if not quick:
        return cells

    # One smallest cell per requested workload gives a fast path check while
    # preserving the fresh-process/cold-cache timing semantics.
    first: dict[str, Cell] = {}
    for cell in cells:
        first.setdefault(cell.workload, cell)
    return [first[name] for name in selected]


def _worker_environment(cache_dir: str) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_ENABLE_X64"] = "false"
    # A unique cache directory is essential: a fresh process alone could load
    # an executable from JAX's persistent cache and cease to be a cold cell.
    env["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    return env


def _tail(text: str | None, limit: int = 4000) -> str:
    value = (text or "").strip()
    return value if len(value) <= limit else "..." + value[-limit:]


def _decode_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _fallback_row(
    cell: Cell,
    *,
    status: str,
    timeout: float,
    wall: float,
    error: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "workload": cell.workload,
        "model_size": cell.model_size,
        "dimension": cell.dimension,
        "batch": cell.batch,
        "hidden_features": "x".join(map(str, cell.hidden_features)),
        "transforms": cell.transforms,
        "bins": cell.bins,
        "data_samples": cell.data_samples,
        "trainer_steps": cell.trainer_steps,
        "ladder": cell.ladder,
        "mc_iters": cell.mc_iters,
        "worker_wall_s": f"{wall:.3f}",
        "timeout_s": f"{timeout:.3f}",
        "error": error,
    }


def _run_cell(
    cell: Cell, *, warm_repeats: int, timeout: float
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--spec-json",
        json.dumps(asdict(cell), separators=(",", ":")),
        "--warm-repeats",
        str(warm_repeats),
    ]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="jflows-compile-cache-") as cache_dir:
        process = subprocess.Popen(
            command,
            cwd=ROOT.parent,
            env=_worker_environment(cache_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the complete worker process group so a backend compiler
            # helper cannot survive a timeout and retain host/device memory.
            _kill_process_group(process)
            stdout, stderr = process.communicate()
            wall = time.perf_counter() - started
            detail = _tail(stderr) or _tail(stdout) or "worker exceeded timeout"
            return _fallback_row(
                cell,
                status="timeout",
                timeout=timeout,
                wall=wall,
                error=detail,
            )
        except KeyboardInterrupt:
            _kill_process_group(process)
            process.communicate()
            raise
    wall = time.perf_counter() - started
    decoded = _decode_result(stdout)
    if decoded is None:
        detail = _tail(stderr) or _tail(stdout) or "worker returned no result record"
        return _fallback_row(
            cell,
            status="error",
            timeout=timeout,
            wall=wall,
            error=f"returncode={process.returncode}: {detail}",
        )
    decoded["worker_wall_s"] = f"{wall:.3f}"
    decoded["timeout_s"] = f"{timeout:.3f}"
    if process.returncode and decoded.get("status") == "ok":
        decoded["status"] = "error"
        decoded["error"] = _tail(stderr) or f"worker returncode={process.returncode}"
    elif stderr and decoded.get("status") != "ok":
        decoded["error"] = decoded.get("error") or _tail(stderr)
    return decoded


def _synchronize(value: Any) -> None:
    import jax

    jax.block_until_ready(value)


def _array_elements(model: Any) -> int:
    import equinox as eqx
    import jax

    return sum(
        int(leaf.size)
        for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_array))
        if eqx.is_array(leaf)
    )


def _molecular_points(potential: Any, batch: int, key: Any) -> Any:
    import jax
    import jax.numpy as jnp

    reference = potential.reference_internal()
    noise = 1e-3 * jax.random.normal(key, (batch, potential.dimension))
    return potential.domain.wrap(jnp.broadcast_to(reference, noise.shape) + noise)


def _build_operation(cell: Cell):
    """Return ``(compiled_callable, arguments, model, validator)``."""

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    if cell.workload == "jflows_ncsf":
        from jflows.flow import NCSF

        lower = -jnp.pi * jnp.ones((cell.dimension,))
        upper = jnp.pi * jnp.ones((cell.dimension,))
        flow = NCSF(
            jax.random.key(1),
            lower,
            upper,
            bins=cell.bins,
            transforms=cell.transforms,
            hidden_features=cell.hidden_features,
        )
        samples = jax.random.uniform(
            jax.random.key(2),
            (cell.batch, cell.dimension),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        operation = eqx.filter_jit(
            lambda candidate, value: candidate.call_and_ladj(value)
        )

        def validate(output):
            mapped, ladj = output
            assert mapped.shape == samples.shape and ladj.shape == (cell.batch,)

        return operation, (flow, samples), flow, validate, samples.dtype

    if cell.workload == "jflows_md_mixed_nsf":
        from jflows_md import Mixed_NSF
        from jflows_md.core.domain import Mixed_Domain

        domain = Mixed_Domain(GLYCEROL_EUCLIDEAN, GLYCEROL_PERIODIC)
        flow = Mixed_NSF(
            jax.random.key(3),
            domain,
            bins=cell.bins,
            transforms=cell.transforms,
            hidden_features=cell.hidden_features,
        )
        euclidean_key, periodic_key = jax.random.split(jax.random.key(4))
        samples = jnp.concatenate(
            (
                jax.random.normal(
                    euclidean_key,
                    (cell.batch, domain.euclidean_dim),
                ),
                jax.random.uniform(
                    periodic_key,
                    (cell.batch, domain.periodic_dim),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                ),
            ),
            axis=-1,
        )
        operation = eqx.filter_jit(
            lambda candidate, value: candidate.call_and_ladj(value)
        )

        def validate(output):
            mapped, ladj = output
            assert mapped.shape == samples.shape and ladj.shape == (cell.batch,)

        return operation, (flow, samples), flow, validate, samples.dtype

    if cell.workload == "jflows_train_forward_klx_g":
        from jflows.flow import NCSF
        from jflows.potential import Nlog_Uniform, potential_from
        from jflows.train import train_forward_KLX_G

        lower = -jnp.pi * jnp.ones((cell.dimension,))
        upper = jnp.pi * jnp.ones((cell.dimension,))
        source = Nlog_Uniform(lower, upper)
        phase = jnp.linspace(-0.35, 0.35, cell.dimension)
        amplitude = jnp.linspace(0.05, 0.15, cell.dimension)
        target = potential_from(
            lambda value: jnp.sum(
                amplitude * (1.0 - jnp.cos(value - phase)), axis=-1
            )
        )
        flow = NCSF(
            jax.random.key(10),
            lower,
            upper,
            bins=cell.bins,
            transforms=cell.transforms,
            hidden_features=cell.hidden_features,
        ).zeros()
        sample_key = jax.random.key(11)

        def operation(candidate, key):
            # Keep construction of x_valid inside the timing boundary.  The
            # public trainer then creates its target-like batch with the
            # current flow's score-free, final-target-rejuvenated surrogate
            # before invoking the generic trainer's compiled stage scan.
            x_valid = source.samples(key, cell.data_samples)
            return train_forward_KLX_G(
                x_valid,
                source,
                target,
                candidate,
                n_batch=cell.batch,
                steps=cell.trainer_steps,
                lr=1e-3,
                ladder=cell.ladder,
                mc_step=TRAIN_MC_STEP,
                mc_iters=cell.mc_iters,
                coeff_lambda=1.0,
                mc_adjust=False,
                checkpoint=False,
                e_clip=1000.0,
                g_clip=100.0,
                seed=12,
            )

        def validate(output):
            trained, ess = output
            assert ess.shape == (cell.trainer_steps,)
            assert type(trained) is type(flow)

        return operation, (flow, sample_key), flow, validate, lower.dtype

    if cell.workload == "jflows_md_train_molecular_forward_klx_g":
        from jflows.potential import potential_from
        from jflows_md import Mixed_NSF, Molecular_Source
        from jflows_md.core.domain import Mixed_Domain
        from jflows_md.train import train_molecular_forward_KLX_G

        domain = Mixed_Domain(GLYCEROL_EUCLIDEAN, GLYCEROL_PERIODIC)
        source = Molecular_Source(domain)
        euclidean_center = jnp.linspace(
            -0.2, 0.2, domain.euclidean_dim
        )
        torsion_center = jnp.linspace(-0.4, 0.4, domain.periodic_dim)
        target = potential_from(
            lambda value: 0.5
            * jnp.sum(
                (value[:, : domain.euclidean_dim] - euclidean_center) ** 2,
                axis=-1,
            )
            + 0.1
            * jnp.sum(
                1.0
                - jnp.cos(
                    value[:, domain.euclidean_dim :] - torsion_center
                ),
                axis=-1,
            )
        )
        flow = Mixed_NSF(
            jax.random.key(20),
            domain,
            bins=cell.bins,
            transforms=cell.transforms,
            hidden_features=cell.hidden_features,
        ).zeros()
        sample_key = jax.random.key(21)

        def operation(candidate, key):
            # The molecular trainer consumes an existing particle set rather
            # than running AIS, so construct that data inside the measured
            # public one-step call as part of the end-to-end workload.
            samples = source.samples(key, N=cell.data_samples)
            return train_molecular_forward_KLX_G(
                samples,
                samples,
                source,
                target,
                candidate,
                n_batch=cell.batch,
                steps=cell.trainer_steps,
                lr=1e-3,
                coeff_lambda=1.0,
                energy_origin=0.0,
                e_clip=1000.0,
                g_clip=100.0,
                seed=22,
                checkpoint=False,
            )

        def validate(output):
            trained, ess, kept, updated = output
            expected = (cell.trainer_steps,)
            assert ess.shape == kept.shape == updated.shape == expected
            assert type(trained) is type(flow)

        return operation, (flow, sample_key), flow, validate, source.mean.dtype

    from jflows_md import Molecular_Potential

    potential = Molecular_Potential.from_bundle(GLYCEROL_BUNDLE)
    samples = _molecular_points(potential, cell.batch, jax.random.key(5))
    if cell.workload == "glycerol_energy":
        operation = eqx.filter_jit(lambda model, value: model(value))

        def validate(output):
            assert output.shape == (cell.batch,)

        return operation, (potential, samples), potential, validate, samples.dtype

    if cell.workload == "glycerol_energy_grad":
        operation = eqx.filter_jit(
            lambda model, value: (model(value), model.grad(value))
        )

        def validate(output):
            energy, gradient = output
            assert energy.shape == (cell.batch,)
            assert gradient.shape == samples.shape

        return operation, (potential, samples), potential, validate, samples.dtype

    if cell.workload == "glycerol_mixed_mala_chunk":
        from jflows_md.utils import mixed_mala_step

        keys = jax.random.split(jax.random.key(6), MALA_ITERS)

        def one_chunk(model, state, iteration_keys):
            def body(current, key):
                updated, accepted = mixed_mala_step(
                    key,
                    current,
                    model,
                    model.domain,
                    step=MALA_STEP,
                    images=MALA_IMAGES,
                )
                return updated, accepted.astype(updated.dtype).mean()

            return jax.lax.scan(body, state, iteration_keys)

        operation = eqx.filter_jit(one_chunk)

        def validate(output):
            updated, acceptance = output
            assert updated.shape == samples.shape
            assert acceptance.shape == (MALA_ITERS,)

        return (
            operation,
            (potential, samples, keys),
            potential,
            validate,
            samples.dtype,
        )

    raise ValueError(f"unknown workload {cell.workload!r}")


def _all_finite(value: Any) -> bool:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    leaves = [leaf for leaf in jax.tree.leaves(value) if eqx.is_array(leaf)]
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def _worker(cell: Cell, warm_repeats: int) -> dict[str, Any]:
    # Import JAX only inside the worker.  The controller remains JAX-free and
    # every cell gets a genuinely fresh runtime/backend process.
    import jax

    jax.config.update("jax_enable_x64", False)
    operation, arguments, model, validate, dtype = _build_operation(cell)
    if str(dtype) != "float32":
        raise TypeError(f"benchmark requires JAX's float32 default, got {dtype}")

    # Finish model/input transfers and setup work before starting the cold-JIT
    # clock.  This keeps the metric focused on tracing/lowering/compilation and
    # the first execution, not bundle loading or deterministic initialization.
    _synchronize(arguments)

    first_started = time.perf_counter()
    first = operation(*arguments)
    _synchronize(first)
    cold_ms = (time.perf_counter() - first_started) * 1000.0
    validate(first)
    if not _all_finite(first):
        raise FloatingPointError("compiled result contains a nonfinite value")

    warm_ms = []
    for _ in range(warm_repeats):
        started = time.perf_counter()
        output = operation(*arguments)
        _synchronize(output)
        warm_ms.append((time.perf_counter() - started) * 1000.0)
    validate(output)
    if not _all_finite(output):
        raise FloatingPointError("warm result contains a nonfinite value")

    device = jax.devices()[0]
    memory = device.memory_stats() or {}
    return {
        "status": "ok",
        "workload": cell.workload,
        "model_size": cell.model_size,
        "dimension": cell.dimension,
        "batch": cell.batch,
        "hidden_features": "x".join(map(str, cell.hidden_features)),
        "transforms": cell.transforms,
        "bins": cell.bins,
        "data_samples": cell.data_samples,
        "trainer_steps": cell.trainer_steps,
        "ladder": cell.ladder,
        "mc_iters": cell.mc_iters,
        "array_elements": _array_elements(model),
        "backend": jax.default_backend(),
        "device": getattr(device, "device_kind", str(device)),
        "jax_version": jax.__version__,
        "dtype": str(dtype),
        "peak_host_rss_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:.3f}",
        "peak_gpu_in_use_mb": f"{memory.get('peak_bytes_in_use', 0) / 2**20:.3f}",
        "peak_gpu_reserved_mb": f"{memory.get('peak_bytes_reserved', 0) / 2**20:.3f}",
        "cold_compile_first_ms": f"{cold_ms:.3f}",
        "warm_mean_ms": f"{statistics.fmean(warm_ms):.3f}",
        "warm_std_ms": f"{statistics.pstdev(warm_ms):.3f}",
        "warm_min_ms": f"{min(warm_ms):.3f}",
        "warm_repeats": warm_repeats,
        "error": "",
    }


def _worker_main(spec_json: str, warm_repeats: int) -> int:
    payload = json.loads(spec_json)
    payload["hidden_features"] = tuple(payload.get("hidden_features", ()))
    cell = Cell(**payload)
    try:
        result = _worker(cell, warm_repeats)
        code = 0
    except Exception as exc:  # worker errors must still become a CSV row
        result = _fallback_row(
            cell,
            status="error",
            timeout=math.nan,
            wall=math.nan,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        code = 1
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)
    return code


def _controller(args: argparse.Namespace) -> int:
    selected = tuple(dict.fromkeys(args.workload or WORKLOADS))
    unknown = sorted(set(selected) - set(WORKLOADS))
    if unknown:
        raise SystemExit(f"unknown workload(s): {', '.join(unknown)}")
    cells = _cells(selected, args.quick)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Opt-in jflows/jflows_md cold-compile benchmark", flush=True)
    print(
        f"cells={len(cells)} warm_repeats={args.warm_repeats} "
        f"timeout={args.timeout:.0f}s/cell output={output}",
        flush=True,
    )
    print(
        "Each cell uses a fresh process and fresh JAX persistent-cache directory.",
        flush=True,
    )

    failures = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        handle.flush()
        for index, cell in enumerate(cells, 1):
            print(
                f"[{index}/{len(cells)}] START {cell.label} "
                f"(timeout {args.timeout:.0f}s)",
                flush=True,
            )
            row = _run_cell(
                cell,
                warm_repeats=args.warm_repeats,
                timeout=args.timeout,
            )
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            handle.flush()
            status = row.get("status", "error")
            if status == "ok":
                print(
                    f"[{index}/{len(cells)}] OK    {cell.label}: "
                    f"cold+first={row['cold_compile_first_ms']} ms, "
                    f"warm={row['warm_mean_ms']} ms",
                    flush=True,
                )
            else:
                failures += 1
                print(
                    f"[{index}/{len(cells)}] {str(status).upper():<7} "
                    f"{cell.label}: {_tail(str(row.get('error', '')))}",
                    flush=True,
                )

    print(
        f"DONE: {len(cells) - failures} ok, {failures} failed/timed out; CSV={output}",
        flush=True,
    )
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        action="append",
        choices=WORKLOADS,
        help="benchmark only this workload (repeatable; default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only the smallest cell of each selected workload",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=10,
        help="synchronized warm executions per cell (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=480.0,
        help="hard wall-clock limit for each fresh worker in seconds (default: 480)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV,
        help=f"incremental CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec-json", default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warm_repeats < 1:
        raise SystemExit("--warm-repeats must be at least one")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.worker:
        if not args.spec_json:
            raise SystemExit("worker mode requires --spec-json")
        return _worker_main(args.spec_json, args.warm_repeats)
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
