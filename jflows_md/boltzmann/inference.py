"""Inference scheme of a stored molecular Boltzmann generator run.

``run_inference`` replays the frozen stage maps of a complete run on fresh
source particles (Algorithm "inference" of the manuscript): for every stage
``k`` it pushes the population through the stored selected map ``G_k^{-1}``,
evaluates the stage weight ``w_k`` on the pushforward, resamples with the
screened weights (multinomial, inverse CDF), and rejuvenates every particle
with ``mc_steps`` MALA steps under the stage target ``U_{t_k,t_k} = (1 - t_k) U_0 + t_k U^{rho_{t_k}}``.
No training update takes place. Populations are processed in chunks and kept
on disk as ``.npy`` memmaps, so the sample count is not limited by device
memory. The screen fraction defaults to the package's ``SCREEN_FRACTION``.

Outputs below ``output_dir``: ``run.json`` (manifest), one
``stage_XXXXXX/samples.npy`` with ``log_weights.npy`` (the stage weights on
the pushforward) and ``metadata.json`` (screened ESS, MALA acceptance) per
stage. The last stage's samples are the inference set for the target.
"""

import json
import math
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


from ..utils.rejuvenation import mixed_mala
from ..utils.screen import (
    SCREEN_FRACTION,
    compute_ESS_log,
    linear_weights_from_log,
)
from . import _diagonal, _rg
from .load import _stage, load_stage_flow, validate


__all__ = ["run_inference"]


@eqx.filter_jit
def _pushforward_chunk(samples, source, target, flow, domain):
    proposal, ladj = flow.inv_and_ladj(samples)
    proposal = domain.wrap(proposal)
    return proposal, source(samples) - target(proposal) + ladj


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ranges(total: int, chunk_size: int):
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def _memmap(path: Path, shape, dtype=np.float32):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def run_inference(
    run_dir,
    output_dir,
    template,
    source,
    target,
    *,
    sample_count: int,
    mc_dt: float,
    mc_steps: int,
    chunk_size: int = 10_000,
    mc_image_radius: int = 3,
    seed: int = 0,
    screen_fraction: float = SCREEN_FRACTION,
    max_stages: int | None = None,
    printer=print,
) -> dict:
    """Replay the frozen stage maps of ``run_dir`` on ``sample_count`` source particles.

    ``template`` is the flow template of the run (as for ``load_stage_flow``),
    ``source`` and ``target`` the potentials the run was trained with. Returns
    the manifest; the inference set is ``stage_<last>/samples.npy`` under
    ``output_dir``. Refuses to write into a non-empty ``output_dir``.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"inference root is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_a = output_dir / "work_a.npy"   # resampled population of the current stage
    work_b = output_dir / "work_b.npy"   # pushforward, then the MALA output, of the current stage
    record = validate(run_dir)
    if record["status"] != "complete" or not record["stages"]:
        raise ValueError("the stored run is not complete")
    items = record["stages"][: max_stages] if max_stages else record["stages"]
    domain = target.domain
    dimension = domain.dimension
    shape = (sample_count, dimension)
    ranges = _ranges(sample_count, chunk_size)
    base_key = jax.random.key(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed]))

    def log(message):
        printer(f"[{time.strftime('%H:%M:%S')}] {message}")

    manifest = {
        "format": "jflows-md-inference-2",
        "status": "running",
        "training_run": str(run_dir),
        "sample_count": sample_count,
        "dimension": dimension,
        "config": {
            "seed": seed, "chunk_size": chunk_size, "mc_dt": mc_dt,
            "mc_steps": mc_steps, "mc_image_radius": mc_image_radius,
            "screen_fraction": screen_fraction, "saved_flow_role": "selected",
        },
        "stages": [],
    }
    _atomic_json(output_dir / "run.json", manifest)
    log(
        f"START inference | samples={sample_count:,} | dimension={dimension} | "
        f"stages={len(items)} | chunk={chunk_size:,} | MALA {mc_dt}x{mc_steps} | "
        f"screen {screen_fraction:.2%}"
    )

    population = _memmap(work_a, shape)
    for index, (start, stop) in enumerate(ranges):
        values = source.samples(jax.random.fold_in(jax.random.fold_in(base_key, 101), index), N=stop - start)
        population[start:stop] = np.asarray(jax.block_until_ready(values))
    population.flush()
    del population
    current = work_a   # the file the next stage reads; a stage's samples.npy after the first stage
    rg_param_0 = record["config"].get("rg_param_0")
    rg_param_1 = record["config"].get("rg_param_1")

    def rejuvenate(source_path, destination_path, potential, namespace):
        """``mc_steps`` MALA steps of every particle of ``source_path`` under ``potential``."""
        origin = np.load(source_path, mmap_mode="r")
        moved = _memmap(destination_path, shape)
        acceptance = np.zeros(mc_steps, dtype=np.float64)
        for index, (start, stop) in enumerate(ranges):
            values, history = mixed_mala(
                jax.random.fold_in(jax.random.fold_in(base_key, namespace), index),
                jnp.asarray(np.asarray(origin[start:stop])), potential, domain,
                dt=mc_dt, steps=mc_steps, image_radius=mc_image_radius, chunks=1,
            )
            values = np.asarray(jax.block_until_ready(values))
            if not np.isfinite(values).all():
                raise FloatingPointError(f"MALA produced undefined coordinates at stage {number}")
            moved[start:stop] = values
            acceptance += np.asarray(history, dtype=np.float64) * (stop - start)
        moved.flush()
        del origin, moved
        return acceptance / sample_count

    def resample_to(weights, source_path, destination_path):
        cdf = np.cumsum(weights)
        if not math.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
            raise ValueError(f"invalid resampling weights at stage {number}")
        origin = np.load(source_path, mmap_mode="r")
        resampled = _memmap(destination_path, shape)
        for start, stop in ranges:
            uniforms = rng.random(stop - start) * cdf[-1]
            resampled[start:stop] = origin[np.searchsorted(cdf, uniforms, side="right")]
        resampled.flush()
        del origin, resampled

    for item in items:
        stage_started = time.perf_counter()
        number = int(item["stage"])
        saved = _stage(run_dir, item)
        t_start, t_end = float(saved["t_start"]), float(saved["t"])
        rg_start, rg_end = _rg(rg_param_0, rg_param_1, t_start), _rg(rg_param_0, rg_param_1, t_end)
        if saved["rg_start"] is not None and not (
            np.allclose(saved["rg_start"], rg_start) and np.allclose(saved["rg_end"], rg_end)
        ):
            raise ValueError(f"stage {number} regularization does not match the run config")
        source_bridge, _ = _diagonal(source, target, rg_param_0, rg_param_1, t_start)
        target_bridge, _ = _diagonal(source, target, rg_param_0, rg_param_1, t_end)
        flow = load_stage_flow(run_dir, number, "selected", template)
        stage_dir = output_dir / f"stage_{number:06d}"
        stage_dir.mkdir()
        log(f"STAGE {number:02d} START | t={t_start:.6f}->{t_end:.6f} | saved selection={saved['selected']}")

        population = np.load(current, mmap_mode="r")
        pushed = _memmap(work_b, shape)
        log_weights = np.zeros(sample_count, dtype=np.float32)
        for start, stop in ranges:
            proposal, weight = jax.block_until_ready(_pushforward_chunk(
                jnp.asarray(np.asarray(population[start:stop])), source_bridge,
                target_bridge, flow, domain,
            ))
            pushed[start:stop] = np.asarray(proposal)
            log_weights[start:stop] = np.asarray(weight)
        pushed.flush()
        del population
        np.save(stage_dir / "log_weights.npy", log_weights, allow_pickle=False)
        device_weights = jnp.asarray(log_weights)
        ess = float(compute_ESS_log(device_weights, screen_fraction))
        weights = np.asarray(linear_weights_from_log(device_weights, screen_fraction), dtype=np.float64)
        del device_weights
        log(f"stage {number:02d} pushforward ESS={ess:.6f} (screened {screen_fraction:.2%})")

        del pushed
        resample_to(weights, work_b, work_a)
        del weights
        acceptance = rejuvenate(work_a, work_b, target_bridge, 301 + number)

        samples_path = stage_dir / "samples.npy"
        os.replace(work_b, samples_path)   # the stage's population; work_b is recreated next stage
        current = samples_path

        metadata = {
            "stage": number,
            "t_start": t_start,
            "t": t_end,
            "saved_selection": saved["selected"],
            "saved_flow_path": saved["selected_flow_path"],
            "sample_count": sample_count,
            "pushforward_ess": ess,
            "mala_acceptance_mean": float(acceptance.mean()),
            "rg_start": rg_start,
            "rg_end": rg_end,
            "elapsed_seconds": time.perf_counter() - stage_started,
            "samples_path": str(samples_path.relative_to(output_dir)),
            "log_weights_path": str((stage_dir / "log_weights.npy").relative_to(output_dir)),
        }
        _atomic_json(stage_dir / "metadata.json", metadata)
        manifest["stages"].append(metadata)
        _atomic_json(output_dir / "run.json", manifest)
        log(
            f"STAGE {number:02d} COMPLETE | ESS={ess:.6f} | "
            f"MALA acceptance={float(acceptance.mean()):.4f} | "
            f"{metadata['elapsed_seconds'] / 60.0:.2f} min"
        )

    for path in (work_a, work_b):
        if path.exists():
            path.unlink()
    manifest["status"] = "complete" if items and float(items[-1]["t"]) == 1.0 else "partial"
    manifest["inference_samples_path"] = str(Path(current).relative_to(output_dir))
    _atomic_json(output_dir / "run.json", manifest)
    log(f"FINISH status={manifest['status']} | inference set {manifest['inference_samples_path']}")
    return manifest
