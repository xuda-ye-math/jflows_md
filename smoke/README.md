# jflows_md smoke tests

Run the complete pure-JAX suite from the repository root:

```bash
conda activate jflows
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

This assumes the sibling `jflows` and current `jflows_md` checkouts were
registered with `conda develop` as described in the root README. To test
unregistered live trees instead, set
`PYTHONPATH=/path/to/jflows:/path/to/jflows_md` explicitly.

The tests verify bundle hashes and metadata, pure-JAX energies and forces
against stored OpenMM Reference results for all three molecules, BAT/chart
round trips and Jacobians, ADP L-only support, full small-molecule parity
support, source shapes, JIT compatibility, and the flat `jflows_md.utils`
interface for mixed-domain MALA and a two-level potential-space SMC bridge. The
`Mixed_NSF` regression checks cover identity initialization, round trips,
log-determinants, periodic-representative invariance, seam continuity, and
finite gradients. A public compatibility test verifies the committed `jflows`
signatures and compiled stage-training scheme, chunked public sampling calls,
and the transform primitives used by `jflows_md`. Molecular controller tests
verify fixed-shape chunk execution, disjoint PRNG streams, versioned artifact
round trips, and the default float32 runtime. A bounded two-particle
glycerol test compiles the real energy/gradient and one MALA step. A tiny
synthetic `R^2 x T^1` case compiles and runs the
mixed KL+X trainer, one adaptive Boltzmann-generator stage, and the G-native
score-free AIS surrogate through a nonidentity mixed flow; it does
not launch a molecular training run.

Rebuild the bundles only when their versioned model definition changes:

```bash
conda activate jflows
python bundles/build_molecular_bundles.py
```

## Opt-in compilation benchmark

`benchmark_compile.py` is intentionally excluded from `run_all.py`. It measures
cold compile plus first execution and warm execution for local `jflows` NCSF,
`jflows_md` Mixed_NSF, glycerol energy, glycerol energy plus gradient, one
small compiled mixed-MALA chunk, and two bounded end-to-end public trainer
paths:

- `jflows.train.train_forward_KLX_G` on a 36-dimensional periodic toy target;
- `jflows_md.train.train_molecular_forward_KLX_G` on a synthetic
  `R^25 x T^11` target.

Both trainer workloads use float32, 16 source particles, a batch of 8, one
Adam step, and both small (two transforms, width 32) and medium (four
transforms, width 64) flows. The generic trainer additionally uses two levels
and one ULA step per level. Its AIS path is a biased, score-free,
final-target-rejuvenated target surrogate: nominal incremental weights are
paired with rejuvenation at the final target at every level. Classical SMC is a
separate algorithm whose rejuvenation follows the matching intermediate
potential; this benchmark does not run classical SMC. The mixed-domain
molecular trainer consumes a supplied particle set and therefore has no AIS
or SMC stage. These are synthetic compilation workloads only: no real
molecular training or scaled run is launched.

Start with one bounded cell per workload:

```bash
conda activate jflows
python smoke/benchmark_compile.py --quick
```

Run the full bounded grid only when wanted:

```bash
conda activate jflows
python smoke/benchmark_compile.py --timeout 480 --warm-repeats 10
```

Each cell runs in a fresh subprocess with a fresh JAX persistent-cache
directory, synchronizes every timed output, and is killed as a process group if
its timeout expires. Results are flushed after every cell to
`smoke/compile_benchmark.csv`, so a timeout or interrupted sweep
still leaves usable partial data. The CSV records success/timeout status,
backend, device, float dtype, batch and model size, array-element count,
cold-compile-plus-first latency, warm mean/standard deviation/minimum, trainer
data/step/level settings, and any worker error. For the trainer cells, the cold
timing starts before source-particle construction and ends only after the
public trainer result is synchronized, so it includes sampler/data
construction and compilation plus execution of the optimizer kernel. Warm
calls repeat the complete public path and synchronize before recording time.
The CSV also records process peak host RSS, backend-reported peak GPU bytes in
use/reserved, and `JAX_DISABLE_MOST_OPTIMIZATIONS` when that option is supplied
in the launch environment. Use repeatable `--workload` arguments to select
individual workloads; `--help` lists their exact names.

`compile_benchmark.csv` is a local generated artifact rather than a distributed
baseline: timings depend on the exact package revision, JAX/XLA build, cache,
and device. Regenerate it after any execution-strategy change instead of
comparing against a CSV produced by another revision. This opt-in benchmark is
not part of the scientific smoke suite.
