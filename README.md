# jflows_md

`jflows_md` 0.5.0 is the mixed-domain molecular companion to `jflows` 0.5.0.
It supplies bundle-backed molecular potentials, flows on
`R^p x T^q`, molecular KLX/KLXX training, sampling kernels, linear
regularization sharpening, and complete-stage resume.

The numerical core is intentionally a raw-performance layer. Constructors and
compiled kernels assume valid shapes, dtypes, schedules, chunk counts, and
physical parameters. Bundle construction and structural validation are
offline concerns; persistence is separate from computation.

## Reduced potential and inverse temperature

`Molecular_Potential` owns the inverse temperature. For internal coordinates
`q` and their canonical Cartesian representative `x(q)`, it evaluates

```text
U(q) = beta E(x(q)) - log J(q),
beta = 1 / (k_B T).
```

The trainers and Boltzmann controllers consume reduced potentials and do not
apply another temperature factor. A regularized potential modifies the
Cartesian energy first and then applies the same `beta`; the coordinate
Jacobian is never regularized.

```python
import jax

from jflows_md import Mixed_NSF, Molecular_Potential

target = Molecular_Potential.from_bundle("adp_ff96_obc1")
source = target.source()
samples = source.samples(jax.random.key(0), 32)

flow = Mixed_NSF(
    jax.random.key(1),
    target.domain,
    bins=32,
    transforms=6,
    hidden_features=(256, 256),
).zeros()

soft = target.regularized((50.0, 0.10))  # (e [kJ/mol], r [nm])
```

For `rg_param=(e,r)`, `r` floors only the regular and exception Amber
Coulomb/Lennard-Jones pair distances. If `E_r` is that floor-aware energy and
`E_ref,r` its value at the bundle reference, the excess `d=E_r-E_ref,r` is
mapped by

```text
R_e(d) = d                         if d <= e
       = e [1 + log(d/e)]          if d > e.
```

The regularized reduced potential is
`U_rg(q)=beta[E_ref,r+R_e(d)]-log J(q)`. The pair `(e,r)` is the complete
regularization state; no optimizer clipping parameter participates in this
definition.

## Layout

```text
jflows_md/
├── artifacts.py              generic template-based artifacts
├── boltzmann/
│   ├── __init__.py           pure adaptive BG computation
│   ├── write.py              atomic complete-stage writer
│   └── load.py               load, inspect, fork, and resume
├── bundle_build/             optional OpenMM-side construction
├── core/                     BAT, Amber/OBC, domain, and spline kernels
├── flow.py                   Mixed_Identity and Mixed_NSF
├── openmm/                   native OpenMM potential and samplers
├── potential.py              physical and regularized potentials
├── source.py                 Gaussian x uniform-torus source
├── system.py                 minimal runtime-bundle loading
├── train.py                  compiled molecular KLX/KLXX trainers
└── utils/
    ├── anneal.py             SMC and AIS
    ├── quench.py             quench-and-temper
    └── rejuvenation.py       wrapped mixed-domain MALA
```

Eager Python controllers split work into chunks; compiled kernels operate on
fixed array shapes. Filesystem paths, manifests, and resume policy never enter
the computation functions.

The complete low-/medium-/high-level interface manual is in
[`doc/`](doc/README.md). Narrow executable contracts are organized under
[`smoke/`](smoke/).

## Native OpenMM

The same bundle can instantiate an independent Cartesian OpenMM potential.
It evaluates `beta E(x)` without JAX or the internal-coordinate Jacobian, and
its `(e,r)` regularization matches `Molecular_Potential.regularized`. Native
Langevin and replica-exchange runs accept either the physical or regularized
potential.

```python
from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering

target = OpenMM_Potential.from_bundle("adp_ff96_obc1")
soft = target.regularized((50.0, 0.10))

trajectory, energy = langevin(soft, steps=10000, sample_interval=100)
replicas, energy, swap_acceptance = parallel_tempering(
    target,
    (300.0, 360.0, 432.0, 518.4),
    rounds=1000,
    steps_per_round=100,
)
```

The samplers use fresh OpenMM systems and contexts; `platform="Reference"`,
`"CPU"`, `"CUDA"`, or `"OpenCL"` can be selected explicitly.

## Direct training

`train_forward_KLX_G` and `train_forward_KLXX_G` train the molecular `G`
direction: the flow maps target-like samples toward the source and `flow.inv`
generates target-like proposals. Both functions return only
`(trained_flow, batch_ess_history)`.

```python
from jflows_md.train import train_forward_KLX_G

trained, batch_ess = train_forward_KLX_G(
    target_samples,
    source_samples,
    source,
    target,
    flow,
    batch_size=128,
    train_steps=1000,
    lr=1e-3,
)
```

Direct trainers default to `initialize_from_identity=False`. Adaptive
Boltzmann functions default to `True`, so each accepted-stage attempt starts
from an identity parameterization unless warm-starting is explicitly selected.

## Linear sharpening

Both adaptive entry points require two regularization states:

```python
from jflows_md.boltzmann import boltzmann_forward_KLX_G

particles, stages = boltzmann_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size=4096,
    batch_size=256,
    train_steps=1000,
    lr=1e-3,
    ladder=16,
    mc_dt=1e-4,
    mc_steps=4,
    rg_param_0=(20.0, 0.15),
    rg_param_1=(1000.0, 0.0),
)
```

The controller uses

```text
rg(t) = rg_param_0 + t [rg_param_1 - rg_param_0].
```

Let `U_s` be the source reduced potential. For an accepted step `a -> b`, it
trains and selects a proposal between

```text
B_a    = (1-a) U_s + a U_rg(a)
B_b^-  = (1-b) U_s + b U_rg(a).
```

It then sharpens the particles exactly to

```text
B_b^+  = (1-b) U_s + b U_rg(b)
log w_sharp = B_b^- - B_b^+,
```

followed by resampling and MALA at `B_b^+`. Thus the emitted population at
`t=1` targets `target.regularized(rg_param_1)`.

The flow and sharpening transitions form one adaptive stage. Both the
selected-flow ESS and the sharpening ESS must reach `bg_param["tau_ess"]`
(default `0.6`). If either gate fails, the controller applies the configured
`shrink_factor` to `b-a` and retries the complete stage. A stage is emitted
and made available to persistence only after both gates pass.

Each stage record deliberately distinguishes:

- `flow_rg` and `flow_endpoint="pre_sharpen"`: the law used to train the
  proposal flow;
- `population_rg`: the law of the emitted post-sharpen particles;
- `sharpen_ess_hist`: attempt-aligned sharpening ESS (`NaN` when the flow
  gate rejected before sharpening was evaluated);
- `sharpen_ess` and `sharpen_mala_acceptance`: accepted sharpening
  diagnostics.

The per-stage flows are proposal maps for this particle algorithm. They do not
compose into a deterministic source-to-final generator because sharpening is
a stochastic transition between stages.

## Complete-stage resume

Persistence mirrors `jflows` 0.5 and remains outside the compute API.
`jflows_md.boltzmann.write` atomically writes both flows, the post-sharpen
population, histories, and stage metadata before publishing the stage in
`run.json`. `jflows_md.boltzmann.load.run` resumes from the last published
stage; an incomplete unpublished directory is ignored and recomputed.

```python
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

def iterate(samples, continuation, accepted_t, start_stage):
    return iterate_boltzmann(
        samples,
        source,
        target,
        continuation,
        objective="forward_klx",
        accepted_t=accepted_t,
        start_stage=start_stage,
        # supply the same numerical controls used for a direct run
        **controls,
    )

config = {
    "target": "adp_ff96_obc1",
    "seed": 0,
    "rg_param_0": (20.0, 0.15),
    "rg_param_1": (1000.0, 0.0),
}

particles, stages = run(
    "runs/adp",
    "adp-regularization",
    config,
    x_valid,
    flow,
    iterate,
)

# After interruption, `flow` is the same architecture template.
particles, stages = run(
    "runs/adp",
    "adp-regularization",
    config,
    None,
    flow,
    iterate,
    resume=True,
)
```

Resume requires the same problem identifier, complete numerical configuration,
and exact static flow template; mismatches are rejected. The saved population,
not flow-only replay, is the continuation state. Use
`manifest`, `validate`, `load`, `load_stage_flow`,
`load_validation_samples`, `load_training_history`, and `fork` for inspection
and run management.

## Bundles and installation

Runtime bundles contain exactly:

```text
coordinates.json
manifest.json
reference.pdb
system.json
system.xml
validation.json
```

A source checkout includes three named bundles: alanine dipeptide, glycerol,
and diethanolamine. Wheels contain Python code only, so installed users pass an
external bundle directory. `available_bundles()` returns an empty tuple when
the default data directory is absent.

Install the runtime against `jflows` 0.5:

```bash
pip install -e /path/to/jflows
pip install -e /path/to/jflows_md
```

OpenMM-side construction is optional:

```bash
pip install -e "/path/to/jflows_md[openmm]"
pip install -e "/path/to/jflows_md[bundles]"
python -m jflows_md.bundle_build \
  glycerol molecule.prmtop molecule.rst7 generated/glycerol
```

From a source checkout, the equivalent wrapper is
`python -m bundles.build_molecular_bundles ...`.

## Verification

Run the complete local suite from the repository root only when the accelerator
is available:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

`smoke/benchmark_compile.py` is an opt-in compilation benchmark and is not part
of `run_all.py`.
