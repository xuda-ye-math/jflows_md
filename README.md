# jflows_md

`jflows_md` 0.5.2 is the mixed-domain molecular companion to `jflows` 0.5.2.
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

Direct trainers default to `initialize_from_identity=False`. Adaptive trained
Boltzmann functions default to `True`, so each accepted-stage attempt starts
from an identity parameterization unless warm-starting is explicitly selected.
`boltzmann_identity` performs no flow training at all.

## Linear sharpening

All three adaptive entry points require two regularization states. The
identity-only form is:

```python
from jflows_md.boltzmann import boltzmann_identity

particles, stages = boltzmann_identity(
    x_valid,
    source,
    target,
    ladder=16,
    mc_dt=1e-4,
    mc_steps=4,
    rg_param_0=(20.0, 0.15),
    rg_param_1=(1000.0, 0.0),
)
```

It omits every flow, training, optimizer, pool, and checkpoint argument.
Identity removes only the learned proposal: SMC endpoint selection, complete
population identity ESS, resampling, pre-sharpen MALA, exact regularization
sharpening, its ESS gate, and final MALA remain active.

The trained form is:

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
`jflows_md.boltzmann.write` atomically writes the post-sharpen population,
histories, and stage metadata before publishing the stage in `run.json`.
Trained runs additionally write both flow artifacts; identity runs write no
`.eqx` files. `jflows_md.boltzmann.load.run` resumes from the last published
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

A source checkout includes these seven audited named bundles:

| CLI preset | Bundle | Model | Domain | Fixed centers |
|---|---|---|---|---:|
| `adp` | `adp_ff96_obc1` | ff96; OBC1/ACE | `R^42 x T^18` | 1 (L-Ala) |
| `glycerol` | `glycerol_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^25 x T^11` | 0 |
| `diethanolamine` | `diethanolamine_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^33 x T^15` | 0 |
| `nma` | `nma_ff96_obc1` | ff96; OBC1/ACE | `R^21 x T^9` | 0 |
| `s_2_butanol` | `s_2_butanol_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^28 x T^11` | 1 (S) |
| `rr_2_3_butanediol` | `rr_2_3_butanediol_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^31 x T^11` | 2 (R,R) |
| `cyclohexane` | `cyclohexane_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^33 x T^15` | 0 |

NMA keeps its amide cis/trans coordinate periodic and unrestricted; the
force field assigns its energetic preference. Cyclohexane likewise has no
stereochemical support restriction, so ring conformations and chair inversion
are not split into artificial components. The two alcohol presets restrict
the chart to their named absolute configurations.
Candidate manifests pin AmberTools 26.0.0 and their structure/parameter
provenance. The two alcohol `parmchk2` files add no terms; cyclohexane records
the zero-penalty transfer of its `c6` terms from the GAFF2 `c3` types.

Wheels contain Python code only; bundle data must be downloaded or prepared
separately. An installed user can search an external directory and select a
bundle by name:

```python
from jflows_md import Molecular_Bundle, available_bundles

bundle_root = "/path/to/downloaded/bundles"
names = available_bundles(bundle_root)
bundle = Molecular_Bundle.load(names[0], root=bundle_root)
```

The same `root=` keyword is accepted by `Molecular_Potential.from_bundle` and
`OpenMM_Potential.from_bundle`. `available_bundles()` returns an empty tuple
when the requested directory is absent or contains no bundles.

Install matching JAX and OpenMM accelerator wheels separately. Choose one CUDA
toolkit generation:

```bash
pip install "jax[cuda13]" "openmm[cuda13]"
# or: pip install "jax[cuda12]" "openmm[cuda12]"
```

Installing a JAX CUDA extra does not install the OpenMM CUDA plugin, and the
OpenMM extra does not install the JAX plugin. These extras select the
independently packaged JAX and OpenMM CUDA wheels; they must use the same CUDA
generation. Then install both projects:

```bash
pip install -e /path/to/jflows
pip install -e /path/to/jflows_md
```

Inspect the installed runtimes without creating a JAX client or OpenMM
context:

```python
import jflows_md

jflows_md.backend()
```

Base OpenMM-side construction is optional and CPU-capable; select a CUDA extra
above when native GPU execution is required:

```bash
pip install -e "/path/to/jflows_md[openmm]"
pip install -e "/path/to/jflows_md[bundles]"
python -m jflows_md.bundle_build \
  glycerol molecule.prmtop molecule.rst7 generated/glycerol
```

From a source checkout, the equivalent wrapper is
`python -m bundles.build_molecular_bundles ...`.

The command-line presets above carry explicit coordinate configurations.
Programmatic `jflows_md.bundle_build.write_bundle(...)` calls may provide
`zmatrix`, `fixed_stereocenters`, and `signed_volume_diagnostics` for any
additional explicitly prepared molecule, but that low-level route is not a
claim of curated target support. Multiple fixed tetrahedral centers use
multiple half-chart torsions; no center is inferred from the molecule name.
For named chiral presets, caller-supplied CIP-priority atom orders make bundle
construction reject a reference with the wrong requested R/S configuration.
Legacy schema-v2 achiral and single-center bundles remain runtime compatible.
For source compatibility, a target-only builder call with all new coordinate
options at their defaults still reproduces the historical schema-v2 defaults
for `adp`, `glycerol`, and `diethanolamine`; any non-default coordinate option
uses the new schema-v3 route.
See `doc/01-low-level.md` for the configuration contract.

## Verification

Run the complete local suite from the repository root only when the accelerator
is available:

```bash
PYTHONPATH=/path/to/jflows:/path/to/jflows_md \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python /path/to/jflows_md/smoke/run_all.py
```

`smoke/benchmark_compile.py` is an opt-in compilation benchmark and is not part
of `run_all.py`.
