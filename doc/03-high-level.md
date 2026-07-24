# High-level interfaces

The high level coordinates adaptive molecular stage selection, SMC
target construction, one-stage flow training, trained-versus-identity selection,
regularization sharpening, population rejuvenation, and complete-stage
persistence. It is the normal entry point for difficult molecular targets.
The identity-only route omits flow training while retaining stage selection,
regularization sharpening, and rejuvenation.

Public computation imports:

```python
from jflows_md.boltzmann import (
    boltzmann_identity,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
    iterate_boltzmann,
)
```

Persistence remains separate:

```python
from jflows_md.boltzmann.write import create, stage, finish
from jflows_md.boltzmann.load import (
    manifest, validate, load, fork, run,
    load_stage_flow, load_validation_samples, load_training_history,
)
```

## Stage interpolation and regularization model

The controller evolves two quantities together:

1. the source-to-target interpolation parameter $t$; and
2. the regularization state $\operatorname{rg}(t)$.

The coefficient $t$ is a dimensionless stage-interpolation parameter. It does
not change the physical `temperature_kelvin` or inverse temperature $\beta$
(`beta`), which remain properties of the molecular target.

For endpoint pairs `rg_param_0` and `rg_param_1`, the `_rg` helper computes

$$
\operatorname{rg}(t)
=
\mathtt{rg\_param\_0}
+t\left(\mathtt{rg\_param\_1}-\mathtt{rg\_param\_0}\right).
$$

Let $U_{\mathrm{source}}$ denote `source`, and let
$U_{\operatorname{rg}(t)}$ denote `target.regularized(rg(t))`. At the start
$a$ of one accepted stage, the code variable `source_bridge` is

$$
B_a=(1-a)U_{\mathrm{source}}+aU_{\operatorname{rg}(a)}.
$$

For a candidate stage point $b$, flow training first uses `target_soft`, the
pre-sharpen target

$$
B_b^{-}=(1-b)U_{\mathrm{source}}+bU_{\operatorname{rg}(a)}.
$$

After a flow is selected and resampled, the regularization advances to
`target_sharp`, the post-sharpen target

$$
B_b^{+}=(1-b)U_{\mathrm{source}}+bU_{\operatorname{rg}(b)}.
$$

The code variable `sharpen_log_weight` stores the exact sharpening log weight,

$$
\log w_{\mathrm{sharpen}}=B_b^{-}-B_b^{+}.
$$

The samples are then resampled and MALA-rejuvenated at `target_sharp`. At
$t=1$, a complete run therefore targets
`target.regularized(rg_param_1)`. Choose `rg_param_1` to represent the intended
physical or nearly physical endpoint.

## Design philosophy

The controller keeps four decisions distinct:

```text
candidate stage selection   -> minimum SMC ESS >= tau_smc
trained/identity map choice  -> larger ESS over the complete validation set wins
flow-stage acceptance       -> selected validation ESS >= tau_ess
sharpening acceptance        -> sharpening ESS >= tau_ess
```

SMC ESS decides whether a candidate stage is feasible before training.
Proposal ESS over the complete validation set decides whether the trained flow is better than
identity and whether the training attempt passes. Sharpening ESS independently
decides whether the same candidate stage point can safely advance the
regularization. Optimizer batch ESS and MALA acceptance are diagnostics only.

## Adaptive-staging identity generator

```python
boltzmann_identity(
    x_valid,
    source,
    target,
    ladder,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
)
```

`boltzmann_identity` is the molecular flow-free baseline. It has no `flow`,
`pool_size`, `batch_size`, `train_steps`, `lr`, checkpoint, optimizer, or
initialization arguments. For each candidate stage point it performs:

```text
current post-sharpen population at U_a
  -> select candidate b by molecular SMC ESS
  -> evaluate full-population identity weights from U_a to U_b^-
  -> require identity ESS >= tau_ess
  -> resample and mixed MALA at U_b^-
  -> evaluate exact U_b^- to U_b^+ sharpening weights
  -> require sharpening ESS >= tau_ess
  -> resample and mixed MALA at U_b^+
  -> emit a finite post-sharpen population
```

Thus identity removes only the trained transport map. Sharpening still takes
effect and can independently shrink a candidate interval. The same
`bg_param["tau_ess"]` gates both identity reweighting and sharpening, while
`tau_smc` gates stage selection. `chunks` partitions SMC, identity and
sharpening weights, and both MALA calls.

The returned `(samples, stages)` follows the same completion rule as the
trained generators. Each record has `objective="identity"`,
`selected="identity"`, `rg_start`, `rg_end`, `population_rg=rg_end`, identity
ESS histories, SMC/MALA histories, and sharpening ESS/MALA histories. It has
no flow, trained ESS, batch ESS, or initialization fields.

## Adaptive-staging KLX generator

```python
boltzmann_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
)
```

KLX uses the candidate SMC population as `target_samples` and the current
selection pool as `source_samples` for `train_forward_KLX_G`. It adds the
target-sample X penalty to forward KL but does not construct a quench and temper
hat pool.

## Adaptive-staging KLXX generator

```python
boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps,
    *,
    rg_param_0,
    rg_param_1,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
)
```

KLXX uses the same controller and flow-validation logic, then constructs an
independent hat pool for every training attempt:

1. draw `selection_pool.shape[0]` source samples;
2. run `mixed_quench_and_temper` at the pre-sharpen candidate target `U_b^-`;
3. pass the result to `train_forward_KLXX_G`; and
4. record the QT MALA history as `hat_mala_acceptance`.

`melt`, `opt_alpha`, and `opt_steps` affect only hat-pool construction.
`coeff_alpha` and `coeff_beta` control the hat/proposal mixture in the KLXX X
penalty.

## Shared controls

<div align="center">

<table>
<thead>
<tr><th>Control</th><th>Role</th></tr>
</thead>
<tbody>
<tr><td><code>x_valid</code></td><td>complete current validation set; comparison and emitted samples</td></tr>
<tr><td><code>pool_size</code></td><td><code>0</code> uses the complete population for SMC/training selection; positive values sample that many rows with replacement</td></tr>
<tr><td><code>batch_size</code>, <code>train_steps</code>, <code>lr</code></td><td>medium-level Adam controls</td></tr>
<tr><td><code>ladder</code></td><td>uniform SMC levels between the current and candidate stage potentials</td></tr>
<tr><td><code>mc_dt</code>, <code>mc_steps</code></td><td>mixed MALA controls in SMC, post-flow rejuvenation, sharpening, and KLXX hat freshening</td></tr>
<tr><td><code>mc_image_radius</code></td><td>periodic wrapped-normal image radius</td></tr>
<tr><td><code>chunks</code></td><td>row partitions for SMC, QT, validation weights, and population MALA</td></tr>
<tr><td><code>initialize_from_identity</code></td><td>identity-start each training stage; <code>False</code> warm-starts from the preceding selected map</td></tr>
<tr><td><code>checkpoint</code></td><td>rematerialize the trainer loss during backpropagation</td></tr>
<tr><td><code>u_clip</code></td><td>absolute molecular-energy screen passed to the trainer; nonfinite loss rows are always excluded</td></tr>
<tr><td><code>g_clip</code></td><td>stable global gradient-norm limit passed to the trainer</td></tr>
<tr><td><code>lr_warmup</code></td><td>linear learning-rate warmup length passed to the trainer</td></tr>
<tr><td><code>seed</code></td><td>base key namespace for stage, attempt, and operation keys</td></tr>
<tr><td><code>rg_param_0</code>, <code>rg_param_1</code></td><td>initial and final <code>(e,r)</code> regularization states</td></tr>
</tbody>
</table>

</div>

If `pool_size=0`, SMC and training use all current validation rows. If it is
positive, only stage selection and trainer pool construction use the
separately sampled selection pool. Trained-versus-identity validation and the
post-stage population still use the complete `x_valid` population.

## Adaptive stage policy

`bg_param` overrides any subset of the defaults:

```python
{
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 2.0,
    "tau_smc": 0.75,
    "tau_ess": 0.6,
    "t_tol": 1e-3,
    "max_stages": 25,
    "max_retry": 8,
}
```

<div align="center">

<table>
<thead>
<tr><th>Field</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>t_safe</code></td><td>first proposed endpoint</td></tr>
<tr><td><code>shrink_factor</code></td><td>factor applied to a rejected candidate increment</td></tr>
<tr><td><code>enlarge_factor</code></td><td>growth applied to the preceding accepted increment</td></tr>
<tr><td><code>tau_smc</code></td><td>minimum per-level SMC ESS needed before training</td></tr>
<tr><td><code>tau_ess</code></td><td>shared threshold for selected-flow ESS and sharpening ESS</td></tr>
<tr><td><code>t_tol</code></td><td>snap a near-final candidate stage point to one</td></tr>
<tr><td><code>max_stages</code></td><td>maximum accepted stages</td></tr>
<tr><td><code>max_retry</code></td><td>training/sharpening attempts per stage</td></tr>
</tbody>
</table>

</div>

Stage selection can shrink a candidate repeatedly before one training
attempt. If selected validation ESS or sharpening ESS then fails, the endpoint
is shrunk again and the complete attempt is repeated. Exhausting selection,
retry, or stage limits returns an incomplete run rather than inventing a final
stage.

## Identity fallback and initialization

At every stage the controller evaluates two maps on the complete current
population:

- the newly trained candidate; and
- `entry_flow.zeros()`, an independent identity map.

The map with larger validation ESS is selected. If identity wins, the emitted
proposal begins from the unchanged population. The selected ESS must still
reach `tau_ess`.

This fallback is independent of optimizer initialization. With
`initialize_from_identity=True`, the trainer also starts from identity. With
`False`, the next stage starts from the preceding selected continuation flow.
Retries within a stage reuse the same immutable stage-entry template.

Identity initialization improves stability and keeps comparisons fair, but it
does not guarantee a higher trained ESS.

## Accepted-stage sequence

One successful stage performs:

```text
current post-sharpen population at U_a
  -> select candidate b by SMC ESS
  -> train KLX or KLXX proposal for U_b^-
  -> compare trained and identity validation ESS
  -> resample selected proposal
  -> mixed MALA at U_b^-
  -> evaluate exact U_b^- to U_b^+ sharpening weights
  -> require sharpening ESS >= tau_ess
  -> resample sharpening weights
  -> mixed MALA at U_b^+
  -> require a finite completed population
  -> emit population, record, and continuation flow
```

The stage flow is a proposal map for the pre-sharpen transition. Because a
stochastic sharpening transition lies between stages, saved stage flows do not
compose into a deterministic source-to-final generator. The saved population
is the authoritative continuation state.

## Return values and completion

All three public generators return:

```python
samples, stages
```

`samples` is the latest post-sharpen population. `stages` contains only fully
accepted stages. A successful physical endpoint requires:

```python
complete = bool(stages and stages[-1]["t"] == 1.0)
```

Do not infer completion from a nonempty list alone.

`iterate_boltzmann(...)` exposes the same controller as an iterator. Each
yield is:

```python
samples, stage_record, continuation_flow
```

It is primarily used by the persistence controller, but it is also useful for
custom complete-stage consumers.

## Stage records

An accepted record separates the pre-sharpen flow endpoint from the
post-sharpen population endpoint.

<div align="center">

<table>
<thead>
<tr><th>Field</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>t_start</code>, <code>t</code></td><td>accepted stage interval</td></tr>
<tr><td><code>rg_start</code>, <code>rg_end</code></td><td>regularization pair before and after sharpening</td></tr>
<tr><td><code>flow_rg</code></td><td>regularization used by the trained proposal; equals <code>rg_start</code></td></tr>
<tr><td><code>population_rg</code></td><td>regularization of emitted particles; equals <code>rg_end</code></td></tr>
<tr><td><code>flow_endpoint</code></td><td>always <code>"pre_sharpen"</code></td></tr>
<tr><td><code>valid_selected_ess</code></td><td>accepted maximum of trained and identity validation ESS</td></tr>
<tr><td><code>valid_trained_ess</code></td><td>trained proposal ESS on the complete population</td></tr>
<tr><td><code>valid_identity_ess</code></td><td>identity proposal ESS on the complete population</td></tr>
<tr><td><code>valid_sample_count</code></td><td>complete population size</td></tr>
<tr><td><code>selected</code></td><td><code>"trained"</code> or <code>"identity"</code></td></tr>
<tr><td><code>flow</code></td><td>selected incremental proposal map</td></tr>
<tr><td><code>continuation_flow</code></td><td>map used as the next warm-start template</td></tr>
<tr><td><code>initialized_from_identity</code></td><td>trainer initialization policy</td></tr>
<tr><td><code>objective</code></td><td><code>"forward_klx"</code> or <code>"forward_klxx"</code></td></tr>
<tr><td><code>elapsed_seconds</code></td><td>wall time for the accepted stage including retries</td></tr>
</tbody>
</table>

</div>

Attempt-aligned and sampler histories are:

- `t_hist`;
- `batch_ess_hist`;
- `valid_trained_ess_hist`;
- `valid_identity_ess_hist`;
- `sharpen_ess_hist`;
- `attempt_status_hist`;
- `selection_history`;
- `smc_ess` and `smc_acceptance`;
- `mala_acceptance`;
- `hat_mala_acceptance`, which is `None` for KLX; and
- `sharpen_ess` and `sharpen_mala_acceptance`.

`sharpen_ess_hist` contains `NaN` when an attempt failed the flow ESS gate
before sharpening was evaluated.

## Minimal adaptive-staging workflow

```python
import jax

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import boltzmann_forward_KLXX_G

target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
source = target.source()
x_valid = source.samples(jax.random.key(1), N=200000)
flow = Mixed_NSF(
    jax.random.key(0),
    target.domain,
    bins=32,
    transforms=6,
    euclidean_bound=8.0,
    hidden_features=(256, 256),
    mask_strategy="balanced",
).zeros()

samples, stages = boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size=50000,
    batch_size=10000,
    train_steps=500,
    lr=1e-3,
    ladder=8,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-4,
    mc_steps=20,
    rg_param_0=(20.0, 0.15),
    rg_param_1=(1000.0, 0.0),
    chunks=32,
    monitor=Monitor(50, "[molecular KLXX] "),
)

complete = bool(stages and stages[-1]["t"] == 1.0)
```

The numerical values are an interface example, not universal molecular
tuning.

## Complete-stage persistence

The computation functions do not accept filesystem arguments. Persistence is
implemented by `jflows_md.boltzmann.write` and
`jflows_md.boltzmann.load` around `iterate_boltzmann` or the flow-free
`iterate_identity`.

### Writer API

```python
create(run_dir, problem_id, config, samples, flow)
stage(run_dir, run_record, stage_record, samples)
finish(run_dir, run_record, status)
```

`create` requires an empty destination and writes the initial population and
manifest. A trained run also writes its initial flow. `stage` writes
post-sharpen samples, history, and metadata before publishing the stage in
`run.json`; trained stages additionally write selected and continuation flows.
`finish` marks the manifest `complete` or `exhausted`. An identity run passes
`flow=None`, stores no `.eqx` file, and cannot be mixed with trained stages.

### Loader API

```python
manifest(run_dir)
validate(run_dir)
load(run_dir, template)
fork(run_dir, destination, problem_id=None)
load_stage_flow(run_dir, stage, role, template)
load_validation_samples(run_dir, stage=None, mmap_mode=None)
load_training_history(run_dir, stage)
```

`load` returns `(samples, continuation_flow, records)`. Flow deserialization
requires the same static architecture template used to create a trained run.
For an identity run, call `load(run_dir)` and the continuation is `None`.
`load_stage_flow` uses one-based stage numbers and `role="selected"` or
`"continuation"`. `stage=None` in `load_validation_samples` reads the initial
population.

`fork` copies a validated run to a new destination, optionally changes its
problem identifier, and resets its status to `running`. It never modifies the
source run.

The stable semantic aliases are `inspect_run = manifest`,
`validate_run = validate`, and `fork_run = fork`.

### Resume controller

```python
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

def iterate(samples, continuation, accepted_t, start_stage):
    return iterate_boltzmann(
        samples,
        source,
        target,
        continuation,
        objective="forward_klxx",
        accepted_t=accepted_t,
        start_stage=start_stage,
        **controls,
    )

samples, stages = run(
    "runs/glycerol",
    "glycerol-regularization",
    config,
    x_valid,
    flow,
    iterate,
)

samples, stages = run(
    "runs/glycerol",
    "glycerol-regularization",
    config,
    None,
    flow,
    iterate,
    resume=True,
)
```

On resume, `problem_id`, the fully serialized `config`, and the static flow
template must match. The controller reloads the last published post-sharpen
population and continuation flow, then starts at the next stage. An incomplete
unpublished stage directory is not part of the manifest and is ignored.

For identity persistence, the callback calls `iterate_identity`, and both the
initial and resume calls pass `flow=None`. The run stores the accepted
post-sharpen population and histories only; resume restores the population,
accepted `t` values, and next stage number without reconstructing a flow.

## Stored run tree

```text
run_dir/
├── run.json
├── initial_flow.eqx
├── initial_samples.npy
└── stages/
    ├── stage_000001/
    │   ├── continuation.eqx
    │   ├── history.npz
    │   ├── samples.npy
    │   ├── selected.eqx
    │   └── stage.json
    └── stage_000002/
        └── ...
```

An identity run has the thinner tree:

```text
run_dir/
├── run.json
├── initial_samples.npy
└── stages/
    └── stage_000001/
        ├── history.npz
        ├── samples.npy
        └── stage.json
```

The manifest publishes only complete stages. `validate` checks the stage
schedule, regularization interpolation, objective-specific histories, flow
paths, population paths, and completion status.

## Failure and completion rules

- A candidate rejected by SMC is shrunk before flow or identity evaluation.
- A candidate rejected by selected validation ESS is not sharpened.
- A candidate rejected by sharpening ESS emits no stage.
- A stage is persistable only after both ESS gates pass and post-sharpen MALA
  completes.
- A run ending below `t=1` is `exhausted`, not complete.
- Resume continues from the last manifest-listed complete stage.

## Executable references

- `smoke/test_mixed_training.py`: both objectives under adaptive-staging
  execution on a bounded path.
- `smoke/test_boltzmann_identity.py`: flow-free identity computation with
  observed pre- and post-sharpen MALA targets.
- `smoke/test_boltzmann_identity_artifacts.py`: flow-free save/load/resume with
  sharpening histories and no `.eqx` artifacts.
- `smoke/test_sharpening_gate.py`: combined flow/sharpening ESS rejection.
- `smoke/test_initialization.py`: identity and warm-start behavior.
- `smoke/test_boltzmann_checkpoints.py`: writer/loader and post-sharpen resume.
- `smoke/test_boltzmann_integration.py`: computed interruption/resume
  equivalence.
- `smoke/test_jflows_md_chunking.py`: eager controller and chunk forwarding.
