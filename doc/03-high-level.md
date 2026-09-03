# High-level interfaces

The high level builds a staged molecular Boltzmann generator from the
medium-level trainers. It chooses stage points, trains an incremental map on
the current population, compares that map with the exact identity by the
screened population ESS, advances the particles, and records every accepted
stage. It is the normal entry point for difficult molecular targets. The
identity-only route omits flow training while retaining the stage schedule,
the ESS gate, resampling, and rejuvenation. The inference scheme replays the
frozen stage maps of a complete stored run on fresh source particles.

Public computation imports:

```python
from jflows_md.boltzmann import (
    boltzmann_identity,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLL1_G,
    boltzmann_FAB_G,
    boltzmann_forward_KLXX_G,
    boltzmann_FABX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G_fixed,
    iterate_identity,
    iterate_boltzmann,
)
from jflows_md.boltzmann.inference import run_inference
```

Persistence remains separate:

```python
from jflows_md.boltzmann.write import create, stage, finish
from jflows_md.boltzmann.load import (
    manifest, validate, load, fork, run,
    load_stage_flow, load_validation_samples, load_training_history,
)
```

## Interpolation and stage model

The stage targets are the linear interpolations

```text
U_t = (1 - t) U_source + t U_target,    0 <= t <= 1,
```

of the source and the regularized molecular potential
(`linear_combination([target, source], [t, 1 - t])`). The coefficient $t$ is
a dimensionless stage-interpolation parameter. It does not change the
physical `temperature_kelvin` or inverse temperature $\beta$ (`beta`), which
remain properties of the molecular target, and the energy cut and cap of the
target are fixed for the whole run.

A stage from `t_start` to `t_end` trains one incremental flow between those
two stage distributions. It is not a new global source-to-final map. The
selected flows form an ordered proposal chain, but the returned population
also passes through weighting, resampling, and MALA after every selected map:

```text
x_valid at t = 0
  --stage 1 map / reweight / resample / MALA--> particles at t_1
  --stage 2 map / reweight / resample / MALA--> particles at t_2
  ...
  --stage K map / reweight / resample / MALA--> particles at t = 1
```

Consequently, `valid_selected_ess` is an incremental stage ESS. It does not
measure the global source-to-final chain. The inference scheme below replays
the same chain on fresh source particles.

## Design philosophy

The controller keeps two decisions distinct:

```text
trained/identity map choice  -> larger screened ESS over the complete population wins
stage acceptance             -> selected ESS >= tau_valid
```

The screened ESS of the incremental importance weights over the complete
current population decides whether the trained flow is better than the
identity and whether the attempt passes. Optimizer batch ESS and MALA
acceptance are diagnostics only.

## Adaptive-staging identity generator

```python
boltzmann_identity(
    x_valid,
    source,
    target,
    mc_dt,
    mc_steps_2,
    *,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    screen_fraction=SCREEN_FRACTION,
)
```

`boltzmann_identity` is the molecular flow-free baseline. It has no `flow`,
`pool_size`, `batch_size`, `steps_total`, `lr`, `ladder`, checkpoint,
optimizer, or initialization arguments. For each candidate `a -> b` it:

```text
current population at U_a
  -> evaluate the identity log weights U_a(x) - U_b(x) on every particle
  -> require their screened ESS >= tau_valid, shrinking b - a on rejection
  -> resample the complete population with the screened weights
  -> mc_steps_2 mixed MALA steps at U_b
  -> emit a finite population
```

Thus identity removes only the trained transport map. `chunks` partitions
the identity weights and the MALA advance.

The returned `(samples, stages, effective_seconds)` follows the same completion rule as the
trained generators. Each record has `t`, `t_start`, `valid_selected_ess`
(equal to `valid_identity_ess`), `valid_identity_ess`, `valid_sample_count`,
`selected="identity"`, `objective="identity"`, `t_hist`,
`valid_identity_ess_hist`, `attempt_status_hist`, `mala_acceptance`, and
`elapsed_seconds`. It has no flow, trained ESS, batch ESS, or initialization
fields.

## Adaptive-staging KLX generator

```python
boltzmann_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
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
    screen_fraction=SCREEN_FRACTION,
)
```

Every attempt calls `train_forward_KLX_G` with the current population as
`x_valid` and the stage potentials `U_a` and `U_b` as source and target. The
trainer manufactures its target batch inside every step by the SMC target
surrogate through the current flow (`ladder` levels, `mc_steps_1` MALA steps
per intermediate level and `mc_steps_2` on the last)
and minimizes the forward KL plus `coeff_lambda` times the variation of the
log-ratio; `coeff_lambda = 0` is the forward KL. It constructs no
quench-and-temper pool.

## Adaptive-staging KLXX generator

```python
boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    melt,
    opt_alpha,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    monitor=None,
    bg_param=None,
    chunks=1,
    mc_image_radius=3,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
)
```

KLXX uses the same controller and selection logic and calls
`train_forward_KLXX_G` on every attempt. The trainer builds its
quench-and-temper pool from the current population before its optimizer
scan (`pool_size = 0`: the complete population; `pool_size > 0`: a random
subset of that many rows), quenched and tempered at the candidate target
`U_b` with `mc_steps_2` MALA steps; `coeff_qt > 0` resamples the tempered
pool by `exp(-coeff_qt U_b)` and rejuvenates it again. At every step
it adds `coeff_theta` times the variation of the log-ratio over the mixture
batch drawn with proportion `coeff_alpha` from the rejuvenated pool
(`mc_steps_2` MALA steps) and `1 - coeff_alpha` from the detached pushforward
of the source batch.

`melt`, `opt_alpha`, and `opt_steps` affect only the pool construction.
`coeff_theta`, `coeff_alpha`, and `coeff_qt` control the mixture term.

## Other objectives

`boltzmann_forward_KLL1_G` has the signature of `boltzmann_forward_KLX_G`
and runs `train_forward_KLL1_G` on every attempt; `boltzmann_FAB_G` has the
same signature without `coeff_lambda` and runs `train_FAB_G`;
`boltzmann_FABX_G` has the signature of `boltzmann_forward_KLXX_G` without
`coeff_lambda` and runs `train_FABX_G`. The controller, the selection, and
the advance are the same for every objective; `iterate_boltzmann` takes the
objective as one of `"forward_klx"`, `"forward_kll1"`, `"fab"`,
`"forward_klxx"`, and `"fabx"`.

## Shared controls

<div align="center">

<table>
<thead>
<tr><th>Control</th><th>Role</th></tr>
</thead>
<tbody>
<tr><td><code>x_valid</code></td><td>complete current population; the trainers draw their source batches from it, and the trained-versus-identity comparison and the emitted samples use all of it</td></tr>
<tr><td><code>pool_size</code></td><td>KLXX only: <code>0</code> quenches and tempers the complete population; a positive value draws a random subset of that many rows first</td></tr>
<tr><td><code>batch_size</code>, <code>steps_total</code>, <code>lr</code></td><td>medium-level Adam controls</td></tr>
<tr><td><code>ladder</code></td><td>SMC levels of the target-batch manufacture inside every training step</td></tr>
<tr><td><code>mc_dt</code>, <code>mc_steps_1</code></td><td>MALA step size, and the MALA steps used only on the intermediate SMC levels of every manufactured batch</td></tr>
<tr><td><code>mc_steps_2</code></td><td>MALA steps of every other rejuvenation: the last SMC level, the quench-and-temper rows of the KLXX mixture batch, the temper of the KLXX pool, and the stage advance</td></tr>
<tr><td><code>mc_image_radius</code></td><td>periodic wrapped-normal image radius</td></tr>
<tr><td><code>chunks</code></td><td>row partitions for the population pushforward and weights, the KLXX quench and temper, and the population MALA</td></tr>
<tr><td><code>reject_requested</code></td><td>a <code>Manual_Reject</code> (or any callable returning and clearing a flag); when it fires after a trainer returns, the attempt is recorded as <code>"rejected-manual"</code> and the endpoint shrinks as after a failed ESS gate. Off by default; ignored on a fixed schedule</td></tr>
<tr><td><code>rg_param_0</code>, <code>rg_param_1</code></td><td>regularization path <code>rho_s = (1 - s) rho_0 + s rho_1</code>, each a pair <code>(e [kJ/mol], r [nm])</code> of <code>target.regularized</code>; equal values fix the regularization, <code>None</code> (default) uses <code>target</code> as given</td></tr>
<tr><td><code>initialize_from_identity</code></td><td>identity-start each training stage; <code>False</code> warm-starts from the preceding continuation flow</td></tr>
<tr><td><code>checkpoint</code></td><td>rematerialize the trainer loss during backpropagation</td></tr>
<tr><td><code>u_clip</code></td><td>absolute molecular-energy screen passed to the trainer; nonfinite loss rows are always excluded</td></tr>
<tr><td><code>g_clip</code></td><td>stable global gradient-norm limit passed to the trainer</td></tr>
<tr><td><code>lr_warmup</code></td><td>linear learning-rate warmup length passed to the trainer</td></tr>
<tr><td><code>screen_fraction</code></td><td>fraction removed at the top of every ESS and every resampling weight, in the trainer and in the population comparison and advance (default <code>SCREEN_FRACTION = 1e-4</code>)</td></tr>
<tr><td><code>seed</code></td><td>base key namespace for stage, attempt, and operation keys</td></tr>
</tbody>
</table>

</div>

`boltzmann_forward_KLX_G` runs the controller with `pool_size = 0`,
`coeff_theta = 1.0`, `coeff_alpha = 0.5`, `coeff_qt = 0.0`, `melt = 0.0`,
`opt_alpha = 1.0`, and `opt_steps = 0`; only the KLXX objective reads those
values.

## Adaptive stage policy

`bg_param` overrides any subset of the defaults:

```python
{
    "t_safe": 0.1,
    "shrink_factor": 0.7,
    "enlarge_factor": 2.0,
    "tau_valid": 0.6,
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
<tr><td><code>tau_valid</code></td><td>minimum selected trained-or-identity population ESS</td></tr>
<tr><td><code>t_tol</code></td><td>snap a candidate stage point within <code>t_tol</code> of one to one</td></tr>
<tr><td><code>max_stages</code></td><td>maximum accepted-stage index attempted</td></tr>
<tr><td><code>max_retry</code></td><td>training attempts per stage</td></tr>
</tbody>
</table>

</div>

An unknown key in `bg_param` raises `KeyError`. The first endpoint is
`t_safe`; every later endpoint is the last accepted `t` plus `enlarge_factor`
times the last accepted increment, capped at one. If the selected ESS fails
`tau_valid`, the endpoint is shrunk to `t_start + shrink_factor * (t_end -
t_start)` and the complete attempt is repeated with a fresh operation key.
Exhausting the retry or stage limits returns the accepted prefix rather than
inventing a final stage.

## Regularization path

With `rg_param_0` and `rg_param_1` the stage targets lie on the diagonal of
the regularization path `rho_s = (1 - s) rho_0 + s rho_1`:
`U_{t,t} = (1 - t) U_0 + t U^{rho_t}` with `target.regularized(rho_t)`. The
stage `t_{k-1} -> t_k` trains, selects, and advances the population from
`U_{t_{k-1},t_{k-1}}` to `U_{t_k,t_k}` in one arrow, so the regularization
tightens together with the interpolation and every accepted stage supplies
one weight vector, as at a fixed regularization; the KLXX pool is built under
`U_{t_k,t_k}`. `rg_param_0 = rg_param_1` fixes the regularization; the
default `None` uses `target` as given. The persistence layer stores
`rg_start` and `rg_end`, and `run_inference` rebuilds the diagonal targets
from the run config's `rg_param_0` and `rg_param_1`, checking them against
the stored stage values.

## Manual rejection

A running attempt can be rejected by hand from the log, before its ESS gate:

```python
from jflows_md.boltzmann import Manual_Reject

with Manual_Reject() as reject:          # Ctrl+C rejects; twice within 3 s stops
    print(reject.command)                # "Ctrl+C (detached: kill -INT <pid>); ..."
    samples, stages, seconds = boltzmann_forward_KLXX_G(
        ..., reject_requested=reject,
    )
```

Ctrl+C (SIGINT) only raises a flag; a second Ctrl+C within
`STOP_WINDOW_SECONDS = 3` stops the program as usual. The trainer runs its
scan one compiled step at a time (`REJECT_CHECK_STEPS = 1`) and stops at the
first step boundary after the signal, so the rejection is received within
one gradient step; the quench-and-temper pool, the validation weights, and
the population advance check the flag between their chunks. `iterate_boltzmann` discards a stale flag at the start of
every attempt and reads it once after the trainer returns; a raised flag
records the attempt as `"rejected-manual"` in `attempt_status_hist` (the log
says after how many steps), skips the selection and the advance, and
shrinks the endpoint by `shrink_factor`, so the retry budget and the
persistence are those of an ESS rejection. The default
`reject_requested=None` disables the feature, and a fixed schedule (`t_list`)
ignores it with one log line.

## Fixed stage schedule

`iterate_boltzmann(..., t_list=(0.25, 0.5, 0.75, 1.0))` replaces the
adaptive policy by a fixed schedule: every stage trains one attempt from the
current `t` to the next endpoint of `t_list` above it and is accepted
whatever its selected ESS; `bg_param` is ignored, and the trained-versus-
identity selection and the advance are unchanged. `boltzmann_forward_KLX_G_fixed`
and `boltzmann_forward_KLXX_G_fixed` are the corresponding wrappers: they
take `t_list` as the positional argument after `mc_steps_2`, have no
`bg_param`, and otherwise match `boltzmann_forward_KLX_G` and
`boltzmann_forward_KLXX_G`. Endpoints at or below the current `t` are
skipped, so a schedule may start at `0.0`.

## Identity fallback and initialization

At every attempt the controller evaluates two maps on the complete current
population:

- the newly trained candidate: one chunked pass computes its pushforward
  `G^{-1}(x)` (wrapped) and the log weights
  `U_a(x) - U_b(G^{-1}(x)) + log|det J_{G^{-1}}(x)|`; and
- `entry_flow.zeros()`, an independent identity map, with the log weights
  `U_a(x) - U_b(x)`.

The map with the larger screened ESS is selected (the trained map only when
strictly larger). If the identity wins, the advance starts from the unchanged
population. The selected ESS must still reach `tau_valid`. The pushforward of
the trained map is computed once per attempt and reused for the selection
weights and the advance.

This fallback is independent of optimizer initialization. With
`initialize_from_identity=True`, the trainer also starts from identity. With
`False`, the next stage starts from the preceding continuation flow. Retries
within a stage reuse the same immutable stage-entry template.

Identity initialization improves stability and keeps comparisons fair, but it
does not guarantee a higher trained ESS.

## Accepted-stage sequence

One successful stage performs:

```text
current population at U_a
  -> propose the endpoint b
  -> train the KLX or KLXX map from U_b to U_a on the current population
  -> push the population through the trained map once; evaluate the trained
     and identity log weights and their screened ESS
  -> select the larger ESS; require it >= tau_valid, else shrink b and retry
  -> resample the selected pushforward with the screened weights
  -> mc_steps_2 mixed MALA steps at U_b
  -> require a finite completed population
  -> emit population, record, and continuation flow
```

The stage flow is a proposal map for the transition `U_a -> U_b`. Because
resampling and MALA lie between stages, the stored stage flows do not compose
into a deterministic source-to-final generator; the stored population is the
continuation state, and the inference scheme replays the chain with the same
resample-and-MALA transitions. The generator emits one line per attempt
through the monitor's printer (or `print`),

```text
[stage 001 attempt 01 | t: 0.000000 -> 0.100000] validation ESS=... (trained=..., identity=...) ACCEPTED
```

and prefixes the trainer's monitor with `[stage 001 attempt 01]`.

## Return values and completion

Every public generator returns:

```python
samples, stages, effective_seconds
```

`samples` is the population after the last accepted stage. `stages` contains
only accepted stages. `effective_seconds` is the training time of the
accepted attempts alone, the sum of the records' `accepted_attempt_seconds`,
so the wall time of rejected attempts is removed. A successful endpoint
requires:

```python
complete = bool(stages and stages[-1]["t"] == 1.0)
```

Do not infer completion from a nonempty list alone.

`iterate_boltzmann(samples, source, target, flow, *, objective, pool_size,
batch_size, steps_total, lr, ladder, mc_dt, mc_steps_1, mc_steps_2,
initialize_from_identity, coeff_lambda, coeff_theta, coeff_alpha, coeff_qt,
melt, opt_alpha, opt_steps, monitor, bg_param, chunks, mc_image_radius,
checkpoint, u_clip=inf, g_clip=inf, lr_warmup=0,
screen_fraction=SCREEN_FRACTION, seed=0, accepted_t=(0.0,), start_stage=1)`
exposes the same controller as an iterator with `objective="forward_klx"` or
`"forward_klxx"`; `iterate_identity(samples, source, target, *, mc_dt,
mc_steps_2, mc_image_radius=3, monitor=None, bg_param=None, chunks=1, seed=0,
screen_fraction=SCREEN_FRACTION, accepted_t=(0.0,), start_stage=1)` is the
flow-free form. Each yield is:

```python
samples, stage_record, continuation_flow
```

with `continuation_flow=None` for the identity. `accepted_t` and
`start_stage` let the persistence controller resume a stored run. Both
iterators are also useful for custom complete-stage consumers.

## Stage records

Each trained record is the `record` dictionary assembled by
`iterate_boltzmann`:

<div align="center">

<table>
<thead>
<tr><th>Field</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>t_start</code>, <code>t</code></td><td>accepted stage interval</td></tr>
<tr><td><code>valid_selected_ess</code></td><td>accepted maximum of the trained and identity population ESS (screened)</td></tr>
<tr><td><code>valid_trained_ess</code></td><td>trained proposal ESS on the complete population</td></tr>
<tr><td><code>valid_identity_ess</code></td><td>identity ESS on the complete population</td></tr>
<tr><td><code>valid_sample_count</code></td><td>complete population size</td></tr>
<tr><td><code>initialized_from_identity</code></td><td>trainer initialization policy</td></tr>
<tr><td><code>selected</code></td><td><code>"trained"</code> or <code>"identity"</code></td></tr>
<tr><td><code>flow</code></td><td>selected incremental proposal map (the trained candidate or <code>entry_flow.zeros()</code>)</td></tr>
<tr><td><code>continuation_flow</code></td><td>map used as the next stage's entry template; equal to <code>flow</code></td></tr>
<tr><td><code>t_hist</code></td><td>attempted endpoints</td></tr>
<tr><td><code>batch_ess_hist</code></td><td>trainer batch ESS per attempt, shape <code>(attempts, steps_total)</code></td></tr>
<tr><td><code>valid_trained_ess_hist</code>, <code>valid_identity_ess_hist</code></td><td>one population ESS per attempt</td></tr>
<tr><td><code>attempt_status_hist</code></td><td><code>"accepted"</code>, <code>"rejected"</code>, or <code>"rejected-manual"</code> per attempt</td></tr>
<tr><td><code>mala_acceptance</code></td><td>acceptance history of the <code>mc_steps_2</code> MALA advance, shape <code>(mc_steps_2,)</code></td></tr>
<tr><td><code>rg_start</code>, <code>rg_end</code></td><td><code>rho</code> at <code>t_start</code> and <code>t</code> on the regularization path, <code>None</code> without one</td></tr>
<tr><td><code>objective</code></td><td><code>"forward_klx"</code> or <code>"forward_klxx"</code></td></tr>
<tr><td><code>elapsed_seconds</code></td><td>wall time for the accepted stage including retries</td></tr>
<tr><td><code>accepted_attempt_seconds</code></td><td>wall time of the accepted attempt alone (training, selection, advance); the generators sum it into <code>effective_seconds</code></td></tr>
</tbody>
</table>

</div>

An identity record carries `t`, `t_start`, `valid_selected_ess`,
`valid_identity_ess`, `valid_sample_count`, `selected`, `t_hist`,
`valid_identity_ess_hist`, `attempt_status_hist`, `mala_acceptance`,
`objective="identity"`, and `elapsed_seconds`. The persistence layer adds
`validation_samples_path` and, for trained records, `selected_flow_path` and
`continuation_flow_path`, relative to the run root.

## Minimal adaptive-staging workflow

```python
import jax

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Potential
from jflows_md.boltzmann import boltzmann_forward_KLXX_G

target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
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

samples, stages, effective_seconds = boltzmann_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size=0,
    batch_size=5000,
    steps_total=500,
    lr=1e-3,
    ladder=8,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    coeff_qt=0.0,
    chunks=32,
    u_clip=1e3,
    g_clip=1e2,
    lr_warmup=25,
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
the manifest `run.json` (format `jflows-md-stage-resume-2`). A trained run
also writes its initial flow and the static flow template. `stage` writes the
post-stage population (`samples.npy`), the history (`history.npz`: `t_hist`,
`batch_ess_hist`, `valid_trained_ess_hist`, `valid_identity_ess_hist`,
`mala_acceptance`; the identity form omits the two trained histories), and
the metadata (`stage.json`) before publishing the stage in `run.json`;
trained stages additionally write the selected and continuation flows.
`stage` rejects a record whose `t_start` differs from the last published `t`
or whose flows do not match the run template. `finish` marks the manifest
`complete` or `exhausted`. An identity run passes `flow=None`, stores no
`.eqx` file, and cannot be mixed with trained stages.

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
    "runs/alanine_dipeptide",
    "alanine_dipeptide-klxx",
    config,
    x_valid,
    flow,
    iterate,
)

samples, stages = run(
    "runs/alanine_dipeptide",
    "alanine_dipeptide-klxx",
    config,
    None,
    flow,
    iterate,
    resume=True,
)
```

On resume, `problem_id`, the fully serialized `config`, and the static flow
template must match. The controller reloads the last published population
and continuation flow, then starts at the next stage. An incomplete
unpublished stage directory is not part of the manifest and is ignored.

For identity persistence, the callback calls `iterate_identity`, and both the
initial and resume calls pass `flow=None`. The run stores the accepted
population and histories only; resume restores the population, accepted `t`
values, and next stage number without reconstructing a flow.

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

The manifest publishes only complete stages. `validate` checks the format,
the status (`running`, `exhausted`, or `complete`), the consecutive stage
schedule, the objective-specific metadata and histories, the flow paths, the
population paths, and that a `complete` run ends at `t = 1`.

## Inference scheme

```python
run_inference(
    run_dir,
    output_dir,
    template,
    source,
    target,
    *,
    sample_count,
    mc_dt,
    mc_steps,
    chunk_size=10_000,
    mc_image_radius=3,
    seed=0,
    screen_fraction=SCREEN_FRACTION,
    max_stages=None,
    printer=print,
)
```

`run_inference` (also exported as `jflows_md.run_inference`) replays the
frozen selected stage maps of a complete stored run on `sample_count` fresh
source particles. `template` is the flow template of the run (as for
`load_stage_flow`), `source` and `target` the potentials the run was trained
with. The stored run must validate with status `complete`; `max_stages`
replays only the first stages of it. For every stage `k` the scheme:

```text
population at U_{t_{k-1}}
  -> push through the stored selected map G_k^{-1} (role "selected"),
     chunk by chunk; log weight U_{t_{k-1}}(x) - U_{t_k}(G_k^{-1}(x)) + ladj
  -> screened ESS of the stage weights (screen_fraction)
  -> multinomial resampling with the screened weights (inverse CDF, NumPy
     generator seeded by seed)
  -> mc_steps mixed MALA steps of size mc_dt under U_{t_k}, chunk by chunk
  -> population at U_{t_k}
```

No training update takes place. Populations are processed in chunks of
`chunk_size` rows and kept as `.npy` memmaps below `output_dir`, so the
sample count is not limited by device memory. The screen fraction defaults
to the package's `SCREEN_FRACTION = 1e-4`. A non-finite MALA output raises
`FloatingPointError`; invalid resampling weights raise `ValueError`.

`output_dir` must be empty or absent (`FileExistsError` otherwise). It
receives:

```text
output_dir/
├── run.json                  manifest, format "jflows-md-inference-1"
└── stage_000001/
    ├── samples.npy           the population after this stage
    ├── log_weights.npy       the stage weights on the pushforward
    └── metadata.json
```

The manifest carries `format`, `status` (`running` while replaying, then
`complete` when the last replayed stage ends at `t = 1`, otherwise
`partial`), `training_run`, `sample_count`, `dimension`, the `config`
(`seed`, `chunk_size`, `mc_dt`, `mc_steps`, `mc_image_radius`,
`screen_fraction`, `saved_flow_role="selected"`), the list `stages` of the
per-stage metadata, and `inference_samples_path`, the path of the last stage's
samples relative to `output_dir`. Each stage's `metadata.json` holds `stage`,
`t_start`, `t`, `saved_selection` and `saved_flow_path` (the `selected` and
`selected_flow_path` of the stored stage), `sample_count`, the screened
`pushforward_ess`, `mala_acceptance_mean`, `elapsed_seconds`, `samples_path`,
and `log_weights_path`. `run_inference` returns the manifest; the last
stage's samples are the inference set for the target. Progress lines go to
`printer`.

## Failure and completion rules

- A candidate rejected by the selected population ESS emits no stage; the
  endpoint is shrunk and the attempt repeated.
- A stage is persistable only after the `tau_valid` gate passes and the
  post-stage MALA completes with a finite population.
- A run ending below `t=1` is `exhausted`, not complete.
- Resume continues from the last manifest-listed complete stage.
- The inference scheme refuses a stored run that is not `complete`, and
  reports `partial` when `max_stages` stops it before `t=1`.

## Executable references

- `smoke/test_boltzmann_md.py`: the identity, KLX, and KLXX generators on a
  toy target, rejection of a retired policy key, complete-stage persistence
  through `run`/`validate`/`load`, and `run_inference` on the stored run.
- `smoke/test_initialization.py`: identity and warm-start defaults.
- `smoke/test_api_consistency.py`: the generator signatures and defaults.
