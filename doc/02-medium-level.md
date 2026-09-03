# Medium-level interfaces

The medium level trains one inverse molecular flow between a source and a
target potential. It combines molecular G-direction objectives, the
flow-proposal SMC that manufactures the target batch inside every step,
compiled Adam scans, deterministic key derivation, optional monitoring, and
optional rematerialization. It does not select a stage point or persist a
multi-stage run.

Public imports:

```python
from jflows.train import Monitor
from jflows_md.train import (
    train_forward_KLX_G,
    train_forward_KLL1_G,
    train_FAB_G,
    train_forward_KLXX_G,
    train_FABX_G,
)
from jflows_md.artifacts import (
    save_flow, load_flow,
    save_samples, load_samples,
    save_history, load_history,
)
```

## Shared contract

Both trainers consume:

- `x_valid`: the fixed source-side population from which every optimizer
  step draws its source batch;
- `source` and `target`: reduced potentials for the current stage;
- `flow`: a `Mixed_NSF` or compatible mixed-domain G-flow;
- `domain`: the mixed domain of the target (`target.domain`);
- `batch_size`, `steps_total`, and `lr`: Adam controls;
- `ladder`, `mc_dt`, `mc_steps_1`, and `mc_steps_2`: the SMC levels and the
  Langevin budgets of the target-batch manufacture;
- `seed`: the deterministic trainer key namespace; and
- optional `monitor`: normally `jflows.train.Monitor`.

Both return exactly:

```python
trained_flow, batch_ess_hist
```

`batch_ess_hist.shape == (steps_total,)`. It is the screened ESS of the full
proposal-to-target log weights of the pushforward batch, computed before the
SMC resampling and rejuvenation of that batch. It is a monitor, not an
acceptance gate and not a result over the complete validation set.

The supplied flow is used as-is unless `initialize_from_identity=True`, in
which case training starts from `flow.zeros()`. Equinox flows are immutable, so
always rebind the returned flow.

## Execution model

The committed compilation boundaries are part of the interface behavior:

```text
KLX
  -> one outer eqx.filter_jit call
  -> one lax.scan containing all steps_total Adam steps

KLXX
  -> eager chunked mixed_quench_and_temper pool construction
  -> one compiled lax.scan containing all steps_total Adam steps
```

Inside the scan every step:

```text
draw batch_size rows without replacement from x_valid
  -> flow_target_batch through the current flow (ladder levels,
     mc_steps_1 per intermediate level, mc_steps_2 on the last): target
     batch y, pushforward y_bar,
     proposal log weights
  -> evaluate the loss and its gradient on the detached batch
  -> apply one guarded Adam update
  -> record the screened batch ESS
```

Keeping the KLXX quench and temper outside the enclosing optimizer JIT makes
its `chunks` partition active at the pool boundary. The population size of
`x_valid`, the batch size, the training length, the flow architecture, and
the checkpoint choice participate in compilation.

## Common controls

<div align="center">

<table>
<thead>
<tr><th>Control</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>batch_size</code></td><td>source rows drawn without replacement from <code>x_valid</code> at every step</td></tr>
<tr><td><code>steps_total</code></td><td>number of Adam updates</td></tr>
<tr><td><code>lr</code></td><td>Adam learning rate</td></tr>
<tr><td><code>ladder</code></td><td>SMC levels used to manufacture one target batch</td></tr>
<tr><td><code>mc_dt</code>, <code>mc_steps_1</code></td><td>MALA step size, and the MALA steps used only on the intermediate SMC levels <code>1 .. M-1</code> of every manufactured batch</td></tr>
<tr><td><code>mc_steps_2</code></td><td>MALA steps of every other rejuvenation: the last SMC level, the quench-and-temper rows drawn for the KLXX and FABX mixture batch, and the temper of the quench-and-temper pool (and its rejuvenation after the <code>coeff_qt</code> resampling)</td></tr>
<tr><td><code>mc_image_radius</code></td><td>periodic wrapped-normal image radius of the MALA kernels</td></tr>
<tr><td><code>coeff_lambda</code></td><td>coefficient of the target-batch variation term; <code>0</code> gives the forward KL</td></tr>
<tr><td><code>screen_fraction</code></td><td>fraction of the largest log-ratios removed from the loss, and of the largest log weights removed from the batch ESS and the SMC resampling weights (default <code>SCREEN_FRACTION = 1e-4</code>)</td></tr>
<tr><td><code>seed</code></td><td>integer or JAX value folded into the trainer namespace</td></tr>
<tr><td><code>checkpoint</code></td><td>rematerialize the loss calculation during reverse-mode differentiation</td></tr>
<tr><td><code>u_clip</code></td><td>exclude nonfinite target-energy rows and, when finite, rows above this absolute energy threshold</td></tr>
<tr><td><code>g_clip</code></td><td>stable global gradient-norm limit; infinity disables clipping</td></tr>
<tr><td><code>lr_warmup</code></td><td>number of steps used to linearly warm the learning rate from <code>lr / lr_warmup</code> to <code>lr</code></td></tr>
<tr><td><code>initialize_from_identity</code></td><td>replace the supplied flow by <code>flow.zeros()</code> before training</td></tr>
<tr><td><code>t_start</code>, <code>t_end</code></td><td>stage labels supplied to the monitor</td></tr>
</tbody>
</table>

</div>

`checkpoint=True` changes the memory/runtime tradeoff by recomputing the loss
during the backward pass. It does not change the mathematical objective.

Two screens act on every batch. The energy screen `u_clip` removes rows with
nonfinite target energies and, when `u_clip` is finite, rows above it
(clashes). The top screen removes the `screen_fraction` largest log-ratios
`z = log(pi/nu)` among the kept rows, at least one row: a kept row whose
log-ratio is far above the rest is a hole of the pushforward density at an
ordinary target point and would dominate the batch. Rows with nonfinite
log-ratios never enter the loss. The same `screen_fraction` screens the
reported batch ESS and the resampling weights of the SMC levels.

Adam (`beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`) commits parameters, both
moment trees, and its update counter atomically only when the loss,
gradients, and complete candidate state are finite and at least one row
survived the screens. A rejected step therefore cannot poison a later finite
update.

## Monitoring

The molecular trainers use the generic monitor:

```python
monitor = Monitor(every=20, prefix="[molecular KLX] ")
```

The trainer calls `monitor.report(step, loss, ess, steps_total, t_start,
t_end)` from the compiled scan. The printed loss is the current optimizer
objective, and ESS is the screened batch ESS of the pushforward. `t_start`
and `t_end` are display labels; they do not alter the provided source or
target potentials.

## Molecular KLX trainer

```python
train_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    domain,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    coeff_lambda=1.0,
    mc_image_radius=3,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    initialize_from_identity=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    t_start=0.0,
    t_end=1.0,
    reject_requested=None,
    target_data=None,
)
```

Each step draws `batch_size` source rows `x` from `x_valid` and manufactures
the target batch `y` through the current flow with `flow_target_batch`
(`ladder` levels; `mc_steps_1` MALA steps on each intermediate level,
`mc_steps_2` on the last). The batch is
detached: sample locations are not differentiated through. For the target
rows `y`, define

```text
z(y) = U_source(G(y)) - U_target(y) - log|det J_G(y)|.
```

The per-step objective is the screened mean of `z` plus `coeff_lambda` times
its exact sorted variation,

```text
mean(z) + coeff_lambda * mean_{i != j}(abs(z_i - z_j)),
```

where the second term is the exact mean of `|z_i - z_j|` over all pairs of
the kept rows evaluated by one sort (`jflows.train._variation`, the Gini mean
difference); there is no random pairing. The first term is the forward KL up
to a target-only constant. `coeff_lambda = 0` is the forward KL.

The batch ESS of the step is the screened ESS of the proposal log weights
returned by the SMC,

```text
log w = U_source(x) - U_target(G^{-1}(x)) + log|det J_{G^{-1}}(x)|,
```

evaluated on the pushforward of the source batch before the levels moved it.

Example:

```python
flow, batch_ess = train_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    target.domain,
    batch_size=256,
    steps_total=1000,
    lr=1e-3,
    ladder=4,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    coeff_lambda=1.0,
    monitor=Monitor(100, "[KLX] "),
)
```

Generate target-side proposals with `flow.inv(x_valid)`.

## Data-driven training

`target_data` (KLX, KLL1, KLXX) replaces the SMC target surrogate by a given
target sample set: every step draws its `batch_size` target rows from that
array (without replacement within the step), the loss and its screens are
unchanged, and the reported batch ESS is that of the pushforward of the
source batch through the current map. For KLXX the pushforward half of the
mixture is that same detached pushforward. FAB and FABX have no data-driven
form, since their batches come from `pi^2 / nu`. `reject_requested` is the
manual rejection of `doc/03-high-level.md`, read within one gradient step.

## Molecular KLL1 and FAB trainers

`train_forward_KLL1_G` has the signature of `train_forward_KLX_G` and
replaces the variation by the dispersion of the log-ratio, the mean absolute
deviation of `z` from its screened batch mean (`jflows.train._dispersion`),
so the loss is the forward KL plus `coeff_lambda` times that dispersion.

`train_FAB_G` has the same signature without `coeff_lambda`. Each step
manufactures its batch from `pi^2 / nu` with `flow_fab_batch` (`ladder`
levels to the target, then `ladder` further levels on to `pi^2 / nu`) and
minimizes the screened mean log-ratio over it, whose parameter gradient is
that of the alpha = 2 divergence; there is no penalty term and no replay
buffer. The reported batch ESS is the phase-1 proposal ESS, as for the
forward KL family. Its intermediate levels differentiate through the flow,
so a step costs more than a KLX step at the same `mc_steps_1`.

## Molecular KLXX trainer

```python
train_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    domain,
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
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    mc_image_radius=3,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    chunks=1,
    initialize_from_identity=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    screen_fraction=SCREEN_FRACTION,
    t_start=0.0,
    t_end=1.0,
    reject_requested=None,
    target_data=None,
)
```

KLXX augments KLX with a second variation term evaluated on a mixture batch.
Before the optimizer scan it builds the quench-and-temper pool with
`mixed_quench_and_temper`:

- `pool_size=0`: quench and temper the complete `x_valid` population;
- `pool_size>0`: draw a random subset of that many rows (with replacement)
  from `x_valid` first.

The temper of the pool uses `mc_steps_2` MALA steps, `melt`, `opt_alpha`, and
`opt_steps` control the melt and the L-BFGS quench, and `coeff_qt > 0`
resamples the tempered pool by `exp(-coeff_qt * U_target)` and rejuvenates it
again, which removes tempered rows left at high energy. `chunks` is passed
directly into the quench and temper.

During each training step:

1. draw `batch_size` source rows from `x_valid` and manufacture the target
   batch `y` through the current flow with `flow_target_batch`, keeping its
   pushforward as the detached `y_bar` (no second inverse pass);
2. draw `y_hat` from the pool (with replacement) and rejuvenate it at
   `target` with `mc_steps_2` MALA steps;
3. resample `batch_size` rows of the concatenation of `y_hat` and `y_bar`
   with weights `coeff_alpha` on every `y_hat` row and `1 - coeff_alpha` on
   every `y_bar` row, the mixture batch `y_mix`; and
4. minimize the target KLX term plus `coeff_theta` times the variation of
   the log-ratio over the mixture batch.

The complete loss is

```text
mean(z) + coeff_lambda * mean_{i != j}(abs(z_i - z_j))
        + coeff_theta  * mean_{i != j}(abs(z_mix,i - z_mix,j)),
```

with `z` on the target batch, `z_mix` on the mixture batch, both variation
terms the exact sorted pair means, and both batches passed through the energy
screen and the top screen separately. `coeff_lambda` weights the
target-batch variation, `coeff_theta` the mixture variation, and
`coeff_alpha` is the quench-and-temper proportion of the mixture.

```python
flow, batch_ess = train_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    target.domain,
    pool_size=0,
    batch_size=256,
    steps_total=1000,
    lr=1e-3,
    ladder=4,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    chunks=8,
)
```

## Initialization and fair comparison

`initialize_from_identity=False` preserves the supplied parameterization.
This is useful for explicit warm starts. `True` standardizes the starting map
to the architecture's identity parameterization and is the high-level default.

Identity initialization primarily improves stability and comparison fairness;
it does not guarantee a larger final ESS. Always compare methods with the same
initialization policy when interpreting optimization results.

## Held-out evaluation

The returned batch ESS history is not a validation estimate. Evaluate the
trained G-flow on a complete held-out source population with the screened
ESS the generators use:

```python
from jflows_md.utils import SCREEN_FRACTION, compute_ESS_log

proposal, inverse_ladj = flow.inv_and_ladj(x_valid)
proposal = target.domain.wrap(proposal)
log_weight = source(x_valid) - target(proposal) + inverse_ladj
valid_ess = compute_ESS_log(log_weight, SCREEN_FRACTION)
```

Keep the weights in log form. For difficult molecular targets, report ESS
together with support, stereochemistry, energy, and mode diagnostics.

## Medium-level artifacts

`jflows_md.artifacts` re-exports the generic `jflows` artifact format:

```python
save_flow("flow.eqx", flow)
flow = load_flow("flow.eqx", template)

save_samples("samples.npy", samples)
samples = load_samples("samples.npy")

save_history("history.npz", batch_ess=batch_ess)
history = load_history("history.npz")
```

<div align="center">

<table>
<thead>
<tr><th>Function</th><th>Format</th><th>Important contract</th></tr>
</thead>
<tbody>
<tr><td><code>save_flow</code></td><td>Equinox leaf serialization</td><td>saves array leaves only</td></tr>
<tr><td><code>load_flow</code></td><td>Equinox leaf serialization</td><td>requires a matching architecture template</td></tr>
<tr><td><code>save_samples</code></td><td><code>.npy</code></td><td>pickle disabled</td></tr>
<tr><td><code>load_samples</code></td><td><code>.npy</code></td><td>returns a NumPy array</td></tr>
<tr><td><code>save_history</code></td><td><code>.npz</code></td><td>stores caller-named arrays</td></tr>
<tr><td><code>load_history</code></td><td><code>.npz</code></td><td>returns copied arrays in a dictionary</td></tr>
</tbody>
</table>

</div>

These helpers store individual medium-level objects. Use the high-level
Boltzmann writer/loader when accepted-stage atomicity and resume state are
required.

## Choosing a trainer

<div align="center">

<table>
<thead>
<tr><th>Trainer</th><th>Use when</th><th>Extra input</th></tr>
</thead>
<tbody>
<tr><td><code>train_forward_KLX_G</code></td><td>the forward KL (<code>coeff_lambda=0</code>) or the KLX on SMC-manufactured target batches suffices</td><td>none</td></tr>
<tr><td><code>train_forward_KLL1_G</code></td><td>the L1 dispersion of the log-ratio should replace the variation</td><td>none</td></tr>
<tr><td><code>train_FAB_G</code></td><td>the alpha = 2 divergence on exact two-phase SMC batches is wanted</td><td>none</td></tr>
<tr><td><code>train_forward_KLXX_G</code></td><td>a quench-and-temper pool should regularize the log-ratio on a wider mixture</td><td><code>pool_size</code>, <code>melt</code>, <code>opt_alpha</code>, <code>opt_steps</code>, <code>coeff_theta</code>, <code>coeff_alpha</code>, <code>coeff_qt</code>, <code>chunks</code></td></tr>
<tr><td><code>train_FABX_G</code></td><td>the FAB batch should carry the KLXX mixture term</td><td>the KLXX inputs without <code>coeff_lambda</code></td></tr>
</tbody>
</table>

</div>

Use the high-level generators when stage-size selection,
trained-versus-identity selection, and retry policy should be coordinated
automatically.

## Executable references

- `smoke/test_float32_training.py`: bounded default-float32 direct training,
  the guarded Adam update, gradient clipping, warmup, and the batch ESS
  against the screened ESS of the pushforward weights.
- `smoke/test_initialization.py`: the `initialize_from_identity` defaults of
  the trainers and generators.
- `smoke/test_artifacts.py`: direct artifact round trips.
- `smoke/test_api_consistency.py`: exact public signatures, a two-step
  `train_forward_KLX_G` on `Mixed_Identity`, and return shapes.
