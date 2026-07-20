# Medium-level interfaces

The medium level trains one inverse molecular flow on already prepared,
fixed-shape sample pools. It combines molecular G-direction objectives,
compiled Adam scans, deterministic key derivation, optional monitoring, and
optional rematerialization. It does not select an annealing endpoint, construct
an adaptive SMC bridge, sharpen the regularization, or persist a multi-stage
run.

Public imports:

```python
from jflows.train import Monitor
from jflows_md.train import train_forward_KLX_G, train_forward_KLXX_G
from jflows_md.artifacts import (
    save_flow, load_flow,
    save_samples, load_samples,
    save_history, load_history,
)
```

## Shared contract

Both trainers consume:

- `target_samples`: a fixed target-side pool for the forward objective;
- `source_samples`: a fixed source-side pool used for proposal diagnostics;
- `source` and `target`: reduced potentials for the current stage;
- `flow`: a `Mixed_NSF` or compatible mixed-domain G-flow;
- `batch_size`, `train_steps`, and `lr`: Adam controls;
- `seed`: the deterministic trainer key namespace; and
- optional `monitor`: normally `jflows.train.Monitor`.

Both return exactly:

```python
trained_flow, batch_ess_hist
```

`batch_ess_hist.shape == (train_steps,)`. It measures the pre-update
source-proposal ESS on the optimizer batch. It is a monitor, not an acceptance
gate and not a full-validation result.

The supplied flow is used as-is unless `initialize_from_identity=True`, in
which case training starts from `flow.zeros()`. Equinox flows are immutable, so
always rebind the returned flow.

## Execution model

Each trainer is one outer `eqx.filter_jit` function containing a complete Adam
`lax.scan`:

```text
fixed input pools
  -> compiled scan over train_steps
     -> draw target/source rows
     -> construct the objective batch
     -> evaluate loss and gradients
     -> apply one Adam update
     -> record proposal ESS
  -> trained flow and ESS history
```

The fixed pool shapes, batch size, training length, flow architecture, and
checkpoint choice participate in compilation. High-level SMC, QT, and
sharpening remain outside this medium-level interface.

## Common controls

<div align="center">

<table>
<thead>
<tr><th>Control</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>batch_size</code></td><td>rows drawn without replacement from the target and source pools</td></tr>
<tr><td><code>train_steps</code></td><td>number of Adam updates</td></tr>
<tr><td><code>lr</code></td><td>Adam learning rate</td></tr>
<tr><td><code>coeff_lambda</code></td><td>coefficient of the target-sample X penalty</td></tr>
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

Rows with nonfinite target energies or nonfinite log-density ratios never enter
the molecular loss. Adam commits parameters, both moment trees, and its update
counter atomically only when the loss, gradients, and complete candidate state
are finite. A rejected step therefore cannot poison a later finite update.

## Monitoring

The molecular trainers use the generic monitor:

```python
monitor = Monitor(every=20, prefix="[molecular KLX] ")
```

The trainer calls `monitor.report(...)` from the compiled scan. The printed
loss is the current optimizer objective, and ESS is the current source-batch
proposal ESS. `t_start` and `t_end` are display labels; they do not alter the
provided source or target potentials.

## Molecular KLX trainer

```python
train_forward_KLX_G(
    target_samples,
    source_samples,
    source,
    target,
    flow,
    batch_size,
    train_steps,
    lr,
    coeff_lambda=1.0,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    initialize_from_identity=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    t_start=0.0,
    t_end=1.0,
)
```

For target-side rows `y`, define

```text
x = G(y),
z(y) = U_source(x) - U_target(y) - log|det J_G(y)|.
```

The per-step objective is

```text
mean z + lambda mean |z - z_permuted|.
```

The first term is forward KL up to a target-only constant. The second controls
the spread of the log density ratio on target samples. At each scan step the
trainer independently draws target rows, source rows, and a permutation.

The source rows are mapped with `flow.inv_and_ladj` only to compute the
pre-update proposal ESS:

```text
log w = U_source(x_source) - U_target(G^{-1}(x_source))
        + log|det J_G^{-1}(x_source)|.
```

Example:

```python
flow, batch_ess = train_forward_KLX_G(
    target_samples,
    source_samples,
    source,
    target,
    flow,
    batch_size=256,
    train_steps=1000,
    lr=1e-3,
    coeff_lambda=1.0,
    monitor=Monitor(100, "[KLX] "),
)
```

Generate target-side proposals with `flow.inv(source_samples)`.

## Molecular KLXX trainer

```python
train_forward_KLXX_G(
    target_samples,
    source_samples,
    hat_samples,
    source,
    target,
    flow,
    domain,
    batch_size,
    train_steps,
    lr,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_dt=1e-3,
    mc_steps=1,
    mc_image_radius=3,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    initialize_from_identity=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    lr_warmup=0,
    t_start=0.0,
    t_end=1.0,
)
```

KLXX adds a second X penalty evaluated on a mixture of:

- `y_hat`: rows sampled with replacement from `hat_samples`, then freshened by
  mixed MALA at `target`; and
- `y_bar`: current flow proposals obtained from source rows.

The mixture resampling weights are `coeff_alpha` for every `y_hat` row and
`coeff_beta` for every `y_bar` row. The resulting `batch_size` rows define the
mixture log-density-ratio penalty. The target-sample KLX term remains present.

Conceptually, the objective is

```text
KL term on target_samples
+ lambda X term on target_samples
+ (alpha + beta)^2 X term on the resampled hat/proposal mixture.
```

The direct trainer does not create `hat_samples`; callers prepare that pool
with `mixed_quench_and_temper` or another method. It also has no `chunks`
argument: only the selected `batch_size` hat rows are MALA-freshened inside
the compiled scan.

```python
hat_samples, _ = mixed_quench_and_temper(
    jax.random.key(3),
    source_samples,
    target,
    target.domain,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-4,
    mc_steps=20,
    chunks=8,
)

flow, batch_ess = train_forward_KLXX_G(
    target_samples,
    source_samples,
    hat_samples,
    source,
    target,
    flow,
    target.domain,
    batch_size=256,
    train_steps=1000,
    lr=1e-3,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_dt=1e-4,
    mc_steps=1,
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
trained G-flow on a complete held-out source population:

```python
from jflows.utils import compute_ESS_log

proposal, inverse_ladj = flow.inv_and_ladj(x_valid)
log_weight = source(x_valid) - target(proposal) + inverse_ladj
valid_ess = compute_ESS_log(log_weight)
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

These helpers save individual medium-level objects. Use the high-level
Boltzmann writer/loader when accepted-stage atomicity and resume state are
required.

## Choosing a trainer

<div align="center">

<table>
<thead>
<tr><th>Trainer</th><th>Use when</th><th>Extra input</th></tr>
</thead>
<tbody>
<tr><td><code>train_forward_KLX_G</code></td><td>a fixed target pool adequately represents the stage</td><td>none</td></tr>
<tr><td><code>train_forward_KLXX_G</code></td><td>an independently broadened pool should regularize proposal coverage</td><td><code>hat_samples</code>, domain, and hat-MALA controls</td></tr>
</tbody>
</table>

</div>

Use the high-level generators when target-pool construction, stage-size
selection, trained-versus-identity selection, regularization sharpening, and
retry policy should be coordinated automatically.

## Executable references

- `smoke/test_float32_training.py`: bounded default-float32 direct training.
- `smoke/test_initialization.py`: supplied-flow and identity starts.
- `smoke/test_mixed_training.py`: KLX/KLXX returns and tiny training paths.
- `smoke/test_artifacts.py`: direct artifact round trips.
- `smoke/test_api_consistency.py`: exact public signatures and return shape.
