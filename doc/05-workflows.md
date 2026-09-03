# Molecular workflows

This page collects complete patterns that connect the low-, medium-, and
high-level interfaces. The one executable example is
[`example/methane_9d_raw/`](../example/methane_9d_raw/), a 9D methane
Boltzmann generator driver (`train.py --method id|kl|klxx`) whose
`parameters.py` records every control; the recipes here are documentation,
while the narrow executable evidence lives under `smoke/`.

## Workflow map

<div align="center">

<table>
<thead>
<tr><th>Goal</th><th>Primary interfaces</th><th>Executable evidence</th></tr>
</thead>
<tbody>
<tr><td>inspect a frozen molecular target</td><td><code>Molecular_Bundle</code>, <code>Molecular_Potential</code></td><td><code>test_bundles.py</code>, <code>test_molecular_potential.py</code></td></tr>
<tr><td>construct a mixed flow</td><td><code>Mixed_NSF</code></td><td><code>test_mixed_nsf.py</code></td></tr>
<tr><td>manufacture target samples through a flow</td><td><code>sequential_monte_carlo</code>, <code>sequential_monte_carlo_fab</code>, <code>mixed_mala</code>, <code>mixed_hmc</code></td><td><code>test_smc_hmc.py</code></td></tr>
<tr><td>train one fixed stage</td><td><code>train_forward_KLX_G</code>, <code>train_forward_KLL1_G</code>, <code>train_FAB_G</code>, <code>train_forward_KLXX_G</code>, <code>train_FABX_G</code></td><td><code>test_float32_training.py</code>, <code>test_api_consistency.py</code></td></tr>
<tr><td>run adaptive staging without flow training</td><td><code>boltzmann_identity</code></td><td><code>test_boltzmann_md.py</code></td></tr>
<tr><td>run adaptive staging with flow training</td><td><code>boltzmann_forward_KLX_G</code>, <code>boltzmann_forward_KLL1_G</code>, <code>boltzmann_FAB_G</code>, <code>boltzmann_forward_KLXX_G</code>, <code>boltzmann_FABX_G</code></td><td><code>test_boltzmann_md.py</code></td></tr>
<tr><td>run a fixed stage schedule</td><td><code>boltzmann_forward_KLX_G_fixed</code>, <code>boltzmann_forward_KLXX_G_fixed</code>, <code>iterate_boltzmann(t_list=...)</code></td><td><code>test_boltzmann_md.py</code></td></tr>
<tr><td>resume complete stages</td><td><code>iterate_boltzmann</code>, <code>boltzmann.load.run</code></td><td><code>test_boltzmann_md.py</code>, <code>example/methane_9d_raw/train.py</code></td></tr>
<tr><td>replay a stored run on fresh source particles</td><td><code>run_inference</code></td><td><code>test_boltzmann_md.py</code></td></tr>
<tr><td>run native Cartesian dynamics</td><td><code>OpenMM_Potential</code>, <code>langevin</code>, <code>parallel_tempering</code></td><td><code>test_openmm.py</code></td></tr>
<tr><td>construct a new bundle</td><td><code>jflows_md.bundle_build</code></td><td><code>test_bundles.py</code></td></tr>
</tbody>
</table>

</div>

## Load and inspect a target

```python
import jax

from jflows_md import Molecular_Bundle, Molecular_Potential, available_bundles

bundle_root = "downloaded-bundles"
bundle_names = available_bundles(bundle_root)
bundle = Molecular_Bundle.load(bundle_names[0], root=bundle_root)
target = Molecular_Potential(bundle)
source = target.source()

print(bundle.name, bundle.n_atoms, target.dimension)
print(target.domain.euclidean_dim, target.domain.periodic_dim)

q_ref = target.reference_internal()[None]
x_ref = target.cartesian(q_ref)
u_ref = target(q_ref)
e_ref = target.physical_energy(q_ref)

q = source.samples(jax.random.key(0), N=10000)
support = target.support_mask(target.cartesian(q))
```

`target(q)` is reduced and includes the coordinate Jacobian.
`target.physical_energy(q)` is the underlying Cartesian force-field energy in
kJ/mol. Keep those quantities distinct in diagnostics.

## Construct and check a mixed flow

```python
import jax.numpy as jnp

from jflows_md import Mixed_NSF

flow = Mixed_NSF(
    jax.random.key(1),
    target.domain,
    bins=16,
    transforms=6,
    euclidean_bound=6.0,
    hidden_features=(128, 128),
    mask_strategy="balanced",
).zeros()

q = source.samples(jax.random.key(2), N=128)
y, forward_ladj = flow.call_and_ladj(q)
q_back, inverse_ladj = flow.inv_and_ladj(y)

error = jnp.max(jnp.abs(target.domain.displacement(q_back, q)))
jacobian_error = jnp.max(jnp.abs(forward_ladj + inverse_ladj))
```

Check both domain-aware reconstruction and opposite log-Jacobian signs before
using a new architecture in training.

## Manufacture target samples through a flow

```python
from jflows_md.utils import SCREEN_FRACTION, sequential_monte_carlo

x = source.samples(jax.random.key(3), N=20000)

y, proposal, proposal_log_weight = sequential_monte_carlo(
    jax.random.key(4),
    x,
    source,
    target,
    flow,
    ladder=8,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    mc_image_radius=3,
    domain=target.domain,
    chunks=8,
    screen_fraction=SCREEN_FRACTION,
)
```

The source particles are pushed through `flow.inv`; each of the `ladder`
levels reweights by the `1/ladder`-th power of the proposal-to-target
weight, resamples with the screened weights, and rejuvenates under `target`
(`mc_steps_1` MALA steps on the intermediate levels, `mc_steps_2` on the
last). `proposal` is the pushforward the levels
started from and `proposal_log_weight` its full proposal-to-target log
weight. This is the SMC target surrogate the trainers run inside every
optimizer step.

## Train one KLX stage

```python
from jflows.train import Monitor
from jflows_md.train import train_forward_KLX_G

x_valid = source.samples(jax.random.key(5), N=100000)

flow, batch_ess = train_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    target.domain,
    batch_size=5000,
    steps_total=500,
    lr=1e-3,
    ladder=8,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    coeff_lambda=1.0,
    monitor=Monitor(50, "[KLX] "),
    u_clip=1e3,
    g_clip=1e2,
    lr_warmup=25,
)
```

Every step draws `batch_size` source rows from `x_valid` and manufactures its
target batch through the current flow; `coeff_lambda=0.0` trains the forward
KL. Evaluate a held-out source population after training with the screened
ESS:

```python
from jflows_md.utils import SCREEN_FRACTION, compute_ESS_log

x_test = source.samples(jax.random.key(7), N=100000)
proposal, inverse_ladj = flow.inv_and_ladj(x_test)
proposal = target.domain.wrap(proposal)
log_weight = source(x_test) - target(proposal) + inverse_ladj
valid_ess = compute_ESS_log(log_weight, SCREEN_FRACTION)
```

The flow is G-native; `proposal = flow.inv(x_test)` is the generated
target-side sample.

## Train one KLXX stage

```python
from jflows_md.train import train_forward_KLXX_G

flow, batch_ess = train_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    target.domain,
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
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    chunks=32,
    u_clip=1e3,
    g_clip=1e2,
    lr_warmup=25,
)
```

The quench-and-temper pool is built once from `x_valid` (`pool_size=0`) with
`mixed_quench_and_temper` before the scan; inside every optimizer step
`batch_size` pool rows are rejuvenated with `mc_steps_1` MALA steps and mixed
with the detached pushforward of the source batch.

## Run adaptive-staging KLXX

```python
from jflows_md.boltzmann import boltzmann_forward_KLXX_G

x_valid = source.samples(jax.random.key(9), N=100000)

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
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    chunks=32,
    u_clip=1e3,
    g_clip=1e2,
    lr_warmup=25,
    bg_param={
        "t_safe": 0.25,
        "shrink_factor": 0.7,
        "enlarge_factor": 1.5,
        "tau_valid": 0.4,
        "t_tol": 1e-3,
        "max_stages": 20,
        "max_retry": 5,
    },
    monitor=Monitor(10, "[adaptive-staging KLXX] "),
)

if not stages or stages[-1]["t"] != 1.0:
    raise RuntimeError("adaptive-staging molecular run did not reach its endpoint")
```

At each stage, inspect at least:

```python
for record in stages:
    print(
        record["t_start"],
        record["t"],
        record["selected"],
        record["valid_selected_ess"],
        record["valid_trained_ess"],
        record["valid_identity_ess"],
    )
```

The numerical values above are those of `example/methane_9d_raw/parameters.py`
for the 9D methane target, an interface example rather than universal
molecular tuning.

## Run a fixed stage schedule

`boltzmann_forward_KLXX_G_fixed` takes the same inputs with `t_list` in place
of `bg_param`; every stage trains once to the next endpoint and is accepted
whatever its ESS:

```python
from jflows_md.boltzmann import boltzmann_forward_KLXX_G_fixed

samples, stages, effective_seconds = boltzmann_forward_KLXX_G_fixed(
    x_valid, source, target, flow,
    0, 5000, 500, 1e-3, 8, 1.0, 1e-2, 200, 1e-3, 20, 100,
    (0.25, 0.5, 0.75, 1.0),
    chunks=32, u_clip=1e3, g_clip=1e2, lr_warmup=25,
)
```

`iterate_boltzmann(..., t_list=...)` is the resumable form of the same
schedule.

## Run a regularization path

Every generator and `iterate_boltzmann` accept `rg_param_0` and
`rg_param_1`, the endpoints `(e [kJ/mol], r [nm])` of the regularization path
of `target.regularized`; the stage targets `U_{t,t}` soften the potential
early and tighten it stage by stage along the diagonal:

```python
samples, stages, effective_seconds = boltzmann_forward_KLX_G(
    x_valid, source, target, flow,
    10000, 2000, 1e-3, 8, 1e-3, 20, 100,
    bg_param={"t_safe": 0.15}, chunks=32,
    rg_param_0=(50.0, 0.25), rg_param_1=(100.0, 0.15),
)
for record in stages:
    print(record["t"], record["rg_end"], record["valid_selected_ess"])
```

## Persist and resume adaptive-staging runs

```python
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

controls = dict(
    objective="forward_klxx",
    pool_size=0,
    batch_size=5000,
    steps_total=500,
    lr=1e-3,
    ladder=8,
    mc_dt=1e-3,
    mc_steps_1=20,
    mc_steps_2=100,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    monitor=Monitor(10, "[resume KLXX] "),
    bg_param={"t_safe": 0.25, "enlarge_factor": 1.5, "tau_valid": 0.4},
    chunks=32,
    mc_image_radius=3,
    checkpoint=True,
    u_clip=1e3,
    g_clip=1e2,
    lr_warmup=25,
    screen_fraction=1e-4,
    seed=0,
)

def iterate(samples, continuation, accepted_t, start_stage):
    return iterate_boltzmann(
        samples,
        source,
        target,
        continuation,
        accepted_t=accepted_t,
        start_stage=start_stage,
        **controls,
    )

config = {
    "bundle": target.bundle_name,
    "valid_size": int(x_valid.shape[0]),
    **{key: value for key, value in controls.items() if key != "monitor"},
}

samples, stages = run(
    "runs/alanine_dipeptide",
    "alanine_dipeptide-klxx",
    config,
    x_valid,
    flow,
    iterate,
)
```

To resume, reconstruct the same target, source, flow architecture, controls,
problem identifier, and config, then call:

```python
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

The stored post-stage population is the continuation state. Do not attempt to
reconstruct it by composing stage flows. An identity run uses
`iterate_identity` in the callback and passes `flow=None` to `run`, as
`example/methane_9d_raw/train.py --method id` does.

## Replay a stored run on fresh source particles

```python
import numpy as np

from jflows_md import run_inference

manifest = run_inference(
    "runs/alanine_dipeptide",
    "runs/alanine_dipeptide-inference",
    flow,
    source,
    target,
    sample_count=1_000_000,
    mc_dt=1e-3,
    mc_steps=100,
    chunk_size=10_000,
    mc_image_radius=3,
    seed=0,
)

inference_set = np.load("runs/alanine_dipeptide-inference/" + manifest["inference_samples_path"])
for item in manifest["stages"]:
    print(item["stage"], item["t"], item["pushforward_ess"], item["mala_acceptance_mean"])
```

The stored run must be `complete`. For every stage the scheme pushes the
population through the stored selected map, resamples with the screened
stage weights (`SCREEN_FRACTION = 1e-4` by default), and
rejuvenates with `mc_steps` MALA steps under the stage target, chunk by
chunk on disk memmaps. The output directory must be empty; it receives
`run.json` and one `stage_XXXXXX/` directory per stage with `samples.npy`,
`log_weights.npy`, and `metadata.json`.

## Run native OpenMM Langevin dynamics

```python
from jflows_md.openmm import OpenMM_Potential, langevin

target_mm = OpenMM_Potential.from_bundle("alanine_dipeptide_ff96_obc1")

trajectory, energies = langevin(
    target_mm,
    steps=100000,
    sample_interval=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    seed=0,
    platform="CUDA",
)
```

`trajectory` is Cartesian and measured in nm. `energies` contains the raw
OpenMM potential energy in kJ/mol of the bundle System that the integrator
runs on.

## Run native OpenMM parallel tempering

```python
from jflows_md.openmm import parallel_tempering

replicas, energies, swap_acceptance = parallel_tempering(
    target_mm,
    temperatures_kelvin=(300.0, 360.0, 432.0, 518.4),
    rounds=1000,
    steps_per_round=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    seed=0,
    platform="CUDA",
)
```

The first replica axis is the fixed temperature-slot order supplied by
`temperatures_kelvin`. Inspect adjacent-pair `swap_acceptance` before using a
temperature grid for production sampling. The numerical grid above is an
interface example, not a universal recommendation.

## Build an external bundle

Bundle construction is an offline operation and requires the `bundles`
optional dependency set:

```bash
python -m jflows_md.bundle_build \
  alanine_dipeptide molecule.prmtop molecule.rst7 generated/alanine_dipeptide
```

Construction writes a new six-file directory. It audits the OpenMM System,
extracts the static JAX force-field arrays, defines the internal-coordinate
chart, and stores independent OpenMM validation frames. Never overwrite an
active input bundle.

Preset builds supply their coordinate configuration explicitly. The one
CLI key is `alanine_dipeptide`. For another explicitly
prepared molecule, call `jflows_md.bundle_build.write_bundle(...)` and pass
any required `zmatrix`, `fixed_stereocenters`, and
`signed_volume_diagnostics` values. The builder supports multiple fixed
tetrahedral centers but deliberately does not infer them or their CIP
priorities from topology. A chiral specification can include
`configuration` plus a highest-to-lowest `cip_priority_atoms` tuple so a
reference with the wrong requested R/S configuration is rejected; see the
low-level fixed-stereochemistry contract. The only name-based behavior is the
compatibility route for a target-only `alanine_dipeptide` call with all new
coordinate options at their defaults, which reproduces its old schema-v2
coordinate dictionary.

## Reproducibility checklist

- Record the bundle path/name and manifest.
- Record the target temperature and the regularization path `rg_param_0`,
  `rg_param_1` of the run.
- Record source, flow architecture, initialization policy, and all seeds.
- Record particle, pool, batch, SMC ladder, chunk, `mc_steps_1`,
  `mc_steps_2`, quench-and-temper, screen-fraction, and optimizer controls.
- Require `stages[-1]["t"] == 1.0` before calling an adaptive-staging run
  complete.
- Preserve post-stage populations and complete stage records for resume.
- Evaluate held-out ESS, support, energy, stereochemistry, and molecular
  modes; for the final sample set, replay the complete run with
  `run_inference` and record its manifest.
- For OpenMM, record platform, timestep, friction, temperature grid, exchange
  interval, and swap acceptance.
- Distinguish smoke evidence from production convergence evidence.
