# Molecular workflows

This page collects complete patterns that connect the low-, medium-, and
high-level interfaces. The repository currently has no separate `example/`
directory; the recipes here are documentation, while the corresponding
executable evidence lives under `smoke/`.

## Workflow map

<div align="center">

<table>
<thead>
<tr><th>Goal</th><th>Primary interfaces</th><th>Executable evidence</th></tr>
</thead>
<tbody>
<tr><td>inspect a frozen molecular target</td><td><code>Molecular_Bundle</code>, <code>Molecular_Potential</code></td><td><code>test_bundles.py</code>, <code>test_molecular_potential.py</code></td></tr>
<tr><td>construct a mixed flow</td><td><code>Mixed_NSF</code></td><td><code>test_mixed_nsf.py</code></td></tr>
<tr><td>run a custom particle bridge</td><td><code>sequential_monte_carlo</code>, <code>mixed_mala</code></td><td><code>test_support_and_utils.py</code></td></tr>
<tr><td>train one fixed stage</td><td><code>train_forward_KLX_G</code>, <code>train_forward_KLXX_G</code></td><td><code>test_float32_training.py</code>, <code>test_mixed_training.py</code></td></tr>
<tr><td>run adaptive regularization sharpening without flow training</td><td><code>boltzmann_identity</code></td><td><code>test_boltzmann_identity.py</code></td></tr>
<tr><td>run adaptive regularization sharpening with flow training</td><td><code>boltzmann_forward_KLX_G</code>, <code>boltzmann_forward_KLXX_G</code></td><td><code>test_sharpening_gate.py</code></td></tr>
<tr><td>resume complete stages</td><td><code>iterate_boltzmann</code>, <code>boltzmann.load.run</code></td><td><code>test_boltzmann_integration.py</code></td></tr>
<tr><td>run native Cartesian dynamics</td><td><code>OpenMM_Potential</code>, <code>langevin</code>, <code>parallel_tempering</code></td><td><code>test_openmm.py</code></td></tr>
<tr><td>construct a new bundle</td><td><code>jflows_md.bundle_build</code></td><td><code>test_bundles.py</code></td></tr>
</tbody>
</table>

</div>

## Load and inspect a target

```python
import jax

from jflows_md import Molecular_Bundle, Molecular_Potential, available_bundles

bundle_root = "/path/to/downloaded/bundles"
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

## Run a low-level potential bridge

```python
from jflows_md.utils import sequential_monte_carlo

soft = target.regularized((50.0, 0.10))
x = source.samples(jax.random.key(3), N=20000)

y, level_ess, acceptance = sequential_monte_carlo(
    jax.random.key(4),
    x,
    source,
    soft,
    ladder=16,
    mc_dt=1e-4,
    mc_steps=10,
    mc_image_radius=3,
    domain=target.domain,
    chunks=8,
)
```

This is exact potential-space SMC for the specified schedule: incremental
weighting, resampling, and MALA all use the matching intermediate bridge.

## Train one KLX stage

Prepare fixed target and source pools outside the trainer:

```python
from jflows.train import Monitor
from jflows_md.train import train_forward_KLX_G

source_samples = source.samples(jax.random.key(5), N=50000)
target_samples, _, _ = sequential_monte_carlo(
    jax.random.key(6),
    source_samples,
    source,
    soft,
    ladder=16,
    mc_dt=1e-4,
    mc_steps=10,
    domain=target.domain,
    chunks=8,
)

flow, batch_ess = train_forward_KLX_G(
    target_samples,
    source_samples,
    source,
    soft,
    flow,
    batch_size=512,
    train_steps=1000,
    lr=1e-3,
    coeff_lambda=1.0,
    monitor=Monitor(100, "[KLX] "),
)
```

Evaluate a held-out source population after training:

```python
from jflows.utils import compute_ESS_log

x_valid = source.samples(jax.random.key(7), N=100000)
proposal, inverse_ladj = flow.inv_and_ladj(x_valid)
log_weight = source(x_valid) - soft(proposal) + inverse_ladj
valid_ess = compute_ESS_log(log_weight)
```

The flow is G-native; `proposal = flow.inv(x_valid)` is the generated
target-side sample.

## Train one KLXX stage

```python
from jflows_md.train import train_forward_KLXX_G
from jflows_md.utils import mixed_quench_and_temper

hat_samples, hat_acceptance = mixed_quench_and_temper(
    jax.random.key(8),
    source_samples,
    soft,
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
    soft,
    flow,
    target.domain,
    batch_size=512,
    train_steps=1000,
    lr=1e-3,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_dt=1e-4,
    mc_steps=1,
)
```

The QT pool is constructed once for this direct call; selected hat rows are
freshened inside every optimizer step.

## Run adaptive KLXX with sharpening

```python
from jflows_md.boltzmann import boltzmann_forward_KLXX_G

x_valid = source.samples(jax.random.key(9), N=200000)

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
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    chunks=32,
    bg_param={"tau_smc": 0.75, "tau_ess": 0.6},
    monitor=Monitor(50, "[adaptive KLXX] "),
)

if not stages or stages[-1]["t"] != 1.0:
    raise RuntimeError("adaptive molecular run did not reach its endpoint")
```

At each stage, inspect at least:

```python
for record in stages:
    print(
        record["t_start"],
        record["t"],
        record["selected"],
        record["valid_selected_ess"],
        record["sharpen_ess"],
        record["population_rg"],
    )
```

Use `pool_size=0` when SMC and training selection should consume the complete
current population. A positive value creates a separately sampled selection
pool but leaves final validation and stage output at full size.

## Persist and resume adaptive stages

```python
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

controls = dict(
    objective="forward_klxx",
    pool_size=50000,
    batch_size=10000,
    train_steps=500,
    lr=1e-3,
    ladder=8,
    mc_dt=1e-4,
    mc_steps=20,
    rg_param_0=(20.0, 0.15),
    rg_param_1=(1000.0, 0.0),
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    monitor=Monitor(50, "[resume KLXX] "),
    bg_param={"tau_smc": 0.75, "tau_ess": 0.6},
    chunks=32,
    mc_image_radius=3,
    checkpoint=False,
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
    "rg_param_0": controls["rg_param_0"],
    "rg_param_1": controls["rg_param_1"],
    "controls": {key: value for key, value in controls.items() if key != "monitor"},
}

samples, stages = run(
    "runs/glycerol",
    "glycerol-klxx",
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
    "runs/glycerol",
    "glycerol-klxx",
    config,
    None,
    flow,
    iterate,
    resume=True,
)
```

The saved post-sharpen population is the continuation state. Do not attempt to
reconstruct it by composing stage flows.

## Run native OpenMM Langevin dynamics

```python
from jflows_md.openmm import OpenMM_Potential, langevin

target_mm = OpenMM_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
soft_mm = target_mm.regularized((50.0, 0.10))

trajectory, energies = langevin(
    soft_mm,
    steps=100000,
    sample_interval=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    seed=0,
    platform="CUDA",
)
```

`trajectory` is Cartesian and measured in nm. `energies` contains the active
regularized OpenMM energy in kJ/mol. To run the physical endpoint, pass
`target_mm` instead of `soft_mm`.

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
temperature ladder for production sampling. The numerical ladder above is an
interface example, not a universal recommendation.

## Build an external bundle

Bundle construction is an offline operation and requires the `bundles`
optional dependency set:

```bash
python -m jflows_md.bundle_build \
  glycerol molecule.prmtop molecule.rst7 generated/glycerol
```

Construction writes a new six-file directory. It audits the OpenMM System,
extracts the static JAX force-field arrays, defines the internal-coordinate
chart, and stores independent OpenMM validation frames. Never overwrite an
active input bundle.

## Reproducibility checklist

- Record the bundle path/name and manifest.
- Record the target temperature and both regularization endpoints.
- Record source, flow architecture, initialization policy, and all seeds.
- Record particle, pool, batch, ladder, chunk, MALA, QT, and optimizer controls.
- Require `stages[-1]["t"] == 1.0` before calling an adaptive run complete.
- Preserve post-sharpen populations and complete stage records for resume.
- Evaluate held-out ESS, support, energy, stereochemistry, and molecular modes.
- For OpenMM, record platform, timestep, friction, temperature ladder, exchange
  interval, and swap acceptance.
- Distinguish smoke evidence from production convergence evidence.
