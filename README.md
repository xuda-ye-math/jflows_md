# jflows_md

`jflows_md` is the molecular companion of `jflows`: bundle-backed molecular
potentials, normalizing flows on the mixed domain `R^p x T^q`, molecular
trainers with a flow-proposal SMC target surrogate, mixed-domain MALA and HMC,
adaptive-staging Boltzmann generators with complete-stage resume, and an
inference scheme that replays a stored run on fresh source particles.

The numerical core is a raw-performance layer: constructors and compiled
kernels assume valid shapes, dtypes, schedules, chunk counts, and physical
parameters. Bundle construction is an offline step, and persistence is
separate from computation.

## Reduced potential

`Molecular_Potential` owns the inverse temperature. For internal coordinates
$q$ and their canonical Cartesian representative $x(q)$ it evaluates

$$
U(q)=\beta E(x(q))-\log J(q), \qquad \beta=\frac{1}{k_{\mathrm B}T},
$$

with $E$ the raw force-field energy in kJ/mol. The trainers and the Boltzmann
controllers consume reduced potentials and apply no further temperature
factor; the coordinate Jacobian is never regularized.

```python
import jax

from jflows_md import Mixed_NSF, Molecular_Potential

target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1", temperature_kelvin=300.0)
source = target.source()
samples = source.samples(jax.random.key(0), 32)

flow = Mixed_NSF(
    jax.random.key(1), target.domain, bins=32, transforms=6, hidden_features=(256, 256),
).zeros()

reduced = target.reduced_energy(samples)    # beta E(x(q)), no Jacobian
physical = target.physical_energy(samples)  # E(x(q)) in kJ/mol
```

## Regularization

The Lennard-Jones core grows as $r^{-12}$, so an early flow proposal can place
a few samples at energies far beyond the thermally occupied region, and one
such sample can dominate a batch, a weight vector, or a gradient. The package
addresses this with one regularization of the target and one screen of the
importance weights, and with further safeguards that leave the target
unchanged.

**The regularized potential and its path.** `target.regularized((e, r))` is
the two-parameter potential $U^{\rho}$, $\rho=(e, r)$: the regular and
exception nonbonded pair distances are floored at $r$ (nm), and the excess
$\Delta E$ of the floored energy over the floored energy of the bundle
reference geometry is compressed by $\mathcal C_e(\Delta E)=\Delta E$ below
the threshold $e$ (kJ/mol) and $e\,[1+\log(\Delta E/e)]$ above it, before
$\beta$ and the Jacobian term. The physical potential is the limit
$e\to\infty$, $r\to0$. A Boltzmann generator run follows the path
$\rho_s=(1-s)\rho_0+s\rho_1$ (`rg_param_0`, `rg_param_1`) on the diagonal of
its stage schedule: the stage targets are $U_{t,t}=(1-t)U_0+tU^{\rho_t}$, and
the stage $t_{k-1}\to t_k$ trains, selects, and advances the population from
$U_{t_{k-1},t_{k-1}}$ to $U_{t_k,t_k}$ in one step. A soft start (low $e$,
wide $r$) makes the first increments feasible for the larger molecules;
`rg_param_0 = rg_param_1` is a fixed regularization. Observables are always
evaluated with the physical energy.

**Screened ESS and loss (`screen_fraction`).** Every ESS, every resampling
weight vector, and every loss batch drops the `screen_fraction` largest log
weights (`SCREEN_FRACTION = 1e-4`): a kept row whose log-ratio is far above
the rest is a hole of the pushforward density at an ordinary target point and
would otherwise dominate the ESS or the gradient. One default serves the
trainers, the generators, the SMC, quench and temper, and the inference.

**Safeguards.**

- A non-finite force-field value stays non-finite; the screen gives it weight
  zero and the trainers' energy screen excludes it from the loss.
- `u_clip`: batch rows whose reduced energy exceeds `u_clip` (kT) are excluded
  from the loss and its penalties.
- `g_clip`: the global gradient norm fed to Adam is clipped; a non-finite loss
  or gradient skips the update.
- `coeff_qt`: after the melt, quench, and temper of the KLXX pool, each row is
  reweighted by $\exp(-c_{\mathrm{QT}}U)$, resampled, and rejuvenated again,
  which removes the rows the quench carried away along the singular core.
- MALA at every SMC level of the forward KL family, so that no gradient of the
  flow's log-determinant is taken inside the sampler; the exact two-phase SMC
  is reserved for FAB.

## Layout

```text
jflows_md/
├── artifacts.py              template-based artifacts
├── boltzmann/
│   ├── __init__.py           adaptive-staging and fixed-schedule generators
│   ├── control.py            Manual_Reject: Ctrl+C rejects the running attempt
│   ├── inference.py          replay of a stored run on fresh source particles
│   ├── write.py              atomic complete-stage writer
│   └── load.py               load, inspect, fork, and resume
├── bundle_build/             OpenMM-side bundle construction
├── core/                     BAT coordinates, Amber/OBC force field, domain, splines
├── flow.py                   Mixed_Identity and Mixed_NSF
├── openmm/                   native OpenMM potential and samplers
├── potential.py              reduced potential and the regularized (e, r) potential
├── source.py                 Gaussian x uniform-torus source
├── system.py                 runtime bundle loading
├── train.py                  compiled KLX, KLL1, FAB, KLXX, and FABX trainers
└── utils/
    ├── anneal.py             flow-proposal SMC and the two-phase FAB SMC
    ├── control.py            rejection check shared by the long steps
    ├── quench.py             quench and temper
    ├── rejuvenation.py       wrapped mixed-domain MALA and HMC
    └── screen.py             importance-weight screen for ESS and resampling
```

The interface manual is in [`doc/`](doc/README.md); the executable contracts
are in [`smoke/`](smoke/).

## Native OpenMM

The same bundle instantiates an independent Cartesian OpenMM potential,
`beta E(x)` without JAX or the coordinate Jacobian, with the same
`regularized((e, r))` form. Native Langevin and replica-exchange runs
integrate the bundle's OpenMM System.

```python
from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering

target = OpenMM_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
trajectory, energy = langevin(target, steps=10000, sample_interval=100)
replicas, energy, swap_acceptance = parallel_tempering(
    target, (300.0, 360.0, 432.0, 518.4), rounds=1000, steps_per_round=100,
)
```

`platform="Reference"`, `"CPU"`, `"CUDA"`, or `"OpenCL"` selects the OpenMM
platform.

## Direct training

`train_forward_KLX_G`, `train_forward_KLL1_G`, `train_FAB_G`,
`train_forward_KLXX_G`, and `train_FABX_G` train the map `G` from the target
to the source; `flow.inv` generates target-like proposals. Every step draws a
source batch from `x_valid` and manufactures its target batch through the
current flow with the flow-proposal SMC: `ladder` levels, `mc_steps_1` MALA
steps on the intermediate levels and `mc_steps_2` on the last. Each trainer
returns `(trained_flow, batch_ess_history)`. With `target_data=` (KLX, KLL1,
KLXX) the target batches are drawn from a given target sample set instead,
for data-driven training, and the batch ESS reported is that of the
pushforward of the source batch.

```python
from jflows_md.train import train_forward_KLX_G

trained, batch_ess = train_forward_KLX_G(
    x_valid, source, target, flow, target.domain,
    batch_size=128, steps_total=1000, lr=1e-3, ladder=4,
    mc_dt=1e-4, mc_steps_1=20, mc_steps_2=100,
)
```

Direct trainers default to `initialize_from_identity=False`; the generators
default to `True`, so every stage attempt starts from the identity unless a
warm start is requested.

## Boltzmann generators

The stage targets are the diagonal interpolations `U_{t,t}` above. Each stage
trains the increment `t_{k-1} -> t_k` on the current population, compares the
trained map with the identity by the screened population ESS, accepts the
stage when the selected ESS reaches `bg_param["tau_valid"]` (default `0.6`)
and otherwise shrinks `t_k` towards `t_{k-1}` by `shrink_factor`, and advances
the population by reweight, resample, and `mc_steps_2` MALA steps under the
stage target. Every generator returns `(samples, stages, effective_seconds)`,
the last being the training time of the accepted attempts alone; a run is
complete only when `stages[-1]["t"] == 1.0`.

```python
from jflows_md.boltzmann import (
    boltzmann_identity, boltzmann_forward_KLX_G, boltzmann_forward_KLXX_G,
)

particles, stages, seconds = boltzmann_identity(x_valid, source, target, mc_dt=1e-4, mc_steps_2=100)

particles, stages, seconds = boltzmann_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size=5000, steps_total=500, lr=1e-3, ladder=8,
    mc_dt=1e-3, mc_steps_1=20, mc_steps_2=100,
)

particles, stages, seconds = boltzmann_forward_KLXX_G(
    x_valid, source, target, flow,
    pool_size=0, batch_size=5000, steps_total=500, lr=1e-3, ladder=8,
    melt=1.0, opt_alpha=1e-2, opt_steps=200,
    mc_dt=1e-3, mc_steps_1=20, mc_steps_2=100, coeff_qt=0.5,
)
```

`coeff_lambda=0.0` in the KLX form is the forward KL. `boltzmann_forward_KLL1_G`,
`boltzmann_FAB_G`, and `boltzmann_FABX_G` follow the same pattern, and the
`_fixed` forms take a stage schedule `t_list` in place of `bg_param`. The two
Langevin budgets are `mc_steps_1` on the intermediate SMC levels and
`mc_steps_2` on every other rejuvenation: the last SMC level, the quench-and-
temper rows and pool of KLXX, and the stage advance.

A running attempt can be rejected by hand: `Manual_Reject` installs a Ctrl+C
handler, and with `reject_requested=reject` the attempt is dropped within one
gradient step of the key press, or between chunks of the quench-and-temper
pool, the validation weights, and the advance; the endpoint shrinks as after a
failed ESS gate. A second Ctrl+C within three seconds stops the program.

```python
from jflows_md.boltzmann import Manual_Reject

with Manual_Reject() as reject:
    particles, stages, seconds = boltzmann_forward_KLX_G(..., reject_requested=reject)
```

Each stage record carries `t_start`, `t`, `rg_start`, `rg_end`, the three
population ESS values, `selected`, the selected `flow` and the
`continuation_flow`, the attempt histories, the `mala_acceptance` of the
advance, `elapsed_seconds`, and `accepted_attempt_seconds`.

## Complete-stage resume

`jflows_md.boltzmann.write` publishes every accepted stage atomically (the
population, the histories, the metadata, and the two flow files of a trained
stage) in `run.json`; `jflows_md.boltzmann.load.run` resumes from the last
published stage.

```python
from jflows_md.boltzmann import iterate_boltzmann
from jflows_md.boltzmann.load import run

def iterate(samples, continuation, accepted_t, start_stage):
    return iterate_boltzmann(
        samples, source, target, continuation, objective="forward_klx",
        accepted_t=accepted_t, start_stage=start_stage, **controls,
    )

config = {"target": "alanine_dipeptide_ff96_obc1", "seed": 0, **controls_without_monitor}
particles, stages = run("runs/alanine_dipeptide", "alanine_dipeptide-klx", config, x_valid, flow, iterate)
particles, stages = run("runs/alanine_dipeptide", "alanine_dipeptide-klx", config, None, flow, iterate, resume=True)
```

Resume requires the same problem identifier, configuration, and static flow
template. `manifest`, `validate`, `load`, `load_stage_flow`,
`load_validation_samples`, `load_training_history`, and `fork` inspect and
manage stored runs.

## Inference scheme

`run_inference` replays the frozen selected maps of a complete run on fresh
source particles: for every stage it pushes the population through the stored
map, evaluates the stage weight, resamples with the screened weights, and
rejuvenates with `mc_steps` MALA steps under the stage target, rebuilt from
the run's regularization path. Populations are processed in chunks and kept
as `.npy` memmaps, so the sample count is not limited by device memory.

```python
from jflows_md import run_inference

manifest = run_inference(
    "runs/alanine_dipeptide", "runs/alanine_dipeptide-inference", flow, source, target,
    sample_count=1_000_000, mc_dt=1e-3, mc_steps=100, chunk_size=10_000,
)
final = manifest["inference_samples_path"]
```

The output directory must be empty; it receives `run.json` and, per stage,
`samples.npy`, `log_weights.npy`, and `metadata.json`. The last stage's
samples are the inference set.

## Bundles and installation

A runtime bundle is a directory with `coordinates.json`, `manifest.json`,
`reference.pdb`, `system.json`, `system.xml`, and `validation.json`. The
source checkout ships five, the alanine dipeptide of the chiral tests and the
four alkanes of the alkane family:

| Bundle | Model | Domain | Fixed centers |
|---|---|---|---:|
| `alanine_dipeptide_ff96_obc1` | ff96; OBC1/ACE | `R^42 x T^18` | 1 (L-Ala) |
| `methane_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^7 x T^2` | 0 |
| `ethane_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^13 x T^5` | 0 |
| `propane_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^19 x T^8` | 0 |
| `n_butane_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC; OBC1/ACE | `R^25 x T^11` | 0 |

Wheels contain code only; bundles live in a separate directory and are
selected by name:

```python
from jflows_md import Molecular_Bundle, available_bundles

names = available_bundles("downloaded-bundles")
bundle = Molecular_Bundle.load(names[0], root="downloaded-bundles")
```

`Molecular_Potential.from_bundle` and `OpenMM_Potential.from_bundle` accept
the same `root=`.

Install the JAX and OpenMM accelerator wheels of one CUDA generation, then
`jflows` and this project:

```bash
pip install "jax[cuda13]" "openmm[cuda13]"
python -m pip install -e "${JFLOWS_ROOT:?set JFLOWS_ROOT}"
python -m pip install -e .
```

`jflows_md.backend()` prints the installed runtimes without creating a JAX
client or an OpenMM context. Bundle construction needs the OpenMM extras:

```bash
python -m pip install -e ".[openmm,bundles]"
python -m jflows_md.bundle_build alanine_dipeptide molecule.prmtop molecule.rst7 generated/alanine_dipeptide
```

The one build preset is `alanine_dipeptide`; its fixed stereocenter carries
caller-supplied CIP-priority atom orders, so bundle construction rejects a
reference geometry with the wrong R/S configuration; see
`doc/01-low-level.md` for the coordinate configuration.

## Example

[`example/methane_9d_raw/`](example/methane_9d_raw/) is a complete 9D methane
Boltzmann generator: a driver, its bundle, and result reports. From that
directory, `python train.py --method id`, `--method kl`, or `--method klxx`
runs the identity, the forward KL, or the KLXX generator through the
persistence controller and summarizes the accepted stages in `results/`.

## Verification

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

runs the thirteen smoke modules of `smoke/run_all.py`, each in a fresh
subprocess; [`doc/04-smoke-tests.md`](doc/04-smoke-tests.md) says what each
one establishes.
