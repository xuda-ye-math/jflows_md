# jflows_md documentation

This directory is the standalone user documentation for the mixed-domain
molecular companion to `jflows`. It expands the interface hierarchy introduced
in the project [README](../README.md) and follows the same order as the generic
package:

1. low-level molecular objects and numerical kernels;
2. medium-level single-stage trainers; and
3. high-level adaptive-staging Boltzmann generators.

The live source is authoritative. Public JAX code should import from
the root `jflows_md.backend` report or from `jflows_md.system`,
`jflows_md.source`, `jflows_md.potential`,
`jflows_md.flow`, `jflows_md.utils`, `jflows_md.train`,
`jflows_md.artifacts`, or `jflows_md.boltzmann`. Native OpenMM code should
import from `jflows_md.openmm`. The `jflows_md.core` package is private.

## Documentation tree

```text
doc/
├── README.md                 # orientation, hierarchy, conventions
├── 01-low-level.md           # bundles, potentials, flows, samplers, OpenMM
├── 02-medium-level.md        # direct train_* functions and artifacts
├── 03-high-level.md          # adaptive-staging BG, persistence, resume, inference
├── 04-smoke-tests.md         # executable contract checks by API area
└── 05-workflows.md           # complete molecular workflows
```

Read the numbered files in order. The first three form the API manual; the last
two connect the interfaces to executable repository evidence and complete
usage patterns.

## Interface tree

```text
jflows_md
├── LOW LEVEL
│   ├── backend               JAX/Equinox/OpenMM accelerator report
│   ├── system
│   │   ├── Molecular_Bundle
│   │   └── available_bundles
│   ├── source / potential
│   │   ├── Molecular_Source
│   │   ├── Molecular_Potential
│   │   └── Regularized_Molecular_Potential (the (e, r) surrogate)
│   ├── flow
│   │   ├── Mixed_Identity
│   │   └── Mixed_NSF
│   ├── utils
│   │   ├── wrapped mixed-domain MALA and HMC
│   │   ├── importance-weight screen
│   │   ├── flow-proposal SMC (sequential_monte_carlo / flow_target_batch)
│   │   ├── two-phase FAB SMC (sequential_monte_carlo_fab / flow_fab_batch)
│   │   └── mixed quench and temper
│   └── openmm
│       ├── OpenMM_Potential
│       ├── native Langevin dynamics
│       └── native parallel tempering
├── MEDIUM LEVEL
│   ├── train
│   │   ├── train_forward_KLX_G
│   │   └── train_forward_KLXX_G
│   └── artifacts
│       ├── save_flow / load_flow
│       ├── save_samples / load_samples
│       └── save_history / load_history
└── HIGH LEVEL
    └── boltzmann
        ├── boltzmann_identity
        ├── boltzmann_forward_KLX_G
        ├── boltzmann_forward_KLXX_G
        ├── iterate_identity / iterate_boltzmann
        ├── adaptive stage schedule / trained-or-identity selection / tau_valid gate
        ├── write: create / stage / finish
        ├── load: validate / load / fork / run / stage readers
        └── inference: run_inference
```

The dependency direction is downward. A high-level molecular generator uses a
medium-level trainer; a trainer uses low-level mixed-domain flows, potentials,
and samplers. Direct use of a lower level remains supported when a custom
algorithm needs more control.

## Package tree

The corresponding source layout is:

```text
jflows_md/
├── __init__.py
├── artifacts.py
├── boltzmann/
│   ├── __init__.py
│   ├── inference.py
│   ├── load.py
│   └── write.py
├── bundle_build/             # optional offline bundle construction
├── core/                     # private coordinate and force-field kernels
├── flow.py
├── openmm/
│   ├── __init__.py
│   ├── potential.py
│   └── sampling.py
├── potential.py
├── source.py
├── system.py
├── train.py
└── utils/
    ├── __init__.py
    ├── anneal.py
    ├── quench.py
    ├── rejuvenation.py
    └── screen.py
```

## Core conventions

- JAX molecular coordinates live in `R^p x T^q`: Euclidean coordinates come
  first and periodic torsions come last.
- `Molecular_Potential` is a reduced internal-coordinate potential,
  $U(q)=\beta E(x(q))-\log J(q)$; `regularized((e, r))` returns the
  two-parameter surrogate $U^{\rho}$ used along a regularization path.
- Obtain the mixed domain and matched source from a loaded target. Do not
  construct application code from `jflows_md.core` objects.
- `Molecular_Source` is Gaussian in the Euclidean block and uniform on the
  torsion block. Its variance follows a target temperature override.
- `Mixed_NSF` is G-native in the supplied molecular trainers. Generate
  molecular proposals from source samples with `flow.inv(x)`.
- Random JAX entry points take explicit PRNG keys. Split or fold keys for
  logically independent operations.
- Primitive kernels use `dt` and `steps` (`mixed_mala`) or `dt`,
  `leapfrog_steps`, and `trajectories` (`mixed_hmc`); `mixed_quench_and_temper`
  uses `mc_dt` and `mc_steps`; the SMC routines, the trainers, and the
  generators split the Langevin budget into `mc_steps_1` (the intermediate SMC
  levels only) and `mc_steps_2` (every other rejuvenation: the last SMC level
  at the target, the quench-and-temper rows of the mixture batch, and the
  quench-and-temper pool) and use `steps_total`, `batch_size`, and
  `pool_size`.
- `chunks` is a number of row partitions. It controls eager JAX work in mixed
  MALA and HMC, the population SMC, quench and temper, and validation-weight
  evaluation.
- Mixed Langevin is MALA and always includes the Metropolis correction; every
  SMC level rejuvenates with it, at the target in `sequential_monte_carlo`
  (the forward KL family) and at the level's own distribution of the path in
  `sequential_monte_carlo_fab` (FAB), whose intermediate levels therefore
  differentiate the flow.
- Every ESS and every resampling weight is screened
  (`jflows_md.utils.screen`): infinite log weights and the `screen_fraction`
  largest get weight zero; `SCREEN_FRACTION = 1e-4` is the one default of
  training and inference.
- Native OpenMM potentials act on Cartesian positions in nm. They do not
  include the internal-coordinate Jacobian.
- Direct artifacts, complete-stage persistence, and the inference scheme are
  separate APIs.

## Reading routes

- Loading a bundle, defining a target, or building a custom sampler: start
  with [Low-level interfaces](01-low-level.md).
- Training one fixed molecular stage: continue to
  [Medium-level interfaces](02-medium-level.md).
- Advancing through accepted stages toward the molecular target, or replaying
  a stored run on fresh source particles: continue to
  [High-level interfaces](03-high-level.md).
- Finding the executable contract for one feature: use
  [Smoke tests](04-smoke-tests.md).
- Starting from an end-to-end recipe: use
  [Molecular workflows](05-workflows.md).

## Runtime setup

From a source checkout, install this project and its declared dependencies:

```bash
python -m pip install -e .
python your_script.py
```

For standalone JAX programs, set device preallocation before importing JAX:

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

OpenMM is optional for ordinary pure-JAX bundle use. Install the `openmm`
extra for native OpenMM potentials and samplers, or `bundles` for the complete
offline bundle-construction stack. Accelerator plugins are separate: install
either `jax[cuda12]` and `openmm[cuda12]`, or `jax[cuda13]` and
`openmm[cuda13]`, explicitly.
