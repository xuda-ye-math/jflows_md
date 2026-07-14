# jflows_md

Mixed-domain molecular potentials, normalizing flows, samplers, and Boltzmann
generators for [`jflows`](https://github.com/xuda-ye-math/jflows), built with
JAX and Equinox.

> **Status: experimental.** Molecular energies and forces are checked against
> OpenMM, while production-scale Boltzmann-generator training remains an
> experiment-level responsibility.

## Features

- **Mixed molecular domains.** `Mixed_NSF` acts on
  $\mathbb{R}^{p}\times\mathbb{T}^{q}$: ordinary rational-quadratic splines
  transform whitened bond/angle coordinates and circular C1 splines transform
  torsions.
- **Bundle-defined targets.** `Molecular_Potential` loads the force field,
  OBC1 implicit solvent, internal-coordinate chart, chirality support, and
  OpenMM validation frames from one six-file runtime bundle.
- **Three supplied targets.** The source checkout carries alanine dipeptide
  with the FAB force model (60D), glycerol (36D), and diethanolamine (48D).
- **Molecular sampling.** Mixed-domain MALA, potential-space SMC, and the
  G-native score-free AIS surrogate use explicit `ladder`, `mc_dt`,
  `mc_steps`, and `chunks` controls.
- **Boltzmann-generator training.** Molecular KL+X and KL+X+X trainers use the
  adaptive controller, optimizer-only `e_clip`, global `g_clip`, per-step
  batch ESS, full-validation proposal ESS, final-versus-identity selection,
  and MALA. KL+X+X additionally constructs a mixed-domain quench-and-temper
  coverage pool.
- **Small public surface.** Frequently used objects live directly under
  `jflows_md`; coordinate, force-field, chirality, and spline implementation
  details live under `jflows_md.core`.

## Quick example

```python
import jax

from jflows_md import Mixed_NSF, Molecular_Potential

target = Molecular_Potential.from_bundle("adp_ff96_obc1")
source = target.source()
q = source.samples(jax.random.key(0), N=32)

energy = target(q)          # [32]
gradient = target.grad(q)   # [32, 60]

flow = Mixed_NSF(
    jax.random.key(1),
    target.domain,
    bins=32,
    transforms=6,
    hidden_features=(256, 256),
).zeros()
```

The reduced potential is

```text
U(q) = beta E_bundle(x(q)) - log J_config(q).
```

It contains the full Amber/OBC energy and the Cartesian-to-internal Jacobian
for the configurational measure after quotienting global translation and
rotation. The canonical Cartesian frame is a representative, not a
six-constraint ensemble.

An explicit soft bridge can be constructed without mutating the physical
target:

```python
soft = target.regularized(
    50.0,
    energy_scale_kj_mol=50.0,
    tail_fraction=0.01,
)
```

The cutoff is the Cartesian energy excess above the bundle reference in
kJ/mol. The lin-log scale controls compression above the cutoff and
`tail_fraction` retains a coercive linear component. The coordinate Jacobian
is never regularized.

## Package layout

```text
jflows_md/
├── README.md
├── LICENSE
├── pyproject.toml
├── bundles/
│   ├── build_molecular_bundles.py
│   ├── adp_ff96_obc1/
│   ├── glycerol_gaff2_am1bcc_obc1/
│   └── diethanolamine_gaff2_am1bcc_obc1/
├── jflows_md/
│   ├── artifacts.py
│   ├── boltzmann.py
│   ├── flow.py
│   ├── potential.py
│   ├── source.py
│   ├── system.py
│   ├── train.py
│   ├── utils.py
│   └── core/
└── smoke/
```

Molecular data deliberately lives outside the import package. A source
checkout can resolve the supplied bundles by name. A wheel contains Python
code only, so wheel users pass an external bundle directory, for example:

```python
target = Molecular_Potential.from_bundle("/data/molecules/adp_ff96_obc1")
```

Each bundle contains exactly:

```text
coordinates.json
manifest.json
reference.pdb
system.json
system.xml
validation.json
```

Only the four JSON files are needed by the JAX potential. `reference.pdb` and
`system.xml` make independent OpenMM checks straightforward.

## Installation

Create a fresh virtual environment and let pip select current releases. On
Linux with an NVIDIA CUDA 13 driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade \
  "jax[cuda13]" equinox "openmm[cuda13]" parmed mdtraj \
  scipy matplotlib h5py scikit-learn
```

Clone both source trees and install them for ordinary imports:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/xuda-ye-math/jflows.git "$HOME/src/jflows"
git clone https://github.com/xuda-ye-math/jflows_md.git "$HOME/src/jflows_md"
pip install -e "$HOME/src/jflows"
pip install -e "$HOME/src/jflows_md"
```

The runtime package does not require AmberTools. To construct bundles from
Amber topology inputs in the same environment, install the optional tools:

```bash
pip install -e "$HOME/src/jflows_md[bundles]"
```

This extra includes `ambertools-unofficial`, an unofficial wheel distribution
of AmberTools. It is useful for preparing topology inputs but is not required
for evaluation or training.

Verify the accelerator stack interactively:

```bash
python
```

```python
>>> from pathlib import Path
>>> import jax
>>> import jflows
>>> import jflows_md
>>> print(jax.default_backend(), jax.devices())
>>> print(Path(jflows.__file__).resolve())
>>> print(Path(jflows_md.__file__).resolve())
```

## Public API

- `jflows_md.flow`: `Mixed_Identity`, `Mixed_NSF`
- `jflows_md.potential`: `Molecular_Potential`
- `jflows_md.source`: `Molecular_Source`
- `jflows_md.system`: `Molecular_Bundle`, `available_bundles`
- `jflows_md.train`: `train_forward_KLX_G`, `train_forward_KLXX_G`
- `jflows_md.boltzmann`: `boltzmann_forward_KLX_G`,
  `boltzmann_forward_KLXX_G`
- `jflows_md.artifacts`: `mixed_flow_metadata`, `save_mixed_flow`,
  `load_mixed_flow`
- `jflows_md.utils`: mixed MALA, mixed quench-and-temper, potential-space SMC,
  and G-native AIS

`jflows_md` uses public `jflows` potential/flow bases, spline transforms,
monitoring, importance weights, ESS, resampling, potential algebra, and
L-BFGS. Molecular mixed-domain kernels remain in this companion package.

Training and Boltzmann drivers use `batch_size`, `pool_size`, `train_steps`,
`mc_dt`, `mc_steps`, `opt_alpha`, `opt_steps`, and `chunks`. Direct MALA uses
`dt`, `steps`, and `image_radius`. Increasing `chunks` means more sequential
row partitions and fewer physical samples in each compiled call. `ladder` is
the number of bridge levels.

The mixed flow follows `jflows` direction conventions. `F` maps source to
target and `G = F^{-1}` maps target to source. Molecular forward training is
G-native and therefore does not take a direction string.

The adaptive full-validation gate compares the trained flow with exact
identity. The higher ESS is the sole candidate and is accepted exactly when it
clears `tau_ess`. The `checkpoint` argument controls backward-pass
rematerialization; persistent flow artifacts are enabled separately with
`flow_dir`.

Each accepted level reports `valid_selected_ess`, `valid_trained_ess`,
`valid_identity_ess`, and `valid_sample_count`. Attempt diagnostics use
`t_hist`, `batch_ess_hist`, `valid_trained_ess_hist`,
`valid_identity_ess_hist`, and `attempt_status_hist`. Molecular optimizer
screens additionally expose `kept_fraction_hist` and
`update_applied_hist`. Passing `flow_dir` saves each trained attempt and the
selected level flow. `load_mixed_flow` reloads any one of those flow files.

Potential-space SMC rejuvenates at the matching intermediate potential on
every level. Flow-proposal AIS applies fractional geometric weights while
rejuvenating at the final target at every level; it is the deliberate
score-free surrogate used by `jflows`.

## Supplied molecular targets

| Bundle | Physical model | Mixed coordinate domain |
|---|---|---|
| `adp_ff96_obc1` | Amber ff96/OBC1, L-ADP | R^42 x T^18 (60D) |
| `glycerol_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC/OBC1 | R^25 x T^11 (36D) |
| `diethanolamine_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC/OBC1, neutral | R^33 x T^15 (48D) |

All supplied targets use 300 K, mbondi2 radii, ACE nonpolar solvation,
solvent and solute dielectric constants 78.5 and 1.0, zero salt, `NoCutoff`,
and no constraints. A PDB alone does not define the Hamiltonian. ADP uses a
chiral half-chart to select L-ADP; glycerol and diethanolamine retain both
signs of their diagnostic determinant.

## Build a bundle

Given an Amber topology and matching coordinate file:

```bash
python bundles/build_molecular_bundles.py \
  glycerol molecule.prmtop molecule.rst7 generated/glycerol
```

The target argument is one of `adp`, `glycerol`, or `diethanolamine`. Add
`--name NAME` to set the bundle name. The builder writes the six-file format
shown above. OpenMM and ParmEd are required; AmberTools is needed only when
the input topology must first be prepared.

## Validation

Run the accelerator-backed smoke suite from the repository root with both
source trees importable:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

The suite covers the bundle format, OpenMM energy/force parity,
mixed-coordinate round trips and Jacobians, chirality support, spline seams,
float32 execution, finite-safe optimizer and weight cases, MALA/SMC/AIS,
flow artifact loading, chunking, and the public `jflows` boundary. It does not
launch production molecular training.

## License

`jflows_md` is released under the MIT License. See [`LICENSE`](LICENSE).
