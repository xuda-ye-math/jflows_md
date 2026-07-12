# jflows_md

Mixed-domain molecular potentials, normalizing flows, samplers, and Boltzmann
generators for [`jflows`](https://github.com/xuda-ye-math/jflows), built with
JAX and Equinox.

> **Status: experimental.** The package currently targets Linux and
> accelerator-backed JAX. Molecular targets are validated against OpenMM
> reference energies and forces, but production-scale training remains a
> separate experiment milestone.

## Features

- **Mixed molecular domains.** `Mixed_NSF` acts on
  $\mathbb{R}^{p}\times\mathbb{T}^{q}$: ordinary rational-quadratic splines
  transform whitened bond/angle coordinates, while circular C1 splines
  transform torsions.
- **Frozen physical targets.** `Molecular_Potential` loads a verified bundle
  that fixes the force field, OBC1 implicit solvent, internal-coordinate chart,
  chirality support, validation frames, provenance, and integrity hashes.
- **Three repository benchmarks.** The source checkout carries outer runtime
  bundles for FAB-compatible alanine dipeptide (60D), glycerol (36D), and
  neutral diethanolamine (48D).
- **Molecular sampling.** Mixed-domain MALA, potential-space SMC, and the
  G-native score-free AIS surrogate use the same `ladder`, `step`, `iters`,
  and `chunk` conventions as `jflows`.
- **Boltzmann-generator training.** Molecular forward KL, KL+X, and KL+X+X
  trainers share the adaptive controller, optimizer-only `e_clip`, global
  `g_clip`, honest proposal-side stage ESS, fixed-checkpoint flow selection,
  and MALA. Masked losses use finite-safe selection, gradient clipping remains
  stable when a float32 sum of squares overflows, and an Adam update is
  committed atomically only when its loss, gradients, moments, and resulting
  parameters are finite. The live target-pool ratio moment is deliberately
  labeled separately from stage ESS. KL+X+X adds a mixed-domain
  quench-and-temper coverage pool. No sharpening is part of the target.
- **Public compatibility boundary.** `jflows_md` imports only public `jflows`
  interfaces. Low-level coordinate, force-field, chirality, and spline code
  stays under `jflows_md.core`.

## Quick example

```python
import jax

from jflows_md import Mixed_NSF, Molecular_Potential

# Name lookup uses the outer bundles/ directory in a source checkout.
target = Molecular_Potential.from_bundle("adp_ff96_obc1")
source = target.source()
q = source.samples(jax.random.key(0), 32)

energy = target(q)          # shape [32]
gradient = target.grad(q)   # shape [32, 60]

flow = Mixed_NSF(
    jax.random.key(1),
    target.domain,
    bins=32,
    transforms=6,
    hidden_features=(256, 256),
).zeros()
```

The pulled-back reduced potential is

```text
U(q) = beta E_bundle(x(q)) - log J_config(q).
```

It contains the complete Amber/OBC energy and the Cartesian-to-internal
Jacobian for the standard configurational measure after quotienting global
translation and rotation. The canonical Cartesian frame is only a
representative; it is not a six-constraint gauge-slice ensemble. The target
contains neither a sharpened surrogate nor clipped evaluation energies.

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

The outer `bundles/` directory is intentional. Molecular data is not hidden
inside the import package. Editable source-checkout installs can use the short
registry names shown above. Built wheels intentionally contain Python code
only; wheel users must obtain a bundle directory separately and pass its path,
for example
`Molecular_Potential.from_bundle("/data/molecules/adp_ff96_obc1")`.
The checked-in source-tree bundles are sufficient for runtime evaluation;
OpenMM, ParmEd, and AmberTools are required only to rebuild or independently
validate them.

## Fresh environment for GitHub readers

Use a fresh pip-only virtual environment and let pip select the latest
compatible releases. On Linux with an NVIDIA CUDA 13 driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade \
  "jax[cuda13]" equinox "openmm[cuda13]" parmed mdtraj \
  scipy matplotlib h5py scikit-learn
```

The versioned extras are intentional: `jax[cuda13]` installs JAX's CUDA-13
plugin stack, while `openmm[cuda13]` installs the OpenMM Python API together
with its matching CUDA platform. Use the corresponding CUDA 12 extras when
needed. The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and [OpenMM installation guide](https://docs.openmm.org/development/userguide/application/01_getting_started.html)
document these pip interfaces.

Clone both source trees into a common workspace:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/xuda-ye-math/jflows.git "$HOME/src/jflows"
git clone https://github.com/xuda-ye-math/jflows_md.git "$HOME/src/jflows_md"
```

For readers who prefer conventional editable package imports, the public
GitHub setup may be completed with:

```bash
pip install -e "$HOME/src/jflows"
pip install -e "$HOME/src/jflows_md"
```

The runtime package does not require AmberTools. Readers working from a source
checkout who want to construct new molecular bundles directly can instead
request the optional bundle toolchain:

```bash
pip install -e "$HOME/src/jflows_md[bundles]"
```

The `bundles` extra adds OpenMM, ParmEd, and
`ambertools-unofficial`. The AmberTools wheel is an unofficial repackaging;
every generated bundle must therefore record its exact version, data hashes,
and resulting manifest.

Verify the accelerator libraries, dependency closure, and editable checkouts:

```bash
python -m openmm.testInstallation
pip check
python
```

Then enter:

```python
>>> from pathlib import Path
>>> import jax
>>> import jflows
>>> import jflows_md
>>> print(jax.default_backend(), jax.devices())
>>> print(Path(jflows.__file__).resolve())
>>> print(Path(jflows_md.__file__).resolve())
```

Molecular programs then use ordinary Python imports:

```bash
python molecular_driver.py
```

## Public API

The public modules mirror the organization of `jflows`:

- `jflows_md.flow`: `Mixed_Identity`, `Mixed_NSF`
- `jflows_md.potential`: `Molecular_Potential`
- `jflows_md.source`: `Molecular_Source`
- `jflows_md.system`: `Molecular_Bundle`, `available_bundles`
- `jflows_md.train`: `Molecular_Monitor`,
  `train_molecular_forward_KLX_G`, `train_molecular_forward_KLXX_G`
- `jflows_md.boltzmann`: `molecular_boltzmann_forward_KLX_G`,
  `molecular_boltzmann_forward_KLXX_G`
- `jflows_md.artifacts`: exact flow-architecture metadata, source hashing, and
  verified stage loading
- `jflows_md.utils`: mixed MALA, mixed quench-and-temper, potential-space SMC,
  and G-native AIS

Frequently used objects are lazily exposed directly from `jflows_md`. User
programs should not depend on `jflows_md.core`.

The mixed flow follows `jflows` direction conventions. `F` maps source to
target and `G = F^{-1}` maps target to source. Molecular forward training fixes
the flow as `G`, so its public driver and AIS surrogate do not accept a
direction string. Increasing `chunk` means more sequential row partitions and
therefore fewer physical samples in each compiled molecular call.

`selection_steps` is an optional sparse schedule of post-update flow snapshots
used by the full-validation stage ESS gate. When enabled, the gate also tests
exact identity (reported as step -1), the accepted pre-update warm start (step
0), and the final trained flow. Stage records expose `selected_checkpoint`,
`selected_step`, `checkpoint_labels`, `checkpoint_steps`, and
`checkpoint_ess`. Snapshot storage is proportional to the number of requested
checkpoints times the flow size, each distinct static schedule compiles a
separate trainer executable, and schedules are therefore limited to 32 sparse
entries. The separate Boolean `checkpoint` argument means JAX backward-pass
rematerialization; it does not save or select flow snapshots.

Potential-space SMC is classical: every ladder level rejuvenates at its
matching intermediate potential. Flow-proposal AIS instead applies fractional
geometric weights while rejuvenating at the final target at every level. It is
a deliberately biased, score-free target surrogate rather than exact AIS/SMC.

## Frozen molecular targets

The built-in registry contains only quotient-measure coordinate-schema-2
targets. The earlier schema-1 names used a canonical gauge-slice measure and
are retired rather than silently reinterpreted. The coordinate reader can
still open an explicit v1 bundle for forensic reproducibility. Likewise,
schema-1 flow artifacts require an explicit target from their original
bundle/source revision; automatic built-in target discovery is a schema-2
contract.

| Bundle | Physical model | Mixed coordinate domain |
|---|---|---|
| `adp_ff96_obc1` | FAB-compatible Amber ff96/OBC1, L-ADP only | R^42 x T^18 (60D) |
| `glycerol_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC/OBC1, neutral | R^25 x T^11 (36D) |
| `diethanolamine_gaff2_am1bcc_obc1` | GAFF2/AM1-BCC/OBC1, explicitly neutral | R^33 x T^15 (48D) |

Every target uses 300 K, mbondi2 radii, ACE nonpolar solvation, solvent and
solute dielectric constants 78.5 and 1.0, zero salt, `NoCutoff`, and no
constraints. A PDB is included for inspection but does not define a complete
Hamiltonian. ADP uses a chiral half-chart to exclude its unwanted enantiomer;
glycerol and neutral diethanolamine retain both signs of their diagnostic
determinant because those signs are not fixed stereocentres.

## Validation and maintenance

Run the accelerator-backed smoke suite from the repository root with both live
source trees visible:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

The suite covers bundle integrity, OpenMM energy/force parity, mixed-coordinate
round trips and Jacobians, chirality support, spline seam behavior, float32
execution, finite-safe optimizer and weight edge cases, pre-compilation
validation, MALA/SMC/AIS, artifact loading, chunking, and public `jflows`
compatibility. It does not launch production molecular training. See
[`smoke/README.md`](smoke/README.md) for the opt-in compilation benchmark.

Construct or verify bundles from the repository root with:

```bash
pip install -e ".[bundles]"
python bundles/build_molecular_bundles.py
```

The no-argument command verifies the frozen historical targets. To construct a
new small-molecule target with the currently installed AmberTools toolchain,
give it a new scientific name and an unused output directory:

```bash
python bundles/build_molecular_bundles.py \
  --only glycerol \
  --name glycerol_gaff2_am1bcc_obc1_at26 \
  --output generated/glycerol_gaff2_am1bcc_obc1_at26
```

The builder uses immutable seed artifacts already stored in each bundle and
requires OpenMM and ParmEd. Glycerol and diethanolamine rebuilding additionally
requires AmberTools; the optional `bundles` extra supplies its command-line
programs through `ambertools-unofficial`. AmberTools is not a runtime or
training dependency. The existing small-molecule targets are frozen to their
historical AmberTools 24.8 outputs, so a current version-26 toolchain must use
the explicit `--name`/`--output` mode rather than overwrite them. Frozen
verification creates temporary candidates and accepts only exact matches to
pinned manifests. Any seed, toolchain, Hamiltonian, or validation change
requires a new bundle name and an explicit review of its provenance and hashes.

## License

`jflows_md` is released under the MIT License. See [`LICENSE`](LICENSE).
