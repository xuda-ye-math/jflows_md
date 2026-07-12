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
- **Three included benchmarks.** FAB-compatible alanine dipeptide (60D),
  glycerol (36D), and neutral diethanolamine (48D) are distributed as outer
  runtime bundles.
- **Molecular sampling.** Mixed-domain MALA, potential-space SMC, and the
  G-native score-free AIS surrogate use the same `ladder`, `step`, `iters`,
  and `chunk` conventions as `jflows`.
- **Boltzmann-generator training.** The molecular forward KL+X trainer and
  adaptive controller use `e_clip` as an optimizer screen, `g_clip` as global
  gradient clipping, and MALA by default. No sharpening is part of the target.
- **Public compatibility boundary.** `jflows_md` imports only public `jflows`
  interfaces. Low-level coordinate, force-field, chirality, and spline code
  stays under `jflows_md.core`.

## Quick example

```python
import jax

from jflows_md import Mixed_NSF, Molecular_Potential

target = Molecular_Potential.from_bundle("fab_adp_ff96_obc1_v1")
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
U(q) = beta E_bundle(x(q)) - log |det(dx/dq)|.
```

It contains the complete Amber/OBC energy and the Cartesian-to-internal
Jacobian. It contains neither a sharpened surrogate nor clipped evaluation
energies.

## Package layout

```text
jflows_md/
├── README.md
├── LICENSE
├── pyproject.toml
├── bundles/
│   ├── build_molecular_bundles.py
│   ├── fab_adp_ff96_obc1_v1/
│   ├── glycerol_gaff2_am1bcc_obc1_v1/
│   └── diethanolamine_neutral_gaff2_am1bcc_obc1_v1/
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
inside the import package, and callers may also pass an explicit bundle path.
The checked-in bundles are sufficient for runtime evaluation; OpenMM,
ParmEd, and AmberTools are required only to rebuild or independently validate
them.

## Fresh environment for GitHub readers

The supported project environment separates the dependency stacks:

- install OpenMM and ParmEd from conda-forge for the molecular-science layer;
- install the current JAX, Equinox, and Matplotlib stack with pip. The
  [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
  recommends pip for NVIDIA CUDA wheels.

The [OpenMM installation guide](https://docs.openmm.org/development/userguide/application/01_getting_started.html)
also documents pip packages; using conda-forge here is a project policy, not
an upstream limitation. Create the dependency environment first:

```bash
conda create -n jflows -c conda-forge python=3.11 pip openmm parmed
conda activate jflows
python -m pip install --upgrade "jax[cuda13]" equinox matplotlib
```

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

### Local project runtime

Those editable installs are a convenience for external users, not the project
workstation. Local runs keep both packages uninstalled and expose both roots
explicitly through `PYTHONPATH`. Local edits then take effect on the next
Python process while the environment remains package-free:

```bash
conda activate jflows && \
PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md python -c \
  "from pathlib import Path; import jflows, jflows_md; print(Path(jflows.__file__).resolve()); print(Path(jflows_md.__file__).resolve())"
```

Run molecular programs with the same two-root search path:

```bash
conda activate jflows && \
PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md python molecular_driver.py
```

## Public API

The public modules mirror the organization of `jflows`:

- `jflows_md.flow`: `Mixed_Identity`, `Mixed_NSF`
- `jflows_md.potential`: `Molecular_Potential`
- `jflows_md.source`: `Molecular_Source`
- `jflows_md.system`: `Molecular_Bundle`, `available_bundles`
- `jflows_md.train`: `train_molecular_forward_KLX_G`
- `jflows_md.boltzmann`: `molecular_boltzmann_forward_KLX_G`
- `jflows_md.artifacts`: source hashing and verified stage loading
- `jflows_md.utils`: mixed MALA, potential-space SMC, and G-native AIS

Frequently used objects are lazily exposed directly from `jflows_md`. User
programs should not depend on `jflows_md.core`.

The mixed flow follows `jflows` direction conventions. `F` maps source to
target and `G = F^{-1}` maps target to source. Molecular forward training fixes
the flow as `G`, so its public driver and AIS surrogate do not accept a
direction string. Increasing `chunk` means more sequential row partitions and
therefore fewer physical samples in each compiled molecular call.

Potential-space SMC is classical: every ladder level rejuvenates at its
matching intermediate potential. Flow-proposal AIS instead applies fractional
geometric weights while rejuvenating at the final target at every level. It is
a deliberately biased, score-free target surrogate rather than exact AIS/SMC.

## Frozen molecular targets

| Bundle | Physical model | Mixed coordinate domain |
|---|---|---|
| `fab_adp_ff96_obc1_v1` | FAB-compatible Amber ff96/OBC1, L-ADP only | R^42 x T^18 (60D) |
| `glycerol_gaff2_am1bcc_obc1_v1` | GAFF2/AM1-BCC/OBC1, neutral | R^25 x T^11 (36D) |
| `diethanolamine_neutral_gaff2_am1bcc_obc1_v1` | GAFF2/AM1-BCC/OBC1, explicitly neutral | R^33 x T^15 (48D) |

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
conda activate jflows && \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
  python smoke/run_all.py
```

The suite covers bundle integrity, OpenMM energy/force parity, mixed-coordinate
round trips and Jacobians, chirality support, spline seam behavior, float32
execution, MALA/SMC/AIS, artifact loading, chunking, and public `jflows`
compatibility. It does not launch production molecular training. See
[`smoke/README.md`](smoke/README.md) for the opt-in compilation benchmark.

Rebuild bundles only when their versioned physical definition changes:

```bash
conda activate jflows && \
PYTHONPATH=/mnt/projects/jflows:/mnt/projects/jflows_md \
  python bundles/build_molecular_bundles.py
```

The builder uses immutable seed artifacts already stored in each bundle and
requires OpenMM, ParmEd, and AmberTools. After rebuilding, review every
manifest, provenance record, validation result, and hash before committing.

## License

`jflows_md` is released under the MIT License. See [`LICENSE`](LICENSE).
