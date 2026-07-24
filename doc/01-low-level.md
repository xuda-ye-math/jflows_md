# Low-level interfaces

The low level contains the molecular objects and numerical kernels used to
assemble custom mixed-domain pipelines. It covers frozen bundles, the JAX
source and potential, mixed normalizing flows, mixed MALA/SMC/AIS/QT, and the
independent native OpenMM backend.

## Public imports

```python
from jflows_md.system import Molecular_Bundle, available_bundles
from jflows_md.source import Molecular_Source
from jflows_md.potential import KB_KJ_MOL_K, Molecular_Potential
from jflows_md.flow import Mixed_Identity, Mixed_NSF
from jflows_md.utils import (
    mixed_mala_step,
    mixed_mala,
    sequential_monte_carlo,
    potential_space_smc,
    annealed_importance_sampling,
    mixed_quench_and_temper,
    wrapped_normal_relative_error_bound,
)
from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering
```

The package root lazily exports the common JAX objects. Module imports remain
the clearest choice in reusable code. `OpenMM_Potential` and the native
samplers intentionally live only under `jflows_md.openmm` so OpenMM stays an
optional dependency.

## Runtime backend report

```python
import jflows_md

jflows_md.backend()
```

The report includes the client-free `jflows.backend()` output, the installed
OpenMM version, and whether a version-matched OpenMM CUDA or HIP plugin has
visible accelerator hardware. It inspects package and platform metadata only;
it does not create a JAX backend client, load an OpenMM platform, or construct
an OpenMM context.

Typical output ends with:

```text
OpenMM 8.5.2
OpenMM GPU backend: CUDA 13 (8.5.2) — available
```

## Molecular bundles

```python
bundle_root = "downloaded-bundles"
names = available_bundles(bundle_root)
bundle = Molecular_Bundle.load(names[0], root=bundle_root)
```

A bundle freezes one audited molecular system, coordinate chart, reference
configuration, and validation set. `Molecular_Bundle.load(path_or_name,
root=None, verify=True)` accepts either a direct directory or a short name
selected beneath `root`. When `root` is omitted, a source checkout searches
its repository bundle directory. With verification enabled, the directory
must contain exactly six files:

```text
bundle/
├── coordinates.json          internal-coordinate chart and source parameters
├── manifest.json             identity, model, temperature, schema metadata
├── reference.pdb             reference Cartesian structure
├── system.json               static Amber/OBC parameters for JAX
├── system.xml                native OpenMM System
└── validation.json           audited Cartesian energies and forces
```

<div align="center">

<table>
<thead>
<tr><th>Bundle property</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>bundle.path</code></td><td>resolved bundle directory</td></tr>
<tr><td><code>bundle.name</code></td><td>manifest name</td></tr>
<tr><td><code>bundle.n_atoms</code></td><td>number of Cartesian atoms</td></tr>
<tr><td><code>bundle.dimension</code></td><td>dimension of the rigid-motion-quotient internal chart</td></tr>
<tr><td><code>bundle.manifest</code></td><td>bundle and model metadata</td></tr>
<tr><td><code>bundle.system</code></td><td>JAX force-field arrays</td></tr>
<tr><td><code>bundle.coordinates</code></td><td>mixed-coordinate specification</td></tr>
<tr><td><code>bundle.validation</code></td><td>stored OpenMM reference frames, energies, and forces</td></tr>
</tbody>
</table>

</div>

The source checkout contains seven audited targets:

| Bundle | Atoms / dimension | Mixed domain | Fixed centers |
|---|---:|---|---:|
| `adp_ff96_obc1` | 22 / 60 | `R^42 x T^18` | 1 |
| `glycerol_gaff2_am1bcc_obc1` | 14 / 36 | `R^25 x T^11` | 0 |
| `diethanolamine_gaff2_am1bcc_obc1` | 18 / 48 | `R^33 x T^15` | 0 |
| `nma_ff96_obc1` | 12 / 30 | `R^21 x T^9` | 0 |
| `s_2_butanol_gaff2_am1bcc_obc1` | 15 / 39 | `R^28 x T^11` | 1 |
| `rr_2_3_butanediol_gaff2_am1bcc_obc1` | 16 / 42 | `R^31 x T^11` | 2 |
| `cyclohexane_gaff2_am1bcc_obc1` | 18 / 48 | `R^33 x T^15` | 0 |

The ff96 targets are ADP and NMA; the other five use GAFF2/AM1-BCC. All use
OBC1/ACE, `NoCutoff`, and no constraints. The four candidate manifests pin
AmberTools 26.0.0 and structure/parameter provenance: the alcohol `parmchk2`
files are empty, while cyclohexane records zero-penalty `c6` transfers from
the GAFF2 `c3` types. Installed wheels contain code only, so a wheel user
downloads or prepares bundle data separately. Use
`available_bundles(root)` to list selectable directory names, then pass one of
those names and the same `root` to `Molecular_Bundle.load`,
`Molecular_Potential.from_bundle`, or `OpenMM_Potential.from_bundle`.

## Mixed molecular coordinates

The JAX target uses a rigid-motion quotient. A molecule with Cartesian
positions `x` is represented by a mixed vector

```text
q = (q_euclidean, q_periodic) in R^p x T^q.
```

Bond and angle coordinates occupy the Euclidean block. Torsions occupy the
periodic block and are wrapped to `[-pi, pi)`. The target owns the public
domain object:

```python
target = Molecular_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
domain = target.domain

dimension = target.dimension
euclidean_dim = domain.euclidean_dim
periodic_dim = domain.periodic_dim
wrapped = domain.wrap(q)
delta = domain.displacement(q1, q0)
```

Treat the concrete domain type as an opaque public value obtained from the
target. Application code should not import `jflows_md.core.domain`.

### Fixed stereochemical components

A coordinate-schema-v3 bundle may replace any number of selected periodic
torsions by Euclidean half-chart coordinates

```text
tau_i = s_i pi sigmoid(eta_i),  s_i in {-1, +1}.
```

Its `fixed_stereocenters` list records a unique label, the selected Z-matrix
torsion, its allowed sign, four signed-volume atoms, and the allowed volume
sign for each center. `target.support_mask(x)` is the conjunction of those
signed-volume tests. An empty list leaves molecular support unrestricted.
`signed_volume_diagnostics` is separate metadata and never restricts support.

New bundle construction requires this choice explicitly; it does not infer
stereochemistry from a target name or molecular graph. The narrow
backward-compatibility exception is a target-only call for `adp`, `glycerol`,
or `diethanolamine` with every new coordinate option at its default; that call
reproduces the historical schema-v2 coordinate dictionary exactly. Any
non-default coordinate option selects the explicit schema-v3 route. Pass the
Z-matrix placement and every fixed center to `write_bundle`:

```python
from jflows_md.bundle_build import write_bundle

write_bundle(
    output,
    name=name,
    target=target_name,
    prmtop_path=prmtop,
    coordinate_path=coordinates,
    model=model,
    canonical_smiles=smiles,
    expected_formula=formula,
    expected_charge=charge,
    minimize=False,
    zmatrix={
        "root": 6,
        "prefix": (6, 8, 14, 10),
        "overrides": {
            8: (6, -1, -1),
            14: (8, 6, -1),
            10: (8, 14, 6),
        },
    },
    fixed_stereocenters=(
        {
            "label": "alanine_ca_L",
            "torsion_index": 0,
            "atoms": (8, 6, 14, 10),
            "configuration": "S",
            "cip_priority_atoms": (6, 14, 10, 9),
        },
    ),
)
```

The builder checks that the selected torsion geometrically represents the
listed center, that its three listed substituents are bonded to that center,
and that the reference is away from chart boundaries. If `configuration` and
`cip_priority_atoms` are supplied, the four substituents must be listed from
highest to lowest audited CIP priority; construction computes the reference
handedness and rejects an R/S mismatch. The builder does not infer CIP
priorities from the topology. It derives the chart torsion and signed-volume
signs from the accepted reference configuration. These restrictions change
only the internal-coordinate support and Jacobian; the Amber/OBC force-field
arrays and Cartesian physical energy are unchanged.

The curated NMA bundle leaves the ACE-C-N-C torsion periodic, so both cis and
trans regions remain in support. The curated cyclohexane bundle has no fixed
center: its ring-closing force-field bond is retained while chair inversion
remains accessible. These policies are target-specific and do not imply
automatic stereochemistry or ring-state inference for arbitrary molecules.

Schema-v2 bundles remain readable. Their zero- or one-center singleton fields
are normalized to the same runtime representation, so existing achiral and
ADP bundles require no migration.

## Molecular source

`Molecular_Source` is Gaussian on the Euclidean block and uniform on the
torsion block:

```text
U_source(q) = 1/2 sum_i (q_i - mean_i)^2 / variance_i.
```

The omitted periodic energy is constant on the torus. Construct the matched
source directly from a target:

```python
source = target.source()
x = source.samples(jax.random.key(0), N=10000)
energy = source(x)
```

`samples` returns `[N, target.dimension]`. The source uses explicit JAX keys
and accepts the canonical sample-count keyword `N`.

When the target temperature is overridden, `target.source()` preserves its
mean and scales the Euclidean variance by
`temperature_kelvin / bundle_temperature_kelvin`.

## JAX molecular potential

```python
target = Molecular_Potential.from_bundle(
    "adp_ff96_obc1",
    temperature_kelvin=300.0,
)
```

For internal coordinates `q`, the target evaluates

```text
U(q) = beta E(x(q)) - log J(q),
beta = 1 / (k_B T).
```

`E` is the pure-JAX Amber bonded/nonbonded plus OBC1/ACE energy in kJ/mol.
`J` is the coordinate Jacobian for the rigid-motion-quotient chart. The
potential consumes `[N,d]` and returns `[N]`, so it can be used by generic
`jflows` losses and potential algebra.

`KB_KJ_MOL_K = 0.00831446261815324` is the package conversion constant used
for `beta` when energies are measured in kJ/mol and temperature in kelvin.

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>target(q)</code></td><td>reduced internal-coordinate potential</td><td><code>[N]</code></td></tr>
<tr><td><code>target.grad(q)</code></td><td>JAX gradient of the reduced potential</td><td><code>[N,d]</code></td></tr>
<tr><td><code>target.cartesian(q)</code></td><td>canonical Cartesian representative</td><td><code>[N,A,3]</code> in nm</td></tr>
<tr><td><code>target.physical_energy(q)</code></td><td>unreduced Cartesian energy</td><td><code>[N]</code> in kJ/mol</td></tr>
<tr><td><code>target.energy_terms(q)</code></td><td>reduced force-field terms, Jacobian, and total potential</td><td>dictionary of <code>[N]</code> arrays</td></tr>
<tr><td><code>target.reference_internal()</code></td><td>bundle reference in the mixed chart</td><td><code>[d]</code></td></tr>
<tr><td><code>target.support_mask(x)</code></td><td>stereochemical support of Cartesian frames</td><td>Boolean batch</td></tr>
<tr><td><code>target.source()</code></td><td>temperature-matched molecular source</td><td><code>Molecular_Source</code></td></tr>
</tbody>
</table>

</div>

### `(e, r)` regularization

```python
soft = target.regularized((50.0, 0.10))
```

The pair is `(energy_threshold_kj_mol, pair_distance_floor_nm)`. The distance
floor affects only regular and exception Amber Coulomb/Lennard-Jones pair
distances. Bonded terms, OBC1/ACE, the coordinate map, and the physical target
remain unchanged.

Let `E_r` be the floor-aware energy and `E_ref,r` its value at the bundle
reference. With `d = E_r - E_ref,r`, the mapped excess is

```text
R_e(d) = d                         for d <= e,
       = e [1 + log(d/e)]          for d > e.
```

The regularized reduced potential is

```text
U_rg(q) = beta [E_ref,r + R_e(d)] - log J(q).
```

`soft(q)` evaluates that reduced surrogate. `soft.regularized_energy(q)`
returns the mapped Cartesian energy, while `soft.physical_energy(q)` still
returns the original physical energy. Regularization defines a training and
sampling potential; it does not mutate `target`.

## Mixed flows

### `Mixed_Identity`

`Mixed_Identity(domain)` wraps periodic coordinates and otherwise preserves
the input. Its forward and inverse log-Jacobians are zero. It is the explicit
identity fallback used by molecular Boltzmann stages.

### `Mixed_NSF`

```python
flow = Mixed_NSF(
    jax.random.key(1),
    target.domain,
    bins=32,
    transforms=6,
    euclidean_bound=8.0,
    hidden_features=(256, 256),
    slope=1e-3,
    activation=jax.nn.silu,
    mask_strategy="balanced",
).zeros()
```

`Mixed_NSF` stacks spline coupling layers without linearly mixing incompatible
coordinate types. Euclidean coordinates use rational-quadratic splines with
identity tails; torsions use circular splines. Either block can condition the
other.

`mask_strategy="random"` uses a random half-mask. `"balanced"` places members
of each nontrivial Euclidean/periodic block on both sides when possible, then
alternates complementary masks.

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>flow(x)</code></td><td>native forward map</td><td>wrapped <code>[N,d]</code></td></tr>
<tr><td><code>flow.call_and_ladj(x)</code></td><td>forward map and log-Jacobian</td><td><code>(y, ladj)</code></td></tr>
<tr><td><code>flow.inv(y)</code></td><td>inverse map</td><td>wrapped <code>[N,d]</code></td></tr>
<tr><td><code>flow.inv_and_ladj(y)</code></td><td>inverse map and inverse log-Jacobian</td><td><code>(x, ladj)</code></td></tr>
<tr><td><code>flow.t()</code></td><td>composed public transform</td><td><code>ComposedTransform</code></td></tr>
<tr><td><code>flow.zeros()</code></td><td>identity-parameterized copy</td><td>new <code>Mixed_NSF</code></td></tr>
</tbody>
</table>

</div>

Molecular KLX/KLXX trainers are G-native. If `x` is sampled from the source,
generate target-side proposals with `y = flow.inv(x)`.

## Wrapped mixed-domain MALA

```python
y, accepted = mixed_mala_step(
    key, samples, target, target.domain,
    dt=1e-4,
    image_radius=3,
)

y, acceptance_hist = mixed_mala(
    key, samples, target, target.domain,
    dt=1e-4,
    steps=100,
    image_radius=3,
    chunks=8,
)
```

The Euclidean proposal is Gaussian. The torsion proposal is wrapped, and the
Metropolis ratio uses a finite image sum for the wrapped-normal transition
density. `mixed_mala_step` returns one Boolean per chain. `mixed_mala` returns
a chunk-weighted mean acceptance history with shape `(steps,)`.

`wrapped_normal_relative_error_bound(dt, image_radius)` reports the analytic
truncation bound used to choose the periodic image radius. Mixed MALA is always
adjusted and has no ULA mode.

## Potential-space SMC

```python
samples, level_ess, level_acceptance = sequential_monte_carlo(
    key,
    samples,
    source,
    target,
    ladder=8,
    mc_dt=1e-4,
    mc_steps=10,
    mc_image_radius=3,
    domain=target.domain,
    chunks=8,
)
```

The uniform schedule uses levels `1/ladder, ..., 1`. At each level SMC:

1. applies the incremental geometric-bridge weight;
2. records normalized ESS;
3. resamples back to the original particle count; and
4. applies mixed MALA at the matching intermediate potential.

The return shapes are `[N,d]`, `(ladder,)`, and `(ladder, mc_steps)`.
`potential_space_smc(..., t_list=...)` uses caller-supplied absolute levels
instead of a uniform ladder.

## Flow-proposal AIS

```python
samples, initial_log_weights = annealed_importance_sampling(
    key,
    source_samples,
    source,
    target,
    flow,
    ladder=8,
    mc_dt=1e-4,
    mc_steps=10,
    domain=target.domain,
    chunks=8,
    return_initial_log_weights=True,
)
```

The molecular flow is G-native, so the initial proposal is `flow.inv(x)`.
The routine divides the proposal log weight across `ladder` resampling levels,
but rejuvenates at the final target after every level. It is therefore the
same score-free flow-proposal surrogate used by generic `jflows`, not exact
potential-space AIS.

## Mixed quench and temper

```python
hat_samples, acceptance_hist = mixed_quench_and_temper(
    key,
    samples,
    target,
    target.domain,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-4,
    mc_steps=50,
    mc_image_radius=3,
    chunks=8,
)
```

The routine optionally adds Gaussian noise to the Euclidean block and redraws
the torsions uniformly, minimizes each chunk with L-BFGS and Armijo search,
wraps the result, then tempers with mixed MALA. It returns the final population
and the MALA acceptance history.

## Chunking semantics

`chunks` always means a number of row partitions, not a row count. Larger
values reduce the number of rows entering one eager compiled kernel. The same
spelling is used for mixed MALA, SMC, AIS, quench and temper, and high-level
validation weights.

Chunking does not change the target distribution or objective. Different
chunk counts consume JAX keys in different partition patterns, so stochastic
samples need not be bitwise identical.

## Native OpenMM backend

The OpenMM backend consumes the same bundle but stays in Cartesian coordinates
and executes energies, forces, Langevin dynamics, and replica exchange in
OpenMM contexts.

```python
from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering

target_mm = OpenMM_Potential.from_bundle("glycerol_gaff2_am1bcc_obc1")
soft_mm = target_mm.regularized((50.0, 0.10))
```

The native reduced potential is `beta E(x)`. It does not include `-log J`
because its argument is Cartesian position rather than the mixed quotient
coordinate.

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>target_mm.create_system()</code></td><td>fresh OpenMM System</td><td><code>openmm.System</code></td></tr>
<tr><td><code>target_mm.physical_energy(x)</code></td><td>physical Cartesian energy</td><td>kJ/mol scalar or batch</td></tr>
<tr><td><code>target_mm.forces(x)</code></td><td>OpenMM force of the active potential</td><td>kJ/mol/nm Cartesian array</td></tr>
<tr><td><code>target_mm(x)</code></td><td>reduced Cartesian potential</td><td><code>beta E(x)</code></td></tr>
<tr><td><code>target_mm.regularized((e,r))</code></td><td>independent OpenMM realization of the JAX surrogate</td><td>regularized potential</td></tr>
<tr><td><code>soft_mm.regularized_energy(x)</code></td><td>active mapped Cartesian energy</td><td>kJ/mol scalar or batch</td></tr>
</tbody>
</table>

</div>

The regularized system replaces Amber regular/exception pair interactions by
the same floor-aware expression as the JAX implementation, then wraps the
complete energy in the same reference-relative logarithmic map. OpenMM
differentiates that composed energy to produce regularized forces. There are no
JAX callbacks in native simulation.

### Native Langevin

```python
trajectory, energy = langevin(
    soft_mm,
    positions_nm=None,
    steps=10000,
    sample_interval=100,
    timestep_fs=1.0,
    friction_per_ps=1.0,
    temperature_kelvin=None,
    seed=0,
    platform="CUDA",
)
```

`positions_nm=None` starts from the bundle reference. The default temperature
is the potential temperature. The function uses OpenMM's
`LangevinMiddleIntegrator` and returns sampled positions with shape
`(ceil(steps/sample_interval), A, 3)` and potential energies in kJ/mol.

### Native parallel tempering

```python
replicas, energy, swap_acceptance = parallel_tempering(
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

Each temperature owns an OpenMM context and Langevin integrator. After every
round, alternating adjacent temperature pairs attempt a replica exchange with
the standard canonical Metropolis ratio. Positions move between fixed
temperature slots; the slot velocities remain governed by their local
thermostats.

The returns have shapes `(rounds, replicas, A, 3)`, `(rounds, replicas)`, and
`(replicas-1,)`. The last array is the acceptance fraction for each adjacent
temperature pair. Either a physical or regularized `OpenMM_Potential` can be
passed to both native samplers.

## Executable references

- `smoke/test_bundles.py`: bundle contract and named targets.
- `smoke/test_molecular_potential.py`: JAX energy, force, coordinate, and
  temperature behavior.
- `smoke/test_regularization.py`: `(e,r)` equations.
- `smoke/test_mixed_nsf.py`: mixed flow inversion, Jacobians, and seams.
- `smoke/test_stereochemistry.py`: legacy and multi-center coordinate support.
- `smoke/test_support_and_utils.py`: mixed MALA, SMC, AIS, QT, and support.
- `smoke/test_openmm.py`: JAX/OpenMM parity and both native samplers.
