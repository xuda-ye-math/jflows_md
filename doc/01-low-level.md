# Low-level interfaces

The low level contains the molecular objects and numerical kernels used to
assemble custom mixed-domain pipelines. It covers frozen bundles, the JAX
source and potential, mixed normalizing flows, mixed MALA and HMC, the
importance-weight screen, the flow-proposal SMC, quench and temper, and the
independent native OpenMM backend.

## Public imports

```python
from jflows_md.system import Molecular_Bundle, available_bundles
from jflows_md.source import Molecular_Source
from jflows_md.potential import (
    KB_KJ_MOL_K, Molecular_Potential, Regularized_Molecular_Potential,
)
from jflows_md.flow import Mixed_Identity, Mixed_NSF
from jflows_md.utils import (
    mixed_mala_step,
    mixed_mala,
    mixed_hmc_step,
    mixed_hmc,
    screen_log_weight,
    compute_ESS_log,
    linear_weights_from_log,
    SCREEN_FRACTION,
    sequential_monte_carlo,
    flow_target_batch,
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

The source checkout contains five targets:

| Bundle | Atoms / dimension | Mixed domain | Fixed centers |
|---|---:|---|---:|
| `alanine_dipeptide_ff96_obc1` | 22 / 60 | `R^42 x T^18` | 1 |
| `methane_gaff2_am1bcc_obc1` | 5 / 9 | `R^7 x T^2` | 0 |
| `ethane_gaff2_am1bcc_obc1` | 8 / 18 | `R^13 x T^5` | 0 |
| `propane_gaff2_am1bcc_obc1` | 11 / 27 | `R^19 x T^8` | 0 |
| `n_butane_gaff2_am1bcc_obc1` | 14 / 36 | `R^25 x T^11` | 0 |

Alanine dipeptide uses ff96; the alkanes use GAFF2/AM1-BCC. All use
OBC1/ACE, `NoCutoff`, and no constraints. Installed wheels contain code only, so a wheel user
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
target = Molecular_Potential.from_bundle("alanine_dipeptide_ff96_obc1")
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
backward-compatibility exception is a target-only call for `alanine_dipeptide`
with every new coordinate option at its default; that call
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

These policies are target-specific and do not imply automatic
stereochemistry inference for arbitrary molecules.

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
    "alanine_dipeptide_ff96_obc1",
    temperature_kelvin=300.0,
)
```

`Molecular_Potential(bundle, *, temperature_kelvin=None)` is the constructor
behind `from_bundle(path_or_name, *, root=None, verify=True,
temperature_kelvin=None)`. With `temperature_kelvin=None` the bundle
temperature is used.

For internal coordinates $q$, the target evaluates

$$
\begin{aligned}
U(q)&=R\bigl(\beta E(x(q))\bigr)-\log J(q), \\
\beta&=\frac{1}{k_{\mathrm B}T}.
\end{aligned}
$$

`E` is the pure-JAX Amber bonded/nonbonded plus OBC1/ACE energy in kJ/mol.
`J` is the coordinate Jacobian for the rigid-motion-quotient chart. $R$ is
the energy regularization described below. The potential consumes `[N,d]`
and returns `[N]`, so it can be used by generic `jflows` losses and potential
algebra.

`KB_KJ_MOL_K = 0.00831446261815324` is the package conversion constant used
for $\beta$ when energies are measured in kJ/mol and temperature in kelvin.

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>target(q)</code></td><td>reduced internal-coordinate potential <code>beta E(x(q)) - log J(q)</code></td><td><code>[N]</code></td></tr>
<tr><td><code>target.grad(q)</code></td><td>JAX gradient of the reduced potential</td><td><code>[N,d]</code></td></tr>
<tr><td><code>target.cartesian(q)</code></td><td>canonical Cartesian representative</td><td><code>[N,A,3]</code> in nm</td></tr>
<tr><td><code>target.physical_energy(q)</code></td><td>raw Cartesian force-field energy</td><td><code>[N]</code> in kJ/mol</td></tr>
<tr><td><code>target.reduced_energy(q)</code></td><td>reduced energy <code>beta E(x(q))</code> without the Jacobian</td><td><code>[N]</code></td></tr>
<tr><td><code>target.energy_terms(q)</code></td><td>reduced force-field terms, <code>logdet</code>, and the total <code>potential</code></td><td>dictionary of <code>[N]</code> arrays</td></tr>
<tr><td><code>target.regularized((e, r))</code></td><td>the <code>Regularized_Molecular_Potential</code> <code>U^rho</code> (see below)</td><td>potential</td></tr>
<tr><td><code>target.reference_internal()</code></td><td>bundle reference in the mixed chart</td><td><code>[d]</code></td></tr>
<tr><td><code>target.support_mask(x)</code></td><td>stereochemical support of Cartesian frames</td><td>Boolean batch</td></tr>
<tr><td><code>target.source()</code></td><td>temperature-matched molecular source</td><td><code>Molecular_Source</code></td></tr>
</tbody>
</table>

</div>

### The regularized potential `U^rho`

```python
soft = target.regularized((50.0, 0.25))   # rho = (e [kJ/mol], r [nm])
```

`Molecular_Potential.regularized(rg_param)` returns the
`Regularized_Molecular_Potential` of the manuscript: the regular and exception
nonbonded pair distances are floored at `r`, the excess of the floored energy
over the floored energy of the bundle reference geometry is compressed by
`C_e(dE) = dE` below `e` and `e (1 + log(dE / e))` above it; the reduced
potential is `beta (E_star + C_e) - log J`.
`soft.regularized_energy(q)` is `E_star + C_e(dE)` in kJ/mol,
`soft.physical_energy(q)` the raw energy, and `soft.reference_energy_kj_mol`
the floored reference energy. `e -> inf, r -> 0` recovers `target`. The
generators build their diagonal stage targets `(1 - t) U_0 + t U^{rho_t}`
from this object along a regularization path (see `doc/03-high-level.md`).

## Mixed flows

### `Mixed_Identity`

`Mixed_Identity(domain)` wraps periodic coordinates and otherwise preserves
the input. Its forward and inverse log-Jacobians are zero and `zeros()`
returns the flow itself. The molecular Boltzmann generators compare a trained
map against `flow.zeros()` of the supplied architecture, so `Mixed_Identity`
is the flow-shaped identity for custom pipelines and tests.

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
adjusted and has no ULA mode. A non-finite Metropolis log-acceptance is
mapped to `-inf`, so that proposal is rejected and the input particle
retained.

## Wrapped mixed-domain HMC

```python
y, accepted = mixed_hmc_step(
    key, samples, target, target.domain,
    dt=1e-3,
    leapfrog_steps=10,
)

y, acceptance_hist = mixed_hmc(
    key, samples, target, target.domain,
    dt=1e-3,
    leapfrog_steps=10,
    trajectories=1,
    chunks=8,
)
```

One trajectory draws a random momentum for every coordinate, integrates the
Hamiltonian flow by `leapfrog_steps` leapfrog steps of size `dt` with the
periodic coordinates wrapped after every position update (a volume-preserving
map on the torus, so the accept/reject test is the Euclidean one), and accepts
the endpoint with probability `min(1, exp(H(start) - H(end)))`. A non-finite
trajectory is rejected. `mixed_hmc` runs `trajectories` trajectories per
particle, chunked along the rows, and returns the moved particles and the
mean acceptance per trajectory with shape `(trajectories,)`, the same
contract as `mixed_mala`. The flow-proposal SMC uses one trajectory on each
intermediate level.

## Importance-weight screen

```python
screened = screen_log_weight(log_weight, fraction=SCREEN_FRACTION)
ess = compute_ESS_log(log_weight, fraction=SCREEN_FRACTION)
weights = linear_weights_from_log(log_weight, fraction=SCREEN_FRACTION)
```

`jflows_md.utils.screen` carries the screen that every ESS and every
resampling weight in `jflows_md` passes through. `screen_log_weight` sets
infinite or NaN log weights and the `fraction` largest ones to `-inf`, so
they get weight zero; at least one weight is removed
(`k = max(1, ceil(fraction * count))`). A log weight far above the rest marks
a hole of the pushforward density at an ordinary target point; kept, it would
dominate an ESS and be copied into most of a resampled population.
`compute_ESS_log` and `linear_weights_from_log` are the `jflows` reductions
applied to the screened log weights: the normalized ESS in `[0, 1]` and the
normalized linear weights for resampling.

Every module imports these two names from this module, so a log weight is
screened exactly once, at the point where it is reduced to an ESS or to
resampling weights, and never at the point where it is created. One default,
`SCREEN_FRACTION = 1e-4`, serves the trainers, the generators, the SMC, quench
and temper, and `run_inference` alike. Every caller passes the fraction it
was given.

## Flow-proposal SMC

```python
samples, proposal, proposal_log_weights = sequential_monte_carlo(
    key,
    samples,
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

`sequential_monte_carlo` (alias `smc`) manufactures target samples from
source samples through a trained flow, the molecular counterpart of
`jflows.utils.sequential_monte_carlo`. The flow is always the inverse map
`G` (target -> source), so the source particles are pushed through
`flow.inv_and_ladj` and wrapped; the pushforward and its full
proposal-to-target log weight

```text
log w = U_source(x) - U_target(G^{-1}(x)) + log|det J_{G^{-1}}(x)|
```

are kept. Each of the `ladder` levels `m = 1, ..., M`:

1. reweights the current particles by the `1/M`-th power of the
   proposal-to-target weight (the pushforward weight at level 1, the weight
   refreshed through `G` at the moved particles afterwards);
2. resamples with the screened weights
   (`linear_weights_from_log(log_weight, screen_fraction)`); and
3. rejuvenates under `target` with wrapped MALA steps of size `mc_dt`:
   `mc_steps_1` steps on the levels `1 .. M-1`, `mc_steps_2` steps on the
   level `M`. The energy of the resampled particles is carried into the
   kernel, so each MALA step costs one energy-and-force evaluation.

Rejuvenation targets the final target at every level, so the routine is the
SMC target surrogate of the trainers rather than an exact sampler on the
geometric path. The result is `(samples, proposal, proposal_log_weights)`:
the target samples, the pushforward the levels started from, and the full
proposal-to-target log weights on that pushforward (the batch ESS diagnostic
of the trainers). With `domain=None` the domain is taken from `target.domain`
or `flow.domain`.

`flow_target_batch(key, samples, source, target, flow, domain, *, ladder=1,
mc_dt=1e-3, mc_steps_1=100, mc_steps_2=100, mc_image_radius=3,
screen_fraction=SCREEN_FRACTION)`
is the pure, single-chunk form of the same levels and kernels, traceable
inside a scan; the trainers call it in every optimizer step.
`sequential_monte_carlo` is its eager, chunked form for populations, with the
pushforward, the weights, and the rejuvenations run chunk by chunk.

`sequential_monte_carlo_fab` (alias `smc_fab`) and `flow_fab_batch` take the
same arguments and are the exact two-phase form used by the FAB trainers.
Phase 1 is the routine above. Phase 2 runs `ladder` further levels along
`rho_k = pi (pi / nu)^(k/M)`, `k = 1, ..., M`, where `nu` is the pushforward
density of the source through `G^{-1}`,

```text
log nu(y) = -U_source(G(y)) + log|det J_G(y)|
```

each level reweighting by `log(pi / nu) / M`, resampling with the screened
weights, and rejuvenating under its own `rho_k` (`_path_potential`, the
potential `(1 - s) U_target - s log nu` at `s = -k/M`) with `mc_steps_1`
MALA steps, `mc_steps_2` on the last level, whose distribution is
`pi^2 / nu`. These levels evaluate `log nu` through the flow, so a MALA step
there costs a flow inverse and its Jacobian. The result is
`(samples, proposal, proposal_log_weights)` with the phase-2 particles and
phase 1's pushforward and log weights.

## Mixed quench and temper

```python
hat_pool, acceptance_hist = mixed_quench_and_temper(
    key,
    samples,
    target,
    target.domain,
    melt=1.0,
    opt_alpha=1e-2,
    opt_steps=200,
    mc_dt=1e-3,
    mc_steps=100,
    mc_image_radius=3,
    chunks=8,
    coeff_qt=0.0,
    screen_fraction=SCREEN_FRACTION,
)
```

The construction executes:

```text
input population
  -> (melt > 0) Gaussian scatter of the Euclidean block with standard
     deviation melt, uniform redraw of the periodic block
  -> per-particle L-BFGS quench into the basins of target
     (alpha=opt_alpha, steps=opt_steps, Armijo search), wrapped
  -> mc_steps MALA steps under target
  -> (coeff_qt > 0) resample the tempered particles by
     exp(-coeff_qt * target(y)) with screened weights, then mc_steps further
     MALA steps under target
  -> pool and MALA acceptance history (of the last MALA run)
```

`coeff_qt > 0` weights each tempered particle by `exp(-coeff_qt U(y))`,
resamples, and rejuvenates the pool again; the default `0` returns the
tempered pool. A tempered particle left at high energy, such as a clash the
quench carried away along the singular nonbonded core, is removed by the
resampling. `chunks` partitions the quench, the weights, and both MALA runs.
The KLXX trainer builds its quench-and-temper pool with this routine.

## Chunking semantics

`chunks` always means a number of row partitions, not a row count. Larger
values reduce the number of rows entering one eager compiled kernel. The same
spelling is used for mixed MALA and HMC, the population SMC, quench and
temper, and the high-level validation weights. `flow_target_batch` is the
single-chunk form and has no `chunks` argument.

Chunking does not change the target distribution or objective. Different
chunk counts consume JAX keys in different partition patterns, so stochastic
samples need not be bitwise identical.

## Native OpenMM backend

The OpenMM backend consumes the same bundle but stays in Cartesian coordinates
and executes energies, forces, Langevin dynamics, and replica exchange in
OpenMM contexts.

```python
from jflows_md.openmm import OpenMM_Potential, langevin, parallel_tempering

target_mm = OpenMM_Potential.from_bundle(
    "alanine_dipeptide_ff96_obc1",
    temperature_kelvin=None,
)
```

The native reduced potential is `beta E(x)`, evaluated on the OpenMM energy
for reference checks. It does not include `-log J` because its argument is
Cartesian position rather than the mixed quotient coordinate.
`OpenMM_Potential(bundle, *, temperature_kelvin=None)` also accepts a bundle
name or path in place of a loaded `Molecular_Bundle`, and
`target_mm.regularized((e, r))` is the OpenMM form of the `(e, r)` surrogate
(pair floor and reference-relative compression as OpenMM custom forces), used
by the smoke tests to check the JAX surrogate's energies and forces.

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>target_mm.create_system()</code></td><td>fresh OpenMM System</td><td><code>openmm.System</code></td></tr>
<tr><td><code>target_mm.physical_energy(x, platform=None)</code></td><td>physical Cartesian energy</td><td>kJ/mol scalar or batch</td></tr>
<tr><td><code>target_mm.forces(x, platform=None)</code></td><td>OpenMM force of the bundle System</td><td>kJ/mol/nm Cartesian array</td></tr>
<tr><td><code>target_mm.reduced_energy(x, platform=None)</code></td><td>regularized reduced energy</td><td><code>R(beta E(x))</code></td></tr>
<tr><td><code>target_mm(x, platform=None)</code></td><td>same as <code>reduced_energy</code></td><td><code>R(beta E(x))</code></td></tr>
</tbody>
</table>

</div>

Each evaluation builds a fresh System and Context from the bundle's
`system.xml`; `platform` selects the OpenMM platform by name. There are no
JAX callbacks in native simulation.

### Native Langevin

```python
trajectory, energy = langevin(
    target_mm,
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
temperature pair. Both native samplers integrate the bundle's OpenMM System
through `target_mm.create_system()` and report the raw OpenMM potential
energy in kJ/mol; the energy cut and cap act on `reduced_energy` evaluations,
not on the integrator's forces.

## Executable references

- `smoke/test_bundles.py`: bundle contract and named targets.
- `smoke/test_molecular_potential.py`: JAX energy, force, coordinate, and
  temperature behavior, and the energy cut and cap.
- `smoke/test_mixed_nsf.py`: mixed flow inversion, Jacobians, and seams.
- `smoke/test_stereochemistry.py`: legacy and multi-center coordinate support.
- `smoke/test_smc_hmc.py`: mixed HMC, the flow-proposal SMC and the FAB SMC
  in both forms,
  the importance-weight screen, and quench and temper with `coeff_qt`.
- `smoke/test_float32_alanine_dipeptide_compile.py`: float32 alanine dipeptide energy and
  gradient compilation and one-step mixed MALA.
- `smoke/test_openmm.py`: JAX/OpenMM parity of the physical and regularized
  energies and both native samplers.
