# Smoke-test reference

The `smoke/` directory is the executable contract for the public molecular
interfaces. Tests are bounded package checks: they establish equations,
shapes, compilation paths, persistence behavior, and cross-backend parity.
They are not production training runs and their tiny ESS values are not
scientific benchmark results.

Run the suite from a source checkout with both repositories visible:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python smoke/run_all.py
```

`run_all.py` starts every module of its `TESTS` tuple in a fresh Python
subprocess, in the order of the table below, with a 300 s timeout each, and
prints `ALL jflows_md SMOKE TESTS PASSED` at the end.

## Test map

<div align="center">

<table>
<thead>
<tr><th>Module</th><th>Primary contract</th></tr>
</thead>
<tbody>
<tr><td><code>test_bundles.py</code></td><td>package version and public namespace, client-free JAX/OpenMM backend report, six-file bundle loading, the five named targets, external-root selection, and non-overwriting builder behavior</td></tr>
<tr><td><code>test_api_consistency.py</code></td><td>builder signatures; trainer and generator parameter defaults (<code>initialize_from_identity</code>, <code>u_clip</code>, <code>g_clip</code>, <code>lr_warmup</code>, <code>screen_fraction</code>, <code>coeff_qt</code>); the exact <code>boltzmann_identity</code> signature; the five trainers and five trained generators carrying <code>mc_steps_1</code> and <code>mc_steps_2</code>, the FAB losses without <code>coeff_lambda</code>; a two-step <code>train_forward_KLX_G</code> on <code>Mixed_Identity</code>; float32 bundle sources</td></tr>
<tr><td><code>test_artifacts.py</code></td><td>template-based mixed-flow, sample, and history round trips</td></tr>
<tr><td><code>test_mixed_nsf.py</code></td><td>mixed flow inversion, log-Jacobians, wrapping, masks, and seam behavior</td></tr>
<tr><td><code>test_smc_hmc.py</code></td><td>mixed HMC on the torus, <code>flow_target_batch</code> with and without rejuvenation (zero steps reduce the levels to reweight and resample), agreement of <code>sequential_monte_carlo</code> with the batch form, <code>flow_fab_batch</code> and <code>sequential_monte_carlo_fab</code> sharing the phase-1 proposal and weights, <code>screen_log_weight</code>, screened ESS and resampling weights, and quench and temper with <code>coeff_qt</code> (rejected without a source)</td></tr>
<tr><td><code>test_float32_training.py</code></td><td>default-float32 guarded Adam, global gradient clipping, learning-rate warmup, a two-step <code>train_forward_KLX_G</code> with monitor, and the batch ESS against the screened ESS of the pushforward weights</td></tr>
<tr><td><code>test_initialization.py</code></td><td>the keyword-only <code>initialize_from_identity</code> defaults of the trainers (<code>False</code>) and generators (<code>True</code>)</td></tr>
<tr><td><code>test_boltzmann_md.py</code></td><td>the identity, KLX, KLXX, KLL1, FAB, and FABX generators on a toy mixed-domain target, rejection of a retired policy key and of an unknown objective, complete-stage persistence through <code>run</code>/<code>validate</code>/<code>load</code>, and <code>run_inference</code> on the stored run</td></tr>
<tr><td><code>test_regularization.py</code></td><td>the regularized potential <code>U^rho</code> on the NMA bundle (limits, reference geometry, compression, pair floor) and the diagonal stage targets of a regularization path on a fixed schedule, their persistence, and their replay by <code>run_inference</code></td></tr>
<tr><td><code>test_molecular_potential.py</code></td><td>JAX energies, forces, quotient Jacobian, temperature scaling, empty force families, and five-bundle parity with the stored OpenMM references</td></tr>
<tr><td><code>test_openmm.py</code></td><td>JAX/OpenMM parity of the physical and regularized energies plus native Langevin and parallel tempering</td></tr>
<tr><td><code>test_stereochemistry.py</code></td><td>schema-2 compatibility of the alanine dipeptide chart, its one fixed center, and the rejection of invalid stereochemical specifications</td></tr>
<tr><td><code>test_float32_alanine_dipeptide_compile.py</code></td><td>float32 alanine dipeptide energy and gradient compilation, chart-boundary and saturated-angle behavior, the chiral ADP chart, and one-step mixed MALA</td></tr>
</tbody>
</table>

</div>

## Routing by interface level

### Low level

Use these when changing bundles, potentials, flows, or sampling kernels:

```text
test_bundles.py
test_molecular_potential.py
test_mixed_nsf.py
test_stereochemistry.py
test_smc_hmc.py
test_float32_alanine_dipeptide_compile.py
test_openmm.py
```

### Medium level

Use these when changing direct training or simple artifacts:

```text
test_float32_training.py
test_initialization.py
test_artifacts.py
```

### High level

Use these when changing adaptive-staging logic, persistence, or the inference
scheme:

```text
test_boltzmann_md.py
test_regularization.py
test_initialization.py
```

`test_api_consistency.py` crosses the three levels and should accompany
public interface changes.

## Focused commands

Run one module directly:

```bash
python smoke/test_openmm.py
```

The OpenMM smoke explicitly selects the Reference platform. It checks:

- physical-energy parity for all five shipped bundles;
- energy and force parity of the `(e, r)` regularized surrogate between the
  OpenMM custom-force form and `Molecular_Potential.regularized`;
- native Langevin on the bundle System; and
- native parallel tempering on the bundle System.

The molecular-potential and OpenMM smokes enable x64 for tight
stored-reference comparisons. Normal training remains accelerator-backed
float32 unless a program deliberately enables x64.

## What to verify in output

A zero exit status is necessary. Also verify that the expected PASS lines
cover the intended feature. Important high-level postconditions include:

- a complete adaptive-staging run has a final stage with `t == 1.0`;
- an identity selection reports `selected == "identity"`;
- `batch_ess_hist` has shape `(attempts, steps_total)`;
- a `bg_param` key outside the policy raises `KeyError`;
- a stored run validates as `complete` and `load` returns as many records as
  `run` produced;
- `run_inference` reports `complete`, a finite inference set of the requested
  shape, and screened `pushforward_ess` values in `(0, 1]`;
- OpenMM and JAX energies agree in kJ/mol before the respective beta factors;
  and
- all sampled arrays are finite with the documented shapes.

## Isolated verification

To keep caches, bytecode, and transient artifacts out of the live repositories,
copy the packages and selected smoke tests to a temporary root:

```bash
jflows_root="${JFLOWS_ROOT:?set JFLOWS_ROOT}"
jflows_md_root=$(pwd)
tmp=$(mktemp -d)
mkdir -p "$tmp/jflows" "$tmp/jflows_md"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  "$jflows_root/jflows" "$tmp/jflows/"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  "$jflows_md_root/jflows_md" \
  "$jflows_md_root/smoke" \
  "$jflows_md_root/bundles" \
  "$tmp/jflows_md/"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$tmp/jflows:$tmp/jflows_md" \
python "$tmp/jflows_md/smoke/test_openmm.py"
```

Remove only the exact confirmed temporary directory after verification.

## Scope of smoke evidence

Smoke tests validate the checked configuration and code path only. They do not
establish:

- convergence of a production molecular simulation;
- physical correctness of a newly constructed external bundle;
- adequate mode coverage or effective sample size at scientific scale;
- optimal thermostat, timestep, temperature grid, or flow architecture; or
- compatibility with a platform or precision not exercised by the test.

Production studies should retain independent holdouts, energy/support checks,
stage histories, and reproducible configuration records.
