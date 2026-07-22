# Smoke-test reference

The `smoke/` directory is the executable contract for the public molecular
interfaces. Tests are bounded package checks: they establish equations,
shapes, compilation paths, persistence behavior, and cross-backend parity.
They are not production training runs and their tiny ESS values are not
scientific benchmark results.

Run the suite from a source checkout with both repositories visible:

```bash
source ~/.envs/jflows/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
python /data/projects/jflows_md/smoke/run_all.py
```

`run_all.py` starts every smoke module in a fresh Python subprocess. The
opt-in compilation benchmark is intentionally excluded.

## Test map

<div align="center">

<table>
<thead>
<tr><th>Module</th><th>Primary contract</th></tr>
</thead>
<tbody>
<tr><td><code>test_bundles.py</code></td><td>client-free JAX/OpenMM backend report, six-file bundle loading, named targets, metadata, and non-overwriting builder behavior</td></tr>
<tr><td><code>test_api_consistency.py</code></td><td>public v0.5 namespace, signatures, aliases, and retired layout</td></tr>
<tr><td><code>test_boltzmann_identity.py</code></td><td>flow-free adaptive identity stages with active sharpening and two MALA endpoints</td></tr>
<tr><td><code>test_boltzmann_identity_artifacts.py</code></td><td>identity stage save/load/resume with sharpening histories and no flow files</td></tr>
<tr><td><code>test_artifacts.py</code></td><td>template-based mixed-flow, sample, and history round trips</td></tr>
<tr><td><code>test_jflows_compatibility.py</code></td><td>the live generic <code>jflows</code> interfaces consumed by <code>jflows_md</code></td></tr>
<tr><td><code>test_edge_cases.py</code></td><td>raw mixed-domain sampler equations and numerical edge cases</td></tr>
<tr><td><code>test_mixed_nsf.py</code></td><td>mixed flow inversion, log-Jacobians, wrapping, masks, and seam behavior</td></tr>
<tr><td><code>test_jflows_md_chunking.py</code></td><td>fixed-shape eager chunks, ladder controllers, and chunk equivalence</td></tr>
<tr><td><code>test_float32_training.py</code></td><td>default-float32 direct trainer compilation and execution</td></tr>
<tr><td><code>test_initialization.py</code></td><td>supplied-flow preservation, identity initialization, and stage starts</td></tr>
<tr><td><code>test_boltzmann_checkpoints.py</code></td><td>complete-stage persistence, validation, loading, forking, and post-sharpen resume</td></tr>
<tr><td><code>test_boltzmann_integration.py</code></td><td>actual computed-stage interruption and resumed-run equivalence</td></tr>
<tr><td><code>test_mixed_training.py</code></td><td>direct KLX/KLXX plus bounded adaptive Boltzmann execution</td></tr>
<tr><td><code>test_sharpening_gate.py</code></td><td>flow ESS and sharpening ESS gates with adaptive shrinking</td></tr>
<tr><td><code>test_molecular_potential.py</code></td><td>JAX energies, forces, quotient Jacobian, temperature scaling, and three-bundle parity</td></tr>
<tr><td><code>test_regularization.py</code></td><td>the exact <code>(e,r)</code> energy map and pair-floor derivatives</td></tr>
<tr><td><code>test_openmm.py</code></td><td>JAX/OpenMM energy and regularized-force parity plus native Langevin and parallel tempering</td></tr>
<tr><td><code>test_support_and_utils.py</code></td><td>stereochemical support, mixed MALA, SMC, AIS, and quench-and-temper</td></tr>
<tr><td><code>test_float32_glycerol_compile.py</code></td><td>bounded real-glycerol energy, flow, and trainer compilation</td></tr>
</tbody>
</table>

</div>

## Routing by interface level

### Low level

Use these when changing bundles, potentials, flows, or sampling kernels:

```text
test_bundles.py
test_molecular_potential.py
test_regularization.py
test_mixed_nsf.py
test_edge_cases.py
test_support_and_utils.py
test_jflows_md_chunking.py
test_openmm.py
```

### Medium level

Use these when changing direct training or simple artifacts:

```text
test_float32_training.py
test_initialization.py
test_mixed_training.py
test_artifacts.py
test_float32_glycerol_compile.py
```

### High level

Use these when changing adaptive stage logic, sharpening, or persistence:

```text
test_sharpening_gate.py
test_boltzmann_identity.py
test_boltzmann_identity_artifacts.py
test_boltzmann_checkpoints.py
test_boltzmann_integration.py
test_mixed_training.py
test_initialization.py
test_jflows_md_chunking.py
```

`test_api_consistency.py` and `test_jflows_compatibility.py` cross the three
levels and should accompany public interface changes.

## Focused commands

Run one module directly:

```bash
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
python smoke/test_openmm.py
```

The OpenMM smoke explicitly selects the Reference platform. It checks:

- physical-energy parity for all three shipped bundles;
- regularized energy and force parity against `Molecular_Potential`;
- native Langevin on physical and regularized systems; and
- native parallel tempering on physical and regularized systems.

The molecular-potential smoke enables x64 for tight stored-reference
comparisons. Normal training remains accelerator-backed float32 unless a
program deliberately enables x64.

## What to verify in output

A zero exit status is necessary. Also verify that the expected PASS lines
cover the intended feature. Important high-level postconditions include:

- a complete adaptive run has a final stage with `t == 1.0`;
- an identity selection reports `selected == "identity"` and stores the
  corresponding flow;
- a sharpening rejection emits no stage;
- `flow_endpoint == "pre_sharpen"`;
- `population_rg == rg_end`;
- KLX histories omit hat-MALA data and KLXX histories include it;
- resume continues from the last manifest-published post-sharpen population;
- OpenMM and JAX energies agree in kJ/mol before the respective beta factors;
  and
- all sampled arrays are finite with the documented shapes.

## Isolated verification

To keep caches, bytecode, and transient artifacts out of the live repositories,
copy the packages and selected smoke tests to a temporary root:

```bash
tmp=$(mktemp -d /tmp/jflows-md-smoke.XXXXXX)
mkdir -p "$tmp/jflows" "$tmp/jflows_md"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows/jflows "$tmp/jflows/"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows_md/jflows_md \
  /data/projects/jflows_md/smoke \
  /data/projects/jflows_md/bundles \
  "$tmp/jflows_md/"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$tmp/jflows:$tmp/jflows_md" \
python "$tmp/jflows_md/smoke/test_openmm.py"
```

Remove only the exact confirmed temporary directory after verification.

## Compilation benchmark

`smoke/benchmark_compile.py` is an opt-in measurement tool:

```bash
python smoke/benchmark_compile.py --quick
python smoke/benchmark_compile.py --timeout 480 --warm-repeats 10
```

It measures cold and warm execution for representative generic and molecular
flows, glycerol potential evaluation, mixed MALA, and bounded trainer calls.
It is not part of `run_all.py` and should not be treated as a correctness test.

## Scope of smoke evidence

Smoke tests validate the checked configuration and code path only. They do not
establish:

- convergence of a production molecular simulation;
- physical correctness of a newly constructed external bundle;
- adequate mode coverage or effective sample size at scientific scale;
- optimal thermostat, timestep, temperature ladder, or flow architecture; or
- compatibility with a platform or precision not exercised by the test.

Production studies should retain independent holdouts, energy/support checks,
stage histories, and reproducible configuration records.
