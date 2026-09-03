# jflows_md smoke suite

Run all tests from the repository root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

The suite checks:

- canonical six-file bundle loading and stored OpenMM parity;
- BAT round trips, Jacobians, molecular support, and Amber/OBC energies;
- mixed Euclidean/torus NSF inversion and seam behavior;
- mixed-domain MALA and HMC, the flow-proposal SMC with HMC intermediate
  levels, the importance-weight screen, and quench and temper with `coeff_qt`;
- two-step KLX/KLXX training in float32;
- the identity, KLX, and KLXX generators on a toy target, complete-stage
  persistence, and the inference scheme on the stored run;
- the regularized `(e, r)` surrogate and the diagonal regularization path, OpenMM/JAX
  energy and force parity, and native Langevin and replica exchange.

The core assumes shape- and type-correct inputs. Numerical safeguards reject
nonfinite MALA proposals, optimizer states, and completed-stage populations.

Build a runtime bundle with the installed optional builder:

```bash
python -m jflows_md.bundle_build \
  alanine_dipeptide molecule.prmtop molecule.rst7 generated/alanine_dipeptide
```

The source-checkout wrapper is
`python -m bundles.build_molecular_bundles ...`.
