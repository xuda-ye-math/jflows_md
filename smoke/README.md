# jflows_md smoke suite

Run all tests from the repository root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python smoke/run_all.py
```

The suite checks:

- canonical six-file bundle loading and stored OpenMM parity;
- BAT round trips, Jacobians, molecular support, and Amber/OBC energies;
- mixed Euclidean/torus NSF inversion and seam behavior;
- fixed-shape chunked MALA, SMC, AIS, and quench-and-temper kernels;
- minimal two-value KLX/KLXX training;
- linear `(e,r)` sharpening and its post-sharpen endpoint;
- OpenMM/JAX energy and regularization parity plus native Langevin and replica exchange;
- combined flow/sharpening ESS rejection and stage shrink;
- rejection of nonfinite MALA proposals, optimizer updates, and final populations;
- template-based artifacts and exact computed-stage interruption/resume;
- the live `jflows` 0.5 interfaces used by this package.

The core assumes shape- and type-correct inputs. Numerical safeguards reject
nonfinite MALA proposals, optimizer states, and completed-stage populations.

Build a runtime bundle with the installed optional builder:

```bash
python -m jflows_md.bundle_build \
  glycerol molecule.prmtop molecule.rst7 generated/glycerol
```

The source-checkout wrapper is
`python -m bundles.build_molecular_bundles ...`.

## Opt-in compilation benchmark

`benchmark_compile.py` is excluded from `run_all.py`. It measures cold and
warm execution for generic NCSF, mixed NSF, glycerol energy/gradient, a mixed
MALA chunk, and bounded generic/molecular KLX calls.

```bash
python smoke/benchmark_compile.py --quick
python smoke/benchmark_compile.py --timeout 480 --warm-repeats 10
```

Each benchmark cell runs in a fresh subprocess and writes local measurements
to `smoke/compile_benchmark.csv`.
