# Methane 9D molecular Boltzmann generator

This is the single minimal molecular Boltzmann generator example distributed
with the source repository. The target is CH4 at 300 K in nine mixed internal
coordinates, using GAFF2/AM1-BCC with OBC1/ACE, with the energy cut and cap of
`Molecular_Potential` at their defaults.

The directory is self-contained:

- `train.py` runs identity, forward KL, or KLXX through the existing
  `--method id`, `--method kl`, and `--method klxx` identifiers (the
  package's KLL1, FAB and FABX generators are not wired into this driver);
- `parameters.py` records the complete training and stage-schedule settings;
- `bundle/` is the verified molecular bundle loaded by the driver; and
- `results/` contains the reports of the completed runs, made with the
  earlier `(e, r)` regularization and sharpening that the package no longer
  has; they are not reproduced by the current driver.

No run was repeated when this example was added. A new run, when explicitly
desired, writes logs and complete-stage artifacts below a new local
`artifacts/` directory.
