# Methane 9D molecular Boltzmann generator

This is the single minimal molecular Boltzmann generator example distributed
with the source repository. The target is CH4 at 300 K in nine mixed internal
coordinates, using GAFF2/AM1-BCC with OBC1/ACE and the fixed regularization
parameter `(100.0, 0.15)`.

The directory is self-contained:

- `train.py` runs identity, forward KL, or KLXX through the existing
  `--method id`, `--method kl`, and `--method klxx` identifiers;
- `parameters.py` records the complete training and stage-schedule settings;
- `bundle/` is the verified molecular bundle loaded by the driver; and
- `results/` contains the copied reports from the completed runs.

No run was repeated when this example was added. A new run, when explicitly
desired, writes logs and complete-stage artifacts below a new local
`artifacts/` directory.
