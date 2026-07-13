# Project status

Last updated: 2026-07-12T22:38:24-04:00

## Current state

- Branch: `main`, tracking `origin/main`.
- Pre-checkpoint HEAD: `5896ba275952333267592e7cf7cb3c0313d5b8f8`
  (`Support empty molecular interaction families`).
- The reviewed molecular refactor and this handoff are currently uncommitted:
  21 tracked files are modified and `status.md` is untracked. Changes cover
  the public package, README, and package smoke suite; no molecular training
  output is being committed.
- Canonical public drivers are `train_forward_KLX_G`,
  `train_forward_KLXX_G`, `boltzmann_forward_KLX_G`, and
  `boltzmann_forward_KLXX_G`. The earlier molecular-prefixed names remain
  compatibility wrappers.
- Direct mixed MALA uses `dt`, `steps`, `image_radius`, and `chunks`;
  composite sampling/training uses `mc_dt`, `mc_steps`,
  `mc_image_radius`, `train_steps`, `batch_size`, `pool_size`,
  `opt_alpha`, and `opt_steps`.
- Molecular BG stages use the same full-validation ESS-only acceptance rule
  and aligned history schema as `jflows`, with additional optimizer kept/update
  histories. `flow_dir` is non-overwriting and stores relative, movable,
  template-free `Mixed_NSF` artifacts plus monitor sidecars and terminal state.
- Review-driven scientific fixes include matched independent Student-t source
  density/sampling, pure-Euclidean MALA without irrelevant torus certification,
  exact saved condition-mask reconstruction, aligned adaptive-controller
  bounds, and presence-correct legacy keyword conflict detection.
- The complete documented `smoke/run_all.py` package suite passed in
  `~/.envs/jflows` with explicit live-source `PYTHONPATH`. This includes all
  three frozen bundle checks, OpenMM-reference energy/force parity,
  Mixed_NSF/seam/Jacobian checks, float32 training, mixed SMC/AIS/MALA,
  BG retry and artifact tests, saving-on/off numerical equivalence, and the
  bounded float32 glycerol compile path. `compileall` and `git diff --check`
  also passed.
- Two independent final reviews and a cross-package review found no remaining
  major or medium scientific/API issue.
- A verified pre-refactor standalone snapshot is available at
  `/mnt/games/jflows_md_071226`.

## Pending

- Pending outside this repository: old `Codes` and `Molecular_BG` experiment
  tests/reruns were deliberately not launched during this package refactor.
- Production molecular BG training, beginning with the paused alkane series,
  requires separate user authorization and belongs in `X-regularization`, not
  this public package repository.
- Low-priority crash-observability polish: write the initial `running` manifest
  before the first SMC/training operation, so a hard interruption cannot leave
  an empty `flow_dir`.
- Remove retired stage and molecular-prefixed API aliases only in a future
  major version.

## Timeline

### 2026-07-12T22:38:24-04:00 — Mixed-domain API and recoverable BG artifacts verified

- Synchronized public names and scientific controller rules with `jflows`
  while retaining only the mixed Euclidean/periodic machinery locally.
- Added full-validation ESS histories and portable persistence for every
  accepted or rejected trained candidate and selected stage map.
- Added exact standalone `Mixed_NSF` reconstruction, including activation,
  dtype, balanced or explicit masks, and legacy key-derived masks.
- Fixed the defensive Student-t sampler/density mismatch and strengthened
  Euclidean-only MALA, empty-interaction, float32, bundle-integrity, and
  terminal-failure regressions.
- The full package smoke suite and repeated independent code reviews passed
  with no remaining major or medium issue.
