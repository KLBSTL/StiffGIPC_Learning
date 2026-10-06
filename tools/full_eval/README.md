# Full evaluation controller

Only this additive directory is new. The controller reuses the existing Linux
`execute`, `execute_base`, `base_seal`, file identities, exclusive GPU lock,
process-group termination, GPU load/memory monitoring and disk/time reserves.
It neither builds nor runs SSH, installs packages, archives or deletes data.
Python standard library and NumPy suffice; no SciPy/psutil dependency is added.

## Frozen finite plan

Three scenes: `cloth_hang_l`, `cloth_fixed_bunny_l`, `bunny_cloth_bunny_l`.
All 100-frame runs start from zero, dt=.01, legacy cumulative .01/minimum6,
PCG rho1e-4, original materials/MAS, full CCD, K1, pool off, raw edge order.
The declared timeout is 120 seconds for 100 frames and 600 seconds for 300 frames.
Positions and active actual velocities are exported, with no heavy diagnostics,
Nsight or cost trace. Original Stiff has no actual-velocity/resolved-config
capability; do not reconstruct velocity or certify equivalence from positions.

For every scene, three original-Stiff calibration runs and two independent
holdouts precede every candidate. Then seven alternating original/combined
pairs; the first three also contain alternating IPC host/Graph comparisons.
This is 75 runs. At most one 300-frame combined stability run per scene follows,
only if all its 25 full runs passed hard checks and the analyzer's frozen
material/holdout gate is eligible. No retries, K4 arms or parameter grid.

The four arms are original frozen Stiff; active IPC host without components;
active IPC conditional Graph without components; active combined Graph with
existing refit/batch/reuse plus ordinary BVH refit at interval 8.

## Explicit commands (Linux)

Run from the new owned repository root. Supply fresh active/base builds; the
baseline seal expects `build/base/gipc`. All common assets must match. Exactly
the eight reviewed, unused baseline sorting caches may differ in inventory;
the build attachment must prove no selected scene references them, and all
eight remain source-hashed. Identical asset trees use the original base seal.
Set the caller's valid `DISPLAY` if native GLUT needs Xvfb.

```text
python tools/full_eval/runner.py seal --active-manifest build/active/manifest.json --base-build-log build_logs/base/build.log --build-identity reports/build_identity_after.json --output reports/full_eval_seal.json
python tools/full_eval/runner.py init --seal reports/full_eval_seal.json --session runs/full_eval_20261006
python tools/full_eval/runner.py run-next --seal reports/full_eval_seal.json --session runs/full_eval_20261006 --gpu 0
```

Every `run-next` processes exactly one declared task (or records its skip), then
performs CPU analysis. Inspect the returned status and new summary before the
next explicit invocation. `analysis_exception=true` stops all advancement;
Exit code 2 signals that condition. No tool invokes the next run automatically.

The plan and source/build/analyzer identities are sealed before initialization.
The required build-identity attachment binds the pre/post-build source inventory,
actual objects/device link/link commands and the exact two executable hashes;
its verifier rechecks saved file identities without compiling or running Ninja.
Every task writes exclusive `analysis/<name>.json`, `summary/NNNN.json`, and
`ledger/NNNN.json`; the ledger binds the previous ledger SHA, task, original
result/evidence hashes and analysis/summary hashes. Existing output, missing
ledger, tampered metadata or changed code is a refusal, never a restart.
The old pool screen's failed `allow_next_round` is not a precondition here.

Quality range failures are retained as diagnostics and do not stop declared
100-frame repetitions. Numerical/configuration/resource failures block future
runs of that scene/arm. The full controller additionally rejects any purported
completed run whose wall time exceeds its declared 120/600-second deadline.
Neither success nor a complete sequence alone certifies physical quality or 2×.
`frames.csv/solver_ms` is CPU steady-clock solver duration including its sync;
do not reuse the older runner's inaccurate CUDA-event wording as a claim.

## Space and receipts

Startup still requires 4 GiB free; running stops below 1 GiB. The owner may archive,
download, verify and remove large raw `.bin` files between explicit steps.
The controller never performs those operations. Keep all original metadata,
`evidence.json`, result, trace CSV, analyses and ledgers unchanged.
Missing raw is accepted only with `run/archive_receipt.json`:

```json
{"schema":"full_eval_raw_archive.v1","archive_path":"transfers/run.tar.gz",
 "archive_sha256":"...","archive_bytes":123,"evidence_sha256":"...",
 "download_verified":true,"local_archive_name":"run.tar.gz",
 "removed_raw_paths":["trace/state_0001.bin"]}
```

The archive must remain below the owned repository, match its SHA/size, and
every removed path must refer to an original evidence-listed raw bin. Missing
metadata is never accepted through an archive receipt. Pairwise CPU comparisons
must be completed while both required trajectories remain available; archived
run reports retain those results for later summaries.
Without an archive receipt, all original files are rehashed before advancement,
including same-sized raw files. With a receipt, the preserved original archive
is fully hashed and retained metadata verified; that branch does not claim to
rehash surviving raw copies. The first per-run analysis verifies original raw.

CPU-only tests: `python tools/full_eval/runner_test.py -v`.
