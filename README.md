# StiffGIPC Learning

IPC + CUDA Graph research code for contact cloth and mixed ABD/FEM scenes.

The first commit preserves the source, configurations, reports, build identities
and input meshes before the 2026-10-06 cleanup. Generated trajectories, binaries,
credentials and remote machine configuration are excluded. File SHA-256 values
are recorded in `PRE_CLEANUP_SNAPSHOT.json`.

The current optimization goal is at least 2x relative to the original StiffGIPC,
with unchanged materials, complete collision safety and separately reported
stopping-rule experiments. This goal has not yet been achieved or certified.

The cleanup will replace the historical overlay chain with a standalone active
source tree and a small reproducible AutoDL test entry point. Historical failed
implementations remain recoverable from Git history, while their reports remain
visible in the current tree.

Upstream code retains its original licenses and author attribution.
