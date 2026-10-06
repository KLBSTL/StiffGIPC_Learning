# StiffGIPC Learning

IPC + CUDA Graph research code for contact cloth and mixed ABD/FEM scenes.

The first commit preserves the source, configurations, reports, build identities
and input meshes before the 2026-10-06 cleanup. Recover old implementations with
the tag [`archive/pre-cleanup-20261006`](https://github.com/KLBSTL/StiffGIPC_Learning/tree/archive/pre-cleanup-20261006).
Generated trajectories, binaries, credentials and machine configuration are
excluded. File hashes are recorded in `PRE_CLEANUP_SNAPSHOT.json`.

The current optimization goal is at least 2x relative to the original StiffGIPC,
with unchanged materials, complete collision safety and separately reported
stopping-rule experiments. This goal has not yet been achieved or certified.

The active source is standalone; it no longer inherits historical overlays.

| Directory | Purpose |
|---|---|
| `StiffGIPC/` | Active simulation, collision and solver implementation |
| `baseline/` | Frozen original Stiff reference, built independently |
| `Assets/`, `MeshProcess/` | Active input meshes and partition dependency |
| `tests/` | Numerical and configuration invariants |
| `tools/bench/` | Portable, bounded, serial GPU experiments and analysis |
| `tools/validator/`, `references/tight_inclusion/` | Independent CPU path audit |
| `docs/` | Current execution plan and component decisions |
| `history/` | Historical reports and program identities, without version copies |

The cleanup removes rejected bounded CCD, BVH eligibility, MAS final-dot,
SpMV quadratic fusion and ordered restriction implementations, their dedicated
scratch and study branches. The failed AL velocity-only and selected full-step
exit probes are also retired. Explicit requests fail with a recovery-tag message. Current
contact-pool reuse remains an experiment with an independent switch, default off.
IPC legacy stopping, materials and complete CCD remain unchanged.

Linux build (CUDA 12.8, RTX 4090):

```sh
python3 tools/build_linux.py --kind active --jobs 2
python3 tools/build_linux.py --kind base --jobs 2
ctest --test-dir build/active --output-on-failure
```

Build attempts use fresh directories and preserve complete command logs. Test
and timing outputs remain outside Git; compact reports and identities are saved.
Each optimization round is analyzed before advancing. See
[`docs/EXECUTION_PLAN_20261006.md`](docs/EXECUTION_PLAN_20261006.md).

The current execution target is the local Windows workstation, as requested on
2026-10-06. See [`docs/LOCAL_EXECUTION_PLAN_20261006.md`](docs/LOCAL_EXECUTION_PLAN_20261006.md)
and [`docs/CURRENT_COMPONENTS.md`](docs/CURRENT_COMPONENTS.md). Windows builds
assign independent object names to every source, including the two PCG sources
whose filenames differ only in case. The latest Step3 Windows build has a sealed
source/object/link identity, passing CPU stopping tests, three GPU fixtures and
20 rejected retired-entry checks. Subsequent Step3 runs completed three fixed-bunny
cloth pairs and one contact-pool pair in each cloth scene. Fixed-bunny diagnostic
Stiff/current ratios have a median of 1.475x. After the user freed GPU memory,
Step4 completed all six hanging runs: the incremental pool ratio has a median of
1.050x over 51 frames and 1.115x over the contact window. The net whole-prefix
saving is 4.78%, below the declared 5% component threshold. Pool remains default
off. Two protected Windows Nsight captures now separate actual kernel work from
CPU waiting; no native solver code changed in Step4. Full quality/100-frame
regression remains pending, and prior material/resource failures are preserved.
These are not certified speedups.
See [`reports/STATUS_20261006.md`](reports/STATUS_20261006.md) for the current result
and [`reports/local_step4/ROUND_REVIEW.md`](reports/local_step4/ROUND_REVIEW.md)
for the per-round analysis and speed-comparison explanation. The 4090 evidence
remains a separate program identity.

Upstream code retains its original licenses and author attribution.
