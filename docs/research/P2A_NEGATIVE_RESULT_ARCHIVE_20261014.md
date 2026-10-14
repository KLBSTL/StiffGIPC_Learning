# P2-A negative-result snapshot — 2026-10-14

This research branch preserves the M0 evidence tools and the P2-A execution
prototype that **did not meet the performance adoption gate**. It is an archive
for comparison and future work, not a recommended faster solver.

Starting commit: `a4b83c040161eeef6149c93ae6167644fdf19384`.
Evaluated Windows binary SHA256:
`abd50f5a9ebf03df2b3737a52d88bc67dc73a8b94bfd8d758cb08af17b5bcf18`.

Two independent IPC options default to off:

- `GIPC_P2_LAZY_SCENE`: defer host BVH bounds until an actual CPU consumer.
- `GIPC_P2_MERGE_ABD_ENERGY`: combine the two original ABD reductions into the
  energy result readback, retaining the original host addition order.

`GIPC_P2_AUDIT` is an independent diagnostic. Materials, Full CCD, PCG math and
termination tolerances were retained. The line-search entry device sync remains.

Local evaluation used `dt=.04`, `rho=1e-4`, legacy cumulative tolerance `.01`,
minimum 6 updates, three paired repeats, and the existing B1 execution bundle.

| Short scene | Combined total solver speedup |
|---|---:|
| Boxes, 17 frames | 1.0026 |
| Mixed, 15 frames | 0.9909 |
| Bunny, 14 frames | 1.0018 |

Matched-checkpoint line-search speedups were only 1.035–1.040. Three-pair
intervals cross 1. Trajectory differences and incomplete extreme-path coverage
prevent a same-physical-error certification. The decision is **STOP expansion
to P2-B; keep both options off**.

24 native CTest tests and 14 Python tests passed; 79 successful scene runs covered
384 frames. 54 restores verified 40 buffers each, 77 ABD energy pairs matched
bitwise, and 140 ordinary BVH equivalence checks passed. Failed resource/path
runs were excluded and retained locally.

Evidence:

- [P2-A report](../../reports/innovation_p2a_20261014/REPORT.md)
- [Consumer and dependency audit](../../reports/innovation_p2a_20261014/SOURCE_AUDIT.md)
- [Final analysis](../../reports/innovation_p2a_20261014/analysis_v2/summary.json)
- [Independent data verification](../../reports/innovation_p2a_20261014/analysis_v2/data_validation.json)
- [M0 report](../../reports/innovation_m0_20261014/REPORT.md)

Sealed reports describe their original evaluation-time state, including local
paths and the then-uncommitted/unpublished status. They were not rewritten for
publication. `verify_p2a.py` is the original run-day verifier and assumed HEAD was
the starting commit; use `verify_p2a_snapshot.py` for the archived source check.
Original raw `runs/`, compiled binaries and native source snapshots remain local
and are ignored by Git. Summaries, audits, identities and reproduction scripts
are included. A new checkout needs its own build and same-binary checkpoints;
never substitute a checkpoint binary ID.
