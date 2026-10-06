# MAS final output and rho fusion

**Retired on 2026-10-06:** the active implementation and private study have been
removed after the paired performance screen did not meet its promotion gate.
The remainder is historical documentation. Reproduce it from
`archive/pre-cleanup-20261006`; the current executable rejects either old enable
or study switch when set to `1`. See `docs/CODE_CLEANUP_STEP2.md`.

This is report candidate 4.2 only. It is off by default and has no measured
performance claim. Frozen v50/v54 sources are not edited.

## Controls and runtime evidence

- `GIPC_MAS_FUSED_DOT=0|1`: strict boolean; unset means 0. Independent of
  host/conditional Graph execution. `mas_fused_dot_options.h` exposes
  `gipc::mas_fused_dot_requested()`.
- Every PCG records `mas_fused_dot_requested`, `mas_fused_dot_effective`,
  `mas_fused_dot_fallback_reason`, and `mas_fused_dot_partial_count`.
- Eligibility is resolved once before PCG: exactly one nonempty MAS owner,
  correct local/global interval, MAS last in the local writer order, and no
  other local preconditioner overlapping its interval. Unsupported systems
  retain the old path with an explicit reason. No mid-PCG fallback occurs.
- Legacy builds without `GROUP` mapping explicitly report unsupported;
  wide/Cholesky keep their existing mapped collect semantics.
- Diagonal-update fusion is not combined with this candidate. Its existing
  `fused_diag_update` statistic reports the actual inactive state.

## Ownership and numerical contract

The original global preconditioner/identity and local preconditioners run in
the original order. MAS keeps its original restriction, matrix action, and
factor precision. Only its final collect changes: each thread computes the
same final FEM double3, writes it once, and forms a rho contribution in FP64.
The original float or double component accumulation order is preserved.

The FEM interval is exactly `[3*get_offset(),3*(get_offset()+totalNodes))`.
The complement kernel visits every global scalar outside this interval once,
after all local writes have completed in the same stream. Thus ABD affine
DOFs, identity/global-preconditioned coordinates, and other disjoint local
results contribute using their final z. FEM is not counted twice. Separate
FEM and complement partials are reduced together by CUB on the device.

There is no read of in-flight atomics and no cross-block software barrier.
The partial reduction grouping differs from the original full-vector dot;
rho and hence later PCG iteration decisions may differ by roundoff. No PCG
or Newton threshold, stop timing, material, contact, or CCD rule changes.
Host still checks previous rho after updating x/r and before the next
preconditioner. Graph retains its existing guard/stop kernel ordering.

## Lifetime and Graph integration

PCGSolver owns `mas_dot_partials` and a separate CUB temporary allocation.
Sizing occurs before any production graph capture. No allocation, host
readback, or environment re-resolution occurs inside the captured apply.
The production graph key includes effective mode, partial count, temporary
byte count, and both new buffer addresses in addition to the existing
preconditioner signatures. Buffer growth/eligibility changes invalidate the
old graph before replay. The same preconditioner is used throughout a PCG.

Initial rho and every subsequent rho use the same chosen path. Zero RHS and
nonfinite/negative rho still go through the existing PCG guards. Tail
threads participate in the block reduction with zero contribution. An
invalid MAS ownership range or unprepared scratch throws rather than
silently returning a partial rho.

## Protected validation entry

Set `GIPC_FIXED_MAS_DOT_STUDY=1` plus the existing `GIPC_FIXED_STUDY_DIR`,
`GIPC_FIXED_STUDY_FRAMES` and `GIPC_FIXED_STUDY_DIRECTIONS`. The public runner
maps `fixed_mas_dot_study` and must exclude other fixed-study modes.

The study exports A/b/M identity and writes `f<frame>_n<direction>_mas_dot_study.json`.
It applies the old preconditioner once per input to freeze its local MAS
action, then compares the new final collect on the exact same local values.
Inputs are actual RHS, zero RHS, and a signed pattern covering the actual
tail and ABD/FEM boundary. Host launch and two component-Graph replays must
produce byte-identical full z and finite rho within a recorded FP64
roundoff bound of an independent CPU long-double dot. This isolates collect
correctness from nondeterministic legacy restriction atomics.

It also runs the full fused initial-preconditioner/PCG path on a private zero
RHS in host and conditional-Graph modes: both must return zero iterations
with finite, zero x/z/r. The Graph zero branch returns before modifying the
production Graph. This is an executable test entry, not a claim that it has
already been run during implementation.

It saves/restores x, z/r/p/Ap, reduction scalars, graph scalars/CUB scratch,
new partials/CUB scratch, and all four MAS R/Z work buffers. It verifies
full system/MAS snapshots, read-only RHS, production Graph handle/key, and
saved buffers after restoration. Exceptions also restore state. The study
uses its own short-lived graph and does not replace the production graph.

This is a component correctness audit, not an entire fused PCG or trajectory
equivalence claim. Before promotion, run host/conditional Graph on the same
frozen full systems with fusion off/on, including mixed ABD/FEM and legacy/
wide/Cholesky paths, then test buffer growth/cache invalidation, zero RHS,
guard fixtures, and full continuous scene quality. Include scratch setup,
MAS preparation, all applications, and graph construction in performance.
Default remains off until those measurements are reviewed.

## Implementation-time verification status

Only source/interface inspection and CPU ownership arithmetic were allowed
for the delegated implementation. No CUDA build, GPU test, or performance
measurement was executed by this implementation task.
