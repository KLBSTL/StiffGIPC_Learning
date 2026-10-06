# IPC bounded-request CCD candidate

This is one default-off execution candidate. It keeps every supplied PT/EE candidate, the original CCD functions and off kernel, materials, solver stopping rules, and the caller's final `min(temp_alpha, step)` / CFL handling. It does not claim a measured speedup, a 5% whole-scene gain, or the 2× target.

## Source and scope

The copied device arithmetic comes from `sources/stiff_perf_v50/StiffGIPC/collision/ACCD.cu`, `edge_edge_ccd` and `point_triangle_ccd`, originally by Kemeng Huang / StiffGIPC. Frozen file SHA256: `b9c7ef6ce46e32313d9fabaaed4b397f626e527c8947d139ce925f600148acea`. The frozen file and its exported device functions are unchanged. The new copies call its existing `edge_edge_distance_unclassified` and `point_triangle_distance_unclassified` helpers. Arithmetic and original distance classification, zero-motion exit, 50,000-iteration limit, convergence break, and `toc>1` return retain their order.

Only the second IPC swept CCD request is intended to use this module. Its broad phase has already been constructed for `temp_alpha`, and its returned step is clipped to that same bound. The first discrete-contact CCD and boundary/TOI paths retain their existing dispatch. Integration belongs to the caller.

## Configuration and interfaces

- `GIPC_BOUNDED_CCD=0|1` and `GIPC_BOUNDED_CCD_VALIDATE=0|1`, both default 0. Any other present string is rejected; validation requires enabled.
- Include `ipc_bounded_ccd.h` for the configuration and host API. Include `ipc_bounded_ccd.inl` in exactly one CUDA translation unit.
- `ipc_bounded_ccd_launch(x,pairs,direction,queue,slackness,count,bound)` launches 256 threads per block, writes exactly `ceil(count/256)` caller-owned block maxima, and leaves final reduction/readback to the existing caller. Count zero performs no allocation or launch. No pair compaction, block tuning, asynchronous scheduling change, or persistent scratch is introduced.
- `ipc_bounded_ccd_compare(x,pairs,direction,slackness,count,bound)` is explicit heavy diagnostics; it returns JSON and owns all temporary outputs. It never writes input positions, directions, pairs, caller reduction buffers, or Graph state.
- `ipc_bounded_ccd_fixture(output_path)` writes JSON and returns 0/1. Per-frame observation/reset and application entry hooks are integrated separately by the caller.

## Cutoff and its limits

Bounded copies first require a finite bound strictly between 0 and 1, finite positions/directions, finite eta strictly between 0 and 1, and zero thickness. Otherwise they invoke the **original exported function**. Supported copies maintain a persistent prefix-valid flag: finite nonnegative distances/gap/increments, finite positive speed, finite updated positions, and nonnegative monotone `toc`. Any prefix violation disables cutoff for the rest of that call. This check does not rewrite the original arithmetic or return a new safety value on invalid data.

The only new return occurs after the original convergence-break test, the original `toc += toc_lower_bound`, and the original `toc>1` exit. It requires `toc>requested_bound` and a finite actual reciprocal roundtrip `1/(1/toc) >= requested_bound`. A volatile intermediate prevents the roundtrip gate from being algebraically simplified to `toc`. It returns 1, indicating no further restriction after the caller's unchanged clip. Production uses the same 256-lane CUB `Max` and tail `valid_items` convention as the old kernel, with neutral value 1.

**The prefix flag does not prove the unexecuted original floating-point suffix remains finite or monotone.** Exact real-arithmetic reasoning is not a guarantee for all original machine executions. Overflow, degeneracy, later invalid intermediates, or compiler differences could invalidate a general equivalence claim. This remains an experimentally guarded candidate: same-state actual clipped-return and final-alpha equivalence must pass before performance screening. It is not a replacement for independent accepted-path CCD or full scene quality checks. Arbitrary degenerate inputs are not certified.

## Independent comparison

For every unchanged input pair the diagnostic evaluates four paths:

1. Original exported `point_triangle_ccd` / `edge_edge_ccd`: the truth.
2. Unbounded arithmetic copy with an iteration counter: must match the original return bitwise before its count is usable.
3. Bounded production specialization without counters.
4. Bounded specialization with counters: must match the production specialization bitwise.

Private outputs retain actual returns, reciprocals, reciprocal roundtrips, clipped steps, type, iterations, cutoff/fallback and invalid-prefix flags. Comparison requires per-pair clipped roundtrips to match bitwise. Unsupported bounds compare the raw returns instead. Separate private old/new 256-thread block reductions and CUB final maxima reproduce the actual `1/max(1/t)` path. A real production-kernel call uses a third private partial buffer; its partials and final result must match the diagnostic production values bitwise. All outputs/partials/scalars are poisoned first. Any unwritten/NaN partial, invalid return, nonfinite input, count-copy discrepancy, or clipped discrepancy fails. Positive infinity is allowed only as a reciprocal/partial representing an original feasible step of zero; reported actual steps must be finite.

JSON includes `old_step`, `new_step`, `old_clipped_step`, `new_clipped_step`, `passed`, per-pair and complete-partial comparison flags, original/new counted-copy checks, per-type counts, cut/fallback/invalid-prefix counts, and `diagnostic_host_ms`. `new_iterations_known` is the explicitly counted bounded work. `new_iterations_unknown_pairs` records fallback calls, whose exported implementation is not instrumented. `fallback_iterations_from_verified_old_clone` accounts for those calls using the old counter copy, and `new_iterations` is their sum; fallback is never silently counted as zero saved work. All iteration totals remain instrumented-copy diagnostics, conditional on return equivalence, not direct timing. The caller must additionally compare its real production step and final CFL alpha.

## Fixture coverage and remaining tests

The fixture has 13 ordinary comparison groups: eleven finite geometry/motion cases under eight bounds, plus counts 0/1/255/256/257. The finite cases cover PT/EE zero motion, approach, separation, near contact, PT grazing, near-parallel EE, and a thin finite triangle. Bounds include .05/.25/.9, 0, 1, negative, NaN, and positive infinity. Normal fixture inputs are copied back and checked bitwise unchanged; at least one real cutoff is required. Two additional one-pair cases use NaN/infinite geometry and zero relative motion. The old function can return a benign scalar before inspecting distances; these cases explicitly require original fallback equality **and safety rejection**, and do not treat old finite output as valid geometry.

The fixture does not exhaust all degeneracies, nonfinite directions, unsupported eta/thickness, or finite-prefix/later-overflow behavior. There is no new Graph capture/replay coverage because this component is called outside the PCG graph. Complete continuous trajectories and independent accepted-path safety checks remain necessary. At implementation handoff only CPU source/static checks have run; GPU compilation, fixture execution, same-state production guards, quality, and performance are pending with the main runner.

## Measured local decision (2026-10-06)

Release program `b1a1ac4506adcf999b073100a6853c625d37dd78594d91df35d33d08146b6497` passed all 15 GPU fixture groups (859 pairs) and eight historical regression gates. The two continuous contact guards (hang51/fixed59) compared 1,998,885 pairs in 67 calls: clipped per-pair, reduction and actual final-alpha results were bitwise equal. Mixed3 did not exercise the path. Three interleaved short-window timing pairs gave median off/on speedups 0.99863x / 0.98730x; the predefined net 5% gate failed. Keep this experiment disabled and archived from promotion. Do not extrapolate instrumented per-pair iteration savings to whole-scene speed or quality. See `reports/active/CONTACT_OPTIMIZATION_RESULTS_20261006.md` for coverage, costs and the next selected direction.
