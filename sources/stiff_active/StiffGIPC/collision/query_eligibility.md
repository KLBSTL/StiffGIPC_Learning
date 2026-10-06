# Conservative BVH query eligibility

This component is an execution candidate. It does not change geometry, bounding boxes, contact distance, leaf collision predicates, CCD, pair orientation, materials, or solver stopping rules. It has no measured performance claim at implementation time.

## Switches and interfaces

- `GIPC_BVH_ELIGIBILITY=0|1`, default `0`.
- `GIPC_BVH_ELIGIBILITY_VALIDATE=0|1`, default `0`; requires the first switch to be `1`.
- Any present value other than exactly `0` or `1`, including an empty string, is rejected.
- `gipc::query_eligibility_config()`, `query_eligibility_reset_stats()`, `query_eligibility_stats_json()` supply configuration and thread-local diagnostic accounting.
- `int gipc::query_eligibility_fixture(const char* output)` writes an independent GPU fixture report; the application owns invocation and process exit.

The four ordinary/swept VF/EE kernels have compile-time `UseEligibility` and `Diagnostics` specializations. Default dispatch uses `<false,false>` with null pointers: no summary allocation, summary loads, or new device predicate branches. The ordinary old leaf checks, the VF root-leaf shortcut, and EE original-ID deduplication remain in place. Production enabled dispatch uses `<true,false>`, without prune-count atomics. Diagnostic `<true,true>` accumulates four local counters per query thread and flushes each nonzero total once on exit, including early returns. Counter 0 records invalid metadata parent arrivals, independently of traversal counters.

## Conservative proof and metadata

A 16-byte summary contains `body`, a separate `homogeneous` flag, `all_fixed`, and `max_original_id`. The leaf body key is **only** `BodyID[element.x]`, matching the existing leaf predicate even if other primitive vertices belong to different body IDs. No integer body value is reserved as a mixed sentinel.

A subtree is rejected only if one of these necessary leaf eligibility conditions is false for every descendant:

1. Query body is not `-1`, every descendant leaf key is equal, and that key equals the query key. Query body `-1` always bypasses this body pruning rule; other negative values obey the old equality predicate.
2. Every query-primitive vertex and every descendant primitive vertex have `btype >= 2`. Values below 2, including negative values, are not fixed.
3. For EE only, every descendant original edge ID is at most the query original edge ID. The existing leaf test rejects both self and smaller original IDs. IDs are never canonicalized or reordered.

Internal homogeneity requires **both** children homogeneous and their keys equal. Fixed flags are ANDed and original IDs use max. This only rejects work that the unmodified leaf predicates would reject.

## Ownership and publication

Each `lbvh` owns one query-local `QueryEligibilityScratch`: summaries, private parent-arrival counts, and optional diagnostic counters. It is deliberately outside ordinary/swept storage swapping. Every nonempty query regenerates all live summaries **after** selecting its current storage, so same-address changes to body IDs, boundary flags, or node-to-original-element mappings cannot reuse stale summaries. VF metadata uses `face_number`, while the query launch uses `vert_number`; EE metadata uses `edge_number`.

Phase 1 initializes all leaves and clears every internal arrival count. Phase 2 starts one worker per leaf and propagates upward. `cuda::atomic_ref<uint32_t,cuda::thread_scope_device>::fetch_add` with `memory_order_acq_rel` publishes completed child writes and lets the second arrival acquire its sibling's summary before merging. The two phases and the query use the existing per-thread default stream. Original BVH `_flags` are never modified. Count zero allocates/launches nothing; one leaf does not launch the merge kernel. EE with at most one leaf has no pair query.

Scratch is private to the tree object and the existing single-thread, same-stream query sequence. Concurrent queries on one tree or using it from different CUDA streams are unsupported, as are concurrent mutations of geometry, tree nodes, or body/boundary arrays. Scratch growth occurs before the query; this component is not added to any PCG graph or its signatures. `FREE_DEVICE_MEM()` releases all three buffers.

## Validation and cost accounting

Validation runs the old and enabled query on the **same current tree and inputs**, with independent pairs, CCD pairs, five counters, and MatIndex buffers. Each path starts at capacity zero and retries independently using the required count. Neither uses the caller's append count or production capacity. Comparison preserves duplicate records and compares full DCD+CCD encodings; each contact type's MatIndex must form a complete local permutation. Swept queries compare their complete CCD tuples. A difference or invalid parent arrival throws; it never silently falls back. Existing ordinary-refit validation, when requested, finishes before this same-tree comparison.

`collision.eligibility_prepare` and `collision.eligibility_query` separate production preparation and query cost. `diagnostic.query_eligibility` and the `diagnostic_query_eligibility` sample tag isolate validation. Stats distinguish ordinary/swept VF/EE, enabled/empty/prepared queries, capacities, independent retry counts, validation passes/failures, pairs compared, and candidate traversal pruning by body/fixed/original ID. Retry traversal passes are included in diagnostic counts. These counts describe tested candidate nodes and pruned candidate roots; they are **not** old-node visitation reduction or a performance estimate.

The standalone fixture covers faces and edges with 0, 1, 2, 3, 31, 33, and 257 leaves; poisoned retained scratch; per-node CPU descendant enumeration; both parent arrivals; heterogeneous vertex body IDs with the x-key rule; body `-1` and other negative IDs; btype negative/1/2/3; same-address body/btype changes; and same-address leaf original-ID remapping. Odd leaf counts create internal/leaf sibling publication cases. A separate real query fixture uses one face and five query points for ordinary and swept paths, with nonempty pairs, all-fixed rejection, and in-place mutations; one-edge queries exercise the no-query guard. It intentionally does not manufacture a large randomized geometry suite. Three production-scene same-state guards must cover nontrivial trees, EE traversal, and normal/swept storage switching before performance tests.

The 2026-10-06 build compiled 37 units after correcting a fixture-only NVCC/MSVC captured-constexpr incompatibility. The independent GPU fixture passed 48 metadata and 10 query cases, followed by eight existing GPU regression cases. The hang 43-frame guard passed all 1,022 comparisons. The fixed-bunny 59-frame guard stopped for the shared GPU memory budget after 22 saved frames and 236 passing comparisons; mixed-body coverage remains pending.

The six-run hang-only timing subset has a paired median off/on ratio of 0.962011: this all-query candidate failed its local speed screen and remains disabled. No 100-frame quality, independent accepted-path CCD, or controlled performance certification follows from these tests. See `reports/active/IPC_ELIGIBILITY_RESULTS_20261006.md` and its original, subset, and review artifacts for accounting and limitations. Preserve this implementation as a diagnostic reference; do not promote based on subtree counters or the ordinary-EE kernel observation alone.
