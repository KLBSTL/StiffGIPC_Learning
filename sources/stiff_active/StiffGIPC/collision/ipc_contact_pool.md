# Swept barrier contact pool candidate

Status: implementation prepared for compilation and GPU validation. This document
does not assert a speedup, a successful fixture, or trajectory equivalence.

## Scope and mathematical contract

`GIPC_CONTACT_POOL=0|1` defaults to `0`. `GIPC_CONTACT_POOL_VALIDATE=0|1`
defaults to `0` and requires enabled. The only substitution is the pair of
ordinary VF/EE contact queries inside IPC `buildCP`. Ordinary BVH construction,
FullCCD construction and queries, safe-step CCD, ground checks and
`isIntersected` remain in the caller. Materials and solver stopping rules are
unchanged. TOI paths must not create a pool generation.

The existing FullCCD VF and EE queries already apply `sqrt(dHat)` broad-phase
separation. For a current primitive box contained in its saved swept box, any
ordinary overlapping pair must also have passed that swept overlap test. Body,
fixed, shared-vertex, and EE original-edge-ID predicates are the same. We cache
the resulting original primitive identities, including duplicate query vertices
and primitive IDs. We do not infer edge IDs from four vertex IDs.

Real arithmetic interpolation alone is insufficient for ABD. Every reuse checks
actual current leaf boxes against saved swept boxes, plus every vertex against
its saved point endpoint box. This is deliberately conservative: even an unused
vertex outside its endpoint box causes fallback. These checks have no epsilon.
Finite trial alpha must be within `[0,max_alpha]`, and the radius must match.
Every trial also compares direction, body, boundary, rest positions, face/edge
connectivity, and surface mapping against generation snapshots, including
same-address writes. Metadata or geometry outside this contract uses the original
queries before any production output has been written.

Both pool classification branches repeat the original instantaneous leaf AABB
overlap and call the exact existing `_checkPTintersection` or
`_checkEEintersection` helper. The second original EE ID is passed as `obj_idx`,
preserving the mollifier `-obj_idx-2` encoding. No canonicalization, deduplication,
distance formula, type rule, count rule or output ordering contract is introduced.
GPU atomic append order remains unspecified, as on the old query path.

## Source and disabled path

`mlbvh.cu` contains two capture-only copies of the current `_selfQuery_*_ccd`
templates. Their only changes are extra arguments and sidecar assignments in
the existing guarded output writes. The old templates and block size 256 remain
unchanged. A CPU structural comparison reconstructs the original templates by
removing those changes and requires exact text equality. The ordinary templates
are not edited. Disabled calls retain the old kernels and allocate no pool
device storage; the caller/module performs only a short host feature check.

The sidecar is indexed by the same global append slot shared by VF and EE.
`{first,second,kind,epoch}` is 16 bytes; kind 0 is vertex/face and kind 1 is
query-edge/leaf-edge. Every capture capacity attempt poisons and rewrites it.
Sealing an overflow or incomplete/wrong-epoch sidecar is an error, not a silent
fallback. Zero capacity uses one unwritten marker allocation so count-only
capture selects the same capture-only code path without `cudaMalloc(0)`.

## Ownership, lifecycle and caller API

The module owns one thread-local state, device snapshots, identity sidecar,
gathered boxes and private validation outputs. It borrows tree objects, positions,
direction, attributes and primitive arrays only during the active generation.
It never stores borrowed BVH node/AABB pointers across storage swaps. Original
primitive IDs index all cached boxes, so a valid tree leaf reordering is allowed.
All kernels/copies follow the application's per-thread default stream; the pool
is not captured into PCG Graphs and adds no pointer to a Graph signature.

Caller sequence:

1. `ipc_contact_pool_begin(f,e,direction,vertex_count,max_alpha,dHat,generation)`
   before the swept build. This invalidates prior generations and references.
2. `capture_pass(capacity)` before **each** FullCCD count/write attempt; pass the
   returned sidecar to both appended `SelfCollitionFullDetect(...,pool)` arguments.
3. `seal(final_count)` after a complete FullCCD attempt, while swept trees and
   origin geometry are still selected. It freezes metadata and swept boxes.
4. After every trial `step_forward`, call `set_trial(alpha)`. After ordinary
   trees are built, `try_discrete(...)` either enqueues pool classification or
   returns false, with collision outputs untouched, for the caller's old queries.
5. `end()` on exit and before a new frame/direction. `reset_stats()` invalidates
   the reference but deliberately does not free capacity. Call `end()` separately.

Buffers retain capacity until thread/module destruction. `pool_bytes_peak`
counts production pool allocations; diagnostic buffers are separately bounded
by `diagnostic_peak_capacity` and are not included in that number. Every trial
performs a small host readback of guard bits. This synchronization, metadata
copies, and any fallback are part of candidate cost, not free preprocessing.

## Same-state validation and energy reference

With validation enabled, every successful guard runs independent old raw
ordinary VF+EE and new pool queries. Each uses private counts, DCD, CCD and
MatIndex storage, poisons outputs, and retries its own overflow from scratch.
No tree is rebuilt or modified for comparison. It verifies all five counts,
the complete sorted signed `{type,DCD[4],CCD[4]}` multiset **with duplicates**,
and a separate `[0,count[type])` MatIndex bijection for each type. It does not
normalize signs, edge direction, mollifier IDs or multiplicity.

`ipc_contact_pool_reference()` exposes a read-only view of the last private
legacy output (`valid,count,counts[5],dcd,ccd,matrix_indices`) for the caller's
separate energy audit. The view is invalidated on every trial, attempt, reset,
end, or production overflow. The caller is responsible for preserving/restoring
its own production arrays if temporarily installing the view. Ground data is
outside this module. No private diagnostic action writes production arrays.

`production_overflow_queries` is observed only with validation enabled, from the
complete reference count versus caller capacity. Such an attempt still validates
the full multiset but cannot support an energy comparison; therefore expected
energy calls are `validation_passed - production_overflow_queries`. Standard
`buildCP` starts count zero. The low-level kernel also retains legacy append
semantics; the fixture explicitly tests nonzero caller count/type offsets.

Frame statistics include requested/validate; attempts, reused_queries,
old_queries; prepare_calls, capture_passes, generations; pool_pairs/vf_pairs/
ee_pairs; fallback_reasons; validation_calls/passed/failed, pairs_compared,
nonempty_validation_calls, diagnostic retries and host time. Counts include
production retry attempts. Multiple guard failures in one attempt may contribute
to multiple reason counters. Costs use `collision.contact_pool.prepare`,
`.guard`, `.classify` and separate `diagnostic.contact_pool`. Capture-only kernel
sidecar writes occur inside the existing swept-query cost scope.

## Fixture and remaining validation

`gipc::ipc_contact_pool_fixture(output)` runs only when explicitly invoked by
the caller. It builds finite synthetic BVHs and exercises the actual raw old and
capture FullCCD kernels plus ordinary/pool classifiers. Cases include zero and
one leaf, a 257-query-point tail, duplicate surface points and duplicate edges,
near-parallel EE/rest mollification, several typed geometries, interval endpoints,
invalid alpha, attribute/direction/connectivity/surface changes at the same
address, actual position outside the swept box, and end/new generation. Source
body `-1`, fixed type 3, and a partially fixed type 1 are included.
The concrete host-double ABD cancellation counterexample computes source and
predicted endpoint 2 versus actual q-step point 4, then supplies that actual point
to the GPU containment guard. This tests the guard response to the arithmetic
counterexample, not the full production ABD integration routine.

Normal fixture output has an 11-slot poisoned prefix and poisoned unused suffix:
complete live DCD/CCD tuples and adjusted MatIndex ranks must equal the independent
old reference, and all other bytes must stay poisoned. A distinct zero-capacity
pass asserts legacy behavior: total count grows, type counts do not, no output is
written, reference is invalid; the next complete attempt must pass. Capture-only
FullCCD outputs are compared as exact full-tuple multisets to old raw FullCCD.
Reported typed counts are actual coverage, not an assertion that every possible
distance subcase occurs.

Not established by this finite fixture: all degenerate geometry cases, arbitrary
memory corruption of a BVH topology, all CCD-distance subtypes, nontrivial ABD
trajectories, ordinary/swept production cache switching, full-scene energy or
trajectory equivalence, or performance. Existing tree validity is a prerequisite;
the module does not validate every BVH parent/child relation. Three scene guards
and paired diagnostics-off timing remain required. A low reuse rate or excessive
guard/preparation cost rejects this candidate; no speed target is claimed here.
