# Ordinary discrete BVH refit candidate

This is an opt-in implementation candidate, not a measured speedup. The active
overlay inherits the frozen v50 LBVH construction and query kernels. No frozen
source, FullCCD kernel, collision threshold, accepted contact set, or solver
tolerance is changed.

## Configuration and host integration

| Environment variable | Default | Accepted values |
| --- | --- | --- |
| `GIPC_DISCRETE_BVH_REFIT` | `0` | exactly `0` or `1` |
| `GIPC_DISCRETE_BVH_REBUILD_INTERVAL` | `8` | decimal integer in `[1,1024]` |
| `GIPC_DISCRETE_BVH_VALIDATE` | `0` | exactly `0` or `1`; requires refit enabled |

`gipc::discrete_bvh_config()` reads and validates the environment once per
process. Interval 8 means one full build followed by at most seven ordinary
refits. The next ordinary build is full again. Interval 1 is the all-rebuild
control. The implementation proposes only the fixed interval 8; the integer
interface exists for the all-rebuild control and future explicit experiments,
not for an automatic parameter search.

Existing `lbvh_f::Construct()` / `lbvh_e::Construct()` calls select the policy.
No new core query call is required. `ConstructRebuild()` always executes the
legacy full-build sequence and records valid ordinary state. `RefitDiscrete()`
selects the ordinary bundle and requires valid ordinary state and an identical
storage signature; it never refits a swept topology. It is also an explicit
fixture entry point.

Core may call `gipc::discrete_bvh_reset_stats()` at frame entry and save
`gipc::discrete_bvh_stats_json()` at frame exit. Statistics are local to the
calling GIPC host thread. Reset affects counters only, never tree age or state.
Disabled mode runs the unchanged full-build and query GPU sequences; it adds
no GPU allocations, launches, queries, copies, or synchronization.

## Ownership and invalidation

`lbvh` owns nodes, bounds, permutations, sorting scratch, and internal-update
flags through `DeviceBuffer` members. Geometry, face/edge indices,
surface-vertex indices, rest positions, and body metadata are borrowed from
GIPC; this module never owns or frees them. Policy states are host metadata,
initialized invalid with the `lbvh` object.

With `REFIT=1`, `LBVHStorage alternate_storage` additionally owns the inactive
tree's complete eight-buffer bundle and its host `scene`. `select_storage()`
swaps all eight owning buffers and `scene`; it does not copy device data,
allocate, launch kernels or synchronize. `DeviceBuffer` is move-only and each
`std::swap` move-assignment target has already been moved empty, so the swap
does not free either live allocation. The public buffers retain their previous
API meaning: they expose the most recently selected build/refit/query operation.
Callers must not retain their raw pointers across any such operation or resize.
The current external edge-triangle reader runs after `GIPC::buildBVH()` and
therefore sees ordinary bounds; display readers see the latest selected tree.
Both caches belong to the GIPC host thread; concurrent access is unsupported,
as it was for the original single tree.

Every enabled ordinary build compares counts, borrowed pointers, and data
addresses/logical sizes/capacities of all eight owned buffers. First use,
signature/storage changes, explicit invalidation, and the fixed interval force
a full ordinary build. Signatures are recorded and compared with their own
bundle selected; swapping a bundle out and back preserves every pointer, size
and capacity, including Morton-sort ping-pong buffers. A count/storage-size
change resizes the owned buffers before rebuilding. `init`,
`MALLOC_DEVICE_MEM`, `FREE_DEVICE_MEM`, explicit topology invalidation, and face
`getSceneSize()` invalidate **both** states. `MALLOC_DEVICE_MEM` and
`FREE_DEVICE_MEM` first restore the ordinary bundle, then release the alternate
swept bundle. Destruction releases both groups exactly once by ownership.

The face/edge/surface-index arrays are immutable between `init` and explicit
`invalidate_discrete_topology()` calls. External in-place remeshing or original-ID
renumbering must call that method even when addresses and counts are unchanged.
The existing simulation updates positions, not this topology. The production
candidate intentionally adds no GPU topology hash/readback on every build.
Validation checks the live nodes/permutation mapping directly.

The first implementation shared storage with FullCCD and consequently observed
zero production refits in the measured IPC prefixes. The bounded revision keeps
independent ordinary and swept caches under the **same** existing flag. Both
FullCCD entry points select the swept bundle. A FullCCD full build uses the
unchanged complete build sequence; a FullCCD refit still refreshes every swept
leaf and internal bound. The first swept refit, or a refit after its own
count/storage/mapping signature becomes invalid, instead performs a complete
FullCCD build. It never refits an uninitialized tree. FullCCD no longer changes
ordinary topology, signature or interval age. Ordinary and FullCCD query entry
points select their corresponding bundle and reject invalid topology/storage;
they do not reuse contacts or infer that geometry bounds remain current without
the normal preceding build/refit call.

With `REFIT=0`, the original shared-storage behavior is retained, including
marking ordinary state swept after FullCCD. No alternate buffer is allocated.
The new host-only members and empty-buffer release calls add no CUDA work to
this path. No change is made to the independent existing FullCCD refit flag.

The swept bundle is allocated lazily for the actual primitive count. For one
nonempty tree of `n` primitives, logical storage is `204*n - 68` bytes. Ordinary
`resize()` uses exact capacity; only the sorting `resize_discard()` arrays grow
by 1.5x. After the first full build, actual extra capacity is therefore
`204*n - 68 + 12*floor(n/2)` bytes, approximately `210*n`. Sum both trees for
the scene (about 20.0 MiB per 100,000 primitives). This includes both sorting
ping-pong arrays, excludes shared CUB workspace, and never reserves pair limits.
The earlier `306*n` estimate incorrectly applied 1.5x to every buffer. Record
actual `swept_cache_capacity_bytes_peak` instead of assuming this estimate;
retained capacity after a prior larger mesh can exceed the current-count value.

## Refit work and safety

Refit recomputes every leaf AABB from current geometry, reorders those fresh
bounds by the existing original-element-ID permutation, clears internal flags,
and recomputes every internal AABB with the unchanged bottom-up kernel. It also
refreshes the public host `scene` box from the root. Nodes and element IDs are
unchanged. It does not reuse candidate/contact lists or weaken distance tests.

The discrete EE kernel still reads `self_eid` and `obj_idx` from each leaf's
original element ID, including the existing `obj_idx < self_eid` filter. Neither
sorted leaf position nor Morton order is substituted for an original edge ID.
The existing 65-entry DFS query stack is unchanged; refit cannot increase tree
depth because it never edits topology. Debug validation checks root/link
consistency, complete reachability, unique original IDs, permutation agreement,
and depth at most 64 before running a query on the candidate tree.

## Explicit diagnostic validation

With both `REFIT=1` and `VALIDATE=1`, each ordinary face/edge query does the
following before writing the production query output:

1. Check the current tree topology and query-stack bound on the CPU.
2. **Actively execute an ordinary refit**, even if the production policy just
   chose a full build. Check finite leaf bounds and exact internal child unions.
3. Run the unchanged query into independent scratch pair/CCD-pair/MatIndex/count
   buffers. Start capacity at zero, count all candidates, grow to the actual
   observed count, and rerun until a complete pass succeeds. No configured
   maximum pair count is preallocated.
4. Execute an unconditional full build at the same geometry and repeat the tree,
   bounds and scratch-query checks. Compare leaf bounds by original element ID,
   all five counters, and sorted **multisets** of `(type, exact DCD int4, exact
   CCD int4)`. Duplicate contacts are retained. For each result, MatIndex must
   be a bijection onto the appropriate type-local matrix index range. Atomic
   output order and the corresponding matrix rank are allowed to permute.
5. Throw on any mismatch; never silently accept incomplete output or switch a
   failed check into a fallback. On success the live tree remains a full ordinary
   tree with age 1; the normal production query then regenerates its own output.

Scratch queries never write GIPC's output/count buffers, so the face/edge
combined append convention is preserved. Geometry, rest geometry and body
metadata are read-only. Scratch allocations are scoped to one validation call;
only CPU pair/bounds snapshots span the two query passes. Allocation/runtime
errors propagate through the existing CUDA error handling.

The validation rebuild changes the subsequent diagnostic trajectory's tree
ordering/age and costs extra GPU/CPU work. It is **not** a performance mode or a
proof of bitwise trajectory neutrality. `diagnostic_refits`,
`diagnostic_rebuilds`, query/growth counts, compared pair counts, maximum tree
depth and `diagnostic_host_ms` are separate from production rebuild/refit counts.
Trace records use sample kind `diagnostic_discrete_bvh_validation` and stage
`diagnostic.discrete_bvh_validation`; their enclosing production scopes are
inclusive and must not be added or counted as production-only cost.

The ordinary production counters retain their original meaning. In particular,
`construct_calls = production_rebuilds + production_refits` in enabled mode;
diagnostic forced refits/rebuilds are excluded. Added counters are
`storage_switches`, `ordinary_cache_restores`, `swept_full_builds`,
`swept_refits`, `swept_refit_fallbacks`, and
`swept_cache_capacity_bytes_peak`. The first five count the named host operations
since the last reset, including an explicitly requested external FullCCD audit
if it calls those operations; they are not ordinary production refit hits or
allocator-call counts. The byte counter measures maximum allocated swept-bundle
capacity observed during the frame. `ordinary_cache_restores` counts a restored
valid-mode cache before checking whether its signature or interval still permits
refit; use `production_refits` for actual policy hits. Enabled mode should report
`swept_rebuilds=0` for ordinary construction; a pointer/count change must instead
produce a signature/invalid rebuild, and the interval must still force rebuilds.

## Build and verification handoff

Owned overlay files: `collision/mlbvh.cu`, `collision/mlbvh.cuh`,
`collision/discrete_bvh.h`, and `collision/discrete_bvh.inl`. The active `.cu`
retains its quoted `"mlbvh.cuh"`, which now resolves to its active sibling.
The inherited `core/GIPC.cuh` uses `<collision/mlbvh.cuh>` and requires the active
include root ahead of v50, as configured by the active overlay build. The new
`.inl` is included once after the unchanged helper functions in active
`mlbvh.cu`; it introduces no extra translation unit.

Before any performance claim, compile the active target and run the explicitly
scheduled short validation scenes. Require nonzero diagnostic refits and
validation comparisons, zero failures, valid pair-type/MatIndex checks, and
capacity-growth evidence whenever nonempty contacts exist. Zero contacts or zero
natural production refits must be reported, not interpreted as broad coverage.
For the separate-cache revision, also exercise ordinary build/query -> FullCCD
build/query -> ordinary refit/query and the existing FullCCD-refit branch.
Require nonzero `production_refits`, `ordinary_cache_restores`, and allocated
swept-cache bytes, with zero ordinary `swept_rebuilds`; check exact typed pair
multisets at nonempty contacts. A lifecycle fixture should additionally change
the topology/mapping address and a buffer size, call explicit invalidation and
MALLOC/FREE, and verify that neither cache is reused until correctly rebuilt.
Same-state diagnostic rebuilds reset ordinary age to 1, so interval expiry must
be checked with validation disabled or a dedicated explicit build sequence.
Then run separate validation-off comparisons against disabled and interval-1
controls using the same scene, state, physical/stopping settings and quality
acceptance. Include rebuild/refit counts and preparation/copy costs in any net
benefit assessment. This implementation handoff itself does not include a build
or GPU test.
