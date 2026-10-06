# Native cleanup step 1: retired collision candidates

Date: 2026-10-06. Scope: the flattened `StiffGIPC/` tree in this learning
repository. The original experiment workspace was not edited. The parent task
confirmed backup commit `eaa1b2131bca06de41531c66ed64e74dd4316da0` and tag
`archive/pre-cleanup-20261006` before this change.

**Source ready; compilation and GPU validation pending.** This change is cleanup
of two previously rejected execution candidates, not a new speedup claim.

## Removed implementation

- `collision/ipc_bounded_ccd.h`
- `collision/ipc_bounded_ccd.inl`
- `collision/ipc_bounded_ccd_observer.h`
- `collision/query_eligibility.h`
- `collision/query_eligibility.inl`

Historical descriptions `ipc_bounded_ccd.md` and `query_eligibility.md` remain,
with explicit retired-component banners. Their runtime functions and private GPU
fixture entry points no longer exist in the current native program. They remain
recoverable at the backup tag; failed experiment evidence is not reclassified.

The bounded branch in `core/GIPC.cu` now always invokes the existing complete
`self_largestFeasibleStepSize(slackness_m,h_ccd_cpNum)` at the same second-query
position. The condition `temp_alpha > 2*alpha_CFL`, all candidate pairs, first
CCD query, `min(temp_alpha,step*ccd_size)` followed by `max(alpha,alpha_CFL)`, and
line-search safety checks remain. Original ACCD functions and reduction kernel
are retained.

The four ordinary/swept VF/EE kernels and the two contact-pool capture kernels
now contain their former eligibility-disabled path directly. Removed items are
summary loads, subtree predicates, diagnostics, template parameters, scratch and
query wrappers. The original leaf predicates, original edge-ID deduplication,
signed tuple encoding, atomic append locations, block size 256 and stack remain.
Both contact-pool capture kernels and sidecar writes are retained.

## Contact pool and shared dependencies

The pool fixture previously called `eligibility_fixture_nodes`. Its small CPU
tree builder was moved into `ipc_contact_pool.inl` as `cp_fixture_nodes`, keeping
the same random seed, leaf-ID permutation, topology and checks. Explicit
`<numeric>` and `<random>` includes replace indirect includes from the removed
module. Pool fixture calls were updated to the smaller raw-query signatures.

Ordinary/swept independent BVH storage, refit, all pool guards and private
validation, ground handling, material/ABD/FEM routines and all PCG/Graph/MAS
components were outside this cleanup. In particular, MAS final-dot, SpMV+pAp,
legacy ordered restriction and the user-selected defaults were not modified.

## Explicit unsupported requests

`core/retired_components.h` adds a small host-only guard, called first in `main`
before fixture dispatch and scene setup. These controls accept absent or `0`,
reject `1` with a retired-component message, and reject other strings as invalid
strict booleans:

- `GIPC_BOUNDED_CCD` / `GIPC_BOUNDED_CCD_VALIDATE`
- `GIPC_BVH_ELIGIBILITY` / `GIPC_BVH_ELIGIBILITY_VALIDATE`
- `GIPC_ELIGIBILITY` / `GIPC_ELIGIBILITY_VALIDATE` spelling aliases

The three corresponding `*_FIXTURE` variables are explicitly rejected whenever
present, including an empty path. A retired fixture request must not fall through
to normal scene execution or be reported as a successful skipped test.

Resolved configuration retains the historical false-valued bounded/eligibility
keys and adds `*_available=false`. Frame output retains only truthful disabled
metadata; bounded `validation_failed=0` means no bounded validation was executed.
The old measured query/iteration counters are absent, not fabricated as zero.
Historical JSON parsing and new-run capability refusal remain the runner's
responsibility. No `tools/bench/config.py`, CMake or runner file was edited here.

## Verification already performed

Read-only Python checks ran through
`E:/Anaconda/envs/DL/python.exe -X utf8 -` against the original activity overlay:

1. Reconstructed the former false/false path of all six query kernels by removing
   only retired template arguments, diagnostic locals and always-true eligibility
   tests; its whitespace-normalized text equals the new kernel text for all six.
2. `_checkPTintersection` and `_checkEEintersection` function texts are unchanged.
3. The migrated CPU fixture tree builder is unchanged except its local name and
   error-message component name.
4. `collision/ACCD.cu`, `ACCD.cuh`, `discrete_bvh.h` and `discrete_bvh.inl` are
   byte-identical to their pre-cleanup providers.
5. Modified native files pass lexical delimiter checks, removed implementation
   paths no longer exist, and `rg` finds no remaining include/call to their
   deleted APIs in current native sources.

These are static preservation checks. They do not prove compilation, native
environment rejection, GPU output equivalence, trajectory quality or performance.

Frozen native identities at handoff:

| File under `StiffGIPC/` | SHA256 |
|---|---|
| `collision/mlbvh.cu` | `aaac9fc714c08a5c58bc28cae8c9cf71ab1d979c4b934ebc335a037696e4c570` |
| `collision/mlbvh.cuh` | `1b84d0923801bf2fd586b9fa48a649f4e077ef2e5ca52c1ed6628a1e3e1ecde4` |
| `collision/ipc_contact_pool.inl` | `0c2e6e14992e3aba7cbcafaf2071982d3a91d6a413bd987e0cfddbbe4f216bf3` |
| `core/GIPC.cu` | `5e4ffa38535ccff16095c9c5d59f005f5b0de9bf4616f8b277c333314e346539` |
| `solver/toi_options.h` | `46b69a89adcc1a70d8e18c0127cc33780bec186a39714b16c344b05ee96ed9c3` |
| `app/gl_main.cu` | `edb3149bde743caa78e47d496841cbc034c56c937ef7f8f1255a0bbf5cf466a5` |
| `core/retired_components.h` | `824f3cc0798bd3d97094991ddc8ac100ac2aa047b47578fb19fa5e8e84c583aa` |

## Required next checks

The parent task owns AutoDL clean build and serial GPU runs. Record the resulting
source/object/link/binary identity rather than reusing the old executable hash.
Check direct retired enable/validate/fixture requests fail before a normal run;
check absent/zero controls still allow supported fixtures. Run existing component
and PCG guards, the six protected historical MAS systems, and contact-pool fixture.
Then run bounded default host/Graph and a contact-window pool same-state guard
with existing material/stopping/resource settings. Compilation or GPU failures
remain failures until repaired and rerun under a new identity.

Only after this first cleanup is validated should the separately reviewed linear
candidate cleanup be considered. No compilation or GPU work was run by the native
cleanup agent, and no runtime success or speedup is asserted by this document.
