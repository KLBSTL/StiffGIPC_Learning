# SRBK SpMV + quadratic candidate

**Retired on 2026-10-06:** the active implementation and private study have been
removed after the paired performance screen did not meet its promotion gate.
The remainder is historical documentation. Reproduce it from
`archive/pre-cleanup-20261006`; the current executable rejects either old enable
or study switch when set to `1`. See `docs/CODE_CLEANUP_STEP2.md`.

Status: implemented as an opt-in candidate; this document does not assert a CUDA build, GPU correctness result, trajectory quality result, or speed improvement. The root runner owns the serial build and experiments.

## Public switches and actual execution

`GIPC_SPMV_FUSED_QUADRATIC=0|1` selects the candidate for both host PCG and the conditional PCG graph. `GIPC_FIXED_SPMV_QUADRATIC_STUDY=0|1` enables the protected component study, together with the existing `GIPC_FIXED_STUDY_DIR`, `GIPC_FIXED_STUDY_FRAMES`, and `GIPC_FIXED_STUDY_DIRECTIONS` selectors. Both switches default to zero and reject any spelling other than exact `0`/`1`. The public header is `solver/spmv_quadratic_options.h`; its functions are `spmv_fused_quadratic_requested()` and `spmv_quadratic_study_requested()`.

Every PCG reports `spmv_fused_quadratic_requested`, `spmv_fused_quadratic_effective`, `spmv_fused_quadratic_fallback_reason`, and `spmv_fused_quadratic_partial_count`. A study may prepare buffers without enabling the production candidate. Empty vectors, incompatible block dimensions, missing matrix storage, and oversized scalar counts are rejected before solving. A supported path remains fixed throughout one PCG; there is no silent mid-solve fallback.

## Operator and ownership

The converter sorts/merges `(row,col)` pairs but its `ge2sym` call is disabled. The current SRBK operator sends `Aij*xj` to row `i` and, whenever `i != j`, `Aij^T*xi` to row `j`. This rule applies to **every stored block**, including lower-triangular entries and both orientations if both are stored. The candidate follows that rule; it neither filters `i<j` nor canonicalizes pairs.

For each stored block it accumulates `xi^T*Aii*xi` for a diagonal or `2*xi^T*Aij*xj` otherwise. It computes this from the loaded block and input vectors before the row-segment reduction changes the local product. It never reads the output `Ap` while atomic writers are active.

The original kernel is retained unchanged. The candidate retains its per-thread transpose-atomic and segmented-row-atomic ordering. Across threads, atomic arrival order is already unspecified; extra instructions can alter final FP64 low bits. The private scalar partial for block `k` has exactly one writer, CUDA thread zero of that block. Tail threads contribute zero and still participate in the final CUB block reduction. A separate kernel zeros all `Ap` before any atomic addition. There is no inter-block spin barrier.

`PCGSolver` owns independent scalar partials and CUB reduction bytes. Their size is `max(1,ceil(stored_blocks/256))`, not the number of vector coordinates. Preparation and allocation occur before capture. The conditional graph key includes the effective switch, partial count, reduction byte count, and both new buffer addresses, in addition to the existing matrix and vector identity. The host path keeps its scalar readback; the Graph path writes directly to its existing curvature slot. Initial rho, subsequent rho, preconditioning, residual update, and stopping/guard ordering are unchanged.

Only a nonvirtual forwarding API was added to `GlobalLinearSystem` and `IterativeSolver`: `spmv_quadratic_unavailable_reason(count)`, `spmv_quadratic_partial_count()`, and `spmv_quadratic(x,y,partials)`. All affected object files must be rebuilt together because `PCGSolver` gained owned members and `Spmv` gained a method. No frozen source is edited.

## Protected study and predeclared acceptance

The selected fixed system exports the stored blocks and runs three inputs: the actual solved Newton direction, zero, and a signed pattern that populates the vector tail. Each compares the original SpMV plus original dot reduction with the candidate under host launch, first component-Graph launch, and component-Graph replay. The independent CPU reference expands each coefficient into the full operator, including the transpose contribution, without using the candidate's factor-of-two identity.

Before every candidate host/Graph invocation, private `Ap`, result scalar, and all new live partials are poisoned with NaNs. The empty-matrix fixture poisons its private `Ap` and direct single-partial scalar too. Thus an absent/stale Graph replay cannot pass by reusing a preceding correct output. This is diagnostic-only and is absent from ordinary timing runs.

Before any GPU experiment, the acceptance rule is fixed in `spmv_quadratic_reference.h`. Let `c_i` be the number of scalar coefficient contributions to output coordinate `i`, `S_i=sum(abs(Aij*xj))`, `epsilon` the FP64 epsilon, and `gamma(k)=k*epsilon/(1-k*epsilon)`. The coordinate comparison includes `2*gamma(c_i+6)*S_i` plus a denormal allowance; the factor two budgets both GPU accumulation and the CPU reference. The scalar comparison uses the absolute sum of fully expanded quadratic terms and `gamma(expanded_terms + dofs + 32)`, with the old-dot comparison additionally budgeting the propagated coordinate errors and its full-vector reduction. The old/new difference must also fit the sum of their independent bounds. Nonfinite values and unreasonable/overflowed bounds fail. The budget is conservative over possible reduction orders, not a tolerance tuned after a failed experiment.

`Ap_bitwise_equal` is recorded but is not the numerical pass criterion. This local floating-point accumulation budget is **not** a relaxation of the simulation's material, stopping, or trajectory-quality thresholds. The CPU report records the actual `long double` mantissa width; on MSVC it may be only FP64, and the bound accounts for that.

The study separately invokes a truly empty private matrix with zero stored blocks under host, first component Graph, and replay, requiring finite zero `Ap` and quadratic. It does not replace the live matrix. Synthetic empty, 1/31/32/33/255/256/257 block ownership is checked on CPU; only the actual frozen matrix's tail and the empty private matrix execute on GPU. Arbitrary synthetic nonempty GPU tail fixtures and production Graph growth/invalidation are not certified by this study. A zero input exercises the new SpMV/quadratic operator, not another complete PCG solve; initial rho is unchanged by this component.

All production PCG vectors, both old/new component scratch buffers, MAS work buffers, RHS, matrix/preconditioner snapshot identity, production graph handles/key, capture counters, and captured parameters are checked unchanged or restored byte-for-byte. Private input/output/dot workspace and component graphs avoid reallocating buffers referenced by the production graph. Failures propagate after restoration. A successful component study does not certify nonzero PCG convergence, scene quality, or speed.

## Validation order and remaining measurements

Pre-GPU review on 2026-10-05 corrected a tail-collective hazard in the initial candidate: invalid tail threads must participate in all three full-mask warp reductions as well as the block reduction. Their zero vectors use head flags to terminate the last valid segment; only valid heads write `Ap`. The first build before that correction is not the final test binary. Rebuild the dependent SpMV object and record the resulting binary identity before any GPU evidence is attributed to this component.

1. CPU mapping/arithmetic and source-contract checks: `E:/Anaconda/envs/DL/python.exe sources/stiff_active/StiffGIPC/linear_system/utils/spmv_quadratic_cpu_check.py` from the task root. This is a CPU-only design/static check, not CUDA execution.
2. Root clean/reconciled build with overlay include closure and object identity.
3. Protected fixed-system study on both cloth scenes, with separate ordinary host and conditional-Graph PCG smoke tests. Retain the existing guards, finite checks, zero RHS fixtures, and component combinations.
4. Nonzero PCG residual/iteration and trajectory-quality comparisons; explicit growth/invalidation coverage remains necessary before claiming broad Graph regression coverage.
5. Only after those pass, paired scene timing with diagnostics off. Report complete frame windows and preparation/reduction cost. Never use incomplete memory-limited prefixes or isolated kernel timings to certify a full-scene speedup.

Potential rejection reasons include register pressure reducing SpMV throughput, scalar partial/reduction overhead exceeding the removed vector dot, and changed curvature rounding affecting PCG trajectories. The component stays disabled by default unless measured benefits and quality gates justify promotion.
