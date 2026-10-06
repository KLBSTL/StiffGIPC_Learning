# World-coordinate penalty causal test

Status: implementation candidate, default off; no performance or quality claim.

Hypothesis: using mixed ABD affine/FEM positional Hessian diagonals to scale a
world-distance AL constraint creates unnecessary stiffness. This experiment
changes only that estimator; fixed-slack AL, material models, history, CCD,
restart guards and all convergence/budget settings remain unchanged.

Estimator: 0.1 times the maximum of movable FEM scalar Hessian diagonal and
`diag((J H_ABD^-1 J^T)^-1)` across movable ABD vertices. `H_ABD` is the full
contact-free 12x12 body block, including inertia, projected shape and motor
terms. The full compliance is invariant to invertible ABD coordinate changes.
This remains a global diagonal heuristic, not an exact contact Schur complement.
The FEM part retains the existing coordinate-axis diagonal convention. Fixed
vertices are excluded; invalid pivots/compliance abort without a fallback.

Energy, gradient, Hessian, slack and multiplier updates all use the same mu.
Each altered assembled matrix uses the normal MAS rebuild before PCG. Mu and
the preconditioner remain fixed within each PCG. Historical/generalized mode
retains its old path. The candidate is exposed as `mu_coordinates=world_block`.

Finite gates in order:

1. Configuration checks and Release provenance build; GPU derivative, mapping,
   coordinate-unit invariance and invalid-pivot tests; existing PCG guards.
2. Saved f25 n7/n8/n9 matrices: normal-contact-curvature-only CPU diagnostics
   with the same RHS and saved MAS. At most 1000 iterations or 90 seconds per
   case, two arms per system. This isolates A with a fixed M and is explicitly
   not a consistent physical solve, rebuilt-MAS speed test or quality gate.
3. Continuous local mixed scene from frame zero, 35 frames, dt=.01, 120 seconds
   per run, unchanged tolerances; first one generalized and one world arm with
   identical execution options. Stiff references are repeated only if the
   candidate completes and merits a quality comparison. No checkpoint restart.
4. Candidate escalation requires at least 10% lower window work/time without
   quality regression. More than 10% greater PCG work or a failed finite budget
   rejects this single candidate; no scaling grid or increased iteration limit.
   No AutoDL promotion until local evidence supports it.

Counter-evidence remains in scope: positive-slack AL directional curvature was
only 0.000665%, 0.321764%, and 5.224823% of the saved difficult directions.
FEM state/RHS and coupling can still explain the remaining PCG difficulty.

Final certification requires the original Stiff repeat envelope, independent
accepted-path CCD, separate cloth/FEM/ABD position/velocity checks, four arms,
100-frame paired performance and 300-frame stability. These short tests cannot
certify the 2x target.

After the first completed pair: PCG fell 19413 to 6032 and FEM inversions
disappeared, but minimum J=0.23554 remains below Stiff=0.49787. Do not promote
the candidate. Authorize exactly two more interleaved diagnostic pairs and
three current Stiff runs to verify causal repeatability/current timing; this
does not waive quality gates or start the 100/300-frame/AutoDL stages. Each
remains 35 frames/120 seconds, with no parameter sweep.

The three pairs support a large PCG reduction but all fail Stiff quality. One
final cost-only 35-frame/120-second run is diagnostic follow-up, with CUDA event
scopes on frames 24-26 and 33-35; it is excluded from paired timings. No further
trajectory candidate, parameter sweep or promotion in this protocol.
