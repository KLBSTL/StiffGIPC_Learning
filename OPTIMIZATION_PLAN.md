# TOI / PCG / CUDA Graph audit, 2026-10-09

Starting revision: `7f5f069` (`StiffGIPC_Learning/main`). Isolated local branch:
`perf/toi-pcg-audit-20261009`. Original uncommitted work remains user-owned.

Objective: find and correct the reasons for increased PCG work and cost, then
measure at least 2x solver speed relative to frozen Stiff with comparable quality.
The requested speed is a target, not a guaranteed or achieved result.

1. Build clean current source, identify frozen baseline and inputs; reproduce
   host / Graph / AL differences with matched settings, serial GPU runs.
2. Audit every statement on the TOI primal/dual, contact, assembly, MAS and PCG
   paths. Separate proven defects, measured bottlenecks and untested hypotheses.
3. Correct one evidenced mechanism at a time, with small numerical fixtures and
   protected full-scene comparisons. Retain negative experiments and failures.
4. Compare complete runs, work counts, per-stage costs, deformation and safety.
   Keep materials, dt, complete CCD and PCG rho tolerance unchanged in execution
   experiments. Report any algorithm/stopping changes separately.
5. Validate a successful candidate across supplied contact scenes and write a
   reproducible report including exact commands, identities and remaining gaps.

Local GPU: RTX 3070 Laptop, 8 GiB, shared WDDM. Resource protections and GPU
serialization remain active. Existing run budgets belong to their historical
experiments; this is a new round, without overwriting or retrying old failed runs.

Progress: clean baseline compilation and reproduction complete; hot-path audit,
FEM sparse pullback and legacy MAS pivot synchronization implemented. v3 passes
19 CTests, 23 CPU contract checks, racecheck and synccheck. Forty-five component
short runs and twenty-seven full 120-frame runs completed under the user's
chosen quiet power mode. Median paired speedups: hang 2.02178x, fixed 1.78932x,
mixed 1.32943x. All-scene 2x and independent physical quality are NOT achieved.

Latest finite round: v4 full compilation and all 19 tests completed; 9924
captured MAS matrices (12146976 output entries) compare bitwise, and both CUDA
sanitizers pass. The contact-pool option measured 1.08301x paired improvement
over v4 best on complete mixed runs, with quality still unqualified. Completed
the fresh v4/base three-scene pairing: hang 2.08432x, fixed 1.85142x,
mixed 1.33158x; mixed pool 1.29109x with large timing variation. Complete TOI
with new execution kernels/legacy MAS was 64.0713s versus Stiff 31.4914s,
2064 versus 750 directions, 116736 versus 24404 PCG. A stricter existing
inner tolerance increased work to 135599 PCG and 72.2921s; rejected.
CPU warm-start/Ritz screens did not show stable residual/cost benefits; they
remain diagnostic prototypes rather than changes to the production PCG.
Do not widen quality bounds after viewing candidate results.
This round is documented and validated; the all-scene 2x objective remains
unfulfilled. Execution optimizations remain opt-in; no default TOI or stopping
change is adopted. Raw evidence and finite decisions are in reports/AUDIT_REPORT.md
and reports/AUDIT_FINAL_STATUS.json. No performance or physical-quality certificate.
