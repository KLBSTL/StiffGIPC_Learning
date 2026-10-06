# IPC compensated TOI entry and state controller

This module exposes the report's already implemented dual-tolerance stopping
rule through an explicit entry and a testable host controller. It is not a new
contact solver or a replacement for the AL/Robust `toi_al` backend. No speed
benefit or complete-scene quality claim follows from implementing this rule.

## Explicit public entry

`preset=toi_compensated` expands to IPC contact/material assembly, conditional
PCG Graph, `ipc_termination=compensated`, cumulative tolerance `.001`, relative
residual tolerance `.03`, and minimum six effective accepted updates. Override
`execution=host` for the paired control. The preset refuses an incompatible
backend/termination override. It enables none of the execution candidates.

The default host/graph/combined entries still use legacy stopping with `.01`;
the movement tolerance stays `.01` and PCG rho stays `1e-4`. The original `toi`
preset still chooses AL/Robust. The report-compatible `.001/.03` settings belong
to this named experiment and do not replace global defaults.

## Module ownership

- `ipc_options.h`: strict environment parsing, threshold namespace and resolved
  stopping semantics; native integer/boolean invalid spellings throw.
- `ipc_budget.h`: the existing scalar beta/unit/weighted recurrence. Invalid
  alpha/residual/weight disables the budget even before activation and remains
  invalid for the rest of the solve.
- `ipc_residual_controller.h`: pure C++17 state and decision module, with no
  CUDA/JSON or geometry ownership. It fixes reference residual, tracks objective
  epochs, accepts final alpha, audits on the next assembly, and gates exits.
- `ipc_residual.inl`: existing GPU free-coordinate residual reduction. ABD and
  FEM norms remain reported separately; mixed native units are not a physical
  position/velocity error certificate.
- `core/GIPC.cu`: owns assembly, PCG, complete CCD, line search and state update;
  it feeds the controller, writes observation records and follows its selector.
- `tests/ipc_residual_controller_test.cpp`: compiles the same production header
  and selector for CPU property/control/configuration tests.

Legacy without shadow observation does not construct a controller or compute
additional residuals. No material, Hessian, preconditioner or safety CCD code
was rewritten for this entry.

## Exact sequence and definitions

After five positive accepted updates, before solving the sixth direction,
freeze `reference=max(current_free_residual, floor)`. It remains fixed when
Kappa/objective changes. Record both objective epoch and reference epoch.

After CCD/CFL/backtracking and the accepted update, feed the final alpha and
mark it pending. The following normal gradient/Hessian assembly measures the
accepted state and current objective, then updates once:

```text
r_new = current_residual / fixed_reference
q = 1 - accepted_alpha
d = max(0, r_new - q*r_previous)
beta_new = q*beta_old
unit_new = q*unit_old + d
budget_new = q*budget_old + w*d,  w = cumulative_tol/residual_tol
```

Initial beta/unit/budget are one. The published identity
`budget=beta+w*(unit-beta)` and bounds `budget>=beta`, `budget>=w*r`,
`budget<=unit` are checked by IpcBudget. A zero alpha spends no beta; a full
alpha clears the recurrence's previous history and leaves `budget=w*r_new`.

Movement exit has priority. The additional compensated exit requires valid,
active, audited, nonpending state; at least six positive accepted updates;
`animation_fullRate>0.99`; `beta<=cumulative_tol`; and
`budget<=cumulative_tol`. Current residual gating is retained explicitly by
the budget API. Animation/min-update checks are separate from mathematical
readiness so logs do not confuse reasons for continuing.

This adds an assembly before residual-based acceptance relative to the old
bottom-of-loop cumulative exit. It can add Newton/PCG work and is stricter than
legacy along the same prefix; it is not a promised TOI acceleration.

## Runtime diagnosis

Each ordinary observation stays in `newton[].ipc_residual`. An explicit legacy
terminal shadow assembly goes in `ipc_terminal_residual`; both are retained in
`ipc_residual_observations`, never overwritten. Observe final alpha through
`audited_alpha`, `pending_before/after_observation`, reference update/epoch,
audits and weight. Nonfinite JSON values may serialize as null; valid=false
must prevent additional exit.

`gated_ready/compensated_ready/history_veto` are scalar mathematics.
`gated_exit_allowed/compensated_exit_allowed/history_exit_veto` additionally
include animation and minimum updates. A compensated terminal row must have
the normal observation and exit assembly timing, and no PCG for that row.
Movement exit is allowed independently of the residual threshold.

At frame end, legacy `final_beta` is preserved. `budget_beta`, `budget`,
`unit_budget`, `budget_reference`, `budget_pending` and `budget_audits` describe
the actual residual controller; they need not equal iteration-count legacy
beta in zero-step cases.

## Checks and limitations

CPU tests cover 800 predetermined audited steps at w=1 and w=1/30, budget
identity/bounds, zero/full alpha, independent history veto, sticky invalid
inputs, duplicate accept, fixed reference under Kappa change, delayed audit,
animation threshold, movement priority, and strict native parsing. There is
no duplicate CPU solver pretending to validate the GPU gradients.

Public local run plans are in `configs/active/ipc_spmv_compensated_20261005_*.json`
and run through `tools/active/spmv_compensated.py`. Three-frame cloth smoke may
never activate the six-update budget. The separately selected fixed-bunny
23-frame diagnostic is from zero, based on existing first-activation logs; it
does not tune the minimum count to force apparent default coverage. Resource
failure must remain a failure and later same-scene diagnostics stop.

Keep full-scene material and actual velocity/position assessment separate from
these control checks. No current observation is a proof of physical error.
