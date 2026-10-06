# User-selected defaults and next TOI investigation

The user explicitly requested default-on factor-inverse action and the world
penalty estimator despite the historical quality gate. This changes the active
development defaults, not the target or acceptance evidence. Preserve the old
switches, materials, stopping tolerances, complete CCD and finite budgets.

Native source: Cholesky action defaults to factor_inverse; TOI diagonal/movable
defaults to world_block. Mass/full-scope paths retain their compatible historical
estimator. Common runner resolves auto to active defaults; legacy MAS/IPC stays
triangular/generalized. Historical requested/expanded values and hashes are
preserved even though omitted defaults have changed.

Finite sequence: Release build; existing components and PCG guards; 2-frame
native-default smoke with precisely the two environment overrides omitted;
three 35-frame combined runs and one current 35-frame Stiff reference, 120-second
budget each. One additional 35-frame diagnostic-only outer probe, same budget,
selected frames 1-3/24-26/33-35. Never include its time in performance ratios.

Observer records solved affine contact-model identity, full objective and energy
parts, dual update, contact/plane changes, safe-trial separation and FEM J at
trial/safe endpoints. Energy probes check protected physics state byte-for-byte;
no position swaps, derivative rebuilds or threshold changes. Affine feasibility
and complementarity summaries are diagnostics, not complete KKT certification.

Only if the observer identifies a specific state/update defect, implement one
bounded correction and compare from zero. Do not revive global velocity_only,
global half steps or a penalty grid. No 100/300-frame acceptance or speed claim
until quality is demonstrated. Default-on does not imply certified quality.

## Selected exit counterfactual, declared before execution

The first probe showed f34/o2 reducing trial FEM J from about .475 to .095
with one direction (.496 m/s), then a full-step heuristic exit. Historical
Robust advance_al uses a flat frame-wide Newton index, so the index scope
is not a newly identified porting error. Do not reset six iterations at every
model or revive the global velocity_only route.

Use one new native diagnostic and two selected-target diagnostics, from zero
to 35 frames, 90 seconds each. Only f34/o2 suppresses the full-step heuristic;
the existing .05 m/s raw direction stop remains, with an eight-iteration cap
(lower than production). If it does not converge, reject at this cap. Record
the target in resolved_config and actual Newton records. It is off by default,
never a performance benchmark. Compare local energy, trial/safe J, later outer
work, and pre-target states; repeat variance limits causal attribution. This
tests whether inadequate inner solution is a useful next correction, without
asserting that extra iterations must repair the physical path.
