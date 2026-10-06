# v53 cloth regression: initial-guess flag only

Question: does the v53 complete-objective initial guess selection degrade cloth scenes?
Use the frozen v53 executable and runner, no solver edits. Within each scene change ONLY choose-start. All runs start from zero, CUDA Graph, native exits, rho=1e-4, velocity tolerance=.05, suite=1, reduced-slack=0, full CCD and pair limit 10 million. Diagnostic trace/physics/PCG audit enabled; no certified timing claims.

Scenes, each 100 frames and 180 s per-run budget:
- cloth_hang_l: nominal dt=.04, MAS Cholesky; anchored cloth, 4 s physical time.
- cloth_sphere7_l: nominal dt=.01, diagonal preconditioner as prior cloth/sphere studies; 1 s.
- cloth_fixed_bunny_l: nominal scene dt, diagonal preconditioner for cloth over a fixed obstacle.

Run OFF r1, ON r1, ON r2, OFF r2 per scene. If a run fails, preserve it, skip remaining runs for that scene, continue independent scenes; do not relax limits. Verify identical effective scenes/initial state/constraints and frozen identity. Audit full-objective selection and native exits, full accepted-path CCD, maximum and p95 cloth edge stretch, triangle area, pinned/fixed displacement, mass-weighted cloth trajectory differences and energy records. Compare ON/OFF differences with each arm's repeat variation, not an invented physical-error tolerance. Render actual shared keyframes with the same camera/limits and visually inspect.

Report no observed regression only within tested scenes/durations, distinguish nonactivation of the new branch from validation under contact, and distinguish different draping from demonstrated physical degradation. No new solver candidates in this turn.

Final: all 12 runs completed; 4239 accepted-path/bridge CCD segments passed with zero flags, all fixed displacement zero, no PCG caps, native exit/initial selection audits passed. Regression detected: fixed-bunny cloth peak edge ratio OFF 1.0441–1.0445 versus ON 1.13235 in both repeats (frame 24); same-edge coordinates and cloth potential independently cross-checked. Sphere has one larger ON peak, hang peak unchanged but trajectory differs. Actual branch selected safe in all six ON runs. Keyframe heatmaps inspected, source/binary/runner unchanged. Keep default OFF; no solver fix was attempted. Report V53_CLOTH_REGRESSION_20261004.md.
