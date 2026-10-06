# Two cloth scenes, requested four configurations

User requested current speed ratios for base, base+CUDA Graph, base+TOI,
base+CUDA Graph+TOI. Use local RTX 3070 Laptop, hanging cloth `cloth_hang_l`
and cloth over a fixed bunny `cloth_fixed_bunny_l`. Eight two-frame startup
checks, followed by 24 timing runs: 100 continuous from-zero frames, dt=.01,
three interleaved rounds per scene/configuration, 120-second budget per run.
Stop a failed arm rather than extending its budget. Quality failures are
reported; this user-requested 100-frame measurement is diagnostic, not promotion
or the final seven-pair/300-frame acceptance sequence.

|arm|program|contact|PCG execution|MAS|
|---|---|---|---|---|
|base|frozen original StiffGIPC|IPC|host|original legacy|
|base+Graph|active|IPC|conditional_graph|legacy|
|base+TOI|active|TOI AL|host|stable Cholesky, default factor_inverse|
|base+Graph+TOI|active|TOI AL|conditional_graph|same stable path|

All four disable refit, batched energy and energy reuse. Restriction remains
serial, matching current defaults. TOI uses default world_block penalty,
choose_start and safe-restart guard. Retain all material/ABD/FEM/cloth,
complete CCD, velocity .05, IPC/remaining fraction .01 and rho 1e-4 settings.
Do not substitute the all-components `toi` preset without overriding the
three unrelated execution components. Graph means the PCG conditional CUDA
Graph, not capture of the whole nonlinear simulation.

The TOI arms require the stable MAS path already adopted by this project.
Their comparison with original base therefore includes this prerequisite;
it is not an isolated pure-TOI algorithm gain. Within IPC and within TOI,
Graph comparison changes only execution. Freeze source/binary/configuration
identities and validate actual observed execution. The frozen base lacks the
new resolved configuration interface; verify its saved manifest, effective
scene parameters and observed host execution separately.

Measure solver_seconds from the common synchronized frame timer, excluding
startup and trace export outside the solver interval. Heavy probes, profiling,
substeps and velocity exports are off. Preserve positions for endpoint quality
and independent CPU rendering; do CPU analysis/rendering after GPU timing ends.
Timing includes graph setup within the solver where it occurs. Keep desktop
load samples and report these timings as diagnostics, not controlled-GPU speed
certification or a claim that quality gates passed.

Report three per-run times, medians, three paired base/configuration ratios and
their median, plus Graph gain with/without TOI. Separately report directions,
PCG iterations, limits/breakdowns, cloth maximum stretch, fixed object drift,
and position differences against the three original-base repeats. Freeze the
base's own endpoint/position repeat envelope; do not relax it. No actual
velocity or independent accepted-path CCD certification is claimed. Render
common final/peak frames with identical camera, scale and stretch colors.
