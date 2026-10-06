# v51 source overlay

This candidate inherits frozen v50 except `StiffGIPC/solver/toi_solver.cu`.
It keeps the original positive frozen-slack Hessian as a majorizing model,
but selects steps with the exactly slack-minimized energy. The gradient at
the assembly point is unchanged because slack was just updated there.
Default remains off. CCD, multiplier updates, dt and stopping are unchanged.

Build using the v50 generated project and `tools/v51_overlay.targets`,
`/p:ForceImportAfterCppTargets=<absolute targets path>` with Release/x64.
The targets file limits both source replacement and `TargetName=gipc_v51`
to the main gipc project; do not pass TargetName globally to dependencies.
Only the changed CUDA translation unit needs recompilation; the v50 executable
and all frozen source files remain untouched. The inherited objects retain
v50 asset paths, whose contents are included in the manifest. The overlay and
the output executable are hashed in `manifests/perf_v51_local.json`.

This is a second candidate after the full reduced-Hessian v50 timed out at
27/40 frames. It is not a claim of reference-algorithm equivalence or a fix.
