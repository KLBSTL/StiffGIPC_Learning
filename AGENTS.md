# Working rules

- Active simulation source is `StiffGIPC/`; build with root `CMakeLists.txt`.
- `baseline/` is the frozen Stiff reference. Do not change its solver or materials.
- Use `tools/bench/` for current portable experiments. Keep GPU runs serial.
- Record source, compile, binary, input and resolved-configuration identities.
- Keep materials, full CCD, legacy cumulative tolerance .01, minimum 6 updates
  and PCG rho tolerance 1e-4 unchanged in execution-only experiments.
- Historical rejected code is recoverable at `archive/pre-cleanup-20261006`.
  Keep its reports rather than reintroducing versions into the active tree.
- Each optimization round needs correctness and quality checks, paired timing,
  an explicit cost analysis, and a finite next-round decision.
- Do not certify performance from partial, instrumented or resource-failed runs.
