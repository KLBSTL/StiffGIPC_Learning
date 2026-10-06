# Fixed-cloth quality diagnosis

The first 4090 guard remains failed: fixed-cloth p99 stretch 1.0138636848759526 exceeded the frozen 1.0135636480371302. The twelve performance-screen runs are not authorized by this diagnostic and are not launched here.

This additive tool leaves `tools/bench`, the original build seal and all guard evidence unchanged. It freezes exactly seven from-zero fixed-bunny-cloth runs (59 frames, dt .01, native minimum 6, Newton/cumulative .01, PCG rho 1e-4, 120 seconds each):

1. active contact pool off, repeat 1
2. official baseline, repeat 1
3. official baseline, repeat 2
4. active contact pool off, repeat 2
5. active contact pool on, repeat 1, validation disabled
6. active contact pool off, repeat 3
7. official baseline, repeat 3

Original bound exceedance is an observation to investigate, not a reason to abort these seven diagnostic controls. Configuration/identity/numeric failures stop the owned run and remaining plan. Every run is CPU-analyzed before the next. All GPU work holds the original serial lock and preserves the same GPU-memory/disk/deadline policies. No tolerance is expanded, speedup certified, default promoted, or long run started. One on run cannot establish repeatability.

The active program uses the existing frozen runner and original manifest SHA. Baseline uses an explicitly separate adapter: only its supported scene/step/tolerance/output environment variables are passed. Its actual `scene.json` confirms scalar dt/Newton/PCG and native MAS selection; baseline source confirms the hard-coded stopping rule. It does not claim an active resolved-config contract, nor request or reconstruct unsupported velocity. Cross-program comparisons use real positions and explicitly report unavailable baseline velocity. Asset/initial-array identities must match before trajectory comparisons.

On the already built Step1 AutoDL checkout, replace the manifest/log paths below with the real preserved paths. Seal creation does not build or install anything. Do not upload later native changes into this checkout before the diagnosis: they would invalidate the original seal.

```sh
/root/miniconda3/bin/python tools/diagnostic/quality_contracts_test.py -v
/root/miniconda3/bin/python tools/diagnostic/fixed_quality.py seal \
  --active-manifest build/active/bench_manifest.json \
  --prior-guard runs/pool_step1_20261006/guards_analysis.json \
  --base-build-log reports/base_build.log \
  --output build/fixed_quality_step1_seal.json
/usr/bin/xvfb-run -a /root/miniconda3/bin/python tools/diagnostic/fixed_quality.py run \
  --seal build/fixed_quality_step1_seal.json --session fixed_quality_step1_20261006
```

The new seal stores independent baseline `baseline/StiffGIPC`, Assets, MeshProcess, CMake and compile-command identities, executable hash, linker log when available, and compiler binaries identified through CMakeCache. Missing compiler/link evidence is explicitly listed. It binds the unchanged active seal and original failed-guard receipt by SHA, plus the new diagnostic tool files. Source/inputs/executables/tools are checked before each run and before/after final analysis; compiler binaries are checked but shared system libraries are not exhaustively frozen.

The new session contains the frozen plan, seven run directories, per-run checks, ordered batch, final analysis and receipt. Reports retain the original bound, signed excess, violating frame numbers, per-frame material metrics, actual export availability, off/base observed repeat ranges, grouped trajectory differences and diagnostic elapsed times. Observed ranges never replace the original bound. Complete raw evidence remains required for independent reanalysis. Both quality and performance certification remain false, even if every new run is within bounds.

Seal creation also requires active and baseline Assets to match file by file. CPU entry point: `quality_analysis.analyze(session_path, seal_dict, batch_dict)`. It verifies run evidence hashes and the exact planned prefix before analysis. The synthetic contract tests never invoke GPU, SSH or native executables and never edit real run folders.
