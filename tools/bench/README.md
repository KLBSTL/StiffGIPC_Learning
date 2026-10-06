# Linux contact-pool short screen

This directory is self-contained except for Python + NumPy, the flattened native build and its Assets. It does not import the old `tools/active` tree and never builds, installs, uploads, cleans, resumes a simulation checkpoint, or starts the next round automatically.

Read-only AutoDL check on 2026-10-06 found `/usr/bin/xvfb-run`, `/usr/bin/Xvfb`, `/usr/bin/nvidia-smi`, CUDA 12.8.93 at `/usr/local/cuda-12.8/bin/nvcc`, and NumPy 2.3.2 under `/root/miniconda3/bin/python`. System `python3` lacked NumPy. Recheck paths on a replacement instance.

After a clean build by the owner, seal the actual executable, complete flattened native source mapping, compiler command list, CMake cache, build log and optional linker command. All paths below are relative to the repository root; select the real paths produced by the build.

```sh
/root/miniconda3/bin/python tools/bench/contracts_test.py -v
/root/miniconda3/bin/python tools/bench/linux_runner.py seal-build --build-dir build/autodl --exe build/autodl/gipc --build-log reports/autodl_build.log --output build/autodl/bench_manifest.json
/usr/bin/xvfb-run -a /root/miniconda3/bin/python tools/bench/linux_runner.py run --manifest build/autodl/bench_manifest.json --session pool_20261006 --stage guards
```

The guard stage performs exactly four from-zero runs: hanging cloth 51/on, fixed-bunny cloth 59/on, mixed bunny 35/off and 35/on. The on guards require typed pair-set and same-state energy/Armijo validation, actual activation, complete state/velocity exports and no hard numeric failure. Cloth uses the byte-identical frozen protocol in `quality_protocol.json` (SHA256 `1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f`). Mixed trajectory and material differences remain pending because a repeat tolerance was not frozen. This new declared policy does not rewrite the historical failed guard and does not certify mixed quality; no mixed hard failure is waived.

Read `runs/pool_20261006/guards_analysis.json`. Only after that analysis succeeds, invoke one round at a time:

```sh
/usr/bin/xvfb-run -a /root/miniconda3/bin/python tools/bench/linux_runner.py run --manifest build/autodl/bench_manifest.json --session pool_20261006 --stage r1
# Review r1_analysis.json before invoking r2.
/usr/bin/xvfb-run -a /root/miniconda3/bin/python tools/bench/linux_runner.py run --manifest build/autodl/bench_manifest.json --session pool_20261006 --stage r2
# Review r2_analysis.json before invoking r3.
/usr/bin/xvfb-run -a /root/miniconda3/bin/python tools/bench/linux_runner.py run --manifest build/autodl/bench_manifest.json --session pool_20261006 --stage r3
```

Every round is four serial runs, with off/on and scene order reversed in round 2. Maximum total is 16 runs, 120 seconds each; timeout, foreign GPU compute, low memory or disk reserve stops the owned process group and ends the stage. Existing session/run/stage outputs are never overwritten or retried. Source, executable, prior result, complete raw file hashes, batch ledger and analysis receipt are rechecked before subsequent rounds. New source requires a new clean build seal and new session.

Each run records requested and expanded config, full GIPC environment, resolved config, build/source/executable identity, logs, process ownership, GPU observations, result, config validation and an evidence inventory. Each stage writes an immutable batch, analysis and receipt. No ambient GIPC variables survive; resume/checkpoint variables are absent. `diagnostics=[]`, no profiler/cost/substep probes; state and actual velocity exports intentionally remain enabled in both arms for quality support.

`screen_summary.json` reports three paired medians for the full 51/59-frame prefixes and fixed contact windows hang 41–50 / fixed bunny 22–59. Both medians must reach 1.05 in both scenes to support cost review. This does not start 100-frame or full acceptance runs, certify performance, supply independent path CCD, or promote the candidate default. Final quality and long-run gates remain separate. Deleting raw before later-round verification breaks the evidence chain.

Offline checks run with `python tools/bench/contracts_test.py -v`; they use synthetic inputs and mock configuration validation where identified. They do not assert native build, real GPU, CCD or physical correctness.
