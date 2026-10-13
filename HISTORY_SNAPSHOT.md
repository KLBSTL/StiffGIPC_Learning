# AutoDL recorded simulation snapshot — 2026-10-13

Branch: `history/autodl-recording105-dt0p04-20261013`.

The root `StiffGIPC/`, `Assets/`, `MeshProcess/`, `tests/` and `CMakeLists.txt`
are the sealed observer snapshot actually compiled and measured on AutoDL.
They are copied from the 2026-10-11 local observer, not from the subsequently
edited working directory. `LOCAL_SOURCE_COMPARISON.json` records all 355
source/input hashes and the CMake hash. There are 37 native translation units.
`.gitattributes` preserves exact bytes across Windows and Linux.

The current execution combination, called “最快版” in the experiment, uses
conditional PCG Graph K1, full CCD refit, energy batch/reuse, BVH refit,
direct FEM pullback, legacy MAS and pivot-only synchronization. All three
candidate stopping rules use this same execution combination. No numerical
solver changes were made while preparing this history branch.

## Fixed experiment configuration

- Scenes: Bunny (`cloth_fixed_bunny_l`), Mixed (`bunny_cloth_bunny_l`),
  Boxes (`boxes1920_cloth_l`), all low resolution.
- dt=0.04; candidates: PCG rho=1e-4; movement threshold=0.01;
  cumulative TOI threshold=0.01; residual gate=0.03; minimum updates=6.
- Rules: Movement only (MOV), Cumulative TOI .01 (TOI),
  Compensated + residual gate (GATE).
- Reference: Movement only with rho=1e-6 and the same outer threshold 0.01.
  It is a stricter numerical reference, not an exact solution.
- 105 frames × 3 repetitions; timing tables use frames 1–100;
  actual-state and common-input checkpoint tables use physical frame 100.
- Complete expanded task configurations and serial ordering:
  [`EXPERIMENT_PLAN.json`](EXPERIMENT_PLAN.json).

The completed matrix contains 12 two-frame smoke runs, 36 full trajectories,
63 checkpoint/recording comparisons and 9 cost diagnostics. All 120 native
runs succeeded: 3780 formal trajectory frames, 67523 per-Newton rows,
28620 error observations and 162 lossless OBJ exports.

## Observed first-100-frame timings

Times are medians of three runs; speedup is the MOV median divided by the
candidate median on the same AutoDL RTX 4090.

| Scene | MOV (s) | TOI (s) | TOI speedup | GATE (s) | GATE speedup |
|---|---:|---:|---:|---:|---:|
| Bunny | 9.216805 | 5.193676 | 1.774620× | 8.091610 | 1.139057× |
| Mixed | 24.738401 | 10.338492 | 2.392844× | 14.361656 | 1.722531× |
| Boxes | 64.663762 | 22.648293 | 2.855127× | 59.749192 | 1.082253× |

MOV here is the current Graph-enabled observer, not frozen original StiffGIPC.
These are instrumented measurements, not a certification of equal-quality
2× acceleration. In the common-input Boxes frame-100 step, pure TOI has
1.966769 mm RMS position difference versus the stricter reference, while
GATE has 0.015503 mm. The report also exposes large reference-repeat dispersion
on complete trajectories.

## Saved evidence

- [Main report, 6 pages](history/autodl_recording105_20261013/AutoDL_3Scene_Main.pdf)
- [Visual appendix, 15 pages](history/autodl_recording105_20261013/AutoDL_3Scene_Appendix.pdf)
- [Individual frame-100 checkpoint PDFs](history/autodl_recording105_20261013/checkpoints/)
- [All 18 numerical datasets](history/autodl_recording105_20261013/data/)
- [Dataset sizes and SHA256](history/autodl_recording105_20261013/DATA_MANIFEST.json)
- [Build, input and backup evidence](history/autodl_recording105_20261013/evidence/)

`per_newton.csv` exceeds GitHub's single-file limit; its exact contents are
saved as lossless `per_newton.csv.gz`. The other 17 CSVs are unchanged.
Decompress with `gzip -dk history/autodl_recording105_20261013/data/per_newton.csv.gz`.
The frozen original baseline remains under `baseline/`.

The complete raw archive includes every position and velocity export,
native binary, compilation evidence and runtime logs. It is retained locally
and on AutoDL rather than placed in Git:

`autodl_recording105_20261013_full.tar.gz`, 4,570,828,245 bytes,
SHA256 `b13b4f11cbeac3225d096c21079874515de7360cd31171d87904a70789534d49`.

The local extraction receipt confirms all 11,323 archive members, including
7,392 non-keyframe raw files added to the review package. No raw frames are missing.
The repository does not contain credentials, private SSH keys or generated binaries.

## Replay on a fresh AutoDL instance

The measured environment used Ubuntu 22.04, CUDA 12.8.93, RTX 4090 (SM89),
Ninja, Eigen3, GLEW, freeglut, OpenGL and nlohmann-json. Python needs numpy,
matplotlib and moderngl (measured: 2.3.2, 3.10.5 and 5.12 respectively).
Use a fresh checkout/output area; controllers preserve existing sealed results.

From the checkout root, with CUDA 12.8 and Python available:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p build/observer
cmake -S . -B build/observer -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  > build/observer/configure.log 2>&1
cmake --build build/observer --target gipc --parallel 2 \
  > build/observer/build.log 2>&1
python -B tools/remote/run_recording105_autodl.py seal
xvfb-run -a python -B -u tools/remote/run_recording105_autodl_v2.py run
python -B tools/remote/run_recording105_autodl_v2.py analyze
```

`TRANSFER_MANIFEST.json` is the original transfer manifest; the v2 controller
and later postprocessing adapters are preserved beside it. The original
controller's Linux metadata omission was corrected by the v2 wrapper before
any formal frames were produced. Use v2 for the GPU run. The original launch
and resume scripts are retained as history; they are not the fresh-run entry point.

For the exact software-EGL visual postprocessor, its recorded checkout path
is `/root/autodl-tmp/codex_recording105_20261013/payload`. From that root:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_ALWAYS_SOFTWARE=1 \
python -B tools/remote/postprocess_recording105_autodl.py render
```

Rendering occurs after the GPU matrix. It does not rerun or deform simulations.
Existing checked PDFs are included in this branch. The local PDF generator
also uses the earlier local comparison dataset, which is in the retained local
experiment archives rather than this three-scene Git snapshot.

## Snapshot validation

Before commit: verify all 355 source/input hashes and CMake against the sealed
manifest; compile every copied Python source; run the portable bench and paper
suite CPU contracts. After staging: verify the native/input Git blobs against
the same SHA256 manifest. The actual clean cloud build and complete GPU matrix
are recorded in the evidence above; archiving does not rerun them.
