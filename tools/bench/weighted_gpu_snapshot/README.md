# P5 weighted quality-check source snapshot

This directory preserves the exact generated CUDA sources compiled for the
latest private fixed-system experiment. `project/` and `overlay/` were copied
byte for byte from its build workspace. The root CMake file selects the
repository's `StiffGIPC/` tree, which matched the experiment's frozen source
tree at archival time. No generated test results, matrices, vectors, binaries,
or performance reports are included.

The replay has four arms: native MAS with the legacy quality-check schedule,
cached rank-eight balanced MAS with the legacy schedule, and those two actions
with the conditional two-iteration cooldown schedule. The cooldown changes
when explicit quality checks occur; the acceptance thresholds are unchanged.
The implementation remains a private diagnostic and is not enabled in the
simulation's production PCG path. This snapshot does not certify an overall
speedup.

## Build on the existing workstation

The preserved project uses CUDA, MSVC, Eigen, and the workstation's existing
include paths. Adjust these paths in `project/CMakeLists.txt` for another
machine. For the original workstation:

```powershell
& 'D:/computer/cmake/bin/cmake.exe' -S tools/bench/weighted_gpu_snapshot -B build/weighted_gpu_source_snapshot -G 'Visual Studio 17 2022' -A x64 -DCMAKE_GENERATOR_INSTANCE=D:/vs2022 -DCMAKE_CUDA_ARCHITECTURES=86 '-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0'
& 'D:/computer/cmake/bin/cmake.exe' --build build/weighted_gpu_source_snapshot --config Release --target weighted_gpu
```

The executable accepts `contracts|profile|measure`, an input-manifest JSON
path, and a fresh output-directory path. It requires locally generated fixed
system and MAS/coarse-space inputs; these experiment data are intentionally
not published. The historical preparation and run scripts remain under
`tools/local/` and `tools/bench/`; they require their local parent captures and
seals. This source snapshot can be compiled independently of those captures.
