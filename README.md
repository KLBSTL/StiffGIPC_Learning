# StiffGIPC core source

This branch contains the current StiffGIPC solver and application source, its
MeshProcess / GKlib / METIS build dependencies, and the project license.
Tests, benchmark scripts, experiment results, reports, scene assets, and build
outputs are excluded.

## Build

Requirements: a C++17 compiler, CUDA Toolkit, CMake, Eigen3, GLEW, GLUT, OpenGL,
and nlohmann_json. On Windows, use Visual Studio 2022 with CUDA integration.
Make the dependency packages available through CMake's package search paths
or a vcpkg toolchain.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --target gipc
```

Set `CMAKE_CUDA_ARCHITECTURES` to the architecture of your GPU. For a vcpkg
installation, also pass `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`.

Runtime scene meshes and configuration files must be supplied separately in
`Assets/`. The application uses `Output/` for generated output. Neither directory
is part of this source snapshot.
