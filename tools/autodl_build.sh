#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
task_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
mkdir -p builds
nvcc --version > builds/autodl_cuda.txt
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version --format=csv > builds/autodl_gpu.csv
if ! grep -q 'release 12.8' builds/autodl_cuda.txt; then
    printf 'Expected CUDA 12.8; see builds/autodl_cuda.txt\n' >&2
    exit 2
fi
for tree in stiff_base stiff_fused; do
    build="builds/autodl-${tree}"
    cmake -S "sources/${tree}" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > "builds/${tree}_configure.log" 2>&1
    cmake --build "$build" --target gipc --parallel 2 > "builds/${tree}_build.log" 2>&1
done
cmake -S tools/validator -B builds/autodl-validator -DCMAKE_BUILD_TYPE=Release > builds/validator_configure.log 2>&1
cmake --build builds/autodl-validator --parallel 2 > builds/validator_build.log 2>&1
builds/autodl-validator/validate_path --self-test > builds/validator_selftest.log 2>&1
if [ "${1:-}" != "--compile-only" ]; then
    GIPC_VALIDATE_COMPONENTS="$task_root/builds/toi_components.json" builds/autodl-stiff_fused/gipc > builds/toi_component_validation.log 2>&1
fi
printf 'AUTODL_BUILD_COMPLETE\n'
