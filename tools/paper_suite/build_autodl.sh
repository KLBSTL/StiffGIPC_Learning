#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 tools/paper_suite/prepare.py verify-payload
python3 - <<'PY'
import re,subprocess
v=subprocess.check_output(['nvcc','--version'],text=True)
m=re.search(r'release (\d+)\.(\d+)',v)
assert m and tuple(map(int,m.groups())) >= (12,4), 'Conditional WHILE graph requires CUDA toolkit 12.4 or newer'
import numpy,matplotlib
PY
for tool in cmake ninja nvcc nvidia-smi xvfb-run; do command -v "$tool" >/dev/null; done
test ! -e build/autodl_active
test ! -e build/autodl_base
mkdir -p build/autodl_active build/autodl_base
arch=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits | tr -d '.[:space:]')
test -n "$arch"
cmake -S . -B build/autodl_active -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" > build/autodl_active/configure.log 2>&1
cmake --build build/autodl_active --parallel 2 > build/autodl_active/build.log 2>&1
cmake -S baseline -B build/autodl_base -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" > build/autodl_base/configure.log 2>&1
cmake --build build/autodl_base --target gipc --parallel 2 > build/autodl_base/build.log 2>&1
ctest --test-dir build/autodl_active --output-on-failure > build/autodl_active/ctest.log 2>&1
python3 tools/paper_suite/run_suite.py seal --active build/autodl_active --base build/autodl_base --output build/autodl_programs.json
echo 'Clean builds and manifest ready. Run smoke/checkpoint gates before performance.'
