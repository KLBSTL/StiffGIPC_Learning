#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
task_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
mkdir -p builds reports runs/autodl
nvcc --version > builds/autodl_cuda.txt
grep -q 'release 12.8' builds/autodl_cuda.txt
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version --format=csv > builds/autodl_gpu.csv
for tree in stiff_base stiff_robust_port; do
    build="builds/autodl-${tree}"
    test ! -e "$build"
    cmake -S "sources/${tree}" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > "builds/${tree}_configure.log" 2>&1
    cmake --build "$build" --target gipc --parallel 2 > "builds/${tree}_build.log" 2>&1
    printf 'BUILD_COMPLETE %s\n' "$tree"
done
python3 -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); p=r/"manifests/robust_port_v32.json"; m=json.loads(p.read_text()); assert all(hashlib.sha256((r/e["path"]).read_bytes()).hexdigest()==e["sha256"] for e in m["files"]); m["binaries"]={str(b.relative_to(r)):{"sha256":hashlib.sha256(b.read_bytes()).hexdigest(),"bytes":b.stat().st_size} for b in [r/"builds/autodl-stiff_base/gipc",r/"builds/autodl-stiff_robust_port/gipc"]}; m["build_platform"]="AutoDL CUDA12.8 sm89 Release"; (r/"manifests/robust_port_v32_autodl.json").write_text(json.dumps(m,indent=2))'
GIPC_VALIDATE_COMPONENTS="$task_root/builds/toi_components.json" builds/autodl-stiff_robust_port/gipc > builds/toi_component_validation.log 2>&1
python3 tools/check_robust_port_components.py builds/toi_components.json
printf 'AUTODL_V32_BUILD_COMPLETE\n'
