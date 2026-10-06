#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
task_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"
mkdir -p builds reports runs/autodl
nvcc --version > builds/autodl_cuda.txt
grep -q 'release 12.8' builds/autodl_cuda.txt
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,utilization.gpu --format=csv > builds/autodl_gpu.csv
baseline=/root/autodl-tmp/stiff_toi_cudagraph_20260929/v32_20261002
test ! -e builds/autodl-stiff_perf_v34_cuda128
for name in stiff_base stiff_robust_port; do
    mkdir -p "builds/autodl-${name}"
    cp -n "$baseline/builds/autodl-${name}/gipc" "builds/autodl-${name}/gipc"
done
cmake -S sources/stiff_perf_v34_cuda128 -B builds/autodl-stiff_perf_v34_cuda128 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > builds/autodl_v34_cuda128_configure.log 2>&1
cmake --build builds/autodl-stiff_perf_v34_cuda128 --target diag_fused_update_tests gipc --parallel 2 > builds/autodl_v34_cuda128_build.log 2>&1
builds/autodl-stiff_perf_v34_cuda128/diag_fused_update_tests > builds/autodl_perf_v34_fixture.json
python3 - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path.cwd()
for filename,version,binaries in [
 ('robust_port_v32','v32-robust-derived-port',['stiff_base','stiff_robust_port']),
 ('perf_v34_cuda128','v34-fused-diag-update',['stiff_base','stiff_perf_v34_cuda128'])]:
 m=json.loads((r/f'manifests/{filename}.json').read_text())
 assert m['implementation_version']==version
 assert all(hashlib.sha256((r/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in m['files'])
 m['binaries']={}
 for tag in binaries:
  p=r/f'builds/autodl-{tag}/gipc'
  m['binaries'][p.relative_to(r).as_posix()]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 m['build_platform']='AutoDL CUDA12.8 sm89 Release'
 (r/f'manifests/{filename}_autodl.json').write_text(json.dumps(m,indent=2))
old=json.loads(pathlib.Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929/v32_20261002/manifests/robust_port_v32_autodl.json').read_text())
new=json.loads((r/'manifests/robust_port_v32_autodl.json').read_text())
assert old['source_digest']==new['source_digest'] and old['binaries']==new['binaries']
PY
printf 'AUTODL_V34_BUILD_COMPLETE\n'
