#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
cd "$(dirname "$0")/.."
mkdir -p builds reports runs/autodl
baseline=/root/autodl-tmp/stiff_toi_cudagraph_20260929/v34_20261003
test ! -e builds/autodl-stiff_perf_v41
ln -s "$baseline/sources/stiff_base" sources/stiff_base
mkdir -p builds/autodl-stiff_base
cp -n "$baseline/builds/autodl-stiff_base/gipc" builds/autodl-stiff_base/gipc
cmake -S sources/stiff_perf_v41 -B builds/autodl-stiff_perf_v41 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 > builds/configure.log 2>&1
cmake --build builds/autodl-stiff_perf_v41 --target gipc diag_fused_update_tests --parallel 2 > builds/build.log 2>&1
builds/autodl-stiff_perf_v41/diag_fused_update_tests > builds/fixture.json
python3 - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path.cwd()
m=json.loads((r/'manifests/perf_v41.json').read_text())
assert all(hashlib.sha256((r/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in m['files'])
for name in ['stiff_base','stiff_perf_v41']:
 p=r/f'builds/autodl-{name}/gipc'
 m['binaries'][p.relative_to(r).as_posix()]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
m['build_platform']='AutoDL CUDA12.8 sm89 Release'
(r/'manifests/perf_v41_autodl.json').write_text(json.dumps(m,indent=2))
PY
echo V41_BUILD_COMPLETE
