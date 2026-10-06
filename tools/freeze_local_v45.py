"""Freeze the local v45 source, executable, and build evidence after compilation."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];target=ROOT/'manifests/perf_v45_local.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for p in sorted((ROOT/'sources/stiff_perf_v45').rglob('*')):
    if p.is_file() and not any(x in {'Output','sorted_mesh','.git','__pycache__'} for x in p.relative_to(ROOT).parts):
        files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
binaries={}
for name in ['gipc.exe','diag_fused_update_tests.exe']:
    p=ROOT/'builds/local-v45/Release'/name
    binaries[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'implementation_version':'v45-restart-linear-diagnostic','files':files,
    'source_digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'binaries':binaries,
    'build_platform':'Windows VS2022 CUDA13.0 SM86 Release',
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
target.write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
