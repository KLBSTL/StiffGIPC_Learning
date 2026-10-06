"""Freeze the local v50 source, executable, and build evidence after compilation."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];target=ROOT/'manifests/perf_v50_local.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for p in sorted((ROOT/'sources/stiff_perf_v50').rglob('*')):
    if p.is_file() and not any(x in {'Output','sorted_mesh','.git','__pycache__'} for x in p.relative_to(ROOT).parts):
        files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
binaries={}
for name in ['gipc.exe','diag_fused_update_tests.exe']:
    p=ROOT/'builds/local-v50/Release'/name
    binaries[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'implementation_version':'v50-reduced-slack','files':files,
    'source_digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'binaries':binaries,
    'build_platform':'Windows VS2022 CUDA13.0 SM86 Release',
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
target.write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
