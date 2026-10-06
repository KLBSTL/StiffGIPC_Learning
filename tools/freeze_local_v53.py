"""Freeze the incremental source overlay and preserve the inherited identity."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((ROOT/'manifests/perf_v50_local.json').read_text())
for f in base['files']:assert sha(ROOT/f['path'])==f['sha256']
for p,f in base['binaries'].items():assert sha(ROOT/p)==f['sha256'],p
files=[f for f in base['files'] if not f['path'].endswith('/solver/toi_solver.cu')]
for name in ['sources/stiff_perf_v53/StiffGIPC/solver/toi_solver.cu','tools/v53_overlay.targets',
             'builds/local-v50/gipc.vcxproj']:
    p=ROOT/name;files.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
files.sort(key=lambda x:x['path'])
binary=ROOT/'builds/local-v53/Release/gipc.exe'
manifest={'implementation_version':'v53-initial-guess','inherited_source_digest':base['source_digest'],
    'files':files,'source_digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
    'binaries':{binary.relative_to(ROOT).as_posix():{'sha256':sha(binary),'bytes':binary.stat().st_size}},
    'build_platform':base['build_platform'],'gpu':base['gpu'],
    'build_method':'v50 CMake/MSBuild project with conditional source overlay, existing objects reused; v50 assets'}
target=ROOT/'manifests/perf_v53_local.json';assert not target.exists()
target.write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
