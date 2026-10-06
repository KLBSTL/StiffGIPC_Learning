"""Freeze MAS action-only optimization; v54 TOI and every other dependency unchanged."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((ROOT/'manifests/perf_v54_local.json').read_text())
for f in old['files']:assert sha(ROOT/f['path'])==f['sha256']
for p,v in old['binaries'].items():assert sha(ROOT/p)==v['sha256']
dest=ROOT/'builds/local-v55/Release';assert not dest.exists();dest.mkdir(parents=True)
shutil.copy2(ROOT/'builds/local-v50/Release/gipc_v55.exe',dest/'gipc.exe')
for p in (ROOT/'builds/local-v54/Release').glob('*.dll'):shutil.copy2(p,dest/p.name)
exclude=['/solver/MASPreconditioner.cu','/solver/mas_cholesky.inl','/solver/mas_cholesky_fixture.inl','tools/v54_overlay.targets']
files=[f for f in old['files'] if not any(f['path'].endswith(e) for e in exclude)]
for name in ['sources/stiff_perf_v55/StiffGIPC/solver/MASPreconditioner.cu','sources/stiff_perf_v55/StiffGIPC/solver/mas_cholesky.inl',
             'sources/stiff_perf_v55/StiffGIPC/solver/mas_cholesky_fixture.inl','tools/v55_overlay.targets']:
    p=ROOT/name;files.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
files.sort(key=lambda x:x['path']);exe=dest/'gipc.exe'
m={'implementation_version':'v55-warp-cholesky','inherited_source_digest':old['source_digest'],'files':files,
   'source_digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
   'binaries':{exe.relative_to(ROOT).as_posix():{'sha256':sha(exe),'bytes':exe.stat().st_size}},
   'build_platform':old['build_platform'],'build_method':'v50 project; v54 TOI and v55 MAS overlay'}
target=ROOT/'manifests/perf_v55_local.json';assert not target.exists();target.write_text(json.dumps(m,indent=2))
print(json.dumps({k:v for k,v in m.items() if k!='files'}))
