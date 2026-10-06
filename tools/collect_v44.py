"""Verify frozen sources/binaries, inventory and package the bounded diagnostic results."""
import hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'manifests/perf_v44_autodl.json').read_text())
for entry in manifest['files']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
for path,entry in manifest['binaries'].items():assert sha(ROOT/path)==entry['sha256'],path
files=[]
for folder in ['runs','reports','manifests','tools']:
    files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [ROOT/path for path in manifest['binaries']]
files += [ROOT/'builds/build.log',ROOT/'builds/configure.log',ROOT/'builds/fixture.json',ROOT/'builds/autodl-stiff_perf_v44/pcg_guard_fixture.json']
inventory=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
target=ROOT/'reports/FINAL_IDENTITY_V44.json';assert not target.exists()
target.write_text(json.dumps({'source_digest':manifest['source_digest'],'source_files_verified':len(manifest['files']),
    'binaries':manifest['binaries'],'files':inventory,'free_bytes':shutil.disk_usage(ROOT).free,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],text=True).strip()},indent=2))
archive=ROOT/'results_v44.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(set(files+[target])):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'source_digest':manifest['source_digest']}))
