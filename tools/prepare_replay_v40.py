import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'manifests/mas_replay_v40.json';bundle=ROOT/'bundles/mas_replay_v40.tar.gz'
assert not manifest.exists() and not bundle.exists()
files=[]
for tree in ['sources/mas_replay_v40','sources/stiff_perf_v37/StiffGIPC']:
    for path in sorted((ROOT/tree).rglob('*')):
        if path.is_file():files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
manifest.write_text(json.dumps({'files':files,'scope':'Frozen v38 kernels with configurable diagnostics only'},indent=2))
with tarfile.open(bundle,'w:gz') as tar:
    for path in [ROOT/f['path'] for f in files if f['path'].startswith('sources/mas_replay_v40/')]+[manifest,ROOT/'tools/benchmark_replay_v40.py']:
        tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps({'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'files':len(files)}))
