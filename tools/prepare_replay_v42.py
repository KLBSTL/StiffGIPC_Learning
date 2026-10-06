import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'manifests/mas_replay_v42a.json';bundle=ROOT/'bundles/mas_replay_v42a.tar.gz'
assert not manifest.exists() and not bundle.exists();files=[]
for tree in ['sources/mas_replay_v42','sources/stiff_perf_v37/StiffGIPC']:
    for p in sorted((ROOT/tree).rglob('*')):
        if p.is_file():files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
runner=ROOT/'tools/benchmark_replay_v42.py'
manifest.write_text(json.dumps({'files':files,'scope':'v42 symmetric Cholesky and deterministic restriction, frozen v40 controls',
                              'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest()},indent=2))
with tarfile.open(bundle,'w:gz') as tar:
    for path in [ROOT/f['path'] for f in files if f['path'].startswith('sources/mas_replay_v42/')]+[manifest,runner]:
        tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps({'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'files':len(files)}))
