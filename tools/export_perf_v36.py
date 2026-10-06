"""Archive v36 outputs and identities only; exclude caches and source copies."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'autodl_perf_v36_20261003_results.tar.gz'
assert not target.exists()
paths=[]
for folder in ['runs/autodl','reports','manifests']:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
paths.extend(p for p in (ROOT/'builds').glob('*') if p.is_file())
paths.extend([ROOT/'builds/autodl-stiff_perf_v36/gipc',ROOT/'builds/autodl-stiff_base/gipc'])
paths.extend(p for p in (ROOT/'tools').glob('*v36*') if p.is_file())
inventory=[]
for p in paths:
    inventory.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
index=ROOT/'reports/AUTODL_PERF_V36_EXPORT_FILES.json'
index.write_text(json.dumps(inventory,indent=2));paths.append(index)
with tarfile.open(target,'w:gz',compresslevel=3) as archive:
    for p in paths:archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(paths)}))
