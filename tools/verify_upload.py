import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'upload_manifest.json').read_text())
bad=[]
for item in manifest['files']:
    path=(root/item['path']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Manifest path outside task')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:bad.append(item['path'])
print(json.dumps({'checked_files':len(manifest['files']),'mismatches':bad}))
if bad:raise SystemExit(1)
