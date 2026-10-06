import hashlib,json,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1];excluded={'Output','sorted_mesh','__pycache__','.git'}
p=argparse.ArgumentParser();p.add_argument('--version',default='v5-unrestricted-trial');a=p.parse_args()
previous=root/'IMPLEMENTATION_MANIFEST.json'
if previous.exists():
    archive=root/'manifests';archive.mkdir(exist_ok=True)
    old=json.loads(previous.read_text())
    saved=archive/(old['source_digest']+'.json')
    if not saved.exists():saved.write_bytes(previous.read_bytes())
manifest={'implementation_version':a.version,'base_commit':'bb2849a7b292099581907937860d96ecfdf42588','files':[]}
for tree in ['sources/stiff_base','sources/stiff_fused']:
    for p in sorted((root/tree).rglob('*')):
        if p.is_file() and not any(x in excluded for x in p.relative_to(root).parts):
            manifest['files'].append({'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
manifest['source_digest']=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()
(root/'IMPLEMENTATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'files':len(manifest['files']),'source_digest':manifest['source_digest']}))
