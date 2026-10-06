"""Separate source identity; never replace the v31 global manifest."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
files=[]
for tree in ['sources/stiff_base','sources/stiff_robust_port']:
    for p in sorted((root/tree).rglob('*')):
        if p.is_file() and not any(x in {'Output','sorted_mesh','__pycache__','.git'} for x in p.relative_to(root).parts):
            files.append({'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
manifest={'implementation_version':'v32-robust-derived-port','base_commit':'bb2849a7b292099581907937860d96ecfdf42588',
          'files':files,'source_digest':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
manifest['binaries']={}
for tag in ['local-base','local-fused-v32']:
    binary=root/'builds'/tag/'Release/gipc.exe'
    if not binary.is_file():
        raise FileNotFoundError(binary)
    manifest['binaries'][binary.relative_to(root).as_posix()]={
        'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size}
out=root/'manifests/robust_port_v32.json'
out.parent.mkdir(exist_ok=True)
if out.exists():
    previous=json.loads(out.read_text())
    archive=out.parent/('robust_port_'+previous['source_digest']+'.json')
    if not archive.exists():archive.write_bytes(out.read_bytes())
out.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'files':len(files),'source_digest':manifest['source_digest']}))
