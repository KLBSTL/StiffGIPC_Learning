import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
excluded={'.git','Output','sorted_mesh','__pycache__','build','builds','runs'}
selected=[]
for tree in ['sources/stiff_base','sources/stiff_fused','tools','configs','references/tight_inclusion','references/graph_v3','references/paper_al','references/robust_local']:
    for p in sorted((ROOT/tree).rglob('*')):
        if p.is_file() and not any(x in excluded for x in p.relative_to(ROOT).parts):selected.append(p)
for name in ['PLAN.md','REPLAN_20260930.md','TODO.md','STATUS.md','FEATURES.md','SOURCE_LOCK.json','IMPLEMENTATION_MANIFEST.json']:
    selected.append(ROOT/name)
selected.extend(sorted((ROOT/'research').glob('*.json')))
manifest={'schema':'stiff-toi-upload-v1','files':[{'path':p.relative_to(ROOT).as_posix(),
    'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in selected]}
source_manifest=json.loads((ROOT/'IMPLEMENTATION_MANIFEST.json').read_text())
uploaded_hashes={x['path']:x['sha256'] for x in manifest['files']}
changed=[x['path'] for x in source_manifest['files'] if uploaded_hashes.get(x['path'])!=x['sha256']]
if changed:raise RuntimeError('Refresh source_inventory.py before packaging: '+', '.join(changed[:5]))
manifest_path=ROOT/'bundles/upload_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=ROOT/'bundles/stiff_toi_cudagraph_source.tar.gz'
if archive.exists():
    import shutil
    previous_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    frozen=ROOT/'bundles/frozen';frozen.mkdir(exist_ok=True)
    previous=frozen/(previous_sha+'.tar.gz')
    if not previous.exists():shutil.copyfile(archive,previous)
with tarfile.open(archive,'w:gz',compresslevel=6) as tf:
    for p in selected:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    tf.add(manifest_path,arcname='upload_manifest.json')
print(json.dumps({'archive':str(archive),'files':len(selected),'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
