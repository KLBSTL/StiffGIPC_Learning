"""Freeze established benchmark inputs without modifying their source folder."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'stiffGIPC'
bench=SOURCE/'benchmarks/stiff4-v1'
families=['cloth_hang','cloth_sphere7','cloth_fixed_bunny','cloth_table','cloth_stack10','bunny_cloth_bunny','boxes1920_cloth']
manifest={'source':str(SOURCE),'files':[],'cases':[]}
mesh_paths=set()
for family in families:
    for resolution in ['l','m','h']:
        case=f'{family}_{resolution}'
        objects=json.loads((bench/'scene_object_manifests'/f'{case}.json').read_text())
        scene=json.loads((bench/'scene_manifests'/f'{case}.json').read_text())
        normalized={'case_id':case,'objects':objects['objects'],
                    'effective_scalar_fields':scene['effective_scalar_fields'],
                    'initial_counts':scene['initial_counts'],'case_spec':scene['case_spec']}
        for obj in normalized['objects']:
            relative=obj['stiff_mesh'];data=(SOURCE/'Assets'/relative).read_bytes()
            actual=hashlib.sha256(data).hexdigest()
            expected=obj.get('stiff_mesh_sha256')
            if expected and expected != actual:raise RuntimeError(f'Frozen mesh hash mismatch: {relative}')
            obj['frozen_sha256']=actual;obj['stiff_mesh']='benchmark_meshes/'+relative
            mesh_paths.add(relative)
        for tree in ['stiff_base','stiff_fused']:
            target=ROOT/'sources'/tree/'Assets/benchmark_scenes'/f'{case}.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(normalized,indent=2),encoding='utf-8')
        manifest['cases'].append({'case_id':case,'initial_counts':scene['initial_counts'],'resolution':scene['case_spec'].get('resolution_spec')})
for relative in sorted(mesh_paths):
    src=SOURCE/'Assets'/relative;data=src.read_bytes();sha=hashlib.sha256(data).hexdigest()
    for tree in ['stiff_base','stiff_fused']:
        target=ROOT/'sources'/tree/'Assets/benchmark_meshes'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if hashlib.sha256(src.read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Input changed during snapshot: {relative}')
    manifest['files'].append({'path':relative,'sha256':sha,'bytes':len(data)})
(ROOT/'research/benchmark_scene_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'frozen_cases':len(manifest['cases']),'meshes':len(mesh_paths),
                  'mesh_bytes':sum(x['bytes'] for x in manifest['files'])}))
