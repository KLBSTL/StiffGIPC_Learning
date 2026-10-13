"""Check downloaded cloud frame accounting, sealed sources, inputs and mesh exports."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

REMOTE=Path('/root/autodl-tmp/codex_recording105_20261013/payload')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main(root,receipt_name):
    root=root.resolve()
    out=root/'reports/autodl_recording105_20261013_v2'
    def local(path):
        rel=Path(path).relative_to(REMOTE)
        return root/rel
    def verify_record(r):
        p=local(r['path'])
        assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
    summary=read(out/'summary.json')
    assert summary['complete'] and summary['selected_scenes']==['Bunny','Mixed','Boxes']
    for name,value in summary['output_sha256'].items():
        assert sha(out/name)==value,name
    frames=rows(out/'per_frame.csv')
    f100=rows(out/'frame100_actual.csv')
    assert len(frames)==3780 and len(f100)==36
    assert [f for f in frames if int(f['physical_frame'])==100]==f100
    counts=Counter(f['run'] for f in frames)
    assert len(counts)==36 and set(counts.values())=={105}
    indices=Counter((r['run'],int(r['physical_frame'])) for r in frames)
    assert len(indices)==3780 and set(indices.values())=={1}
    directions,iterations,records=Counter(),Counter(),Counter()
    for n in rows(out/'per_newton.csv'):
        k=n['run'],int(n['physical_frame'])
        records[k]+=1
        if n['P']:
            directions[k]+=1
            iterations[k]+=int(n['P'])
    for f in frames:
        k=f['run'],int(f['physical_frame'])
        assert 1<=k[1]<=105 and k[1]==int(f['local_frame'])
        assert float(f['dt'])==.04 and np.isfinite(float(f['solver_ms']))
        assert directions[k]==int(f['N']) and iterations[k]==int(f['P']) and records[k]==int(f['newton_records'])
    for r in rows(out/'phase_table_first100.csv'):
        selected=[[f for f in frames if f['scene']==r['scene'] and f['method']==r['method']
                   and int(f['repeat'])==repeat and int(f['physical_frame'])<=100] for repeat in (1,2,3)]
        assert all(len(group)==100 for group in selected)
        for key,field in [('TimeTot','solver_ms'),('assembly','assembly_ms'),('pcg','pcg_ms'),
                          ('ccd','ccd_ms'),('line_search','line_search_ms')]:
            actual=median(sum(float(f[field]) for f in group)/1000 for group in selected)
            assert abs(actual-float(r[key]))<1e-8,(r['scene'],r['method'],key)
        assert float(r['N'])==median(sum(int(f['N']) for f in group) for group in selected)
        assert float(r['P'])==median(sum(int(f['P']) for f in group) for group in selected)
    for r in rows(out/'trajectory_errors.csv'):
        if int(r['frame'])==0:
            assert all(float(r[k])==0 for k in ['position_rms_mm','position_max_mm','velocity_rms_m_per_s','velocity_max_m_per_s'])
    gate=read(out/'checkpoint_gate.json')
    assert gate['passed'] and gate['exact_input_runs']==63
    assert all(all(r['checks'].values()) for r in gate['runs'])
    initial=read(out/'trajectory_input_gate.json')
    assert initial['passed'] and initial['exact_initial_runs']==36
    visual=read(out/'visuals/manifest.json')
    assert visual['renderer']['depth_order_smoke_passed'] and len(visual['source_vertices'])==162
    for item in visual['source_vertices']:
        state=local(item['state'])
        topology=state.parent/'topology.bin'
        obj=out/'visuals'/item['obj']
        assert sha(state)==item['state_sha256'] and sha(topology)==item['topology_sha256']
        assert sha(obj)==item['obj_sha256']
        vertices,faces=[],[]
        with obj.open(encoding='ascii') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:]])
                elif line.startswith('f '):
                    faces.append([int(x)-1 for x in line.split()[1:]])
        raw=np.fromfile(topology,dtype='<u4')
        nv,nf,_=map(int,raw[:3])
        assert np.array_equal(np.asarray(vertices),np.fromfile(state,dtype='<f8').reshape(nv,3))
        assert np.array_equal(np.asarray(faces),raw[3:3+nf*3].reshape(nf,3))
    packet=read(root/'runs/autodl_recording105_20261013_v2/identities.json')
    program=packet['programs']['active']
    for r in program['sources']+program['build_evidence']+[program['exe']]:
        verify_record(r)
    for r in [packet['controller'],summary['identity'],summary['analysis_controller'],visual['report_controller']]:
        verify_record(r)
    original=read(root/'LOCAL_SOURCE_COMPARISON.json')
    for r in original['files']:
        assert sha(root/r['path'])==r['sha256'],r['path']
    repo=Path(__file__).resolve().parents[2]
    local_session=repo/'runs/local_recording105_dt0p04_20261011'
    cloud_session=root/'runs/autodl_recording105_20261013_v2'
    platform_initial=[]
    for scene in (('Bunny','Mixed','Boxes') if local_session.is_dir() else ()):
        name=f'performance_{scene.lower()}_original_r1'
        a=local_session/name/'trace'
        b=cloud_session/name/'trace'
        equal={name:sha(a/name)==sha(b/name) for name in
               ('topology.bin','masses.bin','boundary_types.bin','body_ids.bin','state_0000.bin','velocity_0000.bin')}
        x=np.fromfile(a/'state_0000.bin',dtype='<f8').reshape(-1,3)
        y=np.fromfile(b/'state_0000.bin',dtype='<f8').reshape(-1,3)
        assert x.shape==y.shape
        diff=np.linalg.norm(x-y,axis=1)
        platform_initial.append(dict(scene=scene,bitwise_equal=equal,
            position_rms_mm=1000*float(np.sqrt(np.mean(diff**2))),position_max_mm=1000*float(diff.max())))
    result=dict(passed=True,actual_frames=3780,frame100_rows=36,checkpoint_inputs=63,initial_inputs=36,
        lossless_obj_meshes=162,local_native_and_input_files_identical=355,
        native_translation_units=program['translation_units'],
        native_binary_sha256=program['exe']['sha256'],
        phase_rows_independently_recomputed=9,cross_platform_initial_inputs=platform_initial,
        system_compilers='Hashes retained in cloud identity; system compiler binaries are outside the downloaded payload',
        verifier_sha256=sha(Path(__file__)))
    target=out/receipt_name
    assert not target.exists(),'Keep prior verification receipts'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--receipt-name',default='delivery_verification.json',choices=['delivery_verification.json','delivery_verification_local.json'])
    args=p.parse_args()
    main(args.root,args.receipt_name)
