"""Shared physical-time refinement comparisons; finest grid is not ground truth."""
import json,copy,os,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((ROOT/p).read_text())
load=lambda name,i:np.fromfile(ROOT/f'runs/local/{name}/trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
out={'physical_seconds':.5,'physical_truth_certified':False,'performance_certified':False,'scenes':[]}
reports={d:read('reports/V54_CLOTH_QUALITY.json' if d==1 else f'reports/V54_DT{d}_QUALITY.json') for d in [1,2,4]}
for scene in ['cloth_sphere7_l','cloth_fixed_bunny_l']:
    runs={}
    for div,report in reports.items():
        s=next(s for s in report['scenes'] if s['scene']==scene)
        for r in s['runs']:
            if r['variant'] in ['off','guard']:runs[(div,r['variant'],r['repeat'])]=r
    assert len(runs)==12
    ref=runs[(1,'off',1)]['name'];base=ROOT/f'runs/local/{ref}/trace'
    raw=np.fromfile(base/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;cf=faces[~tet[faces].any(axis=1)];cloth=np.zeros(nv,bool);cloth[np.unique(cf)]=True
    boundary=np.fromfile(base/'boundary_types.bin',dtype='<i4');free=cloth&(boundary!=1);mass=np.fromfile(base/'masses.bin',dtype='<f8')
    x0=load(ref,0);scale=float(np.linalg.norm(np.ptp(x0[cloth],axis=0)))
    scene_ref=read(f'runs/local/{ref}/output/scene.json');scene_ref['effective_run'].pop('dt')
    req_ref=read(f'runs/local/{ref}/requested.json')
    summaries=[]
    for (div,variant,repeat),r in runs.items():
        name=r['name'];z=ROOT/f'runs/local/{name}/trace';req=read(f'runs/local/{name}/requested.json')
        actual=read(f'runs/local/{name}/output/scene.json');assert actual['effective_run'].pop('dt')==.01/div
        assert actual==scene_ref,(name,'material/scene')
        for file in ['topology.bin','state_0000.bin','masses.bin','boundary_types.bin']:assert (base/file).read_bytes()==(z/file).read_bytes(),(name,file)
        for k in ['exe_sha256','source_digest','runner_sha256','pcg_tol','tol','preconditioner','suite','mas_cholesky','inner_exit','robust_velocity_tol','reduced_slack','ccd_pair_limit','execution','backend']:
            assert req[k]==req_ref[k],(name,k)
        assert req['dt']==.01/div and r['native_audit_passed'] and r['pcg_limit_hits']==0
        obs=r['observations'][:50*div];assert len(obs)==50*div
        peak=max(obs,key=lambda o:o['max_stretch']);common=[obs[i*div-1] for i in range(1,51)]
        energy={int(e['frame']):{k:float(v) for k,v in e.items() if k!='frame'} for e in r['physical_energy']}
        common_energy=[{'time':i*.01,**energy[i*div]} for i in range(1,51)]
        summaries.append({'run':name,'div':div,'variant':variant,'repeat':repeat,'dt':.01/div,
            'max_stretch_all_endpoints':peak['max_stretch'],'peak_time':peak['frame']*.01/div,
            'max_stretch_common_times':max(o['max_stretch'] for o in common),
            'min_area_ratio':min(o['min_area_ratio'] for o in obs),'fixed_motion':max(o['fixed_motion_m'] for o in obs),
            'common_stretch':[{'time':i*.01,'max_stretch':o['max_stretch']} for i,o in enumerate(common,1)],
            'common_energy':common_energy,'diagnostic_solver_seconds':r['result']['solver_seconds'],
            'timing_scope':'full original100 frames' if div==1 else 'new refinement through .5s'})
    def compare(left,right):
        dl,vl,rl=left;dr,vr,rr=right;ln=runs[left]['name'];rn=runs[right]['name'];series=[]
        for i in range(1,51):
            delta=load(ln,i*dl)[free]-load(rn,i*dr)[free]
            rms=float(np.sqrt(np.average(np.sum(delta**2,axis=1),weights=mass[free])))
            series.append({'time':i*.01,'rms_percent_scale':100*rms/scale,'max_vertex_m':float(np.linalg.norm(delta,axis=1).max())})
        return {'a':ln,'b':rn,'max_rms_percent_scale':max(x['rms_percent_scale'] for x in series),'final_rms_percent_scale':series[-1]['rms_percent_scale'],'samples':series}
    trends=[]
    for v in ['off','guard']:
        for rep in [1,2]:
            coarse=compare((1,v,rep),(2,v,rep));fine=compare((2,v,rep),(4,v,rep))
            trends.append({'variant':v,'repeat':rep,'h_h2':coarse,'h2_h4':fine,
                'max_difference_shrinks':fine['max_rms_percent_scale']<coarse['max_rms_percent_scale'],
                'max_difference_ratio':fine['max_rms_percent_scale']/coarse['max_rms_percent_scale']})
    repeat_pairs=[{'div':d,'variant':v,**compare((d,v,1),(d,v,2))} for d in [1,2,4] for v in ['off','guard']]
    switch_pairs=[{'div':d,'repeat':rep,**compare((d,'off',rep),(d,'guard',rep))} for d in [1,2,4] for rep in [1,2]]
    out['scenes'].append({'scene':scene,'same_material_initial_state':True,'cloth_scale_m':scale,'runs':summaries,'trends':trends,'repeat_pairs':repeat_pairs,'switch_pairs':switch_pairs})
target=ROOT/'reports/V54_DT_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False))
for s in out['scenes']:
    print(json.dumps({'scene':s['scene'],'runs':[{k:r[k] for k in ['run','max_stretch_all_endpoints','peak_time']} for r in s['runs']],
        'trends':[{k:t[k] for k in ['variant','repeat','max_difference_shrinks','max_difference_ratio']} for t in s['trends']],
        'repeat_rms':[{k:p[k] for k in ['div','variant','max_rms_percent_scale']} for p in s['repeat_pairs']]}))
