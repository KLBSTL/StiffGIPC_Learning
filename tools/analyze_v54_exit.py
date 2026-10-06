"""Same-state exit ablation with explicit first-system and common-prefix checks."""
import json,csv,itertools
from pathlib import Path
import numpy as np
from run_v54_exit import ROOT,read,difference
from analyze_perf_v36 import matrix_from_snapshot
rows=read(ROOT/'reports/V54_EXIT_MATRIX.json');assert len(rows)==10
out={'physical_truth_certified':False,'performance_certified':False,'scenes':[]}
for label,scene in [('fixed','cloth_fixed_bunny_l'),('sphere','cloth_sphere7_l')]:
    group=[r for r in rows if r['scene']==scene];seed=next(r for r in group if r['kind']=='seed');frame=seed['target_frame']
    gate=read(ROOT/'reports'/f'V54_EXIT_GATE_{label}.json');assert gate['passed']
    base=ROOT/'runs/local'/seed['name'];trace=base/'trace';raw=np.fromfile(trace/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;faces=faces[~tet[faces].any(axis=1)]
    cloth=np.zeros(nv,bool);cloth[np.unique(faces)]=True;boundary=np.fromfile(trace/'boundary_types.bin',dtype='<i4')
    objects=read(base/'output/scene.json')['objects']
    assert all(o['body_type']=='ABD' and o['fixed_mode']=='all' for o in objects if o['dimension']==3)
    fixed=(boundary==1)|tet;free=cloth&~fixed
    mass=np.fromfile(trace/'masses.bin',dtype='<f8');x0=np.fromfile(trace/'state_0000.bin',dtype='<f8').reshape(-1,3)
    scale=float(np.linalg.norm(np.ptp(x0[cloth],axis=0)));edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    lens=lambda x:np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
    area=lambda x:np.linalg.norm(np.cross(x[faces[:,1]]-x[faces[:,0]],x[faces[:,2]]-x[faces[:,0]]),axis=1)/2
    rest=lens(x0);restarea=area(x0);summaries=[];states={};frames={}
    _,A,rhs=matrix_from_snapshot(base/f'fixed/f{frame}_n1');A=A.tocsr();A.sum_duplicates()
    for row in group:
        name=row['name'];run=ROOT/'runs/local'/name;req=read(run/'requested.json');seed_req=read(base/'requested.json')
        assert row['result']['status']=='completed'
        for k in ['dt','pcg_tol','tol','preconditioner','suite','mas_cholesky','choose_start','restart_guard','robust_velocity_tol','reduced_slack','ccd_pair_limit','exe_sha256','source_digest','runner_sha256']:
            assert req[k]==seed_req[k],(name,k)
        assert req['inner_exit']==row['inner_exit']
        for file in ['topology.bin','masses.bin','boundary_types.bin']:assert (run/'trace'/file).read_bytes()==(trace/file).read_bytes()
        if row['kind']!='seed':
            assert (run/'trace/state_0000.bin').read_bytes()==(trace/f'state_{frame-1:04d}.bin').read_bytes()
            assert read(run/'output/checkpoint_restore.json')['device_bytes_verified']
        _,B,b=matrix_from_snapshot(run/f'fixed/f{frame}_n1');B=B.tocsr();B.sum_duplicates();delta=A-B
        da={'max_abs':float(np.max(np.abs(delta.data))) if delta.nnz else 0.,'relative_l2':float(np.linalg.norm(delta.data)/max(np.linalg.norm(A.data),1e-30))}
        db=difference(rhs,b)
        assert da['relative_l2']<=1e-10 and da['max_abs']<=1e-10*(1+np.max(np.abs(A.data)))
        assert db['relative_l2']<=1e-10 and db['max_abs']<=1e-10*(1+db['scale'])
        study=read(run/f'fixed/f{frame}_n1_study.json');assert study['system_unchanged'] and study['primary_restored_bitwise']
        x=np.fromfile(run/'final_checkpoint/vertices.bin',dtype='<f8').reshape(-1,3);states[name]=x
        assert np.isfinite(x).all();stretch=lens(x)/rest;ar=area(x)/restarea
        fixed_motion=float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max())
        assert ar.min()>0 and fixed_motion<=1e-12
        allframes=read(run/'output/stats.json')['frames'];f=allframes[-1];frames[name]=f
        exits=[]
        for ff in allframes:
            for i,n in enumerate(ff['newton']):
                if 'pcg' not in n:continue
                assert not n['pcg'].get('iteration_limit') and not n['pcg'].get('breakdown')
                assert n['trial_energy_after']<=n['trial_energy_before']+1.001e-12*max(1,abs(n['trial_energy_before']))
                reason=n.get('inner_exit_reason');velocity=n['trial_newton_axis_velocity_m_s']
                if reason=='velocity_converged':assert velocity<=.05
                elif reason=='full_step':
                    assert req['inner_exit']=='native' and n['line_search_r']==1 and velocity<=5 and i+1>=6
                    if n['restart_full_step_guard_active']:assert n['toi_inner']+1>=6
                else:assert reason is None
        nn=[n for n in f['newton'] if 'pcg' in n]
        for n in nn:
            if n.get('inner_exit_reason'):exits.append({'outer':n['toi_outer'],'inner':n['toi_inner'],'reason':n['inner_exit_reason'],'velocity':n['trial_newton_axis_velocity_m_s']})
        if req['inner_exit']=='velocity_only':assert exits and all(e['reason']=='velocity_converged' for e in exits)
        energy=list(csv.DictReader((run/'trace/physical_energy.csv').open()))[-1]
        energy={k:float(v) for k,v in energy.items()};assert all(np.isfinite(list(energy.values())))
        summaries.append({'run':name,'kind':row['kind'],'repeat':row['repeat'],'A_difference':da,'rhs_difference':db,
            'max_stretch':float(stretch.max()),'min_area_ratio':float(ar.min()),'fixed_motion_m':fixed_motion,'potential':energy,
            'newton':len(nn),'outer':len(f['toi']),'pcg_iterations':sum(n['pcg']['iterations'] for n in nn),'exits':exits,
            'first_outer_final_energy':next(n for n in reversed(nn) if n['toi_outer']==0)['trial_energy_after']})
    pairs=[]
    for a,b in itertools.combinations(summaries,2):
        delta=states[a['run']][free]-states[b['run']][free];rms=float(np.sqrt(np.average(np.sum(delta**2,axis=1),weights=mass[free])))
        pairs.append({'a':a['run'],'b':b['run'],'rms_m':rms,'rms_percent_scale':100*rms/scale,'max_vertex_m':float(np.linalg.norm(delta,axis=1).max())})
    prefixes=[]
    native=next(r for r in summaries if r['kind']=='resume' and r['repeat']==1);nf0=frames[native['run']]
    n0=[n for n in nf0['newton'] if n.get('toi_outer')==0]
    assert n0[-1]['inner_exit_reason'] in ['full_step','velocity_converged']
    for r in summaries:
        if r['kind']!='velocity':continue
        vf=frames[r['run']];v0=[n for n in vf['newton'] if n.get('toi_outer')==0]
        assert len(v0)>=len(n0)
        for k in ['mu','delta']:assert abs(vf['toi'][0][k]-nf0['toi'][0][k])<=1e-12*max(1,abs(nf0['toi'][0][k]))
        errs=[]
        for n,v in zip(n0,v0):
            assert n['toi_inner']==v['toi_inner'] and n['line_search_r']==v['line_search_r']
            for key in ['trial_energy_before','trial_energy_after','trial_newton_axis_velocity_m_s']:
                d=abs(n[key]-v[key]);assert d<=1e-8*max(1,abs(n[key]));errs.append(d)
        prefixes.append({'native':native['run'],'velocity':r['run'],'native_exit':n0[-1]['inner_exit_reason'],'shared_inner_steps':len(n0),'max_logged_difference':max(errs),
            'additional_inner_steps':len(v0)-len(n0),'energy_at_native_stop':v0[len(n0)-1]['trial_energy_after'],
            'energy_after_continuation':v0[-1]['trial_energy_after'],
            'relative_energy_decrease_after_native_stop':1-v0[-1]['trial_energy_after']/v0[len(n0)-1]['trial_energy_after']})
    out['scenes'].append({'scene':scene,'target_frame':frame,'physical_time':frame*.01,'continuation_gate_passed':True,
        'first_system_all_branches_equivalent':True,'runs':summaries,'pairs':pairs,'shared_first_outer':prefixes})
target=ROOT/'reports/V54_EXIT_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False))
for s in out['scenes']:print(json.dumps({'scene':s['scene'],'runs':[{k:r[k] for k in ['run','max_stretch','newton','outer','potential','first_outer_final_energy']} for r in s['runs']],'shared_first_outer':s['shared_first_outer']}))
