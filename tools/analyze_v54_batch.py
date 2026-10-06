"""Window propagation, geometry and native stopping checks; no accuracy claim."""
import json,csv,itertools,argparse
from pathlib import Path
import numpy as np
from run_v54_exit import ROOT,read,difference
from analyze_perf_v36 import matrix_from_snapshot
p=argparse.ArgumentParser();p.add_argument('--perturb',action='store_true');args=p.parse_args()
prefix='V54_PERTURB' if args.perturb else 'V54_BATCH'
rows=read(ROOT/f'reports/{prefix}_MATRIX.json');out={'physical_truth_certified':False,'performance_certified':False,'comparison_groups':'initial state origin' if args.perturb else 'continuous exit rule','runs':[],'fixed_pairs':[]}
for r in rows:
    name=r['name'];run=ROOT/'runs/local'/name;trace=run/'trace';req=read(run/'requested.json');res=r['result'];assert res['status']=='completed'
    origin=ROOT/'runs/local'/('v54_fixed_bunny_guard_r1' if 'bunny' in r['scene'] else 'v54_sphere_guard_r1')
    base=origin/'trace';base_req=read(origin/'requested.json')
    for k in ['dt','pcg_tol','tol','preconditioner','suite','mas_cholesky','choose_start','restart_guard','robust_velocity_tol','reduced_slack','ccd_pair_limit','exe_sha256','source_digest','runner_sha256']:
        assert req[k]==base_req[k],(name,k)
    assert req['inner_exit']==r['inner_exit']
    for file in ['topology.bin','masses.bin','boundary_types.bin']:assert (trace/file).read_bytes()==(base/file).read_bytes()
    raw=np.fromfile(base/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
    tet=np.zeros(nv,bool);tet[tets.ravel()]=True;cf=faces[~tet[faces].any(axis=1)];cloth=np.zeros(nv,bool);cloth[np.unique(cf)]=True
    boundary=np.fromfile(base/'boundary_types.bin',dtype='<i4');fixed=tet|(boundary==1);free=cloth&~fixed
    scene=read(run/'output/scene.json');assert all(o['fixed_mode']=='all' for o in scene['objects'] if o['dimension']==3)
    x0=np.fromfile(base/'state_0000.bin',dtype='<f8').reshape(-1,3);mass=np.fromfile(base/'masses.bin',dtype='<f8');scale=float(np.linalg.norm(np.ptp(x0[cloth],axis=0)))
    edges=np.unique(np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1),axis=0)
    lens=lambda x:np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
    area=lambda x:np.linalg.norm(np.cross(x[cf[:,1]]-x[cf[:,0]],x[cf[:,2]]-x[cf[:,0]]),axis=1)/2
    rest=lens(x0);restarea=area(x0);observations=[];offset=res['resume_completed_physical_frames']
    if req['checkpoint_load']:
        assert read(run/'output/checkpoint_restore.json')['device_bytes_verified']
        cp=Path(req['checkpoint_load']);assert (trace/'state_0000.bin').read_bytes()==(cp/'vertices.bin').read_bytes()
    for i in range(1,res['recorded_frames']+1):
        x=np.fromfile(trace/f'state_{i:04d}.bin',dtype='<f8').reshape(-1,3);assert np.isfinite(x).all();stretch=lens(x)/rest;ar=area(x)/restarea
        fixed_motion=float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max()) if fixed.any() else 0.;assert fixed_motion<=1e-12 and ar.min()>0
        observations.append({'physical_frame':offset+i,'max_stretch':float(stretch.max()),'min_area_ratio':float(ar.min()),'fixed_motion_m':fixed_motion})
    frames=read(run/'output/stats.json')['frames'];calls=[];exits=[]
    for fidx,f in enumerate(frames):
        for i,n in enumerate(f['newton']):
            if 'pcg' not in n:continue
            assert not n['pcg'].get('iteration_limit') and not n['pcg'].get('breakdown');calls.append(n['pcg'])
            assert n['trial_energy_after']<=n['trial_energy_before']+1.001e-12*max(1,abs(n['trial_energy_before']))
            reason=n.get('inner_exit_reason');v=n['trial_newton_axis_velocity_m_s']
            if reason=='velocity_converged':assert v<=.05
            elif reason=='full_step':
                assert req['inner_exit']=='native' and n['line_search_r']==1 and v<=5 and i+1>=6
                if n['restart_full_step_guard_active']:assert n['toi_inner']+1>=6
            else:assert reason is None
            if reason:exits.append({'frame':offset+fidx+1,'reason':reason,'velocity':v})
    energy=[{k:float(v) for k,v in e.items()} for e in csv.DictReader((trace/'physical_energy.csv').open())]
    assert all(np.isfinite(list(e.values())).all() for e in energy)
    out['runs'].append({'run':name,'scene':r['scene'],'kind':r['kind'],'mode':r['inner_exit'],'origin':r.get('origin'),'repeat':r['repeat'],'result':res,
        'newton':len(calls),'outer':sum(len(f['toi']) for f in frames),'pcg_iterations':sum(n['iterations'] for n in calls),
        'max_stretch':max(o['max_stretch'] for o in observations),'observations':observations,'energy':energy,'exits':exits})
    if r['scene']=='cloth_fixed_bunny_l':fixed_topology=(free,mass,scale)
fixed_runs=[r for r in out['runs'] if r['scene']=='cloth_fixed_bunny_l'];free,mass,scale=fixed_topology
for a,b in itertools.combinations(fixed_runs,2):
    series=[]
    count=min(a['result']['recorded_frames'],b['result']['recorded_frames']);offset=a['result']['resume_completed_physical_frames']
    assert offset==b['result']['resume_completed_physical_frames']
    for i in range(count+1):
        def x(name):return np.fromfile(ROOT/f'runs/local/{name}/trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
        d=x(a['run'])[free]-x(b['run'])[free];rms=float(np.sqrt(np.average(np.sum(d*d,axis=1),weights=mass[free])))
        series.append({'physical_frame':offset+i,'rms_m':rms,'rms_percent_scale':rms/scale*100,'max_vertex_m':float(np.linalg.norm(d,axis=1).max())})
    out['fixed_pairs'].append({'a':a['run'],'b':b['run'],'same_mode':a['origin']==b['origin'] if args.perturb else a['mode']==b['mode'],'same_repeat':a['repeat']==b['repeat'],'series':series,
        'max_rms_percent_scale':max(s['rms_percent_scale'] for s in series),'max_vertex_m':max(s['max_vertex_m'] for s in series)})
# First-system equivalence for fixed-window branches to the already verified frame24 seed.
systems=[]
for r in fixed_runs:
    frame=25 if args.perturb else 24
    refname=next(x['run'] for x in fixed_runs if x['origin']==r['origin'] and x['repeat']==1) if args.perturb else 'v54_exit_fixed_seed_r1'
    _,A,rhs=matrix_from_snapshot(ROOT/f'runs/local/{refname}/fixed/f{frame}_n1');A=A.tocsr();A.sum_duplicates()
    _,B,b=matrix_from_snapshot(ROOT/f"runs/local/{r['run']}/fixed/f{frame}_n1");B=B.tocsr();B.sum_duplicates();delta=A-B
    rel=float(np.linalg.norm(delta.data)/np.linalg.norm(A.data));db=difference(rhs,b)
    assert rel<=1e-10 and db['relative_l2']<=1e-10
    assert (float(np.max(np.abs(delta.data))) if delta.nnz else 0.)<=1e-10*(1+np.max(np.abs(A.data)))
    assert db['max_abs']<=1e-10*(1+db['scale'])
    study=read(ROOT/f"runs/local/{r['run']}/fixed/f{frame}_n1_study.json");assert study['system_unchanged'] and study['primary_restored_bitwise']
    systems.append({'run':r['run'],'reference':refname,'A_relative':rel,'rhs_relative':db['relative_l2']})
out['fixed_first_systems']=systems;out['sphere_gate']=read(ROOT/'reports/V54_BATCH_SPHERE_GATE.json')
target=ROOT/f'reports/{prefix}_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'runs':[{k:r[k] for k in ['run','newton','outer','pcg_iterations','max_stretch']} for r in out['runs']],
    'pairs':[{k:v for k,v in p.items() if k!='series'} for p in out['fixed_pairs']]}))
