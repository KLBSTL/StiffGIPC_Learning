"""Cloth-specific paired observables and native initial-guess/exit audits."""
import argparse,collections,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
matrix=read(ROOT/'reports/V53_CLOTH_MATRIX.json');grouped=collections.defaultdict(list)
for row in matrix:grouped[row['scene']].append(row)
report={'scenes':[],'physical_truth_certified':False,'performance_certified':False,
        'scope':'Same-executable flag regression, two repetitions per arm. Differences are not errors against physical ground truth.'}
for scene,rows in grouped.items():
    runs=[];ref=ROOT/'runs/local'/rows[0]['name'];d=ref/'trace'
    raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
    faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True;cloth_faces=faces[~in_tet[faces].any(axis=1)]
    cloth=np.zeros(nv,bool);cloth[np.unique(cloth_faces)]=True
    boundary=np.fromfile(d/'boundary_types.bin',dtype='<i4');fixed=boundary==1;free=cloth & ~fixed
    mass=np.fromfile(d/'masses.bin',dtype='<f8');x0=np.fromfile(d/'state_0000.bin',dtype='<f8').reshape(-1,3)
    scale=float(np.linalg.norm(np.ptp(x0[cloth],axis=0)))
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    restlen=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
    def area(x):
        q=x[cloth_faces];return np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2
    restarea=area(x0);assert (restarea>0).all() and (restlen>0).all() and free.any()
    for row in rows:
        run=ROOT/'runs/local'/row['name'];z=run/'trace';req=read(run/'requested.json');stats=read(run/'output/stats.json')['frames']
        base_req=read(ref/'requested.json')
        for k in ['scene','dt','pcg_tol','tol','preconditioner','suite','mas_cholesky','inner_exit','robust_velocity_tol','reduced_slack','ccd_pair_limit','execution','backend']:
            assert req[k]==base_req[k],(run.name,k)
        assert read(run/'output/scene.json')==read(ref/'output/scene.json')
        for file in ['topology.bin','state_0000.bin','masses.bin','boundary_types.bin','metadata.json']:
            assert (z/file).read_bytes()==(d/file).read_bytes(),(run.name,file)
        assert req['choose_start']==row['choose_start'] and req['runner_sha256']==sha(ROOT/'tools/run_perf_v53.py')
        selections=[];directions=[];exits=0;tol=req['robust_velocity_tol']
        for frame in stats:
            assert frame['toi_reduced_slack'] is False
            for outer in frame.get('toi',[]):
                s=outer.get('initial_guess_selection')
                if s:
                    assert req['choose_start']=='1' and s['full_objective']
                    assert np.isfinite([s['warm_energy'],s['safe_energy']]).all()
                    assert s['selected']==('safe' if s['safe_energy']<s['warm_energy'] else 'warm')
                    selections.append(s)
            for i,n in enumerate(frame['newton']):
                if 'pcg' not in n:continue
                assert n['trial_energy_after']<=n['trial_energy_before']+1.001e-12*max(1,abs(n['trial_energy_before']))
                reason=n.get('inner_exit_reason');velocity=n['trial_newton_axis_velocity_m_s']
                if reason=='full_step':assert n['line_search_r']==1 and velocity<=100*tol and i+1>=6
                elif reason=='velocity_converged':assert velocity<=tol
                elif reason is not None:raise AssertionError(reason)
                exits+=reason is not None;directions.append(n)
        obs=[]
        for i in range(1,row['result']['recorded_frames']+1):
            x=np.fromfile(z/f'state_{i:04d}.bin',dtype='<f8').reshape(-1,3);assert np.isfinite(x).all()
            stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/restlen;ratio=area(x)/restarea
            obs.append({'frame':i,'max_stretch':float(stretch.max()),'p95_stretch':float(np.percentile(stretch,95)),
                'min_area_ratio':float(ratio.min()),'max_area_ratio':float(ratio.max()),
                'fixed_motion_m':float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max()) if fixed.any() else 0.})
        energy=list(csv.DictReader((z/'physical_energy.csv').open()))
        for r in energy:assert all(np.isfinite(float(v)) for v in r.values())
        runs.append({'name':row['name'],'switch':req['choose_start'],'repeat':row['repeat'],'result':row['result'],
            'native_audit_passed':True,'initial_guess_comparisons':len(selections),
            'safe_selections':sum(s['selected']=='safe' for s in selections),'native_exits':exits,
            'pcg_calls':len(directions),'pcg_iterations':sum(n['pcg']['iterations'] for n in directions),
            'pcg_limit_hits':sum(bool(n['pcg'].get('iteration_limit')) for n in directions),
            'max_stretch':max(x['max_stretch'] for x in obs),'max_p95_stretch':max(x['p95_stretch'] for x in obs),
            'min_area_ratio':min(x['min_area_ratio'] for x in obs),'max_fixed_motion_m':max(x['fixed_motion_m'] for x in obs),
            'observations':obs,'physical_energy':energy})
    pairs=[]
    for left,right in itertools.combinations(runs,2):
        series=[]
        for i in range(1,min(left['result']['recorded_frames'],right['result']['recorded_frames'])+1):
            load=lambda name:np.fromfile(ROOT/f'runs/local/{name}/trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
            delta=load(left['name'])[free]-load(right['name'])[free]
            rms=float(np.sqrt(np.average(np.sum(delta**2,axis=1),weights=mass[free])))
            series.append({'frame':i,'cloth_rms_m':rms,'cloth_rms_percent_scale':100*rms/scale,'max_vertex_m':float(np.linalg.norm(delta,axis=1).max())})
        pairs.append({'a':left['name'],'b':right['name'],'same_switch':left['switch']==right['switch'],
            'same_repeat':left['repeat']==right['repeat'],'frames':series,
            'max_cloth_rms_percent_scale':max(x['cloth_rms_percent_scale'] for x in series),
            'max_vertex_m':max(x['max_vertex_m'] for x in series)})
    report['scenes'].append({'scene':scene,'dt':rows[0]['dt'],'preconditioner':rows[0]['preconditioner'],
        'cloth_scale_m':scale,'fixed_vertex_count':int(fixed.sum()),'same_scene_and_initial_state':True,
        'friction_compiled':read(ref/'output/scene.json')['effective_scalar_fields']['friction_compiled'],
        'runs':runs,'pairs':pairs})
a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
for s in report['scenes']:
    print(json.dumps({'scene':s['scene'],'runs':[{k:r[k] for k in ['name','max_stretch','max_p95_stretch','min_area_ratio','max_fixed_motion_m','safe_selections','initial_guess_comparisons','pcg_limit_hits']} for r in s['runs']],
        'pairs':[{k:v for k,v in r.items() if k!='frames'} for r in s['pairs']]}),flush=True)
