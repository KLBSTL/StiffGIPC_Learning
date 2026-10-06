"""Check complete-objective initial guesses and native exits; observe quality."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
run=ROOT/'runs/local'/a.run;target=ROOT/'reports'/f'AUDIT_{a.run}.json';assert not target.exists()
req=json.loads((run/'requested.json').read_text());result=json.loads((run/'result.json').read_text())
frames=json.loads((run/'output/stats.json').read_text())['frames']
tol=req['robust_velocity_tol'];selections=[];exits=0;directions=[]
assert req['choose_start']=='1' and req['reduced_slack']=='0'
for f in frames:
    assert f['toi_reduced_slack'] is False
    for outer in f.get('toi',[]):
        s=outer.get('initial_guess_selection')
        if s:
            assert s['full_objective']
            warm,safe=s['warm_energy'],s['safe_energy']
            assert np.isfinite(warm) and np.isfinite(safe)
            assert s['selected']==('safe' if safe<warm else 'warm')
            selections.append(s)
    for index,n in enumerate(f['newton']):
        if 'pcg' not in n:continue
        velocity=n['trial_newton_axis_velocity_m_s'];r=n['line_search_r']
        assert n['trial_energy_after']<=n['trial_energy_before']+1.001e-12*max(1,abs(n['trial_energy_before']))
        directions.append(n)
        reason=n.get('inner_exit_reason')
        if reason=='full_step':
            assert r==1 and velocity<=100*tol and index+1>=6
        elif reason=='velocity_converged':assert velocity<=tol
        elif reason is not None:raise AssertionError(reason)
        exits+=reason is not None
d=run/'trace';raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
tets=raw[3+3*nf:].reshape(-1,4);faces=raw[3:3+3*nf].reshape(-1,3)
abd=json.loads((d/'metadata.json').read_text())['abd_point_num'];fem=tets[np.all(tets>=abd,axis=1)]
x0=np.fromfile(d/'state_0000.bin',dtype='<f8').reshape(-1,3)
def det(x):
    q=x[fem];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
rest=det(x0);assert np.all(rest!=0)
in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True;cloth=faces[~in_tet[faces].any(axis=1)]
edges=np.unique(np.sort(np.concatenate([cloth[:,[0,1]],cloth[:,[1,2]],cloth[:,[2,0]]]),axis=1),axis=0)
length=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
observations=[]
for path in sorted(d.glob('state_*.bin'))[1:]:
    x=np.fromfile(path,dtype='<f8').reshape(-1,3);assert np.isfinite(x).all()
    J=det(x)/rest;stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/length
    observations.append({'frame':int(path.stem.split('_')[1]),'fem_min_J':float(J.min()),
        'nonpositive_fem_tets':int(np.sum(J<=0)),'cloth_max_stretch':float(stretch.max())})
report={'run':a.run,'result':result,'native_exit_and_initial_guess_checks_passed':True,
    'directions':len(directions),
    'max_newton_records_per_completed_frame':max(len(f['newton']) for f in frames[:result['recorded_frames']]),
    'initial_guess_comparisons':len(selections),
    'safe_initial_guesses':sum(s['selected']=='safe' for s in selections),
    'max_initial_energy_reduction':max((s['warm_energy']-s['safe_energy'] for s in selections if s['selected']=='safe'),default=0),
    'native_inner_exits':exits,
    'pcg_limit_hits':sum(bool(n['pcg'].get('iteration_limit')) for n in directions),
    'pcg_max_iterations':max(n['pcg']['iterations'] for n in directions),
    'completed_frames':len(observations),'endpoint_observations':observations,
    'worst_fem_J':min(x['fem_min_J'] for x in observations),
    'max_nonpositive_tets_per_frame':max(x['nonpositive_fem_tets'] for x in observations),
    'max_cloth_stretch':max(x['cloth_max_stretch'] for x in observations),
    'physical_quality_certified':False,'performance_certified':False}
target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='endpoint_observations'}))
