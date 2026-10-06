"""Collect final v47 evidence without changing frozen simulation inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=read(ROOT/'reports/VERIFICATION_V47_BUNNY.json')['runs'][0]
run=ROOT/'runs/local/v47_bunny_graph'
stage=[json.loads(s) for s in (run/'stage.jsonl').read_text().splitlines()]
partial=[s['pcg'] for s in stage if s['frame']==35 and s['stage']=='pcg_end']
progress=[json.loads(s) for s in (run/'events/progress.jsonl').read_text().splitlines()]
events=read(ROOT/'reports/EVENTS_v47_bunny_graph_CPU.json')['events']
ref=read(ROOT/'reports/REFERENCE_V47_E4.json')
top=np.fromfile(run/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,top[:3])
abd=read(run/'trace/metadata.json')['abd_point_num']
tets=top[3+3*nf:].reshape(-1,4);tets=tets[np.all(tets>=abd,axis=1)]
def det(x):
    t=x[tets];return np.einsum('ij,ij->i',t[:,1]-t[:,0],np.cross(t[:,2]-t[:,0],t[:,3]-t[:,0]))
initial=np.fromfile(run/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)
def shape(path):
    j=det(np.fromfile(path,dtype='<f8').reshape(-1,3))/det(initial)
    return {'min_J':float(j.min()),'nonpositive_tets':int(np.sum(j<=0))}
rows=[]
for r in events:
    e=r['event'];rows.append({'frame':e['frame'],'outer_zero_based':e['outer'],'inner_zero_based':e['inner'],
        'velocity':e['trial_velocity'],'true_residual':r['cpu_true_relative_residual'],
        'r':e.get('line_search_r'),'energy_before':e['energy_before'],'energy_delta':r['actual_energy_delta'],
        'predicted_delta':r['predicted_quadratic_energy_delta'],'components':e['energy_components_before'],
        'component_deltas':r['energy_component_deltas'],'trial_min_J':r['fem_min_J'],
        'trial_nonpositive_tets':r['fem_nonpositive_tets'],'outer_completed_later':r['outer_completed_later']})
manifest=read(ROOT/'manifests/perf_v47_local.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
for path,entry in manifest['binaries'].items():assert sha(ROOT/path)==entry['sha256']
old=read(ROOT/'manifests/perf_v45_local.json')
for f in old['files']:assert sha(ROOT/f['path'])==f['sha256']
for path,entry in old['binaries'].items():assert sha(ROOT/path)==entry['sha256']
record={'identity_verified':True,'frozen_v45_identity_preserved':True,'source_files':len(manifest['files']),
    'source_digest':manifest['source_digest'],'binaries':manifest['binaries'],
    'python_numpy_scipy':{'numpy':np.__version__,'scipy':scipy.__version__},
    'partial_frame_pcg':{'completed':len(partial),'max_iterations':max(r['iterations'] for r in partial),
        'limit_hits':sum(bool(r.get('iteration_limit')) for r in partial),
        'breakdowns':sum(bool(r.get('breakdown')) for r in partial),'last_stage':stage[-1]},
    'last_inner_progress':next(r for r in reversed(progress) if r['stage']=='inner_end'),
    'last_outer_progress':next(r for r in reversed(progress) if r['stage']=='outer_end'),
    'events':rows,'final_completed_frame_shape':shape(run/'trace/state_0034.bin'),
    'reference':ref,'reference_velocity_reduction_fraction':1-ref['reference_fem_axis_velocity']/ref['original_fem_axis_velocity'],
    'accepted_path_ccd':v['ccd']['report'],'smoke':read(ROOT/'reports/V47_SMOKE_COMPARISON.json'),
    'components':{'passed':read(ROOT/'reports/V47_components.json')['passed'],
        'tests':len(read(ROOT/'reports/V47_components.json')['tests'])},
    'helpers':{p.name:sha(p) for p in sorted((ROOT/'tools').glob('*v47.py'))},
    'run_bytes':{name:sum(p.stat().st_size for p in (ROOT/'runs/local'/name).rglob('*') if p.is_file())
        for name in ['v47_smoke_off','v47_smoke_on','v47_bunny_graph']}}
out=ROOT/'reports/FINAL_V47_20261004.json';assert not out.exists();out.write_text(json.dumps(record,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in record.items() if k not in ['events','helpers','binaries','accepted_path_ccd']}))
