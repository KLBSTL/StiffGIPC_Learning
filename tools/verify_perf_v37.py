"""Final identity, diagnostic coverage, and bounded-run verification."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'downloads/autodl_perf_v37_20261003'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
out={}
for version in ['v36','v37']:
    manifest=read(ROOT/f'manifests/perf_{version}.json')
    assert all(sha(ROOT/f['path'])==f['sha256'] for f in manifest['files'])
    out[version+'_source_verified']=len(manifest['files'])
manifest=read(data/'manifests/perf_v37_autodl.json')
out['runs']=[]
for run in sorted((data/'runs/autodl').iterdir()):
    req,res=read(run/'requested.json'),read(run/'result.json')
    assert req['source_digest']==manifest['source_digest']
    assert req['runner_sha256']==sha(data/'tools/run_perf_v37.py')
    assert req['exe_sha256'] in [b['sha256'] for b in manifest['binaries'].values()]
    assert res['timing_is_diagnostic']
    if res['status']=='completed':assert res['recorded_frames']==req['steps'] and res['finite']
    else:assert run.name=='autodl_perf_v37_bunny_mas' and res['recorded_frames']==38
    out['runs'].append({'name':run.name,**res})
reassembly=read(data/'runs/autodl/autodl_perf_v37_host_reassembly/output/stats.json')
audits=[n['state_snapshot_audit'] for f in reassembly['frames'] for n in f['newton'] if 'state_snapshot_audit' in n]
replays=[n['pcg']['fixed_system_replay'] for f in reassembly['frames'] for n in f['newton'] if 'fixed_system_replay' in n['pcg']]
assert all(x['protected_state_restored_bitwise'] and x['assembly']['structure_identical'] for x in audits)
assert all(x['rhs_unchanged_bitwise'] and x['primary_solution_restored_bitwise'] for x in replays)
out['reassembly']={'samples':len(audits),'max_A_relative':max(x['assembly']['matrix']['relative_difference'] for x in audits),
    'max_b_relative':max(x['assembly']['rhs']['relative_difference'] for x in audits),'all_protected_restored':True,
    'active_contact_samples':sum(x['active_contacts']>0 for x in audits)}
out['fixed_replays']={'systems':len(replays),'max_relative_solution_difference':max(y['relative_solution_difference'] for x in replays for y in x['same_execution_runs'])}
out['window_coverage']=[]
for i in range(1,4):
    run=data/f'runs/autodl/autodl_perf_v37_host_r{i}'
    stats=read(run/'output/stats.json')['frames']
    expected=sum(len(f['newton']) for f in stats[20:25])
    actual=len(list((run/'state_window').glob('*_state.json')))
    assert expected==actual==31
    out['window_coverage'].append({'run':run.name,'expected':expected,'actual':actual})
# Cloth-only metric excludes the fixed sphere's mass.
runs=[data/f'runs/autodl/autodl_perf_v37_host_r{i}' for i in range(1,4)]
raw=np.fromfile(runs[0]/'trace/topology.bin','<u4');nv,nf,nt=map(int,raw[:3])
faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
cloth=np.unique(faces[~in_tet[faces].any(axis=1)])
mass=np.fromfile(runs[0]/'trace/masses.bin','<f8')[cloth]
initial=np.fromfile(runs[0]/'trace/state_0000.bin','<f8').reshape(-1,3)[cloth]
scale=np.linalg.norm(np.ptp(initial,axis=0))
out['cloth_repeat']=[]
for candidate in runs[1:]:
    values=[]
    for frame in range(26):
        x,y=[np.fromfile(run/f'trace/state_{frame:04d}.bin','<f8').reshape(-1,3)[cloth] for run in [runs[0],candidate]]
        values.append(float(np.sqrt(np.average(np.sum((x-y)**2,axis=1),weights=mass))/scale))
    out['cloth_repeat'].append({'candidate':candidate.name,'max_normalized_rms':max(values),'frame_of_max':int(np.argmax(values))})
archive=ROOT/'downloads/autodl_perf_v37_results_20261003.tar.gz'
assert sha(archive)=='24759e3a019d49fc2f7c0e740cbf582cb0b934cad4bced6b4301d418150f70a9'
out['archive']={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}
out['analysis_sha256']={p.name:sha(p) for p in (ROOT/'reports').glob('AUTODL_PERF_V37_*.json')}
out['tool_sha256']={p.name:sha(p) for p in (ROOT/'tools').glob('*v37*') if p.is_file()}
target=ROOT/'reports/AUTODL_PERF_V37_FINAL_VERIFICATION.json'
assert not target.exists()
target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k not in ['runs','analysis_sha256','tool_sha256']},indent=2))
