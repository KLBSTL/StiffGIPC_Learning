"""Measured work, phase attribution, repeated trajectories and endpoint quality."""
import collections,csv,json,statistics
import numpy as np
from run_bunny_components import ROOT,read,sha,PRESETS
OUT=ROOT/'reports/BUNNY_COMPONENTS_ANALYSIS.json'
assert not OUT.exists()
matrix=read(ROOT/'reports/BUNNY_COMPONENTS_MATRIX.json')
rows=matrix['runs'];complete=[r for r in rows if r['result']['status']=='completed' and r['result']['recorded_frames']==100]
ref=ROOT/'runs/local'/next(r['name'] for r in complete if r['label']=='base')
d=ref/'trace';raw=np.fromfile(d/'topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3])
faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
load=lambda d,i:np.fromfile(d/f'state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
x0=load(d,0);abd=read(d/'metadata.json')['abd_point_num'];boundary=np.fromfile(d/'boundary_types.bin',dtype='<i4')
mass=np.fromfile(d/'masses.bin',dtype='<f8');in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
cloth_faces=faces[~in_tet[faces].any(axis=1)];cloth=np.zeros(nv,bool);cloth[np.unique(cloth_faces)]=True
groups={'cloth':cloth,'fem':in_tet&(np.arange(nv)>=abd)&(boundary!=1),'abd':(np.arange(nv)<abd)&(boundary!=1)}
scales={k:float(np.linalg.norm(np.ptp(x0[v],axis=0))) for k,v in groups.items()}
edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
length=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1);fem=tets[np.all(tets>=abd,axis=1)]
def det(x):
    q=x[fem];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))
rest=det(x0);assert np.all(rest!=0) and np.all(length>0)
observations={};bylabel=collections.defaultdict(list)
for row in complete:
    run=ROOT/'runs/local'/row['name'];req=read(run/'requested.json');stats=read(run/'output/stats.json')['frames']
    assert len(stats)==100 and req['runner_sha256']==matrix['runner_sha256']==sha(ROOT/'tools/run_bunny_components.py')
    assert req['configuration']==PRESETS[row['label']]
    for f in ['topology.bin','masses.bin','boundary_types.bin','state_0000.bin','metadata.json']:assert (d/f).read_bytes()==(run/'trace'/f).read_bytes()
    assert read(run/'output/scene.json')==read(ref/'output/scene.json')
    m=read(__import__('pathlib').Path(req['manifest']))
    files=[f for f in m['files'] if row['label']!='base' or f['path'].startswith('sources/stiff_base/')]
    for f in files:assert sha(ROOT/f['path'])==f['sha256']
    exe='builds/local-base/Release/gipc.exe' if row['label']=='base' else 'builds/local-v54/Release/gipc.exe'
    assert sha(ROOT/exe)==req['exe_sha256']==m['binaries'][exe]['sha256']
    pcg=[n['pcg'] for f in stats for n in f['newton'] if 'pcg' in n]
    if row['label']!='base':assert all(p['execution']==('conditional_graph' if req['configuration'].get('graph') else 'host') for p in pcg)
    assert not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg)
    phases={k:sum(f['phase_ms'].get(k,0) for f in stats)/1000 for k in stats[0]['phase_ms']}
    total=row['result']['solver_seconds']
    obs={'label':row['label'],'repeat':row['repeat'],'solver_seconds':total,'wall_seconds':row['result']['wall_seconds'],
        'pcg_calls':len(pcg),'pcg_iterations':sum(p['iterations'] for p in pcg),'max_pcg_iterations':max(p['iterations'] for p in pcg),
        'graph_cache_hits':sum(bool(p.get('graph_cache_hit')) for p in pcg),
        'graph_capture_seconds':sum(p.get('graph_capture_instantiate_host_ms',0) for p in pcg)/1000,
        'phases_seconds':phases,'other_seconds':total-sum(phases.values()),'pcg_limit_hits':0,
        'cloth_max_stretch':0.,'fem_min_J':float('inf'),'max_nonpositive_tets':0,'frames':[]}
    obs['pcg_microseconds_per_iteration']=phases['pcg']*1e6/obs['pcg_iterations']
    observations[row['name']]=obs;bylabel[row['label']].append(row['name'])
comparisons={}
for name,obs in observations.items():
    for reference in [bylabel['base'][0],bylabel[obs['label']][0]]:
        if reference==name:continue
        comparisons.setdefault((reference,name),{k:0. for k in groups})
for i in range(1,101):
    positions={}
    for name,obs in observations.items():
        x=load(ROOT/'runs/local'/name/'trace',i);assert x.shape==(nv,3) and np.isfinite(x).all();positions[name]=x
        j=det(x)/rest;s=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/length
        frame={'frame':i,'cloth_max_stretch':float(s.max()),'fem_min_J':float(j.min()),'nonpositive_tets':int((j<=0).sum())}
        obs['frames'].append(frame);obs['cloth_max_stretch']=max(obs['cloth_max_stretch'],frame['cloth_max_stretch'])
        obs['fem_min_J']=min(obs['fem_min_J'],frame['fem_min_J']);obs['max_nonpositive_tets']=max(obs['max_nonpositive_tets'],frame['nonpositive_tets'])
    for (a,b),values in comparisons.items():
        dist=np.sum((positions[a]-positions[b])**2,axis=1)
        for k,mask in groups.items():values[k]=max(values[k],float(100*np.sqrt(np.average(dist[mask],weights=mass[mask]))/scales[k]))
summary={}
for label,names in bylabel.items():
    rr=[observations[n] for n in names];ts=[r['solver_seconds'] for r in rr]
    summary[label]={'repeats':len(rr),'median_seconds':statistics.median(ts),'min_seconds':min(ts),'max_seconds':max(ts),
        'cv_percent':100*statistics.pstdev(ts)/statistics.mean(ts),'median_pcg_iterations':statistics.median(r['pcg_iterations'] for r in rr),
        'median_pcg_calls':statistics.median(r['pcg_calls'] for r in rr),
        'phase_medians_seconds':{k:statistics.median(r['phases_seconds'][k] for r in rr) for k in rr[0]['phases_seconds']},
        'median_pcg_us_per_iteration':statistics.median(r['pcg_microseconds_per_iteration'] for r in rr),
        'cloth_max_stretch_range':[min(r['cloth_max_stretch'] for r in rr),max(r['cloth_max_stretch'] for r in rr)],
        'fem_min_J_range':[min(r['fem_min_J'] for r in rr),max(r['fem_min_J'] for r in rr)],
        'worst_repeat_rms_percent':{k:max((v[k] for (a,b),v in comparisons.items() if a in names and b in names),default=0.) for k in groups}}
for label,s in summary.items():
    s['speed_vs_base']=summary['base']['median_seconds']/s['median_seconds']
    s['speed_vs_host']=summary['host']['median_seconds']/s['median_seconds']
    s['paired_speed_vs_host']=[observations[bylabel['host'][r]]['solver_seconds']/observations[bylabel[label][r]]['solver_seconds'] for r in range(min(len(bylabel['host']),len(bylabel[label])))]
report={'scene':'bunny_cloth_bunny_l','same_scene_initial_state_verified':True,'frames':100,'dt':.01,
    'performance_certified':False,'physical_quality_certified':False,'shared_gpu':True,
    'quality_note':'Geometric comparison to Stiff, not a converged physical reference; maxima include every endpoint.',
    'summary':summary,'runs':observations,'comparisons':[{'a':a,'b':b,'max_rms_percent_scale':v} for (a,b),v in comparisons.items()]}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(summary,indent=2),flush=True)
