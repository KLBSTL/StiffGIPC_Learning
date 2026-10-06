"""Same-input work/quality observations. Certification requires separate gates."""
import argparse
import csv
import itertools
import json
import numpy as np
from config import ROOT, read, sha

def load(run, prefix, frame):
    return np.fromfile(run/'trace'/f'{prefix}_{frame:04d}.bin', dtype='<f8').reshape(-1,3)

def analyze(paths, baseline_count):
    ref=paths[0];d=ref/'trace';raw=np.fromfile(d/'topology.bin',dtype='<u4')
    nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+nf*3].reshape(-1,3);tets=raw[3+nf*3:].reshape(-1,4)
    assert len(tets)==nt
    x0=load(ref,'state',0);abd=read(d/'metadata.json')['abd_point_num']
    fixed=np.fromfile(d/'boundary_types.bin',dtype='<i4')==1
    mass=np.fromfile(d/'masses.bin',dtype='<f8');in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
    objects=read(ref/'output/scene.json')['objects']
    objects_3d=[o for o in objects if o['dimension']==3]
    if objects_3d and all(o['body_type']=='ABD' and o.get('fixed_mode')=='all' for o in objects_3d):
        fixed_abd=in_tet&(np.arange(nv)<abd)
        assert fixed_abd.sum()==abd
        fixed|=fixed_abd
    cloth_faces=faces[~in_tet[faces].any(axis=1)];cloth=np.zeros(nv,bool);cloth[cloth_faces.ravel()]=True
    groups={'cloth':cloth&~fixed,'fem':in_tet&(np.arange(nv)>=abd)&~fixed,
            'abd':(np.arange(nv)<abd)&~fixed}
    scales={k:float(np.linalg.norm(np.ptp(x0[v],axis=0))) for k,v in groups.items() if np.any(v)}
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    lengths=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
    def volumes(x):
        q=x[tets];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
    rest=volumes(x0);fm=np.all(tets>=abd,axis=1);am=np.all(tets<abd,axis=1)
    rows=[]
    for run in paths:
        for file in ('topology.bin','masses.bin','boundary_types.bin','state_0000.bin','metadata.json'):
            assert (d/file).read_bytes()==(run/'trace'/file).read_bytes(),(run,file)
        assert read(ref/'output/scene.json')==read(run/'output/scene.json')
        result=read(run/'result.json');stats=read(run/'output/stats.json')['frames']
        with (run/'trace/frames.csv').open() as f: times=list(csv.DictReader(f))
        frames=[]
        for i in range(1,result['recorded_frames']+1):
            x=load(run,'state',i);assert np.isfinite(x).all()
            j=volumes(x)/rest;s=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/lengths
            st=stats[i-1];pcg=[n['pcg'] for n in st['newton'] if 'pcg' in n];outer=st.get('toi',[])
            phases=st.get('phase_ms',{})
            f={'frame':i,'seconds':float(times[i-1]['solver_ms'])/1000,
               'pcg_calls':len(pcg),'pcg_iterations':sum(p['iterations'] for p in pcg),
               'pcg_limit_hits':sum(bool(p.get('iteration_limit') or p.get('breakdown')) for p in pcg),
               'outers':len(outer),'safe_restarts':sum(o.get('initial_guess_selection',{}).get('selected')=='safe' for o in outer),
               'backtracks':sum(n.get('line_search_backtracks',0) for n in st['newton']),
               'active_added':sum(o.get('active_added',0) for o in outer),'active_removed':sum(o.get('active_removed',0) for o in outer),
               'cloth_max_stretch':float(s.max()) if len(s) else 1.,
               'fem_min_J':float(j[fm].min()) if np.any(fm) else 1.,
               'fem_nonpositive':int((j[fm]<=0).sum()),
               'fem_negative_volume':float(np.sum(np.maximum(-j[fm],0)*abs(rest[fm]))),
               'abd_min_J':float(j[am].min()) if np.any(am) else 1.,
               'fixed_max_drift':float(np.max(np.linalg.norm(x[fixed]-x0[fixed],axis=1))) if np.any(fixed) else 0.,
               'linear_stage_seconds':phases['pcg']/1000 if phases.get('pcg',0)>0 else (None if pcg else 0.),
               'phase_seconds':{k:v/1000 if any(phases.values()) else None for k,v in phases.items()}}
            frames.append(f)
        windows=[]
        for start,end in dict.fromkeys([(1,3),(24,26),(33,35),(1,35),(1,result['recorded_frames'])]):
            subset=[f for f in frames if start<=f['frame']<=end]
            if len(subset)!=end-start+1:continue
            windows.append({'start':start,'end':end,**{k:sum(f[k] for f in subset) if all(f[k] is not None for f in subset) else None for k in
                ('seconds','pcg_calls','pcg_iterations','outers','safe_restarts','backtracks','active_added','active_removed','linear_stage_seconds')}})
        rows.append({'name':run.name,'path':str(run),'result':{k:v for k,v in result.items() if k!='gpu_samples'},
                     'request_sha256':sha(run/'requested.json'),'frames':frames,'windows':windows})
    counts=[len(r['frames']) for r in rows];comparisons=[]
    for ia,ib in itertools.combinations(range(len(paths)),2):
        frames=[]
        for i in range(1,min(counts[ia],counts[ib])+1):
            q={'frame':i}
            for prefix,label in [('state','position'),('velocity','velocity')]:
                if not all((paths[k]/'trace'/f'{prefix}_{i:04d}.bin').exists() for k in [ia,ib]):
                    continue
                delta=np.sum((load(paths[ia],prefix,i)-load(paths[ib],prefix,i))**2,axis=1)
                for group,mask in groups.items():
                    if np.any(mask):q[label+'_'+group+'_rms']=float(np.sqrt(np.average(delta[mask],weights=mass[mask])))
            frames.append(q)
        comparisons.append({'a':paths[ia].name,'b':paths[ib].name,'frames':frames,
            'maxima':{k:max(f[k] for f in frames) for k in frames[0] if k!='frame'} if frames else {}})
    envelope=[]
    for i in range(min(counts[:baseline_count])):
        baseline=[r['frames'][i] for r in rows[:baseline_count]]
        keys=['cloth_max_stretch','fem_min_J','fem_nonpositive','fem_negative_volume','abd_min_J','fixed_max_drift']
        envelope.append({'frame':i+1,**{k:[min(f[k] for f in baseline),max(f[k] for f in baseline)] for k in keys}})
    verdicts=[]
    for row in rows[baseline_count:]:
        violations=[]
        for f,e in zip(row['frames'],envelope):
            for key in ('cloth_max_stretch','fem_nonpositive','fem_negative_volume','fixed_max_drift'):
                if f[key]>e[key][1]+1e-12:violations.append({'frame':f['frame'],'metric':key,'value':f[key],'bound':e[key][1]})
            if f['fem_min_J']<e['fem_min_J'][0]-1e-12:violations.append({'frame':f['frame'],'metric':'fem_min_J','value':f['fem_min_J'],'bound':e['fem_min_J'][0]})
            if f['abd_min_J']<=0:violations.append({'frame':f['frame'],'metric':'abd_inversion'})
        verdicts.append({'run':row['name'],'endpoint_envelope_passes':not violations,'violation_count':len(violations),
                         'violations':violations,'quality_certified':False,
                         'reason':'Requires accepted-path CCD and position/velocity repeat-envelope checks; no automatic relaxation.'})
    return {'same_initial_state_scene_topology':True,'baseline_names':[p.name for p in paths[:baseline_count]],
            'scales_m':scales,'runs':rows,'comparisons':comparisons,'baseline_frame_envelope':envelope,'candidate_gates':verdicts,
            'performance_certified':False,'timing_scope':'Diagnostic only; selected windows include continuous from-zero history.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--baseline-count',type=int,default=3)
    p.add_argument('--output',required=True);a=p.parse_args();paths=[ROOT/r for r in a.runs]
    target=ROOT/a.output
    if target.exists():raise FileExistsError(target)
    report=analyze(paths,a.baseline_count);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'runs':[{'name':r['name'],'windows':r['windows']} for r in report['runs']],
                      'candidate_gates':[{k:v for k,v in r.items() if k!='violations'} for r in report['candidate_gates']]}),flush=True)
