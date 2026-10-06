"""Finite local IPC execution/residual experiment. Never certifies shared GPU speed."""
import argparse
import collections
import json
import statistics
from pathlib import Path
import numpy as np
from config import ROOT, read, sha, digest, expand

TAG='ipc_revision_20261005'
SCENES={'hang':'cloth_hang_l','fixed_bunny':'cloth_fixed_bunny_l'}
WINDOWS={'hang':'1-3,41-43','fixed_bunny':'1-3,39-41,57-59','mixed':'1-3,24-26,33-35'}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def task(key,variant,repeat,steps=100,**changes):
    binary={'base':'base','observed':'base_observed','reference':'reference'}.get(variant,'active')
    graph=variant not in ('base','observed','host')
    config={'scene':SCENES.get(key,'bunny_cloth_bunny_l'),'steps':steps,'dt':.01,
        'timeout_seconds':600 if steps>100 else 120,'execution':'conditional_graph' if graph else 'host',
        'backend':'ipc','mas':'legacy','trace_velocity':True,'refit':False,'batch':False,'reuse':False}
    if variant in ('shadow','gated','gate001'):
        config['ipc_residual_shadow']=True
        if variant!='shadow':config['ipc_termination']='gated'
        if variant=='gate001':config['ipc_cumulative_tol']=.001
    if variant in ('refit','batch','reuse'):config[variant]=True
    if variant=='combined':config.update(refit=True,batch=True,reuse=True)
    config.update(changes)
    expand(config)
    return {'name':f'{TAG}_{key}_{variant}_{repeat}','arm':key+'_'+variant,'scene_key':key,
        'variant':variant,'repeat':repeat,'binary':binary,'config':config}

def prepare(stage):
    tasks=[]
    if stage=='baseline':
        for key in SCENES:
            tasks+=[task(key,'base','observer_check',steps=3)]
            tasks+=[task(key,'observed',f'r{i}') for i in range(1,6)]
    elif stage=='smoke':
        for key in [*SCENES,'mixed']:
            for variant in ['host','graph','shadow','gated','gate001','reference']:
                tasks.append(task(key,variant,'smoke',steps=3))
    elif stage in ('cost','cost_v2'):
        for key in [*SCENES,'mixed']:
            steps={'hang':43,'fixed_bunny':59,'mixed':35}[key]
            tasks.append(task(key,'graph',stage,steps=steps,diagnostics=['cost','operator_probe'],cost_frames=WINDOWS[key]))
        for key in SCENES:
            tasks.append(task(key,'graph','nsys' if stage=='cost' else 'nsys_v2',steps=43 if key=='hang' else 59,
                diagnostics=['cost'],cost_frames='41-43' if key=='hang' else '39-41',profile='node'))
    elif stage=='components':
        for repeat in range(1,4):
            variants=['reference','graph','refit','batch','reuse','combined']
            variants=variants[repeat-1:]+variants[:repeat-1]
            for key in SCENES:
                for variant in variants:tasks.append(task(key,variant,f'r{repeat}',trace_velocity=False))
    elif stage=='residual':
        for repeat in range(1,4):
            for key in SCENES:
                for variant in ['shadow','gated','gate001']:tasks.append(task(key,variant,f'r{repeat}'))
        for key in SCENES:
            steps=43 if key=='hang' else 59
            tasks.append(task(key,'shadow','terminal',steps=steps,ipc_terminal_audit_frame=steps))
    else:raise ValueError(stage)
    plan={'report':f'reports/active/{TAG}_{stage}_batch.json','stop_on_failure':True,'runs':tasks}
    write(ROOT/f'configs/active/{TAG}_{stage}.json',plan)
    print(json.dumps({'stage':stage,'runs':len(tasks)}))

def metrics(folder,frame_limit=None):
    req=read(folder/'requested.json');result=read(folder/'result.json');frames=read(folder/'output/stats.json')['frames']
    if frame_limit is not None:
        if not isinstance(frame_limit,int) or not 1<=frame_limit<=result['recorded_frames']:
            raise ValueError('Prefix metrics require completed exported frames')
        frames=frames[:frame_limit]
    raw=np.fromfile(folder/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    x0=np.fromfile(folder/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)
    abd=read(folder/'trace/metadata.json')['abd_point_num'];in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
    cloth_faces=faces[~in_tet[faces].any(axis=1)]
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    l0=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
    boundary=np.fromfile(folder/'trace/boundary_types.bin',dtype='<i4');mass=np.fromfile(folder/'trace/masses.bin',dtype='<f8')
    fixed=boundary!=0
    scene=read(folder/'output/scene.json');objs=scene['objects']
    if all(o.get('fixed_mode')=='all' for o in objs if o['dimension']==3 and o['body_type']=='ABD'):
        # Only mark ABD fixed if there actually are ABD vertices and all such objects are fixed.
        fixed[:abd]=True
    def volume(x):
        q=x[tets];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
    v0=volume(x0);fm=np.all(tets>=abd,axis=1);am=np.all(tets<abd,axis=1)
    rows=[]
    for i in range(1,len(frames)+1):
        x=np.fromfile(folder/f'trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
        stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/l0
        j=volume(x)/v0 if nt else np.array([])
        rows.append({'frame':i,'max_stretch':float(stretch.max()) if len(stretch) else 1,
            'p99_stretch':float(np.quantile(stretch,.99)) if len(stretch) else 1,
            'fixed_drift_m':float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max()) if fixed.any() else 0,
            'finite':bool(np.isfinite(x).all()),'fem_min_J':float(j[fm].min()) if fm.any() else None,
            'fem_nonpositive':int((j[fm]<=0).sum()),'fem_negative_volume':float(np.maximum(-j[fm],0).dot(abs(v0[fm]))) if fm.any() else 0,
            'abd_min_J':float(j[am].min()) if am.any() else None})
    pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    observations=[n['ipc_residual'] for f in frames for n in f['newton'] if 'ipc_residual' in n]
    return {'name':folder.name,'scene':req['expanded_config']['scene'],'config':req['expanded_config'],
        'status':result['status'],'seconds':result.get('solver_seconds'),'frames':rows,
        'max_stretch':max(r['max_stretch'] for r in rows),'p99_stretch':max(r['p99_stretch'] for r in rows),
        'fixed_drift_m':max(r['fixed_drift_m'] for r in rows),'finite':all(r['finite'] for r in rows),
        'directions':len(pcg),'pcg':sum(p['iterations'] for p in pcg),
        'pcg_failures':sum(bool(p.get('iteration_limit') or p.get('breakdown')) for p in pcg),
        'exits':dict(collections.Counter(f.get('newton_exit') for f in frames)),
        'residual_observations':observations,'history_vetoes':sum(bool(r['history_veto']) for r in observations),
        'observe_host_ms':sum(r['observe_host_ms'] for r in observations),
        'terminal_assembly_host_ms':sum(f.get('ipc_terminal_assembly_host_ms',0) for f in frames),
        'exit_assembly_ms':sum(f.get('ipc_exit_assembly_ms',0) for f in frames),
        'phase_ms':{k:sum(f.get('phase_ms',{}).get(k,0) for f in frames) for k in ['assembly','pcg','ccd','line_search','state_update']},
        'requested_sha256':sha(folder/'requested.json'),'velocity_frames':len(list((folder/'trace').glob('velocity_*.bin')))}

def analyze(stage):
    batch=read(ROOT/f'reports/active/{TAG}_{stage}_batch.json')
    records=[]
    for t in batch['runs']:
        if t['result']['status']!='completed':continue
        r=metrics(ROOT/'runs/active'/t['name']);r.update(variant=t['variant'],repeat=t['repeat'],scene_key=t['scene_key']);records.append(r)
    report={'stage':stage,'performance_certified':False,'quality_certified':False,'runs':records,'summary':{}}
    for key in SCENES:
        summary={}
        for variant in sorted({r['variant'] for r in records if r['scene_key']==key}):
            rows=[r for r in records if r['scene_key']==key and r['variant']==variant and
                (r['repeat'].startswith('r') or r['repeat'].startswith('final_r'))]
            if rows:summary[variant]={'seconds':statistics.median(r['seconds'] for r in rows),'directions':statistics.median(r['directions'] for r in rows),
                'pcg':statistics.median(r['pcg'] for r in rows),'max_stretch_range':[min(r['max_stretch'] for r in rows),max(r['max_stretch'] for r in rows)],
                'history_vetoes':sum(r['history_vetoes'] for r in rows)}
        report['summary'][key]=summary
    write(ROOT/f'reports/active/{TAG}_{stage}_analysis.json',report)
    print(json.dumps(report['summary']))

def quality_protocol():
    rows=read(ROOT/f'reports/active/{TAG}_baseline_analysis.json')['runs'];scenes={}
    for key in SCENES:
        cal=[r for r in rows if r['scene_key']==key and r['variant']=='observed' and r['repeat'] in ('r1','r2','r3')]
        hold=[r for r in rows if r['scene_key']==key and r['variant']=='observed' and r['repeat'] in ('r4','r5')]
        assert len(cal)==3 and len(hold)==2
        bounds={k:max(r[k] for r in cal)+eps for k,eps in [('max_stretch',1e-6),('p99_stretch',1e-6),('fixed_drift_m',1e-10)]}
        checks=[{'name':r['name'],'passed':r['finite'] and not r['pcg_failures'] and all(r[k]<=v for k,v in bounds.items())} for r in hold]
        scenes[key]={'calibration':[r['name'] for r in cal],'bounds':bounds,'holdout':checks,'holdout_passed':all(r['passed'] for r in checks)}
    write(ROOT/f'reports/active/{TAG}_quality_protocol.json',{'scenes':scenes,'physical_budget':'No extra material deformation allowed. Numerical comparison floors only.',
        'position_velocity':'Reported separately as trajectory divergence, not physical truth error; strict independent certification pending CCD.',
        'old_results_reclassified':False,'performance_certified':False})
    print(json.dumps(scenes))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze','quality']);p.add_argument('--stage',choices=['baseline','smoke','cost','cost_v2','components','residual','final','audit','compensation','long']);a=p.parse_args()
    if a.action=='prepare':prepare(a.stage)
    elif a.action=='analyze':analyze(a.stage)
    else:quality_protocol()
