"""Read persisted stage events, including the last unfinished physical frame."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('report',type=Path);a=p.parse_args()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pairs={'assembly_begin':('pcg_begin','assembly'),'pcg_begin':('pcg_end','pcg'),
       'line_search_begin':('inner_end','line_search_and_diagnostics'),
       'ccd_bvh_begin':('ccd_broad_begin','ccd_bvh'),'ccd_broad_begin':('ccd_narrow_begin','ccd_broad'),
       'ccd_narrow_begin':('ccd_narrow_end','ccd_narrow'),'ccd_active_cpu_begin':('ccd_query_end','active_cpu'),
       'safe_update_begin':('outer_end','safe_update')}
summary=[]
for run in sorted((a.data/'runs/autodl').glob('autodl_perf_v44_*')):
    result=read(run/'result.json');request=read(run/'requested.json');rows=[];truncated_last_line=False
    path=run/'stage.jsonl'
    if path.exists():
        lines=path.read_text().splitlines()
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                assert i==len(lines)-1
                truncated_last_line=True
    assert all(x['sequence']==i+1 for i,x in enumerate(rows))
    assert all(y['steady_seconds']>=x['steady_seconds'] for x,y in zip(rows,rows[1:]))
    pending={};duration=collections.defaultdict(float);frames={};unfinished=[]
    for row in rows:
        frame=frames.setdefault(row['frame'],{'seconds':collections.defaultdict(float),'inner_completed':0,'pcg_completed':0,'max_pairs':0,'max_motion_m':0})
        for begin,(end,name) in pairs.items():
            if row['stage']==begin:
                assert begin not in pending,('Nested or missing stage',begin,row)
                pending[begin]=row
            elif row['stage']==end and begin in pending:
                start=pending.pop(begin)
                assert (start['frame'],start.get('outer'),start.get('inner'))==(row['frame'],row.get('outer'),row.get('inner'))
                dt=row['steady_seconds']-start['steady_seconds'];duration[name]+=dt;frame['seconds'][name]+=dt
        if row['stage']=='inner_end':frame['inner_completed']+=1
        if row['stage']=='pcg_end':frame['pcg_completed']+=1
        frame['max_pairs']=max(frame['max_pairs'],row.get('pairs',0))
        frame['max_motion_m']=max(frame['max_motion_m'],row.get('max_motion_m',0))
    partial=path.exists() and result['status']!='completed'
    for begin,row in pending.items():unfinished.append({'stage':begin,'frame':row['frame'],'outer':row.get('outer'),'inner':row.get('inner')})
    assert partial or not pending
    pcg=[x['pcg'] for x in rows if x['stage']=='pcg_end']
    geometry=None
    if (run/'ccd_current/current.json').exists():
        folder=run/'ccd_current';meta=read(folder/'current.json');n=meta['vertices']
        arrays={name:np.fromfile(folder/(name+'.bin'),dtype='<f8').reshape(n,3) for name in ['start','target','direction']}
        assert all(np.isfinite(x).all() for x in arrays.values())
        expected=arrays['start']-arrays['target'];error=float(np.max(np.abs(expected-arrays['direction'])))
        assert error==0
        motion=float(np.max(np.linalg.norm(expected,axis=1)))
        assert abs(motion-meta['max_motion_m'])<1e-12*max(1,motion)
        geometry={'metadata':meta,'direction_error':error,'sha256':{x.name:sha(x) for x in folder.iterdir() if x.is_file()}}
    summary.append({'run':run.name,'result':result,'requested_rho':request['pcg_tol'],'stage_events':len(rows),
        'truncated_last_line':truncated_last_line,'last_event':rows[-1] if rows else None,
        'completed_stage_seconds':dict(duration),'frames':frames,'unfinished_stages':unfinished,
        'pcg_completed_in_stage_window':len(pcg),'pcg_limits':sum(bool(x.get('iteration_limit')) for x in pcg),
        'pcg_breakdowns':sum(bool(x.get('breakdown')) for x in pcg),
        'max_pcg_iterations':max((x['iterations'] for x in pcg),default=0),'ccd_geometry':geometry})
a.report.write_text(json.dumps({'runs':summary,'performance_certified':False,'complete_restart_checkpoint':False},indent=2))
for x in summary:
    print(json.dumps({k:x[k] for k in ['run','result','stage_events','completed_stage_seconds','unfinished_stages','pcg_completed_in_stage_window','pcg_limits','pcg_breakdowns','max_pcg_iterations']}))
