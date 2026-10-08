"""Finite stage-1 orchestration; all GPU work uses the shared protected runner."""
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SESSION=ROOT/'runs/report56_stage1_20261008'
for folder in ('bench','diagnostic','full_eval','local'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from config import expand, read
from linux_runner import write_new, require
from local_identity import active_identity, verify, verify_active_sources, record, tree
from windows_runner import execute, gpu_lock

SCENES=('cloth_sphere7_l','cloth_fixed_bunny_l')
ARMS=('base','host','graph','all')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def config(scene,arm,diagnostics=False):
    result={'scene':scene,'steps':120,'timeout_seconds':180,
            'preset':{'base':'host','host':'host','graph':'graph','all':'combined'}[arm]}
    if arm=='all':result['discrete_bvh_refit']=True
    if diagnostics:
        result.update(diagnostics=['cost','structure_probe','bvh_query_probe'],
            cost_frames='1-120',cost_events=True)
    return expand(result)

def reference_tasks():
    rows=[]
    for repeat in range(3):
        order=ARMS[repeat:]+ARMS[:repeat]
        for scene in SCENES:
            for arm in order:
                rows.append({'name':f'{scene}_{arm}_r{repeat+1}',
                    'binary':'base' if arm=='base' else 'active','arm':arm,
                    'repeat':repeat+1,'config':config(scene,arm)})
    return rows

def guard_summary(folder,result):
    info={'completed':result['status']=='completed','solver_seconds':result.get('solver_seconds'),
          'wall_seconds':result.get('wall_seconds'),'performance_certified':False,
          'physical_quality_certified':False}
    if not info['completed']:return info
    stats=read(folder/'output/stats.json');frames=stats.get('frames',[])
    newtons=[n for f in frames for n in f.get('newton',[])]
    pcgs=[n['pcg'] for n in newtons if 'pcg' in n]
    info.update(directions=len(pcgs),pcg_iterations=sum(p.get('iterations',0) for p in pcgs),
        energy_evaluations=sum(n.get('energy_evaluations',0) for n in newtons),
        energy_backtracks=sum(n.get('energy_backtracks',0) for n in newtons),
        max_energy_backtracks=max((n.get('energy_backtracks',0) for n in newtons),default=0),
        graph_cache_hits=sum(p.get('graph_cache_hit',False) for p in pcgs),
        graph_capture_host_ms=sum(p.get('graph_capture_instantiate_host_ms',0) for p in pcgs),
        active_native_guards_observed=bool(pcgs),
        guards_passed=not stats.get('failure') and
            not any(p.get('iteration_limit') or p.get('breakdown') for p in pcgs) and
            not any(n.get('line_search_failure') for n in newtons) and
            not any(f.get('newton_exit')=='iteration_limit' for f in frames))
    phases={}
    for frame in frames:
        for key,value in frame.get('phase_ms',{}).items():
            if type(value) in (int,float):phases[key]=phases.get(key,0)+value
    info['recorded_phase_ms']=phases
    info['exit_assembly_ms']=sum(f.get('ipc_exit_assembly_ms',0) for f in frames)
    return info

def freeze_receipt():
    require(not (SESSION/'REFERENCE_IDENTITY.json').exists(),'Reference receipt already exists')
    active=read(ROOT/'build/report56_recheck_final_20261008/frozen_native_inputs/snapshot_identity.json')
    old=read(ROOT/'build/report_observer_20261007_v2/report_seal_final_capability_v3.json')
    base=old['programs']['base']
    verify_active_sources(active)
    for identity in (active,base):
        verify(identity['sources']+identity['build_evidence']+[identity['exe']]+identity['dlls']+identity.get('objects',[]))
    require({r['path']:r for r in tree(Path(base['source_root']))}==
            {r['path']:r for r in base['sources']},'Frozen Stiff inventory differs')
    tools=[record(ROOT/p) for p in ('tools/local/windows_runner.py','tools/local/windows_owned_job.py',
        'tools/local/local_identity.py','tools/bench/config.py','tools/bench/validate_run.py')]+[record(__file__)]
    write_new(SESSION/'REFERENCE_IDENTITY.json',{'programs':{'active':active,'base':base},
        'tools':tools,'plan':record(REPORT/'PLAN.md'),'tasks':reference_tasks(),
        'scope':'Shared WDDM diagnostic reference. Frozen native bytes and actual Assets checked; no quality/performance certification.'})

def run_reference():
    seal=read(SESSION/'REFERENCE_IDENTITY.json');verify(seal['tools']+[seal['plan']])
    require(seal['tasks']==reference_tasks(),'Reference task list changed')
    ledger={'status':'running','planned_runs':24,'rows':[]}
    write_new(SESSION/'REFERENCE_LEDGER.json',ledger)
    with gpu_lock(ROOT):
        for task in seal['tasks']:
            identity=seal['programs'][task['binary']]
            if task['binary']=='active':verify_active_sources(identity)
            result=execute(SESSION/'reference',task,identity,0)
            row={'name':task['name'],'arm':task['arm'],'repeat':task['repeat'],
                 'scene':task['config']['scene'],'result':result,
                 'summary':guard_summary(SESSION/'reference'/task['name'],result)}
            ledger['rows'].append(row)
            if not row['summary']['completed'] or not row['summary'].get('guards_passed',False):
                ledger['status']='failed';save(SESSION/'REFERENCE_LEDGER.json',ledger)
                raise RuntimeError('Reference failed; evidence preserved and no automatic retry')
            save(SESSION/'REFERENCE_LEDGER.json',ledger)
    ledger['status']='completed';save(SESSION/'REFERENCE_LEDGER.json',ledger)
    analyze_reference()

def analyze_reference():
    ledger=read(SESSION/'REFERENCE_LEDGER.json');rows=[]
    for scene in SCENES:
        selected=[r for r in ledger['rows'] if r['scene']==scene and r['summary']['completed']]
        times={arm:[r['summary']['solver_seconds'] for r in selected if r['arm']==arm] for arm in ARMS}
        summary={'scene':scene,'runs_per_arm':{k:len(v) for k,v in times.items()},
                 'median_seconds':{k:statistics.median(v) if v else None for k,v in times.items()}}
        if all(len(times[a])==3 for a in ARMS):
            pairs=[]
            for repeat in range(1,4):
                t={r['arm']:r['summary']['solver_seconds'] for r in selected if r['repeat']==repeat}
                pairs.append({'repeat':repeat,'stiff_over_graph':t['base']/t['graph'],
                    'host_over_graph':t['host']/t['graph'],'graph_over_all':t['graph']/t['all'],
                    'stiff_over_all':t['base']/t['all']})
            summary.update(paired_ratios=pairs,paired_medians={key:statistics.median(p[key] for p in pairs)
                for key in pairs[0] if key!='repeat'})
        rows.append(summary)
    save(REPORT/'REFERENCE_RESULTS.json',{'status':ledger['status'],'scenes':rows,
        'scope':'Solver-time sums, three interleaved pairs on shared WDDM; diagnostic only. Separate host/Graph isolates Graph on corrected source.'})
    print(json.dumps({'reference_status':ledger['status'],'summary':str(REPORT/'REFERENCE_RESULTS.json')}),flush=True)

def run_probes(build):
    identity=active_identity(ROOT,build,build/'Release/gipc.exe',build/'build.log')
    write_new(SESSION/'PROBE_IDENTITY.json',identity)
    ledger={'status':'running','planned_runs':2,'rows':[]};write_new(SESSION/'PROBE_LEDGER.json',ledger)
    with gpu_lock(ROOT):
        for scene in SCENES:
            task={'name':scene,'binary':'active','config':config(scene,'all',True)}
            result=execute(SESSION/'probe',task,identity,0)
            summary=guard_summary(SESSION/'probe'/scene,result)
            ledger['rows'].append({'scene':scene,'result':result,'summary':summary})
            if not summary['completed'] or not summary.get('guards_passed',False):
                ledger['status']='failed';save(SESSION/'PROBE_LEDGER.json',ledger)
                raise RuntimeError('Probe run failed; evidence preserved and no automatic retry')
            save(SESSION/'PROBE_LEDGER.json',ledger)
    ledger['status']='completed';save(SESSION/'PROBE_LEDGER.json',ledger)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','reference','analyze','probe'))
    parser.add_argument('--build',type=Path);args=parser.parse_args()
    SESSION.mkdir(parents=True,exist_ok=True)
    if args.stage=='freeze':freeze_receipt()
    elif args.stage=='reference':run_reference()
    elif args.stage=='analyze':analyze_reference()
    else:
        require(args.build is not None,'Probe build required');run_probes(args.build.resolve())
