"""One bounded CCD experiment; immutable plans, same-state gates, serial GPU.

Shared-desktop timings cannot certify 2x. Frozen physical bounds never change.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path
from config import ROOT, expand, read, sha
from ipc_benchmark import write, metrics
from component_tuning import state_comparison
from validate_run import validate

TAG='ipc_bounded_20261006'
SCENES={'hang':'cloth_hang_l','fixed':'cloth_fixed_bunny_l','mixed':'bunny_cloth_bunny_l'}
CONTACT={'hang':(41,50),'fixed':(22,59),'mixed':(1,3)}
PROTOCOL=ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json'
PROTOCOL_SHA='1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f'

def task(scene,arm,repeat,stage,steps,**changes):
    cfg={'scene':SCENES[scene],'preset':'combined','discrete_bvh_refit':True,
        'steps':steps,'dt':.01,'timeout_seconds':120,'trace_velocity':True,
        'bounded_ccd':arm=='on'}
    binary='active'
    if arm=='stiff':cfg.update(preset='base',discrete_bvh_refit=False);binary='base'
    cfg.update(changes);expand(cfg)
    return {'name':f'{TAG}_{stage}_{scene}_{arm}_{repeat}','arm':scene+'_'+arm,
        'scene_key':scene,'variant':arm,'repeat':repeat,'binary':binary,'config':cfg}

def prepare(stage):
    assert sha(PROTOCOL)==PROTOCOL_SHA
    if stage=='guards':
        runs=[task(s,'on','guard',stage,n,bounded_ccd_validate=True)
              for s,n in [('hang',51),('fixed',59),('mixed',3)]]
    elif stage=='screen':
        runs=[]
        for repeat in range(1,4):
            for scene,n in [('hang',51),('fixed',59)]:
                for arm in (('off','on') if repeat%2 else ('on','off')):
                    runs.append(task(scene,arm,f'r{repeat}',stage,n))
    elif stage=='profile':
        runs=[task(s,'off','nodes',stage,n+1,diagnostics=['cost'],cost_frames=str(n),
                   cost_events=False,profile='node',trace_velocity=False)
              for s,n in [('hang',41),('fixed',49)]]
    elif stage=='full':
        screen=read(ROOT/f'reports/active/{TAG}_screen_analysis.json')
        assert screen['full_performance_gate_passed'], 'Short-window net 5% gate not met'
        runs=[]
        for repeat in range(1,4):
            arms=['stiff','off','on'];arms=arms[repeat-1:]+arms[:repeat-1]
            for scene in ('hang','fixed'):
                for arm in arms:runs.append(task(scene,arm,f'r{repeat}',stage,100))
    else:raise ValueError(stage)
    plan={'schema':1,'report':f'reports/active/{TAG}_{stage}_batch.json',
        'stop_on_failure':True,'quality_protocol_sha256':PROTOCOL_SHA,
        'native_identity':sha(ROOT/'builds/active/Release/gipc.exe'),
        'contact_windows':CONTACT,'runs':runs}
    write(ROOT/f'configs/active/{TAG}_{stage}.json',plan)
    print(json.dumps({'stage':stage,'runs':len(runs),'identity':plan['native_identity']}))

def window(run,lo,hi):
    rows=read(run/'output/stats.json')['frames'][lo-1:hi]
    with (run/'trace/frames.csv').open() as stream:times=list(csv.DictReader(stream))[lo-1:hi]
    phase={key:sum(f.get('phase_ms',{}).get(key,0) for f in rows)
           for key in ['assembly','pcg','ccd','line_search','state_update']}
    solver_ms=sum(float(f['solver_ms']) for f in times)
    exit_ms=sum(f.get('ipc_exit_assembly_ms',0) for f in rows)
    pcgs=[n['pcg'] for f in rows for n in f['newton'] if 'pcg' in n]
    return {'frames':[lo,hi],'solver_ms':solver_ms,'phase_ms':phase,
        'exit_assembly_ms':exit_ms,'unclassified_ms':solver_ms-sum(phase.values())-exit_ms,
        'directions':len(pcgs),'pcg':sum(p['iterations'] for p in pcgs),
        'native_contact_frames':[lo+i for i,f in enumerate(rows) if
           f.get('contact_geometry',{}).get('native_narrow_self_pairs',0) or
           f.get('contact_geometry',{}).get('native_narrow_ground_pairs',0) or
           any(n.get('active_pairs',0)>0 for n in f['newton'])]}

def analyze(stage):
    assert sha(PROTOCOL)==PROTOCOL_SHA
    plan=read(ROOT/f'configs/active/{TAG}_{stage}.json')
    batch=read(ROOT/plan['report']);records=[]
    for t in batch['runs']:
        run=ROOT/'runs/active'/t['name'];result=read(run/'result.json')
        rec={k:t[k] for k in ('name','variant','repeat','scene_key','binary')}
        rec.update(status=result['status'],recorded_frames=result['recorded_frames'])
        if result['status']=='completed':
            rec.update(metrics(run));rec['configuration']=validate(run) if t['binary']=='active' else None
            frames=read(run/'output/stats.json')['frames']
            audits=[n['bounded_ccd'] for f in frames for n in f['newton'] if 'bounded_ccd' in n]
            counters=[f.get('bounded_ccd',{}) for f in frames]
            rec['bounded']={key:sum(r.get(key,0) for r in counters) for key in
                ('second_queries','bounded_queries','unsupported_queries','pairs','validation_calls',
                 'validation_failed','cut_pairs','old_iterations','new_iterations','diagnostic_host_ms')}
            rec['audits_passed']=all(a['passed'] for a in audits)
            rec['audit_count']=len(audits)
            lo,hi=CONTACT[t['scene_key']];hi=min(hi,result['recorded_frames'])
            rec['contact_window']=window(run,lo,hi) if hi>=lo else None
            rec['hard_checks_passed']=rec['finite'] and not rec['pcg_failures'] and rec['audits_passed']
            if t['binary']=='active':rec['hard_checks_passed'] &= rec['configuration']['passed']
            scene_key={'fixed':'fixed_bunny'}.get(t['scene_key'],t['scene_key'])
            bounds=read(PROTOCOL)['scenes'].get(scene_key,{}).get('bounds')
            rec['frozen_material_checks']={k:{'actual':rec[k],'bound':v,'passed':rec[k]<=v}
                 for k,v in bounds.items()} if bounds else None
        records.append(rec)
    report={'stage':stage,'plan_sha256':sha(ROOT/f'configs/active/{TAG}_{stage}.json'),
        'quality_protocol_sha256':PROTOCOL_SHA,'performance_certified':False,'quality_certified':False,
        'runs':records,'planned':len(plan['runs']),'attempted':len(batch['runs']),
        'unattempted':[t['name'] for t in plan['runs'] if t['name'] not in {r['name'] for r in records}],
        'summary':{},'full_performance_gate_passed':False}
    if stage in ('screen','full'):
        for scene in ('hang','fixed'):
            rows=[r for r in records if r['scene_key']==scene and r['status']=='completed']
            pairs=[]
            for i in range(1,4):
                off=next((r for r in rows if r['variant']=='off' and r['repeat']==f'r{i}'),None)
                on=next((r for r in rows if r['variant']=='on' and r['repeat']==f'r{i}'),None)
                if off and on:
                    pairs.append({'repeat':i,'speedup':off['seconds']/on['seconds'],
                        'contact_speedup':off['contact_window']['solver_ms']/on['contact_window']['solver_ms'],
                        'pcg_off':off['pcg'],'pcg_on':on['pcg'],
                        'state':state_comparison(ROOT/'runs/active'/off['name'],ROOT/'runs/active'/on['name'])})
            report['summary'][scene]={'pairs':pairs,'median_speedup':statistics.median(p['speedup'] for p in pairs) if pairs else None,
                'median_contact_speedup':statistics.median(p['contact_speedup'] for p in pairs) if pairs else None,
                'all_hard_checks_passed':bool(rows) and all(r['hard_checks_passed'] for r in rows)}
        report['full_performance_gate_passed']=all(len(s['pairs'])==3 and s['median_speedup']>=1.05 and
            s['all_hard_checks_passed'] for s in report['summary'].values())
    write(ROOT/f'reports/active/{TAG}_{stage}_analysis.json',report)
    print(json.dumps({'stage':stage,'attempted':report['attempted'],'summary':report['summary'],
                      'full_performance_gate_passed':report['full_performance_gate_passed']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze'])
    p.add_argument('--stage',required=True,choices=['guards','screen','profile','full']);a=p.parse_args()
    (prepare if a.action=='prepare' else analyze)(a.stage)
