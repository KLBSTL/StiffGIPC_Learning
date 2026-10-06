"""Finite component experiments; timing, debugging and quality are separate.

prepare -> run smoke -> run guards -> run performance -> run quality -> analyze.
Every output name is immutable. A failed scene/arm stops its later repetitions.
Material envelope failures are reported; hard numerical failures stop the arm.
"""
import argparse
import csv
import collections
import json
import math
import statistics
from config import ROOT, expand, read, sha
from ipc_benchmark import write, metrics
from ipc_delivery import state_distance
from run import execute
from validate_run import validate

TAG='ipc_report_components_20261005'
SCENES={'hang':'cloth_hang_l','fixed_bunny':'cloth_fixed_bunny_l','mixed':'bunny_cloth_bunny_l'}


def protocol_path():
    newer=ROOT/f'reports/active/{TAG}_protocol_v2.json'
    return newer if newer.exists() else ROOT/f'reports/active/{TAG}_protocol.json'


def task(scene, variant, repeat, steps=100, **changes):
    config={'scene':SCENES[scene],'preset':'combined','steps':steps,'dt':.01,
            'timeout_seconds':120,'trace_velocity':False,
            'mas_fused_dot':variant in ('dot','both','both_host'),
            'discrete_bvh_refit':variant in ('discrete','both','both_host'),
            'discrete_bvh_rebuild_interval':8}
    binary='active'
    if variant in ('stiff','observed'):
        config['preset']='base';binary='base' if variant=='stiff' else 'base_observed'
    if variant=='both_host':config['execution']='host'
    config.update(changes)
    expand(config)
    return {'name':f'{TAG}_{scene}_{variant}_{repeat}','arm':f'{scene}_{variant}',
            'scene_key':scene,'variant':variant,'repeat':repeat,'binary':binary,'config':config}


def prepare():
    folder=ROOT/'configs/active'
    smoke=[task(scene,variant,'smoke',3) for scene in ('hang','fixed_bunny')
           for variant in ('old','dot','discrete','both')]
    smoke += [task('mixed',variant,'smoke',3) for variant in ('old','both')]
    smoke += [task(scene,'both_host','smoke',3) for scene in ('hang','fixed_bunny')]
    guards=[task(scene,'old','dot_guard',2,diagnostics=['fixed'],fixed_frames='2',
                 fixed_directions='1',fixed_mas_dot_study=True) for scene in ('hang','fixed_bunny')]
    guards += [task(scene,'discrete','bvh_guard',3,discrete_bvh_validate=True)
               for scene in SCENES]
    performance=[]
    variants=['stiff','old','dot','discrete','both']
    for repeat in range(3):
        order=variants[repeat:]+variants[:repeat]
        scenes=('hang','fixed_bunny') if repeat%2==0 else ('fixed_bunny','hang')
        for scene in scenes:
            performance += [task(scene,variant,f'r{repeat+1}') for variant in order]
    quality=[]
    for scene in ('hang','fixed_bunny'):
        for variant in ('observed','old','dot','discrete','both'):
            quality.append(task(scene,variant,'quality',trace_velocity=True,
                                diagnostics=[] if variant=='observed' else ['substeps']))
    quality += [task('mixed',variant,'quality',35,trace_velocity=True,diagnostics=['substeps']) for variant in ('old','both')]
    paths=[]
    for stage,runs in [('smoke',smoke),('guards',guards),('performance',performance),('quality',quality)]:
        path=folder/f'{TAG}_{stage}.json'
        write(path,{'stage':stage,'runs':runs,'report':f'reports/active/{TAG}_{stage}_batch.json'})
        paths.append(path)
    protocol={'plan':'reports/active/IPC_REPORT_COMPONENTS_PLAN_20261005.md',
              'old_quality_protocol_sha256':sha(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json'),
              'prior_exe_sha256':sha(ROOT/'builds/active/Release/gipc.exe'),
              'plan_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
              'guards':'Config/actual execution, finite PCG, same-state dot/tree equivalence before performance',
              'hard_failure':'completion/resource/NaN/PCGcap/breakdown/ABD flip; stop scene-arm later runs',
              'material_failure':'Report against frozen bounds, no promotion or threshold widening; finite diagnostic comparisons still recorded',
              'performance':'3 interleaved rounds, 30 x100 frames, no heavy diagnostics/actual velocity',
              'quality':'10 x100 independent velocity trajectories +2 mixed x35. All active arms use identical substep export; original observed has no native substep instrumentation. Only both cloth substeps are independently CCD-audited',
              'local_only':True,'performance_certified':False,'quality_certified':False,'default_promoted':False}
    write(ROOT/f'reports/active/{TAG}_protocol.json',protocol)
    print(json.dumps({'prepared':True,'counts':{'smoke':len(smoke),'guards':len(guards),'performance':len(performance),'quality':len(quality)}}))


def prepare_contact_guard():
    """Pre-run extension: prove nonempty contacts, retain original plan/protocol."""
    assert not (ROOT/f'reports/active/{TAG}_smoke_batch.json').exists()
    old_path=ROOT/f'configs/active/{TAG}_guards.json'
    plan=read(old_path)
    plan['runs'].append(task('fixed_bunny','discrete','bvh_contact_guard',59,discrete_bvh_validate=True))
    new_path=ROOT/f'configs/active/{TAG}_guards_v2.json'
    write(new_path,plan)
    original=ROOT/f'reports/active/{TAG}_protocol.json'
    protocol=read(original)
    protocol.update(prior_protocol_sha256=sha(original),
                    pre_run_revision='Add one bounded 59-frame fixed-cloth validation from zero: initial windows may have no contacts. No previous run data exist; originals retained.',
                    nonempty_contact_guard_frames=59)
    protocol['plan_sha256'][new_path.relative_to(ROOT).as_posix()]=sha(new_path)
    write(ROOT/f'reports/active/{TAG}_protocol_v2.json',protocol)
    print(json.dumps({'pre_run_extension':True,'guard_runs':len(plan['runs']),'originals_preserved':True}))


def hard_gate(task_row, result):
    run=ROOT/'runs/active'/task_row['name']
    problems=[]
    if result['status']!='completed':return ['run_'+result['status']]
    if not result.get('finite'):problems.append('nonfinite_endpoint')
    m=metrics(run)
    if not m['finite']:problems.append('nonfinite_state')
    if m['pcg_failures']:problems.append('pcg_cap_or_breakdown')
    if any(f['abd_min_J'] is not None and f['abd_min_J']<=0 for f in m['frames']):problems.append('abd_flip')
    if task_row['scene_key'] in ('hang','fixed_bunny'):
        limits=read(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json')['scenes'][task_row['scene_key']]['bounds']
        if m['fixed_drift_m']>limits['fixed_drift_m']:problems.append('fixed_body_drift')
    if task_row['binary']=='active':
        check=validate(run)
        if not check['passed']:problems.append('configuration_or_observed_execution')
        write(run/'component_execution_checks.json',check)
    return problems


def run_stage(stage):
    if stage in ('performance','quality'):
        smoke=read(ROOT/f'reports/active/{TAG}_smoke_batch.json')
        guards=read(ROOT/f'reports/active/{TAG}_guards_batch.json')
        assert all(not r['hard_failures'] for r in smoke['runs'] if r['scene_key']!='mixed'), 'Cloth short smoke is not cleared'
        assert all(not r['hard_failures'] or r['result']['status']=='memory_budget' for r in guards['runs']), 'Local component guards have a numerical failure'
        checks=ROOT/f'reports/active/{TAG}_guard_checks.json'
        assert checks.exists() and read(checks)['passed'], 'Same-system/tree evidence is not verified'
    plan_path=ROOT/f'configs/active/{TAG}_{stage}.json'
    if stage=='guards' and (ROOT/f'configs/active/{TAG}_guards_v2.json').exists():
        plan_path=ROOT/f'configs/active/{TAG}_guards_v2.json'
    plan=read(plan_path)
    output=ROOT/plan['report'];assert not output.exists()
    report={'stage':stage,'plan_sha256':sha(plan_path),'runs':[],'skipped':[],
            'performance_certified':False,'quality_certified':False}
    stopped=set()
    unavailable_scenes=set()
    # A main-scene pass does not certify mixed compatibility. Preserve the
    # resource failure and avoid trying further mixed arms in later stages.
    smoke_path=ROOT/f'reports/active/{TAG}_smoke_batch.json'
    if stage!='smoke' and smoke_path.exists():
        unavailable_scenes.update(r['scene_key'] for r in read(smoke_path)['runs']
                                 if r['result']['status'] in ('memory_budget','disk_reserve','timeout'))
    if stage=='quality':
        prior=ROOT/f'reports/active/{TAG}_performance_batch.json'
        if prior.exists():
            previous=read(prior)['runs']
            stopped.update(r['arm'] for r in previous if r['hard_failures'])
            unavailable_scenes.update(r['scene_key'] for r in previous if r['result']['status']=='memory_budget')
    for row in plan['runs']:
        if row['arm'] in stopped or row['scene_key'] in unavailable_scenes:
            report['skipped'].append({'name':row['name'],'reason':'Earlier resource failure in scene; no retry' if row['scene_key'] in unavailable_scenes else 'Earlier same-scene arm hard failure'})
            continue
        result=execute(row['config'],row['name'],row['binary'])
        problems=hard_gate(row,result)
        report['runs'].append(row|{'result':result,'hard_failures':problems})
        if problems:stopped.add(row['arm'])
        output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'checked':row['name'],'hard_failures':problems}),flush=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'stage':stage,'attempted':len(report['runs']),'skipped':len(report['skipped'])}),flush=True)
    return bool(stopped)


def incomplete_prefixes():
    """Exploratory comparison, keeping all requested 100-frame failures intact."""
    batch=read(ROOT/f'reports/active/{TAG}_performance_batch.json')
    groups={}
    for scene in ('hang','fixed_bunny'):
        tasks=[r for r in batch['runs'] if r['scene_key']==scene and r['repeat']=='r1']
        if len(tasks)!=5:continue
        count=min(t['result']['recorded_frames'] for t in tasks)
        if count<=0:continue
        rows=[]
        for task_row in tasks:
            run=ROOT/'runs/active'/task_row['name']
            m=metrics(run,frame_limit=count)
            with (run/'trace/frames.csv').open() as stream:csv_rows=list(csv.DictReader(stream))[:count]
            assert len(csv_rows)==count and len(m['frames'])==count and m['finite']
            frames=read(run/'output/stats.json')['frames'][:count]
            assert all('phase_ms' in frame and 'newton_exit' in frame for frame in frames)
            counters={kind:{key:sum(f.get('discrete_bvh',{}).get(kind,{}).get(key,0) for f in frames)
                for key in ('construct_calls','disabled_rebuilds','production_rebuilds','production_refits','swept_rebuilds')}
                for kind in ('face','edge')}
            rows.append({'run':run.name,'variant':task_row['variant'],'requested_run_status':task_row['result']['status'],
                'common_frames':count,'prefix_solver_seconds':sum(float(r['solver_ms']) for r in csv_rows)/1000,
                'directions':m['directions'],'pcg':m['pcg'],'pcg_failures':m['pcg_failures'],
                'max_stretch':m['max_stretch'],'p99_stretch':m['p99_stretch'],'fixed_drift_m':m['fixed_drift_m'],
                'finite':m['finite'],'discrete_bvh_counters':counters,
                'discrete_bvh_counters_available':all('discrete_bvh' in f for f in frames),
                'requested_sha256':sha(run/'requested.json'),'stats_sha256':sha(run/'output/stats.json')})
        old=next(r for r in rows if r['variant']=='old')
        stiff=next(r for r in rows if r['variant']=='stiff')
        for row in rows:
            row['old_over_candidate_prefix_time']=old['prefix_solver_seconds']/row['prefix_solver_seconds']
            row['stiff_over_candidate_prefix_time']=stiff['prefix_solver_seconds']/row['prefix_solver_seconds']
        groups[scene]={'frames':count,'rows':rows,
            'scope':'Post-run common exported prefix, one failed attempt per arm, not predeclared timing window or complete 100-frame result',
            'performance_certified':False,'quality_certified':False}
    return groups


def analyze(revision=1):
    protocol=read(protocol_path())
    bounds=read(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json')
    assert sha(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json')==protocol['old_quality_protocol_sha256']
    records=[];failures=[]
    for stage in ('smoke','guards','performance','quality'):
        path=ROOT/f'reports/active/{TAG}_{stage}_batch.json'
        batch=read(path)
        for row in batch['runs']:
            if row['result']['status']!='completed':
                failures.append({'name':row['name'],'stage':stage,'failure':row['hard_failures']});continue
            run=ROOT/'runs/active'/row['name'];m=metrics(run)
            m.update(stage=stage,variant=row['variant'],repeat=row['repeat'],scene_key=row['scene_key'],hard_failures=row['hard_failures'])
            if row['scene_key'] in bounds['scenes'] and m['config']['steps']==100:
                limits=bounds['scenes'][row['scene_key']]['bounds']
                m['frozen_material_checks']={key:{'actual':m[key],'bound':limit,'passed':m[key]<=limit} for key,limit in limits.items()}
                m['frozen_material_passed']=not row['hard_failures'] and all(x['passed'] for x in m['frozen_material_checks'].values())
            frames=read(run/'output/stats.json')['frames']
            pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            m['dot_effective_solves']=sum(bool(p.get('mas_fused_dot_effective')) for p in pcg)
            m['discrete_bvh_frame_stats']=[f.get('discrete_bvh',{}) for f in frames]
            records.append(m)
    summary={};paired={}
    for scene in ('hang','fixed_bunny'):
        rows=[r for r in records if r['stage']=='performance' and r['scene_key']==scene]
        summary[scene]={}
        for variant in ('stiff','old','dot','discrete','both'):
            selected=[r for r in rows if r['variant']==variant]
            if not selected:continue
            summary[scene][variant]={'completed':len(selected),'seconds_median':statistics.median(r['seconds'] for r in selected),
                'directions_median':statistics.median(r['directions'] for r in selected),
                'pcg_median':statistics.median(r['pcg'] for r in selected),
                'max_stretch_range':[min(r['max_stretch'] for r in selected),max(r['max_stretch'] for r in selected)],
                'material_passed':sum(r.get('frozen_material_passed',False) for r in selected)}
        paired[scene]={}
        for variant in ('dot','discrete','both'):
            ratios={}
            for baseline in ('old','stiff'):
                values=[]
                for repeat in ('r1','r2','r3'):
                    a=next((r for r in rows if r['variant']==baseline and r['repeat']==repeat and not r['hard_failures']),None)
                    b=next((r for r in rows if r['variant']==variant and r['repeat']==repeat and not r['hard_failures']),None)
                    if a and b:values.append(a['seconds']/b['seconds'])
                ratios[baseline]={'paired_ratios':values,'median':statistics.median(values) if values else None}
            paired[scene][variant]=ratios
    trajectory=[]
    for scene in ('hang','fixed_bunny','mixed'):
        quality=[r for r in records if r['stage']=='quality' and r['scene_key']==scene]
        reference=next((r for r in quality if r['variant']=='old'),None)
        if not reference:continue
        assert reference['velocity_frames']==reference['config']['steps']+1
        for candidate in quality:
            if candidate['variant']=='old':continue
            assert candidate['velocity_frames']==candidate['config']['steps']+1
            trajectory.append({'scene':scene,'variant':candidate['variant'],'reference':'old',
                'comparison':state_distance(ROOT/'runs/active'/candidate['name'],ROOT/'runs/active'/reference['name']),
                'scope':'One independent trajectory per arm, not repeated same-quality certification'})
    suffix='' if revision==1 else f'_v{revision}'
    output={'runs':records,'failures':failures,'summary':summary,
          'paired_speed':paired,'position_velocity':trajectory,'incomplete_prefix_diagnosis':incomplete_prefixes(),
          'old_quality_protocol_unchanged':True,
          'performance_certified':False,'quality_certified':False,'default_promoted':False}
    if revision>1:
        original=ROOT/f'reports/active/{TAG}_analysis.json'
        output.update(prior_analysis_sha256=sha(original),
            revision_reason='Distinguish missing original-Stiff BVH counters from measured zeros; include disabled rebuilds. Original run outcomes and analysis retained.')
    write(ROOT/f'reports/active/{TAG}_analysis{suffix}.json',output)
    print(json.dumps({'summary':summary,'paired_speed':paired,'failed_runs':failures}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','contact_guard','run','analyze'])
    parser.add_argument('--stage',choices=['smoke','guards','performance','quality'])
    parser.add_argument('--revision',type=int,choices=[1,2],default=1)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='contact_guard':prepare_contact_guard()
    elif args.action=='analyze':analyze(args.revision)
    else:
        if args.stage is None:parser.error('--stage required for run')
        raise SystemExit(int(run_stage(args.stage)))
