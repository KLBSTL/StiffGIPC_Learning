"""Bounded local checks for independent SpMV execution and compensated stopping.

No automatic long retry or certification; identities/output names are immutable.
"""
import argparse
from decimal import Decimal, localcontext
import io
import json
import math
import subprocess
import unittest

from config import ROOT, expand, read, sha
from ipc_benchmark import write, metrics
from ipc_delivery import state_distance
from run import execute
from validate_run import validate
from verify_ipc_light_delivery import CountedResult
import test_contracts

TAG='ipc_spmv_compensated_20261005'
SCENES={'hang':'cloth_hang_l','fixed_bunny':'cloth_fixed_bunny_l'}


def task(scene,variant,execution,steps=3,**changes):
    cfg={'scene':SCENES[scene],'preset':'toi_compensated' if variant in ('compensated','both') else 'graph',
         'execution':execution,'steps':steps,'dt':.01,'timeout_seconds':120,'trace_velocity':True,
         'spmv_fused_quadratic':variant in ('spmv','both')}
    cfg.update(changes);expand(cfg)
    return {'name':f'{TAG}_{scene}_{variant}_{execution}', 'scene_key':scene,
            'variant':variant,'execution':execution,'config':cfg}


def prepare():
    plans={
        'guards':[task(scene,'study','conditional_graph',2,trace_velocity=False,diagnostics=['fixed'],
                  fixed_frames='2',fixed_directions='1',fixed_spmv_quadratic_study=True) for scene in SCENES],
        'smoke':[task(scene,variant,execution) for execution in ('host','conditional_graph')
                 for scene in SCENES for variant in ('legacy','spmv','compensated','both')],
        'activation':[task('fixed_bunny','activation','conditional_graph',23,preset='toi_compensated',
                         ipc_residual_cpu_audit=True),
                      task('fixed_bunny','terminal','conditional_graph',23,ipc_residual_shadow=True,
                         ipc_terminal_audit_frame=23)],
    }
    paths=[]
    for stage,rows in plans.items():
        path=ROOT/f'configs/active/{TAG}_{stage}.json'
        write(path,{'stage':stage,'runs':rows});paths.append(path)
    output=io.StringIO()
    tests=unittest.TextTestRunner(stream=output,resultclass=CountedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful(),output.getvalue()
    native=[]
    for name in ('ipc_budget_test','ipc_residual_controller_test'):
        exe=ROOT/f'builds/active/Release/{name}.exe'
        result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30,check=True)
        native.append({'name':name,'exe_sha256':sha(exe),'stdout':result.stdout.strip(),'exit_code':result.returncode})
    cpu={'passed':True,'config_tests':tests.testsRun,'config_subtests':tests.subtests,'native':native,
         'controller_source_sha256':sha(ROOT/'sources/stiff_active/StiffGIPC/solver/ipc_residual_controller.h'),
         'controller_test_source_sha256':sha(ROOT/'sources/stiff_active/tests/ipc_residual_controller_test.cpp')}
    write(ROOT/f'reports/active/{TAG}_cpu_checks.json',cpu)
    protocol={'plan_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        'cpu_checks_sha256':sha(ROOT/f'reports/active/{TAG}_cpu_checks.json'),
        'prior_program_sha256':'9ae1c09394e9f36aed90499dae578064f28921bd96467f6c5a45206cea2a8df0',
        'old_quality_sha256':sha(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json'),
        'identity_at_prepare':sha(ROOT/'builds/active/Release/gipc.exe'),
        'activation_selection':'First active residual budget in existing fixed-bunny compensation logs is frame23; from zero, no min-update tuning',
        'finite_budget':'two fixed studies, 16 three-frame smoke, at most two 23-frame controller diagnostics; stop scene after resource/numerical failure',
        'trace_roundoff_rule':'Scalar budget arithmetic compared to Decimal.from_float inputs within 256 FP64 eps * max(1,|state|); this is not a physical-quality budget',
        'scope':'CPU/native component and short local checks only; no100/300/AutoDL',
        'performance_certified':False,'quality_certified':False,'default_promoted':False}
    write(ROOT/f'reports/active/{TAG}_protocol.json',protocol)
    print(json.dumps({'prepared':{k:len(v) for k,v in plans.items()},'cpu_checks':cpu['config_tests'],
                      'cpu_subtests':cpu['config_subtests']}))


def audit_controller(frames,cfg):
    """Independent scalar replay and ordering audit from real assembly records."""
    observations=active=audits=exits=terminal=0
    eps=256*math.ulp(1.)
    enabled=cfg['ipc_termination'] in ('gated','compensated') or cfg['ipc_residual_shadow']
    with localcontext() as context:
        context.prec=60
        D=Decimal.from_float
        for frame in frames:
            b=u=z=Decimal(1);reference=None;previous=None;accepted=0;last_alpha=None
            for node in frame['newton']:
                # Observation completeness is part of this audit. A missing
                # array must fail instead of silently looking like no activity.
                if enabled:
                    rows=node['ipc_residual_observations']
                    expected=[node['ipc_residual']]
                    assert expected[0]['kind']=='normal_assembly'
                    if 'ipc_terminal_residual' in node:
                        expected.append(node['ipc_terminal_residual'])
                        assert expected[-1]['kind']=='terminal_diagnostic_assembly'
                    assert rows==expected, 'Missing, duplicate or reordered residual observation'
                else:
                    assert all(key not in node for key in
                               ('ipc_residual','ipc_terminal_residual','ipc_residual_observations'))
                    rows=[]
                for row in rows:
                    observations+=1
                    is_terminal=row['kind']=='terminal_diagnostic_assembly'
                    expected_accepted=accepted+int(is_terminal and node.get('alpha',0)>0)
                    assert row['accepted_updates']==expected_accepted
                    if is_terminal:
                        terminal+=1
                        assert cfg['ipc_terminal_audit_frame']>0
                        assert 'ipc_residual' in node and 'ipc_terminal_residual' in node
                        assert node['ipc_residual']['kind']=='normal_assembly'
                    assert row['valid'], 'Finite diagnostic trajectory lost budget validity'
                    # These gates apply even before a reference is frozen.
                    assert row['animation_ready']==(math.isfinite(row['animation_rate']) and row['animation_rate']>.99)
                    assert row['minimum_updates_ready']==(expected_accepted>=cfg['ipc_min_updates'])
                    assert row['gated_exit_allowed']==(row['animation_ready'] and row['minimum_updates_ready'] and row['gated_ready'])
                    assert row['compensated_exit_allowed']==(row['animation_ready'] and row['minimum_updates_ready'] and row['compensated_ready'])
                    if not row['active']:
                        assert row['relative'] is None and not row['gated_ready'] and not row['compensated_ready']
                        continue
                    active+=1
                    if reference is None:
                        assert row['reference_frozen_now'] and expected_accepted==cfg['ipc_min_updates']-1
                        assert row['audits']==0 and not row['pending_before_observation']
                        reference=D(row['reference']);previous=D(row['absolute'])/reference
                    else:
                        assert D(row['reference'])==reference and not row['reference_frozen_now']
                    r=D(row['absolute'])/reference
                    if row['audited']:
                        audits+=1
                        expected_alpha=node['alpha'] if is_terminal else last_alpha
                        assert expected_alpha is not None and row['audited_alpha']==expected_alpha
                        q=Decimal(1)-D(expected_alpha)
                        d=max(Decimal(0),r-q*previous)
                        w=D(cfg['ipc_cumulative_tol'])/D(cfg['ipc_residual_rel_tol'])
                        b=q*b;u=q*u+d;z=q*z+w*d;previous=r
                    for name,value in (('beta',b),('unit_budget',u),('budget',z)):
                        actual=D(row[name]);bound=D(eps)*max(Decimal(1),abs(value),abs(actual))
                        assert abs(actual-value)<=bound, f'{name} scalar replay mismatch'
                    assert abs(D(row['relative'])-r)<=D(eps)*max(Decimal(1),abs(r))
                    assert not row['pending_after_observation']
                    gated=row['audits']>0 and row['beta']<=cfg['ipc_cumulative_tol'] and row['relative']<=cfg['ipc_residual_rel_tol']
                    assert row['gated_ready']==gated
                    assert row['compensated_ready']==(gated and row['budget']<=cfg['ipc_cumulative_tol'])
                if node.get('exit')=='compensated':
                    exits+=1
                    assert 'pcg' not in node and node['ipc_residual']['compensated_exit_allowed']
                    assert 'ipc_exit_assembly_ms' in frame
                if 'alpha' in node:
                    last_alpha=node['alpha'];accepted+=int(last_alpha>0 and math.isfinite(last_alpha))
    return {'passed':True,'observations':observations,'active_observations':active,'audited_accepts':audits,
            'compensated_exits':exits,'terminal_diagnostic_observations':terminal,
            'actual_activation_covered':active>0,'actual_compensated_exit_covered':exits>0}


def check_run(task_row,result):
    run=ROOT/'runs/active'/task_row['name']
    if result['status']!='completed':return {'passed':False,'failure':result['status']}
    check=validate(run);assert check['passed']
    m=metrics(run);assert m['finite'] and not m['pcg_failures']
    frames=read(run/'output/stats.json')['frames']
    cfg=read(run/'requested.json')['expanded_config']
    controller=audit_controller(frames,cfg)
    studies=[]
    if cfg['fixed_spmv_quadratic_study']:
        files=list((run/'fixed').glob('*_spmv_quadratic_study.json'));assert len(files)==1
        study=read(files[0]);assert study['passed']
        studies=[{'path':files[0].relative_to(ROOT).as_posix(),'sha256':sha(files[0]),'result':study}]
    return {'passed':True,'configuration':check,'controller':controller,'studies':studies,
            'metrics':m,'requested_sha256':sha(run/'requested.json'),'stats_sha256':sha(run/'output/stats.json')}


def run_stage(stage):
    protocol=read(ROOT/f'reports/active/{TAG}_protocol.json')
    assert sha(ROOT/'builds/active/Release/gipc.exe')==protocol['identity_at_prepare']
    path=ROOT/f'configs/active/{TAG}_{stage}.json'
    assert sha(path)==protocol['plan_sha256'][path.relative_to(ROOT).as_posix()]
    assert read(ROOT/f'reports/active/{TAG}_cpu_checks.json')['passed']
    assert read(ROOT/f'runs/active/{TAG}_fixtures/result.json')['passed']
    report_path=ROOT/f'reports/active/{TAG}_{stage}_batch.json';assert not report_path.exists()
    blocked=set()
    if stage!='guards':
        guards=read(ROOT/f'reports/active/{TAG}_guards_batch.json')
        assert all(r['checks']['passed'] for r in guards['runs']) and not guards['skipped']
    if stage=='activation':
        smoke=read(ROOT/f'reports/active/{TAG}_smoke_batch.json')
        blocked.update(r['scene_key'] for r in smoke['runs'] if not r['checks']['passed'])
    report={'stage':stage,'plan_sha256':sha(path),'runs':[],'skipped':[],
            'performance_certified':False,'quality_certified':False}
    for row in read(path)['runs']:
        if row['scene_key'] in blocked:
            report['skipped'].append({'name':row['name'],'reason':'Previous scene hard/resource failure; no retry'});continue
        result=execute(row['config'],row['name'])
        checks=check_run(row,result)
        report['runs'].append(row|{'result':result,'checks':checks})
        if not checks['passed']:blocked.add(row['scene_key'])
        # Always persist both successes and failures, including the final skips.
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'checked':row['name'],'status':result['status'],'passed':checks['passed']}),flush=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'stage':stage,'attempted':len(report['runs']),'skipped':len(report['skipped'])}),flush=True)


def analyze():
    batches={stage:read(ROOT/f'reports/active/{TAG}_{stage}_batch.json') for stage in ('guards','smoke','activation')}
    rows=[r for batch in batches.values() for r in batch['runs']]
    short=[]
    for scene in SCENES:
        for execution in ('host','conditional_graph'):
            group=[r for r in batches['smoke']['runs'] if r['scene_key']==scene and r['execution']==execution and r['checks']['passed']]
            old=next((r for r in group if r['variant']=='legacy'),None)
            if old is None:continue
            for row in group:
                comparison=state_distance(ROOT/'runs/active'/row['name'],ROOT/'runs/active'/old['name'])
                short.append({'scene':scene,'execution':execution,'variant':row['variant'],
                    'steps':3,'seconds':row['result']['solver_seconds'],
                    'legacy_over_candidate':old['result']['solver_seconds']/row['result']['solver_seconds'],
                    'pcg':row['checks']['metrics']['pcg'],'directions':row['checks']['metrics']['directions'],
                    'max_stretch':row['checks']['metrics']['max_stretch'],'p99_stretch':row['checks']['metrics']['p99_stretch'],
                    'position_velocity':comparison,'scope':'One complete three-frame startup observation; no long-scene or paired speed certification'})
    analysis={'exe_sha256':sha(ROOT/'builds/active/Release/gipc.exe'),
        'attempted':len(rows),'completed':sum(r['result']['status']=='completed' for r in rows),
        'failures':[{'name':r['name'],'status':r['result']['status'],'failure':r['checks'].get('failure')} for r in rows if not r['checks']['passed']],
        'unattempted':[r for batch in batches.values() for r in batch['skipped']],
        'short_comparisons':short,
        'controller_audits':[{'name':r['name'],**r['checks']['controller']} for r in rows if r['checks']['passed']],
        'studies':[s for r in rows if r['checks']['passed'] for s in r['checks']['studies']],
        'performance_certified':False,'quality_certified':False,'default_promoted':False}
    write(ROOT/f'reports/active/{TAG}_analysis.json',analysis)
    print(json.dumps({k:analysis[k] for k in ('attempted','completed','failures','unattempted')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--stage',choices=['guards','smoke','activation']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='analyze':analyze()
    else:
        if a.stage is None:p.error('--stage required for run')
        run_stage(a.stage)
