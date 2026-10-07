"""Bounded report-driven local IPC rounds. One stage, then an immutable review.

No defaults, stopping rules, automatic retries, or resource limits are changed.
Candidate implementation is gated by measured cost, never by phase labels.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('tools/local','tools/bench','tools/diagnostic')]
from config import expand,digest,read
from linux_runner import require,write_new,sha,verify_files
from local_identity import active_identity,tool_inventory,record,verify,verify_seal
from windows_runner import execute,gpu_lock
from pool_metrics import metrics,export_evidence
from validate_run import validate
from quality_analysis import validate_base,input_comparison

spec=importlib.util.spec_from_file_location('report_full_analysis',ROOT/'tools/full_eval/analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
SESSION=ROOT/'runs/report_execution_20261007'
REPORT=ROOT/'reports/report_execution_20261007'
BUILD=ROOT/'build/report_observer_20261007_v2'
SEAL=BUILD/'report_seal_final_capability_v3.json'
SCENES={'sphere':'cloth_sphere7_l','fixed':'cloth_fixed_bunny_l'}
FLAGS={'refit':'refit','batch':'batch','reuse':'reuse','discrete':'discrete_bvh_refit'}

def task(scene,arm,repeat,phase):
    combined=arm=='combined' or arm.startswith(('minus_','full_'))
    c=expand(dict(scene=SCENES[scene],steps=100,dt=.01,timeout_seconds=120,
        backend='ipc',execution='host' if arm=='base' else 'conditional_graph',
        mas='legacy',pcg_rho_tol=1e-4,ipc_termination='legacy',ipc_newton_tol=.01,
        ipc_cumulative_tol=.01,ipc_min_updates=6,pcg_graph_chunk=1,
        preset='combined' if combined else 'base',discrete_bvh_refit=combined,
        discrete_bvh_rebuild_interval=8,trace_velocity=arm!='base',
        contact_pool=False,diagnostics=[],profile='none',edge_query_order='raw'))
    if arm.startswith('minus_'):
        c[FLAGS[arm[6:]]]=False
        if arm=='minus_batch': c['reuse']=True
    return dict(name=f'{phase}_{scene}_{arm}_{repeat}',scene_key=scene,arm=arm,
        variant=arm,repeat=repeat,phase=phase,pair=repeat if phase in ('screen','ablation') else None,
        calibration_index=repeat if phase=='calibration' else None,
        verification_index=repeat if phase=='holdout' else None,
        binary='base' if arm=='base' else 'active',config=c,expanded_config_sha256=digest(c))

def tasks(stage):
    result=[]
    if stage=='reference':
        for scene in SCENES:
            result += [task(scene,'base',r,'calibration') for r in (1,2,3)]
            result += [task(scene,'base',r,'holdout') for r in (1,2)]
    elif stage=='screen':
        for scene in SCENES:
            for pair in (1,2,3):
                result += [task(scene,a,pair,'screen') for a in
                    (('graph','combined') if pair%2 else ('combined','graph'))]
    elif stage in ('ablation_sphere','ablation_fixed'):
        scene=stage.split('_')[1]
        for pair in (1,2,3):
            flags=list(FLAGS) if pair%2 else list(reversed(FLAGS))
            for flag in flags:
                arms=('full_'+flag,'minus_'+flag) if pair%2 else ('minus_'+flag,'full_'+flag)
                result += [task(scene,arm,pair,'ablation') for arm in arms]
    else: raise ValueError('Unknown finite stage')
    return result

def init():
    require(not SESSION.exists() and not SEAL.exists(),'Fresh session/seal required')
    old=read(REPORT/'FROZEN_REFERENCE.json')
    base=old['programs']['base']
    verify(base['sources']+base['build_evidence']+[base['exe']]+base['dlls'])
    active=active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log')
    seal=dict(old,programs={'active':active,'base':base},tools=tool_inventory(ROOT))
    write_new(SEAL,seal)
    SESSION.mkdir()
    write_new(SESSION/'plan.json',dict(schema='report_execution.v1',
        stages={s:tasks(s) for s in ('reference','screen','ablation_sphere','ablation_fixed')},
        controller=record(Path(__file__)),seal=record(SEAL),maximum_screen_runs=70,
        candidate_cost_fraction_min=.10,candidate_maximum_revisions=1,
        whole_run_saving_min=.05,quality_certified=False,performance_certified=False))
    print('Fresh observer build and original Stiff sealed; no GPU launched.',flush=True)

def analyze(t):
    out=SESSION/t['name'];r=read(out/'result.json');req=read(out/'requested.json')
    require(r['status']=='completed' and r['recorded_frames']==100,'Incomplete/resource/numerical failure: '+r['status'])
    validation=validate_base(out) if t['binary']=='base' else validate(out)
    require(validation['passed'],'Requested/resolved/native configuration mismatch')
    require(req['expanded_config']==expand(t['config']),'Different runtime configuration')
    require(export_evidence(out,100,'state')['passed'],'State missing or nonfinite')
    if t['binary']=='active':
        require(export_evidence(out,100,'velocity')['passed'],'Actual velocity missing/nonfinite')
    m=metrics(out);require(m['finite'] and not m['pcg_failures'],'Nonfinite state/PCG failure')
    frames=read(out/'output/stats.json')['frames']
    require(len(frames)==100 and not any(f.get('newton_exit')=='iteration_limit' for f in frames),'Newton cap')
    with (out/'trace/frames.csv').open() as f: csv_rows=list(csv.DictReader(f))
    timing=analysis.timing_summary(frames,csv_rows,r,analysis.read_timecost(out/'timeCost.txt'))
    row=dict(t,timing=timing,material=analysis.material_summary(m['frames']),
        material_by_frame=m['frames'],directions=m['directions'],pcg_iterations=m['pcg'],
        exits=m['exits'],configuration=validation,hard_checks_passed=True,
        quality_certified=False,performance_certified=False)
    verify_files(out,read(out/'evidence.json')['files'])
    return row

def load_rows():
    return [read(p) for p in sorted((SESSION/'analysis').glob('*.json'))]

def screen_summary(rows,scene):
    rs=[r for r in rows if r['phase']=='screen' and r['scene_key']==scene]
    ratios=[];identities=[]
    for pair in (1,2,3):
        arms={r['arm']:r for r in rs if r['repeat']==pair}
        if set(arms)!={'graph','combined'}: continue
        identities.append(input_comparison(SESSION/arms['graph']['name'],SESSION/arms['combined']['name']))
        require(identities[-1]['passed'],'Different scene/mesh/initial state')
        ratios.append(arms['graph']['timing']['solver_seconds']/arms['combined']['timing']['solver_seconds'])
    median=statistics.median(ratios) if ratios else None
    saving=1-1/median if median else None
    return dict(paired_ratios=ratios,paired_median=median,net_saving=saving,
        complete=len(ratios)==3,allow_ablation=len(ratios)==3 and saving>=.05,
        input_identities=identities)

def review(stage):
    rows=load_rows();declared=tasks(stage)
    chosen=[r for r in rows if r['name'] in {t['name'] for t in declared}]
    require(len(chosen)==len(declared),'Stage incomplete; no promotion')
    result=dict(stage=stage,completed_runs=len(chosen),quality_certified=False,performance_certified=False)
    if stage=='reference':
        result['scenes']={}
        for scene in SCENES:
            rr=[r for r in chosen if r['scene_key']==scene]
            bounds=analysis.freeze_bounds([r for r in rr if r['phase']=='calibration'])
            checks=[analysis.compare_material(r['material'],bounds) for r in rr if r['phase']=='holdout']
            result['scenes'][scene]=dict(frozen_bounds=bounds,holdouts=checks,
                material_eligible=all(c['passed'] for c in checks),
                actual_velocity_available=False,quality_certification_eligible=False)
    elif stage=='screen': result['scenes']={s:screen_summary(rows,s) for s in SCENES}
    else:
        scene=stage.split('_')[1];result['components']={}
        for flag in FLAGS:
            pairs=[]
            for r in chosen:
                if r['arm']!='minus_'+flag:continue
                ref=next(x for x in chosen if x['arm']=='full_'+flag and x['repeat']==r['repeat'])
                require(input_comparison(SESSION/r['name'],SESSION/ref['name'])['passed'],'Ablation input differs')
                pairs.append(r['timing']['solver_seconds']/ref['timing']['solver_seconds'])
            result['components'][flag]=dict(paired_ratios=pairs,paired_median=statistics.median(pairs),
                scope='Adjacent alternating paired runs; WDDM and workload differences remain diagnostic.')
    write_new(SESSION/'reviews'/f'{stage}.json',result)
    report()
    compact={s:{k:v for k,v in item.items() if k in ('material_eligible','paired_median','net_saving','allow_ablation')}
             for s,item in result.get('scenes',{}).items()}
    print(json.dumps(dict(stage=stage,completed=len(chosen),scenes=compact,
        components=result.get('components',{})),ensure_ascii=False),flush=True)

def run_observer():
    require((SESSION/'reviews/reference.json').exists(),'Frozen reference calibration must precede observer comparison')
    identity=read(ROOT/'build/report_base_observer_20261007_v2/identity.json')
    verify(identity['sources']+identity['build_evidence']+identity['objects']+[identity['exe']]+identity['dlls'])
    rows=load_rows();reference=read(SESSION/'reviews/reference.json');comparisons=[]
    for scene in SCENES:
        for repeat in (1,2):
            t=task(scene,'base',repeat,'observation');t['config']['trace_velocity']=True
            t['expanded_config_sha256']=digest(t['config'])
            out=SESSION/t['name']
            with gpu_lock(ROOT):
                if out.exists():
                    verify_files(out,read(out/'evidence.json')['files'])
                    result=read(out/'result.json')
                    require(result['status']=='configuration_failed' and result.get('exit_code')==0 and
                        result.get('finite') is True and result['recorded_frames']==100,'Prior failed attempt; no retry')
                else:result=execute(SESSION,t,identity,0)
                require(result['status'] in ('completed','configuration_failed') and result['recorded_frames']==100,'Observed baseline failed')
                validation=validate_base(out)
                require(validation['passed'] and export_evidence(out,100,'velocity')['passed'],
                    'Observed configuration or actual velocity mismatch')
                m=metrics(out);require(m['finite'] and not m['pcg_failures'],'Observed baseline numerical failure')
                material=analysis.material_summary(m['frames'])
                original=next(r for r in rows if r['scene_key']==scene and r['phase']=='calibration' and r['repeat']==repeat)
                same_input=input_comparison(out,SESSION/original['name']);require(same_input['passed'],'Observer input differs')
                check=analysis.compare_material(material,reference['scenes'][scene]['frozen_bounds'])
                comparisons.append(dict(scene=scene,repeat=repeat,actual_velocity=True,input=same_input,
                    material=check,directions=m['directions'],pcg_iterations=m['pcg'],
                    original_directions=original['directions'],original_pcg_iterations=original['pcg_iterations'],
                    endpoint_neutrality_certified=False,scope='Read-only frontend change; fresh compiler identity and finite actual velocity verified. Material range comparison does not prove trajectory equivalence.'))
            print(json.dumps(dict(observation=t['name'],seconds=result['solver_seconds'],material_range_passed=check['passed'])),flush=True)
    write_new(SESSION/'reviews/observation.json',dict(program=identity['exe'],comparisons=comparisons,
        actual_velocity_available=True,quality_certified=False,performance_denominator=False))

def run_stage(stage):
    plan=read(SESSION/'plan.json')
    require(plan['controller']==record(Path(__file__)) and plan['seal']==record(SEAL),'Controller or seal changed')
    seal=verify_seal(ROOT,SEAL)
    if stage!='reference': require((SESSION/'reviews/reference.json').exists(),'Reference review first')
    if stage.startswith('ablation_'):
        scene=stage.split('_')[1]
        require(read(SESSION/'reviews/screen.json')['scenes'][scene]['allow_ablation'],'Ablation gate failed')
    for t in tasks(stage):
        receipt=SESSION/'analysis'/f"{t['name']}.json"
        if receipt.exists():
            old=read(receipt);require(old['hard_checks_passed'] and all(old[k]==v for k,v in t.items()),'Prior failed or altered analysis')
            verify_files(SESSION/t['name'],read(SESSION/t['name']/'evidence.json')['files'])
            continue
        require(not (SESSION/t['name']).exists(),'Attempt exists; no automatic retry')
        with gpu_lock(ROOT):
            verify_seal(ROOT,SEAL)
            result=execute(SESSION,t,seal['programs'][t['binary']],0)
            require(result['status']=='completed','Run stopped: '+result['status'])
            row=analyze(t)
            write_new(receipt,row)
        print(json.dumps(dict(name=t['name'],seconds=row['timing']['solver_seconds'],
            pcg=row['pcg_iterations'],directions=row['directions'])),flush=True)
    review(stage)

def report():
    rows=load_rows();write=(REPORT/'RESULTS.json')
    write.write_text(json.dumps(dict(schema='report_execution.results.v1',rows=rows,
        quality_certified=False,performance_certified=False),indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 新报告执行优化：本机阶段结果','',
        'rho=1e-4，100帧，dt=.01。共享WDDM桌面诊断；未认证同质量或2×。',
        '原版未导出实际速度；缺口保留，不以位置差分代替。','',
        '| 场景/阶段/配置 | 求解秒 | 方向数 | PCG总量 | 最大拉伸 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['timing']['solver_seconds']:.4f} | {r['directions']} | {r['pcg_iterations']} | {r['material']['max_stretch']:.9f} |")
    lines += ['','原始数据：`runs/report_execution_20261007/`。未完成和资源失败不得进入比率。']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','run','report','observer'))
    p.add_argument('--stage',choices=('reference','screen','ablation_sphere','ablation_fixed'))
    a=p.parse_args()
    if a.action=='init':init()
    elif a.action=='run': require(a.stage is not None,'Stage required');run_stage(a.stage)
    elif a.action=='observer':run_observer()
    else:report()

if __name__=='__main__':main()
