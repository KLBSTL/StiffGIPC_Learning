"""Read-only round analysis: total cost, work, material gates and trajectories.

The original Stiff denominator remains the original executable. A separately
built velocity observer is an explicit diagnostic, not an equivalent baseline.
"""
from __future__ import annotations
from collections import Counter
import itertools
import json
import statistics
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('tools/local','tools/bench','tools/diagnostic')]
import report_rounds as rounds
from config import read
from local_identity import record
from pool_metrics import state_comparison

SESSION=rounds.SESSION;REPORT=rounds.REPORT

def workload(run):
    fs=read(run/'output/stats.json')['frames'];ns=[n for f in fs for n in f['newton']]
    directions=[n for n in ns if 'pcg' in n]
    alpha=[n['alpha'] for n in directions if 'alpha' in n]
    keys=('energy_evaluations','energy_backtracks','intersection_backtracks','initial_energy_reused')
    return dict(directions=len(directions),pcg_iterations=sum(n['pcg']['iterations'] for n in directions),
        pcg_per_direction=statistics.median(n['pcg']['iterations'] for n in directions),
        counters={k:sum(n[k] for n in directions) if all(k in n for n in directions) else None for k in keys},
        counter_coverage={k:sum(k in n for n in directions) for k in keys},
        accepted_alpha_min=min(alpha) if alpha else None,
        accepted_alpha_median=statistics.median(alpha) if alpha else None,
        accepted_alpha_full_steps=sum(v==1. for v in alpha),exits=dict(Counter(f.get('newton_exit') for f in fs)),
        active_pairs_peak=max((n.get('active_pairs',0) for n in ns),default=0),
        narrow_self_peak=max((f.get('contact_geometry',{}).get('native_narrow_self_pairs',0) for f in fs),default=0),
        narrow_ground_peak=max((f.get('contact_geometry',{}).get('native_narrow_ground_pairs',0) for f in fs),default=0))

def phase(row):
    t=row['timing'];total=1000*t['solver_seconds'];p=dict(t['phase_ms']);p['exit_assembly']=t['exit_assembly_ms']
    known=sum(v for v in p.values() if v is not None)
    return dict(solver_ms=total,phase_ms=p,percent={k:100*v/total if v is not None else None for k,v in p.items()},
                residual_ms=total-known,exhaustive=False,
                note='Diagnostic CUDA phase intervals plus separate exit assembly; residual includes host/untimed work. Missing original exit assembly remains missing.')

def peak_windows(run):
    fs=read(run/'output/stats.json')['frames']
    result={}
    for stage in ('pcg','line_search'):
        top=max(range(1,len(fs)-1),key=lambda i:sum(f['phase_ms'][stage] for f in fs[i-1:i+2]))
        result[stage]=dict(center_frame=top+1,frames=[top,top+2],
                          reference_phase_ms=sum(f['phase_ms'][stage] for f in fs[top-1:top+2]))
    return result

def generate():
    rows=rounds.load_rows();ref=read(SESSION/'reviews/reference.json')
    data=dict(schema='report.execution.analysis.v1',analyzer=record(Path(__file__)),scenes={},
        quality_certified=False,performance_certified=False,
        limitations=['Shared WDDM load is uncontrolled','Independent accepted-path CCD not certified',
                     'Baseline observer neutrality unresolved; velocities are not reconstructed',
                     'Ratios may include trajectory-dependent changes in work'])
    for scene in rounds.SCENES:
        chosen=[r for r in rows if r['scene_key']==scene];bounds=ref['scenes'][scene]['frozen_bounds']
        per_run={r['name']:dict(phase=phase(r),work=workload(SESSION/r['name']),
            material=rounds.analysis.compare_material(r['material'],bounds)) for r in chosen}
        screening=rounds.screen_summary(rows,scene)
        screened=[r for r in chosen if r['phase']=='screen']
        phase_medians={arm:{k:statistics.median(r['timing']['phase_ms'][k] for r in screened if r['arm']==arm)
            for k in ('assembly','pcg','ccd','line_search','state_update')} for arm in ('graph','combined')}
        screening['phase_medians_ms']=phase_medians
        screening['work_medians']={arm:{k:statistics.median(r[k] for r in screened if r['arm']==arm)
            for k in ('directions','pcg_iterations')} for arm in ('graph','combined')}
        screening['per_pair_state_difference']={str(i):state_comparison(SESSION/f'screen_{scene}_combined_{i}',
            SESSION/f'screen_{scene}_graph_{i}') for i in (1,2,3)}
        original=[r for r in chosen if r['binary']=='base']
        repeats={f'{a["name"]}__{b["name"]}':state_comparison(SESSION/a['name'],SESSION/b['name'])
            for a,b in itertools.combinations(original,2)}
        observer={str(i):state_comparison(SESSION/f'observation_{scene}_base_{i}',
            SESSION/f'calibration_{scene}_base_{i}') for i in (1,2)}
        observed_velocity_repeat=state_comparison(SESSION/f'observation_{scene}_base_1',
            SESSION/f'observation_{scene}_base_2')
        data['scenes'][scene]=dict(reference=ref['scenes'][scene],per_run=per_run,screening=screening,
            original_repeated_position_differences=repeats,observer_position_differences=observer,
            observed_velocity_repeat=observed_velocity_repeat,observer_neutrality_certified=False,
            peak_windows=peak_windows(SESSION/f'screen_{scene}_graph_1'))
        cost=REPORT/f'COST_{scene}.json'
        if cost.exists():data['scenes'][scene]['cost_diagnostic']=read(cost)
    (REPORT/'ANALYSIS.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 本轮组件筛选与成本分析','',
        '本机 RTX 3070 Laptop / WDDM，100帧、dt=.01、rho=1e-4。结果属于诊断；未认证同质量加速。',
        '停止规则、FP64、MAS数学作用和材料保持。组件路径的浮点顺序与原版重复波动可能改变轨迹工作量。','',
        '| 场景 | Graph→组合配对中位 | 节省求解时间 | Graph方向/PCG中位 | 组合方向/PCG中位 |',
        '|---|---:|---:|---:|---:|']
    for scene,s in data['scenes'].items():
        screen=s['screening'];w=screen['work_medians'];g=w['graph'];c=w['combined']
        lines.append(f'| {rounds.SCENES[scene]} | {screen["paired_median"]:.4f}× | {100*screen["net_saving"]:.2f}% | {g["directions"]:.0f}/{g["pcg_iterations"]:.0f} | {c["directions"]:.0f}/{c["pcg_iterations"]:.0f} |')
    lines+=['','阶段耗时为各臂三次中位，单位秒。退出轮装配另列在完整JSON；嵌套NVTX与GPU等待不再相加。','',
            '| 场景/臂 | 装配 | 整个线性阶段 | CCD | 线搜索 | 状态更新 |','|---|---:|---:|---:|---:|---:|']
    for scene,s in data['scenes'].items():
        for arm,p in s['screening']['phase_medians_ms'].items():
            lines.append(f'| {scene}/{arm} | '+ ' | '.join(f'{p[k]/1000:.4f}' for k in ('assembly','pcg','ccd','line_search','state_update'))+' |')
    lines+=['','逐次材料冻结比较、能量/回溯工作量、位置和实际速度分体差异、原版重复差异及代表窗口详见`ANALYSIS.json`。',
        '原版实际速度来自独立只读前端构建；构建身份不同且观测中性尚未证明，不替换原版性能分母。',
        '固定兔子原版两次独立复核超出冻结p99范围，禁止自动放宽、质量认证或默认推广。']
    (REPORT/'ROUND_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({s:dict(peaks=v['peak_windows'],screen=v['screening']['paired_median'],
        material_failures=sum(not r['material']['passed'] for r in v['per_run'].values())) for s,v in data['scenes'].items()}),flush=True)

if __name__=='__main__':generate()
