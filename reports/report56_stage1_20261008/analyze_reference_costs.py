"""Independent CPU-only analysis of the immutable 24-run reference evidence.

No sealed runner, configuration, identity, controller, or GPU API is imported.
Stage CUDA-event envelopes include device idle/host submission gaps and are not
pure kernel time. No wait/API envelope is added to the stage or solver totals.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
SESSION=ROOT/'runs/report56_stage1_20261008'
ARMS=('base','host','graph','all')
SCENES=('cloth_sphere7_l','cloth_fixed_bunny_l')
PHASES=('assembly','pcg','ccd','line_search','state_update')
COMPARES=(('base','host'),('host','graph'),('graph','all'),('base','graph'),('base','all'))

def require(value,message):
    if not value:raise ValueError(message)

def record(path):
    raw=path.read_bytes()
    return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def finite(value):return type(value) in (int,float) and math.isfinite(value)

def observed(rows,key):
    values=[r[key] for r in rows if key in r]
    require(all(finite(v) and v>=0 for v in values),'Invalid nonnegative counter: '+key)
    return {'observed_records':len(values),'expected_records':len(rows),
            'complete':bool(rows) and len(values)==len(rows),
            'sum':sum(values) if values else None,'max':max(values) if values else None}

def signed_delta(left,right):
    return None if left is None or right is None else right-left

def inspect_run(entry,identity):
    folder=SESSION/'reference'/entry['name']
    inventory=read(folder/'evidence.json')['files'];files={r['path']:r for r in inventory}
    require(len(files)==len(inventory),'Duplicate run evidence record')
    consumed=['requested.json','result.json','config_validation.json','output/stats.json',
        'output/scene.json','trace/frames.csv','trace/topology.bin','trace/masses.bin',
        'trace/boundary_types.bin','trace/body_ids.bin','trace/state_0000.bin','trace/metadata.json']
    if (folder/'trace/velocity_0000.bin').exists():consumed.append('trace/velocity_0000.bin')
    hashes={}
    for name in consumed:
        current=record(folder/name);expected=files[name]
        require(current['bytes']==expected['bytes'] and current['sha256']==expected['sha256'],
                'Run evidence changed: '+entry['name']+'/'+name)
        hashes[name]=current
    req,res,stats=(read(folder/name) for name in ('requested.json','result.json','output/stats.json'))
    config=req['expanded_config'];frames=stats['frames'];arm=entry['arm']
    require(res==entry['result'],'Ledger result differs from immutable run result')
    require(res['status']=='completed' and res['recorded_frames']==120 and res.get('finite') is True
            and len(frames)==120 and finite(res['wall_seconds']) and 0<res['wall_seconds']<=180,
            'Incomplete/nonfinite/resource-limited run')
    require(read(folder/'config_validation.json')['passed'] is True,'Native configuration failed')
    require(req['exe_sha256']==identity['exe']['sha256'] and req['source_digest']==identity['source_digest'],
            'Run program identity differs from frozen receipt')
    require(req['binary']==('base' if arm=='base' else 'active') and config['scene']==entry['scene'] and
            config['steps']==120 and config['timeout_seconds']==180 and config['dt']==.01 and
            config['mas']=='legacy' and config['pcg_rho_tol']==1e-4 and config['backend']=='ipc' and
            config['ipc_termination']=='legacy' and config['ipc_min_updates']==6 and
            config['ipc_newton_tol']==.01 and config['ipc_cumulative_tol']==.01 and
            config['pcg_graph_chunk']==1 and config['diagnostics']==[] and config['profile']=='none' and
            config['execution']==('host' if arm in ('base','host') else 'conditional_graph') and
            all(config[k]==(arm=='all') for k in ('refit','batch','reuse','discrete_bvh_refit')) and
            config['discrete_bvh_rebuild_interval']==8 and
            not any(config[k] for k in ('bounded_ccd','bvh_eligibility','contact_pool','mas_fused_dot','spmv_fused_quadratic')),
            'Reference numerical/execution protocol differs')
    require(not stats.get('failure') and not any(f.get('newton_exit')=='iteration_limit' for f in frames),
            'Reported solver failure or Newton cap')
    with (folder/'trace/frames.csv').open(encoding='utf-8-sig',newline='') as stream:timings=list(csv.DictReader(stream))
    require([int(row['frame']) for row in timings]==list(range(1,121)),'Timing frame sequence differs')
    solver=[float(row['solver_ms']) for row in timings]
    require(all(finite(t) and t>0 for t in solver),'Invalid solver timing')
    require(abs(sum(solver)/1000-res['solver_seconds'])<=1e-8,'Solver timing sum differs from run result')
    newtons=[n for f in frames for n in f['newton']];pcgs=[n['pcg'] for n in newtons if 'pcg' in n]
    require(pcgs and not any(p.get('iteration_limit') or p.get('breakdown') for p in pcgs) and
            not any(n.get('line_search_failure') for n in newtons),'Reported PCG/line-search failure')
    require(all(finite(n['alpha']) and n['alpha']>0 for n in newtons if 'alpha' in n),'Invalid accepted alpha')
    phases={key:sum(f['phase_ms'][key] for f in frames) for key in PHASES}
    require(all(finite(f['phase_ms'][key]) and f['phase_ms'][key]>=0 for f in frames for key in PHASES),
            'Invalid stage timing')
    exit_assembly=observed(frames,'ipc_exit_assembly_ms')
    graph=[p for p in pcgs if 'graph_cache_hit' in p]
    require(all(type(p['graph_cache_hit']) is bool for p in graph),'Invalid Graph hit counter')
    graph_capture=observed(graph,'graph_capture_instantiate_host_ms')
    geometry=[f.get('contact_geometry',{}) for f in frames]
    contact=[i+1 for i,(f,g) in enumerate(zip(frames,geometry)) if
        g.get('native_narrow_self_pairs',0) or g.get('native_narrow_ground_pairs',0) or
        any(n.get('active_pairs',0)>0 for n in f['newton'])]
    initial={name:hashes['trace/'+name]['sha256'] for name in
        ('topology.bin','masses.bin','boundary_types.bin','body_ids.bin','state_0000.bin','metadata.json')}
    initial['physical_scene_objects']=hashlib.sha256(json.dumps(read(folder/'output/scene.json')['objects'],
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    initial['actual_initial_velocity']=hashes.get('trace/velocity_0000.bin',{}).get('sha256')
    breakdown=sum('breakdown' in p for p in pcgs)
    perframe=[]
    for i,f in enumerate(frames):
        ns=f['newton'];ps=[n['pcg'] for n in ns if 'pcg' in n]
        perframe.append({'frame':i+1,'solver_ms':solver[i],'phase_ms':f['phase_ms'],
            'directions':len(ps),'pcg_iterations':observed(ps,'iterations')['sum'],
            'energy_evaluations':observed(ns,'energy_evaluations')['sum'],
            'energy_backtracks':observed(ns,'energy_backtracks')['sum'],
            'intersection_backtracks':observed(ns,'intersection_backtracks')['sum'],
            'exit_assembly_ms':f.get('ipc_exit_assembly_ms'),
            'contact_observed':i+1 in contact,
            'graph_hits':sum(p.get('graph_cache_hit') is True for p in ps),
            'graph_misses':sum(p.get('graph_cache_hit') is False for p in ps)})
    return {'name':entry['name'],'scene':entry['scene'],'arm':arm,'repeat':entry['repeat'],
        'solver_ms':sum(solver),'wall_seconds':res['wall_seconds'],
        'phase_ms':phases,'phase_fraction_of_solver':{k:v/sum(solver) for k,v in phases.items()},
        'phase_sum_ms':sum(phases.values()),'solver_minus_recorded_phases_ms':sum(solver)-sum(phases.values()),
        'exit_assembly_ms':exit_assembly,
        'directions':len(pcgs),'pcg_iterations':observed(pcgs,'iterations'),
        'energy_evaluations':observed(newtons,'energy_evaluations'),
        'energy_backtracks':observed(newtons,'energy_backtracks'),
        'intersection_backtracks':observed(newtons,'intersection_backtracks'),
        'initial_energy_reuse_records':sum(n.get('initial_energy_reused') is True for n in newtons),
        'graph':{'observed_records':len(graph),'cache_hits':sum(p['graph_cache_hit'] for p in graph),
            'cache_misses':sum(not p['graph_cache_hit'] for p in graph),
            'hit_fraction':sum(p['graph_cache_hit'] for p in graph)/len(graph) if graph else None,
            'capture_instantiate_host_ms':graph_capture,
            'captures_counter_max':max((p.get('graph_captures_total',0) for p in graph),default=None),
            'invalidations_counter_max':max((p.get('graph_invalidations_total',0) for p in graph),default=None),
            'counter_scope':'Cumulative native maxima are not summed across solves; cache hit/miss records are counted once.'},
        'contact_observed_frames':contact,'newton_exits':dict(collections.Counter(f['newton_exit'] for f in frames)),
        'actual_initial_velocity_available':initial['actual_initial_velocity'] is not None,
        'pcg_breakdown_telemetry_complete':breakdown==len(pcgs),
        'initial_inputs':initial,'consumed_evidence':hashes,'per_frame':perframe}

def compare(left,right):
    initial={key:left['initial_inputs'][key]==right['initial_inputs'][key] for key in
        left['initial_inputs'] if key!='actual_initial_velocity'}
    velocity=left['initial_inputs']['actual_initial_velocity'] is not None and right['initial_inputs']['actual_initial_velocity'] is not None
    if velocity:initial['actual_initial_velocity']=left['initial_inputs']['actual_initial_velocity']==right['initial_inputs']['actual_initial_velocity']
    require(all(initial.values()),'Initial physical inputs differ across paired arms')
    phase_delta={key:left['phase_ms'][key]-right['phase_ms'][key] for key in PHASES}
    return {'repeat':left['repeat'],'reference_arm':left['arm'],'target_arm':right['arm'],
        'ratio':left['solver_ms']/right['solver_ms'],'solver_saved_ms':left['solver_ms']-right['solver_ms'],
        'phase_reduction_ms':phase_delta,
        'directions_target_minus_reference':right['directions']-left['directions'],
        'pcg_iterations_target_minus_reference':signed_delta(left['pcg_iterations']['sum'],right['pcg_iterations']['sum']),
        'energy_evaluations_target_minus_reference':signed_delta(left['energy_evaluations']['sum'],right['energy_evaluations']['sum']),
        'energy_backtracks_target_minus_reference':signed_delta(left['energy_backtracks']['sum'],right['energy_backtracks']['sum']),
        'initial_inputs_match':True,'initial_actual_velocity_comparable':velocity}

def profile_frames(rows,scene):
    graph=[row for row in rows if row['scene']==scene and row['arm']=='graph']
    if len(graph)!=3:return None
    choices=[]
    for index in range(3,120):
        fs=[row['per_frame'][index] for row in graph]
        if not all(f['contact_observed'] for f in fs):continue
        choices.append({'frame':index+1,'window':[max(1,index),min(120,index+2)],
            'median_pcg_ms':statistics.median(f['phase_ms']['pcg'] for f in fs),
            'median_ccd_line_search_ms':statistics.median(f['phase_ms']['ccd']+f['phase_ms']['line_search'] for f in fs),
            'directions':[f['directions'] for f in fs],'pcg_iterations':[f['pcg_iterations'] for f in fs]})
    stable=[c for c in choices if len(set(c['directions']))==1]
    return {'linear_peak':max(choices,key=lambda c:c['median_pcg_ms']),
        'linear_heavy_stable_directions':max(stable,key=lambda c:c['median_pcg_ms']) if stable else None,
        'ccd_line_search_peak':max(choices,key=lambda c:c['median_ccd_line_search_ms']),
        'scope':'Three-repeat median of post-contact Graph stage envelopes; selected heavy windows require replay from frame zero and are not whole-scene GPU averages.'}

def analyze(allow_partial=False):
    ledger_path=SESSION/'REFERENCE_LEDGER.json';identity_path=SESSION/'REFERENCE_IDENTITY.json'
    ledger_identity=record(ledger_path);ledger=read(ledger_path);identity=read(identity_path)
    require(ledger['planned_runs']==24 and ledger['status'] in ('running','completed','failed'),'Unexpected ledger')
    require(allow_partial or ledger['status']=='completed','Reference still partial; use --allow-partial explicitly')
    expected=identity['tasks'];entries=ledger['rows']
    require([r['name'] for r in entries]==[r['name'] for r in expected[:len(entries)]],'Ledger is not immutable task prefix')
    if ledger['status']=='completed':require(len(entries)==24,'Completed ledger has missing runs')
    rows=[inspect_run(entry,identity['programs']['base' if entry['arm']=='base' else 'active']) for entry in entries]
    scenes=[]
    for scene in SCENES:
        selected=[row for row in rows if row['scene']==scene]
        groups={arm:[r for r in selected if r['arm']==arm] for arm in ARMS}
        comparisons={}
        for left,right in COMPARES:
            paired=[]
            for repeat in range(1,4):
                a=next((r for r in groups[left] if r['repeat']==repeat),None)
                b=next((r for r in groups[right] if r['repeat']==repeat),None)
                if a and b:paired.append(compare(a,b))
            comparisons[left+'_over_'+right]={'pairs':paired,
                'median_ratio':statistics.median(p['ratio'] for p in paired) if paired else None,
                'ratio_range':[min(p['ratio'] for p in paired),max(p['ratio'] for p in paired)] if paired else None,
                'median_phase_reduction_ms':{k:statistics.median(p['phase_reduction_ms'][k] for p in paired) for k in PHASES} if paired else None}
        scene_row={'scene':scene,'runs_per_arm':{arm:len(rs) for arm,rs in groups.items()},
            'median_solver_seconds':{arm:statistics.median(r['solver_ms'] for r in rs)/1000 if rs else None for arm,rs in groups.items()},
            'median_phase_ms':{arm:{k:statistics.median(r['phase_ms'][k] for r in rs) for k in PHASES} if rs else None for arm,rs in groups.items()},
            'workloads':{arm:[{'repeat':r['repeat'],'directions':r['directions'],'pcg_iterations':r['pcg_iterations']['sum'],
                'energy_evaluations':r['energy_evaluations']['sum'],'energy_backtracks':r['energy_backtracks']['sum'],
                'graph_cache_hits':r['graph']['cache_hits'],'graph_cache_misses':r['graph']['cache_misses'],
                'graph_capture_host_ms':r['graph']['capture_instantiate_host_ms']['sum']} for r in rs] for arm,rs in groups.items()},
            'comparisons':comparisons,'profile_frame_candidates':profile_frames(rows,scene)}
        if all(len(groups[arm])==3 for arm in ARMS):
            all_ratios=comparisons['base_over_all']['median_ratio']
            scene_row['cost_interpretation']={
                'graph_pcg_iteration_changes':[p['pcg_iterations_target_minus_reference'] for p in comparisons['host_over_graph']['pairs']],
                'all_energy_evaluation_changes':[p['energy_evaluations_target_minus_reference'] for p in comparisons['graph_over_all']['pairs']],
                'graph_median_hit_fraction':statistics.median(r['graph']['hit_fraction'] for r in groups['graph']),
                'graph_median_capture_host_fraction_of_solver':statistics.median(r['graph']['capture_instantiate_host_ms']['sum']/r['solver_ms'] for r in groups['graph']),
                'all_median_pcg_fraction_of_solver':statistics.median(r['phase_fraction_of_solver']['pcg'] for r in groups['all']),
                'all_median_ccd_line_search_fraction_of_solver':statistics.median(r['phase_fraction_of_solver']['ccd']+r['phase_fraction_of_solver']['line_search'] for r in groups['all']),
                'additional_all_solver_reduction_for_observed_2x_stiff':max(0,1-all_ratios/2),
                'counter_identity':{arm:[{'repeat':r['repeat'],
                    'energy_evaluations_equal_two_per_direction_plus_backtracks_minus_reuse':
                        r['energy_evaluations']['sum']==2*r['directions']+r['energy_backtracks']['sum']-r['initial_energy_reuse_records']}
                    for r in groups[arm]] for arm in ('host','graph','all')},
                'limits':'Capture host fraction is the observed capture/instantiate envelope only. It does not bound all possible Graph management changes or certify a speedup.'}
        scenes.append(scene_row)
    require(record(ledger_path)==ledger_identity,'Ledger changed during analysis; rerun analysis without changing evidence')
    baselines=[r for r in rows if r['arm']=='base']
    return {'schema':'report56_reference_cost_analysis.v1','reference_status':ledger['status'],
        'completed_runs':len(rows),'planned_runs':24,'input_ledger':ledger_identity,'input_identity':record(identity_path),
        'analyzer':record(Path(__file__)),'scenes':scenes,'runs':rows,
        'performance_certified':False,'physical_quality_certified':False,
        'baseline_telemetry':{'actual_initial_velocity_available':bool(baselines) and all(r['actual_initial_velocity_available'] for r in baselines),
            'pcg_breakdown_telemetry_complete':bool(baselines) and all(r['pcg_breakdown_telemetry_complete'] for r in baselines),
            'energy_evaluations_available':bool(baselines) and all(r['energy_evaluations']['complete'] for r in baselines),
            'energy_backtracks_available':bool(baselines) and all(r['energy_backtracks']['complete'] for r in baselines),
            'scope':'Missing baseline telemetry is a comparison limit; no velocity reconstruction or zero imputation is performed.'},
        'interpretation':{'solver':'CPU synchronized solver_ms sums from 120 complete frames; paired ratios describe shared WDDM observations.',
            'phase':'Native CUDA-event stage envelopes include device work, host submission gaps and device idle; they are not pure GPU kernel costs.',
            'addition':'No CUDA API waits, CPU scopes, Graph capture time or exit assembly is added to solver/stage totals. Exit assembly and uncovered differences are reported separately.',
            'workload':'Direction/PCG differences and missing baseline energy/backtrack telemetry limit attribution to identical numerical trajectories.',
            'quality':'Actual Stiff velocity and full PCG breakdown telemetry are absent; no reconstructed velocity, accepted-path CCD certification, material tolerance relaxation or promotion of the prior failed quality gate.'}}

def markdown(result):
    lines=['# 120 帧四臂参考成本分析','',f"独立 CPU 分析读取 {result['completed_runs']}/24 次完整运行，ledger={result['reference_status']}。所有比率为本机共享 WDDM 诊断观测，不是性能或质量认证。",'',
        '四臂固定 FP64、材料、legacy MAS、rho=1e-4、dt=.01、累计阈值 .01、min6 和完整 CCD。all = Graph K1＋FullCCD refit／批量能量／能量复用／普通 BVH refit（间隔 8）。','']
    for scene in result['scenes']:
        lines += ['## '+scene['scene'],'','| 臂 | 120 帧 solver 秒中位数 | 方向数（3轮） | PCG次数（3轮） | 能量评估（3轮） | 能量回溯（3轮） |','|---|---:|---|---|---|---|']
        for arm in ARMS:
            ws=scene['workloads'][arm]
            values=lambda key:','.join('未观测' if w[key] is None else str(w[key]) for w in ws)
            seconds=scene['median_solver_seconds'][arm]
            lines.append(f"| {arm} | {seconds:.6f} | {values('directions')} | {values('pcg_iterations')} | {values('energy_evaluations')} | {values('energy_backtracks')} |" if seconds is not None else f'| {arm} | 未完成 | | | | |')
        lines += ['','| 配对比率（参考/目标） | 3轮比率中位数 | 最小—最大 | PCG阶段减少ms中位数 | CCD阶段减少ms中位数 | LS阶段减少ms中位数 |','|---|---:|---|---:|---:|---:|']
        for key,c in scene['comparisons'].items():
            if c['median_ratio'] is not None:
                d=c['median_phase_reduction_ms'];lo,hi=c['ratio_range']
                lines.append(f"| {key} | {c['median_ratio']:.4f}× | {lo:.4f}—{hi:.4f}× | {d['pcg']:.3f} | {d['ccd']:.3f} | {d['line_search']:.3f} |")
        if scene.get('cost_interpretation'):
            ci=scene['cost_interpretation'];hg=scene['comparisons']['host_over_graph'];ga=scene['comparisons']['graph_over_all']
            slow=100*(1/scene['comparisons']['base_over_host']['median_ratio']-1)
            lines += ['',f"活动host相对Stiff的配对中位耗时增加{slow:.2f}%。Graph对照中PCG阶段减少{hg['median_phase_reduction_ms']['pcg']/1000:.3f}秒，而PCG次数变化为{ci['graph_pcg_iteration_changes']}；收益主要出现在执行阶段，不能解释为一致减少迭代。",
                f"四组件进一步减少CCD阶段{ga['median_phase_reduction_ms']['ccd']/1000:.3f}秒、LS阶段{ga['median_phase_reduction_ms']['line_search']/1000:.3f}秒，能量评估次数变化为{ci['all_energy_evaluation_changes']}。活动臂的能量计数均满足2×方向数＋回溯数−初值能量复用数；这支持复用减少评估，但不能从组合对照分离batch、refit、reuse各自的收益。",
                f"Graph缓存命中率中位{100*ci['graph_median_hit_fraction']:.2f}%，捕获／实例化host包含时间仅占Graph整场{100*ci['graph_median_capture_host_fraction_of_solver']:.3f}%。仅消除已观测的这部分捕获时间不足以支撑2×目标。all剩余PCG阶段占solver约{100*ci['all_median_pcg_fraction_of_solver']:.2f}%，CCD＋LS约{100*ci['all_median_ccd_line_search_fraction_of_solver']:.2f}%；相对本轮Stiff的观测2×目标，还需把all整场进一步缩短约{100*ci['additional_all_solver_reduction_for_observed_2x_stiff']:.2f}%。这些是诊断预算，不是GPU内核比例或质量认证。"]
        lines += ['','| 臂 | assembly / PCG / CCD / LS / update 阶段ms中位数 | Graph hits / misses（3轮） | capture＋instantiate host ms（3轮） |','|---|---|---|---|']
        for arm in ARMS:
            phases=scene['median_phase_ms'][arm]
            if phases:
                ws=scene['workloads'][arm]
                lines.append('| '+arm+' | '+' / '.join(f'{phases[k]:.3f}' for k in PHASES)+' | '+', '.join(f"{w['graph_cache_hits']}/{w['graph_cache_misses']}" for w in ws)+' | '+', '.join('未观测' if w['graph_capture_host_ms'] is None else f"{w['graph_capture_host_ms']:.3f}" for w in ws)+' |')
        p=scene['profile_frame_candidates']
        if p:
            stable=p['linear_heavy_stable_directions'];peak=p['ccd_line_search_peak']
            lines += ['',f"从零捕获候选：稳定方向数的重线性帧 {stable['frame']}（PCG中位{stable['median_pcg_ms']:.3f}ms，方向{stable['directions']}）；CCD／LS高成本帧 {peak['frame']}（两阶段中位合计{peak['median_ccd_line_search_ms']:.3f}ms，方向{peak['directions']}）。高成本帧只用于诊断，不代表整场平均。"]
        lines.append('')
    lines += ['## 归因与边界','',
        'base/host 反映冻结 Stiff 与已纠错活动 host 的总体差异，不能归为 Graph；host/graph 才是同一活动程序下的 Graph 执行对照；graph/all 反映四个既有执行组件的组合收益。每组同时给出方向数和 PCG次数变化，工作量变化不能伪装成每次操作的纯加速。',
        'phase_ms 是 CUDA event 端点之间的阶段包含时间，会覆盖提交间隙和设备空闲。它不是 Nsight kernel 活跃时间，也不能再加 CUDA API等待、CPU包含时间、Graph捕获时间或终止装配。JSON逐帧保留了solver与阶段差额、终止装配和可观测计数；差额不自动解释为可消除的host开销。',
        '原 Stiff缺少实际速度、能量评估／回溯及完整PCG breakdown观测；缺失计数记为null／未观测，不记作零。所有初始位置、质量、边界、拓扑、body与scene对象按配对核对；初始实际速度仅对可用的活动臂核对。物理质量、独立accepted-path CCD及原失败质量门控没有被本分析重新认证。','',
        '复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_reference_costs.py`。脚本只读取已记录证据并写本报告及JSON，不启动GPU、导入或修改封存runner／controller。','']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    result=analyze(args.allow_partial)
    output=REPORT/'REFERENCE_COST_ANALYSIS.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    output.with_suffix('.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps({'reference_status':result['reference_status'],'completed_runs':result['completed_runs'],
        'scenes':[{k:s[k] for k in ('scene','median_solver_seconds')}|{'paired_median_ratios':{k:c['median_ratio'] for k,c in s['comparisons'].items()}} for s in result['scenes']]}))

if __name__=='__main__':main()
