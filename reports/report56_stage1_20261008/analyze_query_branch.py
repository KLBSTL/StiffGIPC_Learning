"""Read completed Stage-1 probe evidence; write only QUERY_BRANCH_DECISION artifacts.

CPU-only: E:/Anaconda/envs/DL/python.exe <this file>
No GPU, build, native-source mutation, or performance certification.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/report56_stage1_20261008'
CASES=('cloth_sphere7_l','cloth_fixed_bunny_l')
KINDS=('DCD_VF','DCD_EE','FullCCD_VF','FullCCD_EE')
METRICS=('node_aabb_tests','internal_pops','leaf_overlaps','narrow_calls','max_pending_stack',
         'launch_warp_max_node_aabb_tests')
IDENTITIES={}
CHECKS=0

def require(condition,message):
    global CHECKS
    CHECKS+=1
    if not condition:raise ValueError(message)

def identity(path):
    path=path.resolve();data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def load_json(path):
    IDENTITIES[str(path.resolve())]=identity(path)
    return json.loads(path.read_text(encoding='utf-8'))

def load_jsonl(path):
    IDENTITIES[str(path.resolve())]=identity(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def spread(values):
    values=list(values)
    return {'min':min(values),'median':statistics.median(values),'max':max(values)} if values else None

def context(row):
    return {key:row.get(key) for key in ('frame','phase','outer','inner','contact_model_id','linear_system_id','alpha')}

def workload(rows,kind):
    pairs=[(row,q) for row in rows for q in row['sample']['queries'] if q['kind']==kind]
    summaries=[q for _,q in pairs];result={'first_samples':len(pairs),
        'queries_per_sample':sorted({q['query_count'] for q in summaries})}
    require(len(pairs)==120,'Each query kind needs 120 first samples: '+kind)
    for metric in METRICS:
        count=sum(q[metric]['count'] for q in summaries);total=sum(q[metric]['sum'] for q in summaries)
        require(count>0,'Nonempty metric sample required')
        result[metric]={'sampled_count':count,'sampled_sum':total,'weighted_mean':total/count,
            'maximum_across_samples':max(q[metric]['max'] for q in summaries),
            'per_first_launch_quantiles':{p:spread(q[metric][p] for q in summaries) for p in ('p50','p95','p99')}}
    peak=max(pairs,key=lambda pair:pair[1]['node_aabb_tests']['max'])
    ratio_peak=max(pairs,key=lambda pair:pair[1]['warp_max_mean_over_query_mean'])
    result['sampled_query_count']=sum(q['query_count'] for q in summaries)
    result['warp_mean_over_query_mean']=result['launch_warp_max_node_aabb_tests']['weighted_mean']/result['node_aabb_tests']['weighted_mean']
    result['per_first_launch_warp_ratio']=spread(q['warp_max_mean_over_query_mean'] for q in summaries)
    result['node_max_context']=context(peak[0]);result['node_max_top16_records']=peak[1]['heaviest_queries']
    result['warp_ratio_max_context']=context(ratio_peak[0])
    if kind.startswith('FullCCD'):require(result['narrow_calls']['sampled_sum']==0,'FullCCD is candidate-only')
    require(result['max_pending_stack']['maximum_across_samples']<=65,'Stack bound')
    return result

def cost_summary(rows):
    groups=defaultdict(list)
    by_id={row['scope_id']:row for row in rows}
    for row in rows:groups[(row['sample_kind'],row['stage'])].append(row)
    def totals(members):
        return {'records':len(members),'cpu_inclusive_sum_ms':sum(r['cpu_submit_ms'] for r in members),
            'gpu_event_inclusive_sum_ms':sum(r.get('gpu_interval_ms',0) or 0 for r in members),
            'per_frame_record_count':dict(sorted(Counter(r['frame'] for r in members).items()))}
    selected={stage:totals(groups[('production',stage)]) for stage in (
        'ipc.physical_frame','collision.discrete_query','collision.swept_query','collision.self_ccd',
        'collision.discrete_bvh_build','collision.swept_bvh_build_or_refit','ipc.line_search')}
    diagnostic=groups[('diagnostic_bvh_query_workload','diagnostic.bvh_query_workload')]
    selected['diagnostic.bvh_query_workload']=totals(diagnostic)
    parents=Counter(by_id.get(r['parent_scope_id'],{}).get('stage','missing') for r in diagnostic)
    require(dict(parents)=={'collision.swept_query':120,'collision.discrete_query':119},'Exact diagnostic parent coverage')
    require(selected['ipc.physical_frame']['records']==120,'120 physical-frame envelopes')
    require(all(r['parent_scope_id']==0 for r in groups[('production','ipc.physical_frame')]),'Only outer physical-frame envelopes')
    nested={}
    for stage in ('collision.discrete_query','collision.swept_query'):
        children=[r for r in diagnostic if by_id[r['parent_scope_id']]['stage']==stage]
        child_totals=totals(children);parent_totals=selected[stage]
        nested[stage]={'parent':parent_totals,'private_child':child_totals,
            'cpu_parent_minus_private_child_ms':parent_totals['cpu_inclusive_sum_ms']-child_totals['cpu_inclusive_sum_ms'],
            'gpu_event_parent_minus_private_child_ms':parent_totals['gpu_event_inclusive_sum_ms']-child_totals['gpu_event_inclusive_sum_ms'],
            'residual_is_exclusive_production_kernel_cost':False}
    return {'scope':'Inclusive CUDA-event/CPU envelopes include nested diagnostics and submission gaps, not exclusive kernel costs.',
        'envelopes':selected,'diagnostic_parent_stage_counts':dict(parents),
        'query_parent_child_breakdown':nested,
        'all_nonproduction_sample_kinds':dict(Counter(r['sample_kind'] for r in rows if r['sample_kind']!='production'))}

def static_resources(path):
    IDENTITIES[str(path.resolve())]=identity(path)
    lines=path.read_text(encoding='utf-8').splitlines()
    targets={'_selfQuery_vf':'_Z13_selfQuery_vfP','_selfQuery_ee':'_Z13_selfQuery_eeP',
        '_selfQuery_vf_ccd':'_Z17_selfQuery_vf_ccdP','_selfQuery_ee_ccd':'_Z17_selfQuery_ee_ccdP',
        '_edgeTriIntersectionQuery':'_Z25_edgeTriIntersectionQueryP'}
    found={}
    for line,text in enumerate(lines):
        for name,prefix in targets.items():
            if text.strip().startswith('Function '+prefix):
                values={key:int(value) for key,value in re.findall(r'\b(REG|STACK|LOCAL|SHARED):(\d+)',lines[line+1])}
                found[name]={'mangled_symbol':text.strip().removeprefix('Function '),'line':line+1,**values}
    require(set(found)==set(targets),'Five exact production cubin resource entries')
    require(all(v['STACK']>0 for v in found.values()),'Nonzero compiled stack footprint')
    return found

def dynamic_counters(program_sha):
    folder=RUNS/'query_counters_f49';path=folder/'counter.csv'
    plan=load_json(OUT/'NCU_QUERY_CHECK_PLAN.json')
    receipt=load_json(folder/'result.json');validation=load_json(folder/'capture_validation.json')
    config=load_json(folder/'resolved_config.json');manifest=load_json(folder/'build_manifest.json')
    IDENTITIES[str(path.resolve())]=identity(path)
    rows=list(csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()))
    units=[r for r in rows if not r.get('ID')];data=[r for r in rows if r.get('ID')]
    require(len(units)==1 and len(data)==1,'One units row + one actual wide CSV kernel row')
    unit=units[0];row=data[0]
    require(row['Kernel Name'].startswith('_selfQuery_ee('),'Actual production DCD EE kernel')
    require(plan['max_attempts']==1 and plan['max_profiled_kernels']==1 and plan['selected_frame']==49,'Bounded single-kernel plan')
    require(plan['program_sha256']==program_sha==manifest['exe']['sha256'],'Same new observer/probe program identity')
    require(receipt['exit_code']==0 and receipt['recorded_frames']==120 and receipt['finite']
            and receipt['cleanup_owned_job_empty'] and validation['passed'],'Successful GPU capture independent of parser receipt')
    def number(name):return float(row[name].replace(',',''))
    metrics={name:{'value':number(name),'unit':unit[name]} for name in plan['metrics']}
    require(metrics['gpu__time_duration.sum']['unit']=='ns','Counter duration units')
    require(metrics['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum']['unit']=='sector','Counter local traffic units')
    launch={name:number(name) for name in ('launch__grid_size','launch__block_size','launch__sm_count',
        'launch__registers_per_thread','launch__registers_per_thread_allocated','launch__waves_per_multiprocessor',
        'launch__occupancy_limit_registers')}
    require(number('profiler__replayer_passes')==1,'Actual single profiler pass')
    return {'collector_original_receipt':{'status':receipt['status'],'error':receipt.get('error'),'exit_code':receipt['exit_code']},
        'capture_validation_passed':True,'cpu_wide_csv_parsed':True,'gpu_rerun':False,
        'selected_frame':49,'kernel':row['Kernel Name'],'program_sha256':program_sha,
        'profiler_replayer_passes':1,
        'block_size':row['Block Size'],'grid_size':row['Grid Size'],'metrics':metrics,'launch':launch,
        'resolved_config':config,'report_identity':identity(folder/'query.ncu-rep'),
        'conclusion':'Actual dynamic local-memory traffic is observed for one DCD EE launch. Stack-array/spill attribution, query/narrow timing split and whole-run savings remain unproven.'}

def analyze_case(case,ledger_rows):
    evidence={}
    for mode in ('observer_off','probe'):
        folder=RUNS/mode/case
        result=load_json(folder/'result.json');cfg=load_json(folder/'resolved_config.json')
        manifest=load_json(folder/'build_manifest.json');validation=load_json(folder/'config_validation.json')
        files=load_json(folder/'evidence.json')['files']
        require(result['status']=='completed' and result['recorded_frames']==120 and result['exit_code']==0,'Completed closed run')
        require(result['cleanup_owned_job_empty'] and result['finite'] and validation['passed'],'Run/config/finite guards')
        for item in files:
            if item['path'] in ('bvh_query_probe.jsonl','cost.jsonl','result.json','resolved_config.json','build_manifest.json'):
                current=identity(folder/item['path'])
                require(current['bytes']==item['bytes'] and current['sha256']==item['sha256'],'Closed evidence input identity')
        summary=ledger_rows[(mode,case)]['summary']
        require(summary['guards_passed'] and summary['active_native_guards_observed'],'Native guards observed')
        evidence[mode]={'result':{key:result.get(key) for key in ('status','recorded_frames','solver_seconds','wall_seconds',
            'finite','performance_certified','physical_quality_certified','timing_is_diagnostic')},
            'summary':summary,'resolved_config':cfg,'build_manifest_sha256':identity(folder/'build_manifest.json')['sha256']}
    off=evidence['observer_off'];probe=evidence['probe']
    require(off['build_manifest_sha256']==probe['build_manifest_sha256'],'Same binary/source build manifest')
    require(not off['resolved_config']['cost_observation']['active'] and not off['resolved_config']['bvh_query_probe']['enabled'],'Off observers disabled')
    require(probe['resolved_config']['cost_observation']['gpu_events_effective'] and probe['resolved_config']['bvh_query_probe']['enabled'],'Probe observers enabled')
    rows=load_jsonl(RUNS/'probe'/case/'bvh_query_probe.jsonl')
    costs=load_jsonl(RUNS/'probe'/case/'cost.jsonl')
    require(len(rows)==240,'240 closed BVH first samples')
    require({(r['frame'],r['phase']) for r in rows}=={(f,phase) for f in range(1,121) for phase in ('DCD','FullCCD')},'Exactly one sample per frame/phase')
    for row in rows:
        require(row['passed'] and row['sample']['typed_multiset_equal'],'Private typed multiset assertion')
        require(row['production_outputs_untouched'] and row['tree_storage_untouched'] and row['first_successful_query_only'],'Sampling contract flags')
        require(not row['sample']['atomic_array_order_compared'],'Do not use atomic output ordering as oracle')
        require({q['kind'] for q in row['sample']['queries']}=={row['phase']+'_VF',row['phase']+'_EE'},'VF/EE phase identity')
        require(all(not q['raw_records_included'] for q in row['sample']['queries']),'Quantile pooling limitation')
    contexts={phase:[context(r) for r in rows if r['phase']==phase] for phase in ('DCD','FullCCD')}
    require(Counter((r['outer'],r['inner']) for r in contexts['DCD'])==Counter({(-1,0):119,(-1,-1):1}),'DCD exact selected context')
    require(Counter((r['outer'],r['inner']) for r in contexts['FullCCD'])==Counter({(-1,0):120}),'FullCCD exact selected context')
    event_cost=cost_summary(costs)
    host={phase:sum(r['diagnostic_host_ms'] for r in rows if r['phase']==phase) for phase in ('DCD','FullCCD')}
    return {'case':case,'arms':evidence,'workload':{kind:workload(rows,kind) for kind in KINDS},
        'selected_contexts':contexts,'cost':event_cost,'private_diagnostic_host_sum_ms':host,
        'alpha_range_fullccd':spread(r['alpha'] for r in rows if r['phase']=='FullCCD'),
        'gross_observer_arm_solver_ratio':probe['result']['solver_seconds']/off['result']['solver_seconds'],
        'observer_off_reference_five_percent_seconds':.05*off['result']['solver_seconds'],
        'branch_gate':{'whole_net_savings_ge_five_percent_proven':False,'measured_whole_net_savings':None,
            'decision':'defer_stackless_production_branch','reason':'No exclusive full-run traversal cost or matched candidate net saving; first-query counts and diagnostic parent envelopes cannot supply it.'}}

def markdown(result):
    lines=['# Query 分支决定：暂缓 stackless 生产视图','',
        '现有证据尚未建立“扣除视图生成／维护后，完整120帧净省至少5%”的启动门槛。决定暂缓生产 stackless 分支，保留碰撞优化方向。新动态计数支持把 DCD EE 列为下一项因果检查的首选，但未证明净潜力低于或达到5%；缺口是遍历的独占时间与替代视图的净收益，而不是查询正确性烟测。','',
        '原计划要求先满足整场净潜力门槛，再实现保守视图；局部完整 query（含准备）至少15%改进，三对从零120帧整场至少5%、另一主场景退化不超过3%、完整质量门通过后才保留。当前没有 stackless A/B 或视图维护计时。','',
        '## 数据与采样范围','',
        '四次运行已关闭：同一新程序 observer-off/probe 各两场景，全部120帧、exit=0、有限值、配置及既有原生守卫通过。诊断 flag 不构成独立物理质量或性能认证。每场480个 VF/EE launch summary 来自240个私有样本（DCD120、FullCCD120），保留 typed multiset、duplicates／orientation／MatIndex，未将原子输出数组顺序作正确性标准。','',
        'DCD 第1个样本来自初始化、contact_model_id=0、linear_system_id=0、outer=-1、inner=-1；其余119个 DCD 和全部120个 FullCCD 都是 outer=-1、inner=0 的首次成功 production pass。FullCCD 与 DCD 各自选择本帧最先遇到的状态，并非同一状态；没有观察后来 Newton inner、线搜索 backtrack、容量重试的全部工作。第1个 DCD 的 JSON frame 被显式设为1，但不在选中 cost frame 的事件上下文内，故每场 BVH diagnostic cost 只有239条（DCD119 + FullCCD120）。','',
        '| 场景 | cost DCD调用 / 对应首样本 | cost FullCCD调用 / 首样本 | VF / EE query数每样本 | FullCCD alpha范围 |','|---|---:|---:|---:|---:|']
    for case,p in result['cases'].items():
        env=p['cost']['envelopes'];n=p['workload']
        lines.append(f'| {case} | {env["collision.discrete_query"]["records"]} / 119 | {env["collision.swept_query"]["records"]} / 120 | {n["DCD_VF"]["queries_per_sample"][0]} / {n["DCD_EE"]["queries_per_sample"][0]} | {p["alpha_range_fullccd"]["min"]:.6f}–{p["alpha_range_fullccd"]["max"]:.6f} |')
    lines+=['','首样本占上述已记录 query 调用约16–18%，这些比例是调用覆盖率，不能据此将其节点分布倍乘为全部调用成本。私有 replay 使用当前树和 query launch ID 顺序，warp_max 是每32条实际 launch lane（含最后部分 warp）的节点计数最大值，并非 Morton ID 排序后的统计。','',
        '## 实际工作量','',
        '下表均来自真实首样本：均值按 sampled query／warp 数加权；max 是这些样本的精确最大值。节点 p95 为120个“各 launch 内 query p95”的中位数，p99为各 launch p99 的最高值；它们都不是混合全部帧的 pooled 分位数。默认 raw records 关闭，仅摘要和 top16，不能恢复 pooled p95/p99 或逐 query 节点／窄相联合分布。','',
        '| 场景 / query | AABB均值 | launch p95中位 | launch p99最高 | AABB max | 叶均值 | narrow均值 / max | pending栈max | warp_max均值 | warp/query均值比 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case,p in result['cases'].items():
        for kind,q in p['workload'].items():
            a=q['node_aabb_tests'];n=q['narrow_calls']
            lines.append(f'| {case} / {kind} | {a["weighted_mean"]:.2f} | {a["per_first_launch_quantiles"]["p95"]["median"]:g} | {a["per_first_launch_quantiles"]["p99"]["max"]:g} | {a["maximum_across_samples"]} | {q["leaf_overlaps"]["weighted_mean"]:.2f} | {n["weighted_mean"]:.3f} / {n["maximum_across_samples"]} | {q["max_pending_stack"]["maximum_across_samples"]} | {q["launch_warp_max_node_aabb_tests"]["weighted_mean"]:.2f} | {q["warp_mean_over_query_mean"]:.3f} |')
    lines+=['','每 query 平均约53–96次 AABB、6.6–14.5次重叠叶，最大356次 AABB；pending 栈最高7–10，未接近65个源码槽。warp 的最大节点次数均值是 query 均值的1.36–1.75倍，支持存在访问工作不均衡；它不是实际 stall／divergence 时长，也不能推出节省同样百分比。stackless 若仅改变存储／DFS控制，不会自动减少这些 AABB、过滤、叶或输出原子工作。','',
        'DCD narrow_calls 计数的是进入 eligible leaf 的距离分类入口，均值0.384–1.418、最高38；其内部 FP64 距离运算没有独立计时，不能用次数比例把完整 DCD kernel 当成纯遍历。leaf_overlaps 在共享 primitive／body／固定边界过滤前统计，故叶数与 narrow数差距也不是可跳过碰撞的授权。FullCCD narrow_calls=0，因为这四个 swept query 仅产生候选；真正 point-triangle／edge-edge ACCD 在 `_cub_reduct_self_step`，保持完整安全计算。','',
        '## 单帧 Nsight 与整场事件包络','',
        '已有 PROFILE_COST_ANALYSIS 的 sphere49/fixed40 是冻结旧程序、Graph K1、refit／batched energy／energy reuse／discrete refit 均关闭的单帧；本次 probe/observer-off 是新程序且四组件开启。只能把该单帧作热点拆分证据，不能移用为当前四组件整場比例或首样本的直接计时配对。','',
        '| GPU kernel union，仅 capture单帧 | sphere49 ms | fixed40 ms |','|---|---:|---:|']
    profiles=result['profile_cost_analysis']['profiles']
    for name in ('collision.discrete_query','collision.swept_query','collision.safety_edge_triangle','collision.accd_self_narrow'):
        values=[p['extended_symbol_categories'][name]['union_ms'] for p in profiles.values()]
        lines.append(f'| {name} | {values[0]:.6f} | {values[1]:.6f} |')
    lines+=['','DCD kernel含遍历、过滤、距离分类与输出；FullCCD候选 kernel含遍历、过滤与输出；ACCD另列。安全 `_edgeTriIntersectionQuery` 在两帧分别20.939/8.450ms，但本次工作量探针未覆盖它，不能将该安全耗时预支给四 query 的 stackless 视图。Graph 的 CPU范围、内部node时间、GPU活跃并集不相加。','',
        '| 120帧观察量 | sphere ms | fixed ms |','|---|---:|---:|']
    for stage in ('ipc.physical_frame','collision.discrete_query','collision.swept_query','collision.self_ccd','diagnostic.bvh_query_workload'):
        values=[p['cost']['envelopes'][stage]['gpu_event_inclusive_sum_ms'] for p in result['cases'].values()]
        lines.append(f'| GPU-event inclusive sum: {stage} | {values[0]:.3f} | {values[1]:.3f} |')
    values=[sum(p['private_diagnostic_host_sum_ms'].values()) for p in result['cases'].values()]
    lines.append(f'| BVH私有 replay / CPU比较 diagnostic_host（240样本） | {values[0]:.3f} | {values[1]:.3f} |')
    lines+=['','| production query 的实际父／私有子event分解 | 父event ms | 私有BQ子event ms | 父减子event余量 ms | 父CPU ms | 私有子CPU ms | 父减子CPU余量 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case,p in result['cases'].items():
        for stage,v in p['cost']['query_parent_child_breakdown'].items():
            lines.append(f'| {case} / {stage} | {v["parent"]["gpu_event_inclusive_sum_ms"]:.3f} | {v["private_child"]["gpu_event_inclusive_sum_ms"]:.3f} | {v["gpu_event_parent_minus_private_child_ms"]:.3f} | {v["parent"]["cpu_inclusive_sum_ms"]:.3f} | {v["private_child"]["cpu_inclusive_sum_ms"]:.3f} | {v["cpu_parent_minus_private_child_ms"]:.3f} |')
    lines+=['','这里按真实 parent_scope_id 对119个 DCD 与120个 FullCCD 私有子范围进行分解，未将子范围再加到父范围。余量约 DCD1598/1708ms、FullCCD380/739ms：它包含生产遍历、距离分类／候选过滤、Ground query（DCD）、清零／计数读回、事件／调度空隙，以及诊断边界影响，绝非纯遍历核函数时间。仅删掉子event没有恢复未诊断的同工作量执行。它说明剩余查询值得继续调查，不能单独满足整场净5%。','',
        '这些 GPU event 是 stream elapsed（含提交空隙和嵌套范围），不是互斥 kernel union。DCD／FullCCD的 production父包络包含上面的私有 probe；physical frame还包含事件、线性结构探针、读回、CPU比较与文件写入。diagnostic_host从私有比较开始到返回，不含JSON文件写入及完整CostScope收尾。239条诊断event与240条host测量覆盖不同，不能相减或相加得出未观测的纯生产遍历成本。','',
        '| 完整臂观测 | sphere | fixed |','|---|---:|---:|']
    for key,label in (('solver_seconds','solver s'),):
        for mode in ('observer_off','probe'):
            values=[p['arms'][mode]['result'][key] for p in result['cases'].values()]
            lines.append(f'| {mode} {label} | {values[0]:.6f} | {values[1]:.6f} |')
    for key in ('directions','pcg_iterations','energy_evaluations','energy_backtracks'):
        values=[f'{p["arms"]["observer_off"]["summary"][key]} → {p["arms"]["probe"]["summary"][key]}' for p in result['cases'].values()]
        lines.append(f'| off → probe {key} | {values[0]} | {values[1]} |')
    lines+=['','两臂完整执行工作量不同，且 probe同时启用CostScope GPU事件和线性结构观察；solver秒数差只表示两次独立诊断观测，不能归为纯BVH probe开销或未来收益。observer-off只有一次／场景，本机WDDM桌面负载未受控；原生守卫通过也不补齐独立质量协议。','',
        '## 静态资源与动态性能证据','',
        '冻结旧exe的 CUDA13 `cuobjdump --dump-resource-usage` 提供以下实际 cubin 信息，覆盖生产kernel（排除 pool／Ordered诊断版本）。它与本次新诊断程序的资源身份不能混同。','',
        '| 生产 kernel | REG / thread | STACK bytes | LOCAL bytes |','|---|---:|---:|---:|']
    for name,r in result['frozen_cubin_static_resources'].items():lines.append(f'| `{name}` | {r["REG"]} | {r["STACK"]} | {r["LOCAL"]} |')
    dynamic=result['dynamic_ncu_counter_evidence'];m=dynamic['metrics'];launch=dynamic['launch']
    lines+=['','这确认编译后的静态栈帧非零，比仅看到源码 stack[65] 更强；它未区分数组、函数调用与临时对象的贡献，也不证明运行中的动态 spill、访问延迟或瓶颈。Nsight Systems各 query/ACCD 的 localMemoryPerThread=0 与 LOCAL=0 同样不能消除实际 STACK。样本栈高水位低也不能直接缩减当前槽数或保证所有状态安全。','',
        '一次有界 Nsight Compute 捕获已完成：新 observer/probe 相同程序、四组件开启、从零120帧，只捕获 sphere 第49帧首个 production `_selfQuery_ee`，一个kernel，raw计数1pass，child exit=0。原collector因期待long CSV而产生 `counter_check_failed / KeyError: Metric Name`，原回执保留；本CPU分析独立读取wide CSV，跳过单位行，未重跑GPU。','',
        '| 动态指标：一个真实 DCD EE launch | 实际值 |','|---|---:|',
        f'| duration | {m["gpu__time_duration.sum"]["value"]/1e6:.6f} ms |',
        f'| local load sectors | {m["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]["value"]:,.0f} |',
        f'| local store sectors | {m["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"]["value"]:,.0f} |',
        f'| achieved active-warp occupancy | {m["sm__warps_active.avg.pct_of_peak_sustained_active"]["value"]:.2f}% |',
        f'| long-scoreboard / active warp | {m["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]["value"]:.2f}% |',
        f'| launch blocks / SMs / waves per SM | {launch["launch__grid_size"]:g} / {launch["launch__sm_count"]:g} / {launch["launch__waves_per_multiprocessor"]:.2f} |',
        f'| registers requested / allocated per thread | {launch["launch__registers_per_thread"]:g} / {launch["launch__registers_per_thread_allocated"]:g} |','',
        '动态 local流量现在有真实证据，但属于完整DCD EE，尚未定位到DFS栈、距离分类临时量或spill指令。long-scoreboard百分比不能转换为等量wall-time收益；36 blocks少于40 SM也造成launch不足，低occupancy不是stackless可以自动解决的证据。该单launch不是所有调用的平均，也不是与probe首样本同一state的计数对照。','',
        '最小CPU来源检查已完成：[新程序单函数SASS分析](NEW_EE_SASS_ANALYSIS.md)。新d4da6程序只提取production `_selfQuery_ee`，共4968条静态指令，82处local站点；地址和控制流把4处32-bit站点映射为节点栈（0x120 init、0x360 pop、0x9c30左push、0x13500右push）。其余36处STL.64／42处LDL.64位于R1+0x00..0x58，和从0x60开始的uint32节点栈槽区分开，出现在FP64／CALL邻域。','',
        '能区分静态节点栈站点与其他64-bit frame值，却不能严格将所有其他值命名为某个窄相临时对象、caller保存或spill。mlbvh实际编译命令有-lineinfo；当前cuobjdump SASS无源码PC注解，不能写成binary没有line information。NCU没有逐PC执行／local事务／stall计数，4/82的静态站点比例绝不能当动态流量／时间份额。这次有界来源检查到此结束，不再增加GPU计数或扩张追踪。','',
        '## 门槛判断与最小下一步','',
        '| 原门槛证据 | 当前状态 |','|---|---|',
        '| 完整 query 的整场独占成本≥10% | 未建立：单帧配置不同；整场事件含诊断；DCD未拆出距离分类 |',
        '| stack 控制／访存或长尾可被 escape视图因果削减 | 工作不均衡、静态栈、动态local流量已观察；识别栈指令PC，但动态栈份额与候选收益未测 |',
        '| 减少的关键路径成本 − 生成／维护／失效成本≥whole 5% | 未建立：没有escape视图成本或完整调用覆盖 |',
        '| 局部完整query≥15%、三对整场≥5%、另一主场景≤3%退化 | 没有候选对照，不具备验收证据 |',
        '| 完整碰撞安全／质量认证 | replay oracle与原生守卫通过；独立质量协议仍未认证 |','',
        '已完成原提出的最小CPU来源检查。现在更有依据的结论是：DCD EE值得优先调查，但观测到的local成本同时含节点栈和其他frame值，缺少它们的动态份额；DCD父包络余量还混合VF／EE／ground／距离／读回。尚无可信的“可消除时间减维护成本≥whole5%”估计，依原门槛暂缓生产stackless分支，本次有界分析结束。不能把“未达到证明门槛”写成“已证明整个碰撞优化方向无效”。','',
        '未来若完整未诊断调用成本建立净≥5%的可信潜力，则依原计划进入单query候选：私有同树／同state的原DFS与escape对照，计数关闭，保持完整距离／tuple oracle，分列视图生成／拓扑失效／refit复用成本。可用实际长尾坐标是 sphere DCD EE 首样本 frame46（node max356，inner0，contact_model226，linear205）／frame50（warp倍率峰值），fixed frame53（node max280，contact_model298，linear276）；它们是数据坐标，不承诺已保存可重放state。后续Newton／backtrack仍需覆盖。严禁把遍历-only减掉窄相当作生产提速，也不能删除树过滤或碰撞候选。','',
        '达到门槛后进入候选实施，再完成原约定的三对从零120帧关闭诊断A/B和完整质量验收。当前决定仅限stackless启动门，不排除更好的树布局、保守空间查询或窄相实现；这些方向需各自成本与安全证据。','',
        '## 可复现文件','',
        f'- [CPU分析脚本]({str(OUT/"analyze_query_branch.py").replace(chr(92),"/")})',
        f'- [聚合数据与完整选择上下文]({str(OUT/"QUERY_BRANCH_DECISION.json").replace(chr(92),"/")})',
        f'- [单帧成本分析]({str(OUT/"PROFILE_COST_ANALYSIS.md").replace(chr(92),"/")})',
        f'- [冻结静态资源原始日志]({str(OUT/"frozen_resource_usage.log").replace(chr(92),"/")})','',
        f'- [单次NCU原始wide CSV]({str(RUNS/"query_counters_f49/counter.csv").replace(chr(92),"/")})','',
        f'- [新程序SASS来源检查]({str(OUT/"NEW_EE_SASS_ANALYSIS.md").replace(chr(92),"/")})','',
        '复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_query_branch.py`。脚本要求completed ledger／run／evidence身份，读取输入前后核对SHA256，只生成这份Markdown和JSON，不修改native/tools、不调用GPU或构建。']
    return '\n'.join(lines)+'\n'

def main():
    ledger=load_json(RUNS/'PROBE_LEDGER.json')
    require(ledger['status']=='completed' and len(ledger['rows'])==4,'Closed complete four-run ledger')
    ledger_rows={(r['stage'],r['scene']):r for r in ledger['rows']}
    profile=load_json(OUT/'PROFILE_COST_ANALYSIS.json')
    sass=load_json(OUT/'NEW_EE_SASS_ANALYSIS.json')
    require(sass['exe']['sha256']==ledger['program_sha256'] and sass['new_gpu_launches']==0,'CPU-only actual new-program SASS inspection')
    result={'schema':'report56.query_branch_decision.v1','performance_certified':False,'physical_quality_certified':False,
        'read_only_inputs':True,'decision':'defer_stackless_production_branch','program_sha256':ledger['program_sha256'],
        'script':identity(Path(__file__)),'cases':{case:analyze_case(case,ledger_rows) for case in CASES},
        'frozen_cubin_static_resources':static_resources(OUT/'frozen_resource_usage.log'),
        'profile_cost_analysis':{'identity':identity(OUT/'PROFILE_COST_ANALYSIS.json'),'profiles':profile['profiles']},
        'dynamic_ncu_counter_evidence':dynamic_counters(ledger['program_sha256']),
        'new_ee_sass_source_analysis':{k:v for k,v in sass.items() if k not in ('instructions','local_instructions')},
        'percentile_scope':'per-first-launch summaries only; no pooled p95/p99 inferred'}
    require(all(identity(Path(name))==item for name,item in IDENTITIES.items()),'All inputs unchanged during CPU analysis')
    result['input_identities']=IDENTITIES;result['cpu_validation_checks']=CHECKS
    (OUT/'QUERY_BRANCH_DECISION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'QUERY_BRANCH_DECISION.md').write_text(markdown(result),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':CHECKS,'decision':result['decision'],
        'samples':{case:sum(q['first_samples'] for q in p['workload'].values()) for case,p in result['cases'].items()},
        'output':str(OUT/'QUERY_BRANCH_DECISION.md')},ensure_ascii=False))

if __name__=='__main__':main()
