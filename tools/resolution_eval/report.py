"""CPU-only reports for the predeclared 16-run resolution experiment.

Reads retained, hash-bound ledger/analysis/native metadata. It never launches a
solver, reconstructs velocity, widens quality limits, or rewrites old evidence.
Usage: report.py --session PATH [--prior COMPACT_RESULTS.json] [--output DIR]
All three output files must be new. A partial/failed case receives no ratios.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = {'hang_l': 'cloth_hang_l', 'sphere_l': 'cloth_sphere7_l',
         'hang_m': 'cloth_hang_m', 'sphere_m': 'cloth_sphere7_m'}
ARMS = ('original_stiff', 'ipc_host', 'ipc_graph', 'combined_graph')
EDGES = (('original_stiff', 'ipc_host'), ('ipc_host', 'ipc_graph'),
         ('ipc_graph', 'combined_graph'), ('original_stiff', 'combined_graph'))
PHASES = ('assembly', 'pcg', 'ccd', 'line_search', 'state_update')
MATERIAL_DIRECTIONS = {'max_stretch': 1, 'p99_stretch': 1, 'fixed_drift_m': 1,
    'fem_min_J': -1, 'fem_nonpositive_peak': 1, 'fem_nonpositive_frame_sum': 1,
    'fem_negative_volume_peak': 1, 'abd_min_J': -1, 'abd_nonpositive_frames': 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def child(root, name):
    root = Path(root).resolve()
    p = root / name
    require(p.resolve().is_relative_to(root), 'Evidence path escapes session: ' + str(name))
    for q in (p, *p.parents):
        if q == root:
            break
        require(not q.is_symlink() and not bool(getattr(q, 'is_junction', lambda: False)()),
                'Linked evidence rejected: ' + str(q))
    return p


def record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def contact_summary(frames, expected=100):
    """Frame-end native narrow counts are not an independent contact certificate."""
    counters = {}
    masks = {}
    for label, field in (('self', 'native_narrow_self_pairs'), ('ground', 'native_narrow_ground_pairs')):
        values = []
        for frame in frames:
            v = frame.get('contact_geometry', {}).get(field)
            require(v is None or type(v) is int and v >= 0, 'Invalid native narrow contact count')
            values.append(v)
        known = [i + 1 for i, v in enumerate(values) if v is not None]
        nonzero = [i + 1 for i, v in enumerate(values) if v is not None and v > 0]
        complete = len(values) == expected and len(known) == expected
        counters[label] = {'complete': complete, 'observed_frames': len(known),
            'missing_frames': [i for i in range(1, expected + 1) if i not in known],
            'nonzero_frames': nonzero, 'nonzero_frame_count': len(nonzero) if complete else None,
            'observed_nonzero_frame_count': len(nonzero),
            'first_nonzero_frame': nonzero[0] if nonzero else None,
            'first_is_full_window_observation': complete,
            'peak_pairs': max((v for v in values if v is not None), default=None)}
        masks[label] = nonzero
    complete = all(v['complete'] for v in counters.values())
    union = sorted(set(masks['self']) | set(masks['ground']))
    geometric = [f.get('contact_geometry', {}).get('geometric_contact') for f in frames]
    require(all(v is None or type(v) is bool for v in geometric), 'Invalid geometric contact classification')
    geometric_complete = len(geometric) == expected and all(v is not None for v in geometric)
    return {'expected_frames': expected, 'frame_count': len(frames), 'complete': complete,
        'native_narrow': counters, 'native_narrow_nonzero_frames': union,
        'native_narrow_nonzero_frame_count': len(union) if complete else None,
        'native_narrow_first_nonzero_frame': union[0] if union else None,
        'geometric_contact_frame_count': sum(v is True for v in geometric) if geometric_complete else None,
        'scope': 'Frame-end native narrow self/ground pair observations, used as a barrier-activity proxy only. They do not prove collision, nonzero force, the earliest intra-frame contact, or independent accepted-path CCD. geometric_contact uses a tighter distance classification; false is not no barrier. newton.active_pairs is not used to infer absence.'}


def execution_summary(frames, requested, resolved):
    pcgs = [n['pcg'] for f in frames for n in f.get('newton', []) if 'pcg' in n]
    reasons = collections.Counter()
    for p in pcgs:
        for k, v in p.items():
            if k.endswith('fallback_reason') and v not in (None, '', 'none'):
                reasons[k + ':' + str(v)] += 1
    env = requested.get('environment', {})
    native = resolved.get('acceleration_features') if resolved else None
    return {'requested_execution': requested.get('expanded_config', {}).get('execution'),
        'observed_pcg_execution': dict(collections.Counter(p.get('execution', 'unreported') for p in pcgs)),
        'fallback_reason_counts': dict(reasons), 'resolved_acceleration_features': native,
        'requested_accel_suite': env.get('GIPC_ACCEL_SUITE'),
        'requested_mas_static_topology': env.get('GIPC_MAS_STATIC_TOPOLOGY'),
        'mas_static_topology_effective': native.get('GIPC_MAS_STATIC_TOPOLOGY') if native else None,
        'scope': 'Fallback fields are retained verbatim; retired disabled feature reasons are not automatically a requested Graph fallback. Baseline has no fabricated resolved telemetry.'}


def load_rows(session):
    session = Path(session).resolve()
    plan_path = child(session, 'plan.json'); plan = read(plan_path)
    tasks = plan.get('tasks', [])
    require(len(tasks) == 16 and len({t['name'] for t in tasks}) == 16, 'Expected 16 distinct declared tasks')
    require({(t['scene_key'], t['arm']) for t in tasks} == {(c, a) for c in CASES for a in ARMS},
            'Unexpected case/arm grid')
    for i, task in enumerate(tasks, 1):
        require(task['index'] == i and task['name'] == f"p1_{task['scene_key']}_{task['arm']}"
                and task['phase'] == 'paired' and task['pair'] == 1
                and task['config']['scene'] == CASES[task['scene_key']]
                and task['config']['steps'] == 100, 'Declared resolution task differs')
    paths = sorted(child(session, 'ledger').glob('*.json'))
    require(len(paths) <= 16, 'Too many ledger entries')
    rows, evidence, previous = [], [record(plan_path)], None
    for i, path in enumerate(paths, 1):
        require(path.name == f'{i:04d}.json', 'Ledger gap')
        path = child(session, 'ledger/' + path.name)
        entry = read(path); task = tasks[i-1]
        require(entry['task'] == task and entry['index'] == i
                and entry['predecessor_sha256'] == previous, 'Ledger task/order/predecessor differs')
        for key in ('analysis', 'summary'):
            p = child(session, entry[key + '_path'])
            require(sha(p) == entry[key + '_sha256'], 'Bound ' + key + ' changed')
            evidence.append(record(p))
        analysis = read(child(session, entry['analysis_path']))
        require(analysis == entry['analysis'] and analysis['name'] == task['name'], 'Embedded analysis differs')
        row = dict(analysis, index=i, name=task['name'], scene_key=task['scene_key'], arm=task['arm'],
                   ledger_status=entry['status'], binary=task['binary'], result=entry['result'])
        row['complete_hard_pass'] = bool(entry['status'] == 'completed' and analysis.get('hard_checks_passed') is True
            and entry['result'].get('status') == 'completed' and entry['result'].get('recorded_frames') == 100
            and entry['result'].get('exit_code') == 0 and not entry.get('analysis_exception'))
        row['contact'] = None; row['execution_observation'] = None
        if entry['run_created']:
            run = child(session, task['name']); inv_path = child(run, 'evidence.json')
            require(sha(inv_path) == entry['evidence_sha256'], 'Run evidence inventory changed')
            inventory = read(inv_path)['files']; mapped = {r['path']: r for r in inventory}
            require(len(mapped) == len(inventory), 'Duplicate run evidence path')
            def bound(name, optional=False):
                p = child(run, name)
                if optional and not p.exists():
                    return None
                require(name in mapped and p.is_file() and p.stat().st_size == mapped[name]['bytes']
                        and sha(p) == mapped[name]['sha256'], 'Run metadata changed: ' + name)
                evidence.append(record(p)); return read(p)
            result = bound('result.json'); require(result == entry['result'], 'Ledger result differs from original')
            request = bound('requested.json', optional=not row['complete_hard_pass']) or {}
            stats = bound('output/stats.json', optional=not row['complete_hard_pass'])
            resolved = bound('resolved_config.json', optional=task['binary']=='base' or not row['complete_hard_pass'])
            if stats is not None:
                row['contact'] = contact_summary(stats['frames'])
                row['execution_observation'] = execution_summary(stats['frames'], request, resolved)
                if row['complete_hard_pass']:
                    require(len(stats['frames']) == 100, 'Passing run missing native frames')
                    if task['binary'] == 'active':
                        require(row['execution_observation']['mas_static_topology_effective'] is False,
                                'Unexpected MAS static topology: report must not label it off')
            evidence.append(record(inv_path))
        for pair in analysis.get('paired_comparisons', []):
            p = child(session, pair['pair_cache_path'])
            require(sha(p) == pair['pair_cache_sha256'], 'Pair cache changed')
            cached = read(p)
            require(cached['state_difference'] == pair['state_difference'], 'Pair state payload differs')
            evidence.append(record(p))
        rows.append(row); evidence.append(record(path)); previous = sha(path)
    for task in tasks[len(paths):]:
        rows.append({'index':task['index'], 'name':task['name'], 'scene_key':task['scene_key'], 'arm':task['arm'],
                     'binary':task['binary'], 'ledger_status':'not_run', 'complete_hard_pass':False,
                     'hard_checks_passed':False, 'failures':['No finalized ledger entry'], 'contact':None})
    return rows, {'recorded_tasks':len(paths), 'planned_tasks':16, 'ledger_complete':len(paths)==16,
                  'evidence':evidence, 'plan':plan}


def same_inputs(a, b):
    left, right = a.get('input_identity', {}), b.get('input_identity', {})
    required = (set(left) | set(right)) - {'initial_actual_velocity'}
    match = bool(required) and all(left.get(k) is not None and left.get(k) == right.get(k) for k in required)
    both_velocity = left.get('initial_actual_velocity') is not None and right.get('initial_actual_velocity') is not None
    if both_velocity:
        match = match and left['initial_actual_velocity'] == right['initial_actual_velocity']
    return bool(match and a.get('gpu_uuid') and a['gpu_uuid'] == b.get('gpu_uuid'))


def ratio(a, b):
    if not finite(a) or not finite(b) or a <= 0 or b <= 0:
        return {'available':False, 'ratio':None, 'saved_fraction':None, 'saved_seconds':None}
    return {'available':True, 'reference_seconds':a, 'candidate_seconds':b, 'ratio':a/b,
            'saved_fraction':1-b/a, 'saved_seconds':a-b}


def compare_material(reference, candidate):
    result = {}
    for key, direction in MATERIAL_DIRECTIONS.items():
        a, b = reference.get(key), candidate.get(key)
        applicable = finite(a) and finite(b)
        result[key] = {'reference':a, 'candidate':b, 'available':applicable,
            'candidate_minus_reference':b-a if applicable else None,
            'signed_worsening_gap':direction*(b-a) if applicable else None}
    return {'metrics':result, 'quality_certified':False, 'acceptance_threshold':None,
            'scope':'One baseline sample, raw signed differences only; no new certification threshold or repeatability range.'}


def case_summary(rows):
    cases = {}
    for case in CASES:
        arms = {r['arm']: r for r in rows if r['scene_key'] == case}
        complete = set(arms) == set(ARMS) and all(r['complete_hard_pass'] for r in arms.values())
        result = {'all_four_complete_hard_pass':complete, 'comparisons':{}, 'material_vs_single_stiff':{}}
        for left, right in EDGES:
            a, b = arms.get(left, {}), arms.get(right, {})
            comparable = complete and same_inputs(a, b)
            edge = {'reference_arm':left, 'candidate_arm':right, 'comparable':comparable,
                'samples':1 if comparable else 0, 'confidence_interval':None,
                'reason':None if comparable else 'A case arm is incomplete/hard-failed, or same GPU/physical inputs are not established.',
                'quality_certified':False, 'solver':ratio(None,None), 'core_event':ratio(None,None), 'wall':ratio(None,None)}
            if comparable:
                for label, field in (('solver','solver_seconds'), ('core_event','core_cuda_event_seconds'), ('wall','process_wall_seconds')):
                    edge[label] = ratio(a['timing'].get(field), b['timing'].get(field))
            pair_rows = [p for r in arms.values() for p in r.get('paired_comparisons', [])
                         if p['left']==left and p['right']==right and p['pair']==1]
            edge['state_difference'] = pair_rows[-1]['state_difference'] if pair_rows else {
                'available':False, 'reason':'No hash-bound retained state comparison for this edge; no velocity reconstruction.'}
            result['comparisons'][left+'/'+right] = edge
        original = arms.get('original_stiff', {})
        if original.get('material'):
            for arm,r in arms.items():
                if arm != 'original_stiff' and r.get('material'):
                    result['material_vs_single_stiff'][arm] = compare_material(original['material'], r['material'])
        cases[case] = result
    return cases


def prior_summary(path):
    value = read(path)
    result = {'source':record(path), 'quality_certified':False, 'cases':{},
        'scope':'Previous 100-frame experiment: seven principal paired ratios and three component pairs. Different samples/cases, not products to reconstruct the principal ratio; no new CI for this single-sample experiment.'}
    for case in ('hang','fixed','mixed'):
        src = value['scene_statistics'][case]
        result['cases'][case] = {label:src[key]['solver'] for label,key in (
            ('Stiff/combined','main_original_to_combined'), ('host/Graph','graph_component'),
            ('Graph/combined','combined_execution_component'))}
    return result


def analyze(session, prior):
    rows, proof = load_rows(session)
    return {'schema':'resolution_eval_report.v1', 'session':str(Path(session).resolve()),
        'recorded_tasks':proof['recorded_tasks'], 'planned_tasks':16, 'ledger_complete':proof['ledger_complete'],
        'completed_hard_pass':sum(r['complete_hard_pass'] for r in rows), 'quality_certified':False,
        'performance_certified':False, 'confidence_intervals':None,
        'selection':'Hanging cloth and cloth falling onto a fixed sphere L/M cases were chosen before this batch based on historical cloth gains. This is a selected-scene diagnostic, not an unbiased scene-wide speed estimate.',
        'actual_components':'FullCCD refit, batched energy, energy reuse, ordinary BVH refit interval 8; MAS static topology off. Preserve each run resolved/execution observations below; do not attribute gains to MAS topology reuse.',
        'timing_scope':'solver_seconds is frames.csv CPU steady_clock around IPC_Solver plus synchronization; core_event is the nested CUDA-event envelope; wall additionally includes loading/export. Never add them. pcg phase includes matrix conversion, MAS preparation/application, PCG, and solution distribution.',
        'limitations':['One sample per case/arm; no confidence interval or same-quality 2x certification.',
            'No new material acceptance threshold: preserve differences and incomplete cases.',
            'Frozen Stiff has no actual velocity export; independent accepted-path CCD is unavailable.',
            'Native narrow pair observations do not certify contact or barrier force; missing counts are not zero.',
            'Different solver trajectories and PCG/direction counts contribute to observed time differences.',
            'Raw archives are not re-expanded; retained per-run analyses and comparison caches are hash-verified.'],
        'cases':case_summary(rows), 'runs':rows, 'prior_experiment':prior_summary(prior),
        'provenance':proof['evidence'] + [record(__file__)]}


def number(value, digits=4):
    return '未测' if value is None else f'{value:.{digits}f}'


def markdown(data):
    lines = ['# 布料场景分辨率诊断：L/M × 四配置', '',
        f"已记录 {data['recorded_tasks']}/16 项，完整硬检查通过 {data['completed_hard_pass']} 项。每项从零运行 100 帧；每个配置仅一次，无置信区间，不认证同质量 2×。", '',
        '本轮在运行前依据历史布料收益选取悬挂布料与布料落球（固定球）场景，各测 L/M；属于有选择的诊断样本，不能外推全部场景。材料只与本次单个 Stiff 样本列原始差异，不新增或放宽认证门槛。', '',
        '## 四段速度比', '',
        '每行均使用对应两臂的实际时间。节省比例为 1−1/ratio，节省秒数为 reference−candidate；负数表示回退。一个场景任一臂失败或不完整，该场景不计算速度比。', '',
        '| 场景 | 比较 | 求解比 | 求解节省% | 求解节省秒 | core event比 | wall比 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case,result in data['cases'].items():
        for edge,c in result['comparisons'].items():
            v=c['solver']; pct=None if v['saved_fraction'] is None else 100*v['saved_fraction']
            lines.append(f"| {case} | {edge} | {number(v['ratio'])} | {number(pct,2)} | {number(v['saved_seconds'])} | {number(c['core_event']['ratio'])} | {number(c['wall']['ratio'])} |")
    lines += ['', '## 每次运行与阶段成本', '',
        '时间单位秒。求解为 CPU 同步包络；core event 在其中，wall 还含加载及不完全对称的导出成本，三者不能相加。linear 包含矩阵转换、MAS 准备/应用、PCG 与解分发；五阶段不是完整 wall 分解。', '',
        '| 场景 | 配置 | 状态 | 求解 | core event | wall | 装配 | linear | CCD | 线搜索 | 状态更新 | 线性次数 | PCG次数 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['runs']:
        t,w=r.get('timing',{}),r.get('work',{})
        vals=[t.get(k) for k in ('solver_seconds','core_cuda_event_seconds','process_wall_seconds')]
        vals += [None if t.get('phase_ms',{}).get(k) is None else t['phase_ms'][k]/1000 for k in PHASES]
        lines.append('| '+r['scene_key']+' | '+r['arm']+' | '+r['ledger_status']+' | '+' | '.join(number(v) for v in vals)+f" | {w.get('linear_directions','未测')} | {w.get('pcg_iterations','未测')} |")
    lines += ['', '实际组合：FullCCD refit、批量能量、能量复用、普通 BVH refit（周期 8）。MAS 静态拓扑复用关闭；没有单项消融，不能把组合比归因某一组件。每次运行的实际 execution、回退原因和 resolved 开关保存在 JSON；线性次数/PCG 与轨迹变化会进入时间差。', '',
        '## 原生碰撞活动观测', '',
        '表中为帧末原生窄相对非零帧数与首次观测帧，用于描述 barrier 活动线索，不是独立接触/非零力认证，也不保证捕获帧内最早活动。geometric_contact 使用更紧距离分类，其 false 不能解释为无 barrier；不以 newton.active_pairs 的零值判断无接触。缺字段保留未知。', '',
        '| 场景 | 配置 | self非零帧 | ground非零帧 | 任一非零帧 | 首次观测帧 | 数据完整 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in data['runs']:
        c=r.get('contact') or {}; n=c.get('native_narrow',{})
        v=[n.get(k,{}).get('nonzero_frame_count') for k in ('self','ground')]+[c.get('native_narrow_nonzero_frame_count'),c.get('native_narrow_first_nonzero_frame')]
        lines.append('| '+r['scene_key']+' | '+r['arm']+' | '+' | '.join('未测' if x is None else str(x) for x in v)+f" | {c.get('complete',False)} |")
    lines += ['', '## 材料与分体状态差异', '',
        '下列单次材料值与 JSON 中的带符号差异全部保留。不存在 FEM/ABD 的指标不填零最小 J；p99 是各帧空间 p99 的全段最大值。非正单元帧累计不是独立翻转事件数。', '',
        '| 场景 | 配置 | 最大拉伸 | p99拉伸 | 固定点漂移m | FEM最小J | FEM非正峰值 | FEM负体积峰值 | ABD最小J |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['runs']:
        m=r.get('material',{})
        lines.append('| '+r['scene_key']+' | '+r['arm']+' | '+' | '.join(number(m.get(k),8) for k in ('max_stretch','p99_stretch','fixed_drift_m','fem_min_J','fem_nonpositive_peak','fem_negative_volume_peak','abd_min_J'))+' |')
    lines += ['', '分体差异为全段最大帧 RMS / 最大顶点差。原版未导出真实速度，不从位置差分伪造速度；缺缓存明确记为未测。', '',
        '| 场景 | 比较 | 量 | 分体 | max-frame RMS | max vertex | 帧数 |',
        '|---|---|---|---|---:|---:|---:|']
    for case,result in data['cases'].items():
        for edge,c in result['comparisons'].items():
            state=c['state_difference']; wrote=False
            for kind in ('position','velocity'):
                for body,v in state.get(kind,{}).items():
                    if isinstance(v,dict) and 'max_frame_rms' in v:
                        wrote=True;lines.append(f"| {case} | {edge} | {kind} | {body} | {v['max_frame_rms']:.9g} | {v['max_vertex']:.9g} | {v['frames']} |")
            if not wrote: lines.append(f'| {case} | {edge} | 未测 | — | — | — | — |')
    lines += ['', '## 前轮多次配对结果（仅作背景比较）', '',
        '前轮主比较为七对，组件比较为前三对；本轮每臂一次。样本、部分场景与规模不同，不把组件统计中位数相乘重建主比，也不把前轮下界移用到本轮。', '',
        '| 前轮场景 | Stiff/组合七对中位 | host/Graph三对中位 | Graph/组合三对中位 |', '|---|---:|---:|---:|']
    for case,c in data['prior_experiment']['cases'].items():
        lines.append('| '+case+' | '+' | '.join(number(c[k]['paired_median']) for k in ('Stiff/combined','host/Graph','Graph/combined'))+' |')
    lines += ['', '完整数值、原始材料差异、缺失原因、失败/回退、程序/输入身份及来源 SHA 见 [RESULTS.json](RESULTS.json)；逐运行表见 [RUN_INDEX.csv](RUN_INDEX.csv)。本报告没有新模拟、独立接受路径 CCD 或质量认证。', '']
    return '\n'.join(lines)


def csv_text(data):
    fields=['index','name','case','arm','status','hard_pass','solver_s','core_event_s','wall_s',
        'assembly_s','linear_s','ccd_s','line_search_s','state_update_s','exit_assembly_s',
        'linear_directions','pcg_iterations','native_narrow_nonzero_frames','native_narrow_first_frame',
        *MATERIAL_DIRECTIONS,'exe_sha256','config_sha256']
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for r in data['runs']:
        t,w,c=r.get('timing',{}),r.get('work',{}),r.get('contact') or {}
        row={'index':r['index'],'name':r['name'],'case':r['scene_key'],'arm':r['arm'],'status':r['ledger_status'],
             'hard_pass':r['complete_hard_pass'],'linear_directions':w.get('linear_directions'),'pcg_iterations':w.get('pcg_iterations'),
             'native_narrow_nonzero_frames':c.get('native_narrow_nonzero_frame_count'),'native_narrow_first_frame':c.get('native_narrow_first_nonzero_frame')}
        for dest,key in (('solver_s','solver_seconds'),('core_event_s','core_cuda_event_seconds'),('wall_s','process_wall_seconds')):row[dest]=t.get(key)
        for dest,key in zip(('assembly_s','linear_s','ccd_s','line_search_s','state_update_s'),PHASES):
            v=t.get('phase_ms',{}).get(key);row[dest]=None if v is None else v/1000
        v=t.get('exit_assembly_ms');row['exit_assembly_s']=None if v is None else v/1000
        row.update({k:r.get('material',{}).get(k) for k in MATERIAL_DIRECTIONS})
        row.update({k:r.get('program_identity',{}).get(k) for k in ('exe_sha256','config_sha256')})
        writer.writerow(row)
    return stream.getvalue()


def write_reports(data, output):
    output=Path(output)
    files={'RESULTS.json':json.dumps(data,indent=2,allow_nan=False), 'REPORT.md':markdown(data), 'RUN_INDEX.csv':csv_text(data)}
    require(not any((output/name).exists() for name in files),'Report exists; no overwrite')
    output.mkdir(parents=True,exist_ok=True)
    for name,text in files.items():
        with (output/name).open('x',encoding='utf-8-sig' if name.endswith('.csv') else 'utf-8',newline='') as f:f.write(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    p.add_argument('--prior','--previous',dest='prior',type=Path,default=ROOT/'reports/autodl_full_20261006/COMPACT_RESULTS.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/autodl_resolution_20261007')
    args=p.parse_args();data=analyze(args.session,args.prior);write_reports(data,args.output)
    print(json.dumps({'recorded_tasks':data['recorded_tasks'],'completed_hard_pass':data['completed_hard_pass'],
                      'output':str(args.output.resolve()),'gpu_run':False,'quality_certified':False}))


if __name__=='__main__':main()
