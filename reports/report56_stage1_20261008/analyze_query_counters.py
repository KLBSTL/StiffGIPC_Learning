"""Read the single closed NCU report's wide raw CSV. Never rerun GPU or rewrite evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from run_stage1 import ROOT, REPORT, SESSION
for folder in ('bench', 'diagnostic', 'full_eval', 'local'):
    sys.path.insert(0, str(ROOT / 'tools' / folder))
from linux_runner import verify_files
from local_identity import record
from run_query_counters import METRICS

def number(raw):
    value = float(raw.replace(',', ''))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Nonfinite/negative metric')
    return value

def main():
    folder = SESSION / 'query_counters_f49'
    evidence = json.loads((folder / 'evidence.json').read_text())
    verify_files(folder, evidence['files'])
    producer = json.loads((folder / 'result.json').read_text())
    request = json.loads((folder / 'requested.json').read_text())
    captured = json.loads((folder / 'capture_validation.json').read_text())
    validation = json.loads((folder / 'config_validation.json').read_text())
    assert producer['exit_code'] == 0 and producer['recorded_frames'] == 120
    assert producer['cleanup_owned_job_empty'] is True and captured['passed'] and validation['passed']
    assert producer['error'] == "KeyError: 'Metric Name'", 'Do not reinterpret a different producer failure'
    raw = list(csv.DictReader((folder / 'counter.csv').read_text(encoding='utf-8-sig').splitlines()))
    units = [x for x in raw if x.get('ID') == '' and x.get('Kernel Name') == '']
    kernels = [x for x in raw if x.get('ID', '').isdigit()]
    assert len(units) == len(kernels) == 1, 'One units row and one profiled query required'
    unit, kernel = units[0], kernels[0]
    assert kernel['Kernel Name'].startswith('_selfQuery_ee(')
    assert 'ipc.physical_frame' in kernel['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg']
    scopes = [json.loads(x) for x in (folder / 'cost.jsonl').read_text().splitlines() if x]
    assert scopes and all(x['frame'] == 49 and x['gpu_interval_ms'] is None for x in scopes)
    assert unit['gpu__time_duration.sum'] == 'ns'
    assert unit[METRICS[1]] == unit[METRICS[2]] == 'sector'
    metrics = {name: {'value': number(kernel[name]), 'unit': unit[name]} for name in METRICS}
    launch_fields = ('launch__registers_per_thread', 'launch__registers_per_thread_allocated',
        'launch__local_mem_per_thread', 'launch__local_mem_total', 'launch__waves_per_multiprocessor',
        'launch__sm_count', 'launch__grid_size', 'launch__block_size', 'profiler__replayer_passes')
    launch = {name: {'value': number(kernel[name]), 'unit': unit[name]} for name in launch_fields if name in kernel}
    data = {'schema': 'gipc.one_query_counter_analysis.v1', 'status': 'validated_offline',
        'producer_result_status_unchanged': producer['status'],
        'producer_analysis_error': producer['error'], 'gpu_capture_exit_code': 0,
        'recorded_frames': 120, 'selected_frame': 49, 'profiled_kernels': 1,
        'kernel': kernel['Kernel Name'], 'metrics': metrics, 'launch': launch,
        'program_sha256': request['exe_sha256'], 'source_digest': request['source_digest'],
        'inputs': [record(folder / name) for name in
            ('counter.csv', 'query.ncu-rep', 'requested.json', 'result.json', 'evidence.json',
             'capture_validation.json', 'config_validation.json', 'cost.jsonl')],
        'analyzer': record(__file__), 'new_gpu_launches': 0,
        'performance_certified': False, 'physical_quality_certified': False,
        'limits': ['Dynamic local-memory traffic includes stack, calls and compiler temporaries; no attribution solely to traversal stack.',
            'One EE kernel does not measure all VF/EE queries, narrow-phase share or whole-run removable cost.',
            'Occupancy also depends on 36 blocks over 40 SMs; fewer registers alone cannot fill an underpopulated grid.',
            'Local-sector counts and long scoreboard stalls do not prove a stackless speedup or >2x result.',
            'Original producer parser failure is preserved. This is offline format repair, with no GPU retry.']}
    verify_files(folder, evidence['files'])
    (REPORT / 'QUERY_COUNTER_ANALYSIS.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    lines = ['# 单查询动态计数检查', '',
        '真实 GPU 捕获与从零 120 帧运行已完成。原 collector 将 wide CSV 当作 long CSV，离线解析出现 `KeyError: Metric Name`；原失败回执保留，本脚本只修正离线读取，未重跑 GPU。', '',
        '采样：四组件组合、落球 L、f49、第一个匹配的离散 EE 查询；仅一个 kernel，不是整场速度组。', '',
        '| 指标 | 观测 | 单位 |', '|---|---:|---|']
    for name, value in metrics.items():
        lines.append(f"| {name} | {value['value']:.6g} | {value['unit']} |")
    lines += ['', '| launch | 观测 | 单位 |', '|---|---:|---|']
    for name, value in launch.items():
        lines.append(f"| {name} | {value['value']:.6g} | {value['unit']} |")
    lines += ['', f"该次 kernel 的测量时长为 {metrics[METRICS[0]]['value'] / 1e6:.6f}ms。数据证明实际存在 local-memory 访问，不能证明其全部来自 BVH 栈；堆栈、窄相调用、编译临时量与网格不足同时存在。", ''] + data['limits'] + [
        '', '复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_query_counters.py`。原始捕获只允许一次，脚本不会修改 result/evidence 或重启模拟。', '']
    (REPORT / 'QUERY_COUNTER_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': data['status'], 'profiled_kernels': 1, 'new_gpu_launches': 0,
        'duration_ms': metrics[METRICS[0]]['value'] / 1e6, 'local_load_sectors': metrics[METRICS[1]]['value'],
        'local_store_sectors': metrics[METRICS[2]]['value']}))

if __name__ == '__main__': main()
