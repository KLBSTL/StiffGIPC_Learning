"""Offline accounting of a finite IPC cost round; no GPU work or solver edits.

The initialization envelope is an inactivity upper bound, not a saving estimate.
It includes capture/profiler/OS effects and excludes already executing GPU work.
"""
import json
import sqlite3
from collections import Counter
from config import ROOT, read, sha
from ipc_benchmark import metrics, write
from component_tuning import state_comparison
from validate_run import validate
from analyze_ipc_light_cost import analyze, merge, clip, TABLES

TAG = 'ipc_next_cost_20261006'


def initialization_envelope(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA query_only=ON')
        labels = {r['id']: r['value'] for r in con.execute('SELECT * FROM StringIds')}
        ranges = [dict(r) for r in con.execute('SELECT * FROM NVTX_EVENTS')]
        for r in ranges:
            r['name'] = r.get('text') or labels.get(r.get('textId'))
        entries = [r for r in ranges if r['name'] == 'graph.entry' and r.get('end')]
        reads = [r for r in ranges if r['name'] == 'graph.initial_readback' and r.get('end')]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        activities = [dict(r) for t in TABLES.values() if t in tables
                      for r in con.execute(f'SELECT * FROM "{t}"') if r['end'] > r['start']]
        graph = [r for r in con.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL') if r['graphNodeId']]
    rows = []
    for entry in entries:
        selected = [r for r in reads if entry['start'] <= r['start'] < r['end'] <= entry['end']]
        assert len(selected) == 1, 'Expected one initialization readback per PCG graph entry'
        rb = selected[0]
        nodes = [r for r in graph if rb['end'] <= r['start'] < entry['end']]
        if not nodes:
            rows.append({'zero_or_initial_failure_without_replay': True, 'readback_start_ns': rb['start']})
            continue
        end = min(r['start'] for r in nodes)
        start = rb['start']
        pieces = merge(p for r in activities for p in clip(r['start'], r['end'], [(start, end)]))
        busy = sum(b-a for a, b in pieces)
        assert 0 <= busy <= end-start
        rows.append({'readback_start_ns': start, 'first_graph_kernel_start_ns': end,
                     'window_ms': (end-start)/1e6, 'gpu_busy_union_ms': busy/1e6,
                     'gpu_inactive_upper_ms': (end-start-busy)/1e6})
    return {'directions': len(entries), 'rows': rows,
            'inactive_upper_ms': sum(r.get('gpu_inactive_upper_ms', 0) for r in rows),
            'limitations': 'Includes profiler, capture, scheduling and submission effects; neither guaranteed removable nor a whole-scene bound.'}


def main():
    plan_path = ROOT / f'configs/active/{TAG}.json'
    plan = read(plan_path)
    batch = read(ROOT / plan['report'])
    assert batch['plan_sha256'] == sha(plan_path)
    assert [r['name'] for r in batch['runs']] == [r['name'] for r in plan['runs']]
    result = {'plan_sha256': sha(plan_path), 'runs': [], 'profiles': [],
              'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    for task in batch['runs']:
        run = ROOT / 'runs/active' / task['name']
        req, completed = read(run / 'requested.json'), read(run / 'result.json')
        assert completed == task['result'] and completed['status'] == 'completed'
        assert completed['recorded_frames'] == req['expanded_config']['steps']
        execution = validate(run)
        assert execution['passed']
        m = metrics(run)
        scene_key = 'hang' if req['expanded_config']['scene'] == 'cloth_hang_l' else 'fixed_bunny'
        drift_bound = read(ROOT / 'reports/active/ipc_revision_20261005_quality_protocol.json')['scenes'][scene_key]['bounds']['fixed_drift_m']
        assert m['finite'] and m['pcg_failures'] == 0 and m['fixed_drift_m'] <= drift_bound
        frames = read(run / 'output/stats.json')['frames']
        frame = frames[-1]
        pcg = [n['pcg'] for n in frame['newton'] if 'pcg' in n]
        row = {'name': run.name, 'frames': len(frames), 'exe_sha256': req['exe_sha256'],
               'configuration_passed': True, 'finite': m['finite'], 'pcg_failures': m['pcg_failures'],
               'fixed_drift_m': m['fixed_drift_m'], 'max_stretch': m['max_stretch'], 'p99_stretch': m['p99_stretch'],
               'directions': m['directions'], 'pcg': m['pcg'], 'exits': m['exits'],
               'target_frame': {'frame': len(frames), 'directions': len(pcg),
                                'pcg': sum(p['iterations'] for p in pcg), 'phase_ms': frame['phase_ms'],
                                'exit': frame['newton_exit'], 'alpha': [n.get('alpha') for n in frame['newton']],
                                'active_pairs': [n.get('active_pairs') for n in frame['newton']]},
               'evidence': {str(p.relative_to(ROOT)): sha(p) for p in (run / 'requested.json', run / 'result.json', run / 'resolved_config.json', run / 'build_manifest.json')}}
        result['runs'].append(row)
        if req['expanded_config']['profile'] == 'none':
            continue
        database = run / 'nsight.sqlite'
        d = analyze(database, 0)
        assert d['captured_physical_frames'] == 1 and not d['missing_activity_tables']
        cpu_trace = [json.loads(s) for s in (run / 'cost.jsonl').read_text().splitlines()]
        assert cpu_trace and {r['frame'] for r in cpu_trace} == {len(frames)}
        assert all(not r['gpu_events_enabled'] and r['gpu_interval_ms'] is None for r in cpu_trace)
        a = d['aggregate']
        wall = a['cpu_physical_frame_wall']['sum_ms']
        envelope = initialization_envelope(database) if req['expanded_config']['execution'] == 'conditional_graph' else None
        if envelope:
            assert envelope['directions'] == len(pcg)
            envelope['fraction_of_profiled_frame'] = envelope['inactive_upper_ms'] / wall
        kernels = sorted(d['kernel_hotspots']['rows'], key=lambda r: r['sum_ms'], reverse=True)
        result['profiles'].append({'name': run.name, 'frame': len(frames),
            'capture_sha256': sha(run / 'nsight.nsys-rep'), 'sqlite_sha256': d['source_sha256'],
            'cpu_frame_wall_ms': wall, 'gpu_activity': a['gpu_timeline_clipped'],
            'runtime_api': a['runtime_api'], 'exclusive_gpu_owners': a['exclusive_runtime_correlation_stages'],
            'stages': d['stages_alphabetical'], 'kernels': kernels,
            'graph_initialization_envelope': envelope, 'method': d['method']})
    off = ROOT / 'runs/active' / f'{TAG}_fixed39_off'
    cpu = ROOT / 'runs/active' / f'{TAG}_fixed39_cpu'
    result['observer_control'] = {
        'positions_and_actual_velocities': state_comparison(off, cpu),
        'workload': [{'name': r['name'], 'pcg': r['pcg'], 'directions': r['directions'], 'exits': r['exits']} for r in result['runs'][:2]],
        'scope': 'One paired diagnostic; differences do not establish observer causality or statistical neutrality.'}
    result['analyzer_sha256'] = sha(__file__)
    write(ROOT / f'reports/active/{TAG}_analysis.json', result)
    print(json.dumps({'completed': len(result['runs']), 'profiles': len(result['profiles']),
                      'initialization_inactive_upper': [{ 'name': r['name'], **{k: r['graph_initialization_envelope'][k] for k in ('inactive_upper_ms', 'fraction_of_profiled_frame')}} for r in result['profiles'] if r['graph_initialization_envelope']],
                      'observer': result['observer_control']}))


if __name__ == '__main__':
    main()
