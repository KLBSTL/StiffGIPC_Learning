"""One bounded revision and paired local screen; immutable plans/results.

The 23-frame screen cannot certify whole-scene material or speed. The existing
quality protocol is never changed; failed resource requests stay failed.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict

import numpy as np
from config import ROOT, expand, read, sha
from ipc_benchmark import write, metrics
from run import execute
from spmv_compensated import audit_controller
from validate_run import validate

TAG = 'ipc_component_tuning_20261006'
SCENES = {'hang': 'cloth_hang_l', 'fixed_bunny': 'cloth_fixed_bunny_l', 'mixed': 'bunny_cloth_bunny_l'}
ARMS = ('stiff', 'graph', 'combined', 'mas', 'spmv', 'bvh', 'fused')
FLAGS = {'mas': {'mas_fused_dot': True}, 'spmv': {'spmv_fused_quadratic': True},
         'bvh': {'discrete_bvh_refit': True},
         'fused': {'mas_fused_dot': True, 'spmv_fused_quadratic': True, 'discrete_bvh_refit': True}}


def state_comparison(a, b):
    """Report missing exports explicitly; never infer actual velocity from x."""
    meta = read(a / 'trace/metadata.json')
    nv = np.fromfile(a / 'trace/state_0000.bin', dtype='<f8').size // 3
    raw = np.fromfile(a / 'trace/topology.bin', dtype='<u4')
    assert np.array_equal(raw, np.fromfile(b / 'trace/topology.bin', dtype='<u4'))
    nf, nt = map(int, raw[1:3])
    tets = raw[3+3*nf:].reshape(nt, 4)
    abd = meta['abd_point_num']
    fem = np.zeros(nv, bool)
    fem[tets.ravel()] = True
    fem[:abd] = False
    groups = {'cloth': ~fem & (np.arange(nv) >= abd), 'FEM': fem, 'ABD': np.arange(nv) < abd}
    result = {}
    for kind, pattern in (('position', 'state_*.bin'), ('velocity', 'velocity_*.bin')):
        files = sorted((a / 'trace').glob(pattern))
        other = sorted((b / 'trace').glob(pattern))
        if not files or not other:
            result[kind] = {'available': False, 'reason': 'Actual export absent in one or both programs',
                            'run_exported_frames': len(files), 'reference_exported_frames': len(other)}
            continue
        assert [f.name for f in files] == [f.name for f in other], 'Incomplete paired state exports'
        values = {k: [] for k, mask in groups.items() if mask.any()}
        for p, q in zip(files, other):
            x, y = np.fromfile(p, dtype='<f8'), np.fromfile(q, dtype='<f8')
            assert x.size == y.size == 3*nv and np.isfinite(x).all() and np.isfinite(y).all()
            delta = (x-y).reshape(nv, 3)
            for k in values:
                error = np.linalg.norm(delta[groups[k]], axis=1)
                values[k].append((float(np.sqrt(np.mean(error**2))), float(error.max())))
        result[kind] = {k: {'max_frame_rms': max(v[0] for v in rows), 'max_vertex': max(v[1] for v in rows),
                           'frames': len(rows)} for k, rows in values.items()}
    return result


def task(scene, arm, repeat, stage, steps, **changes):
    cfg = {'scene': SCENES[scene], 'preset': 'graph' if arm in ('stiff', 'graph') else 'combined',
           'steps': steps, 'dt': .01, 'timeout_seconds': 120, 'trace_velocity': True}
    cfg.update(FLAGS.get(arm, {}))
    if arm == 'stiff':
        cfg['execution'] = 'host'
    cfg.update(changes)
    expand(cfg)
    return {'name': f'{TAG}_{stage}_{scene}_{arm}_{repeat}', 'scene_key': scene,
            'arm': arm, 'repeat': repeat, 'binary': 'base' if arm == 'stiff' else 'active', 'config': cfg}


def prepare():
    stages = {
        'guards': [task('hang', 'bvh', 'query', 'guards', 3, discrete_bvh_validate=True),
                   task('fixed_bunny', 'bvh', 'query', 'guards', 23, discrete_bvh_validate=True),
                   task('mixed', 'fused', 'compat', 'guards', 3),
                   task('fixed_bunny', 'fused', 'compensated', 'guards', 23,
                        ipc_termination='compensated', ipc_cumulative_tol=.001)],
        'screen': []}
    # Rotate and reverse complete groups; each round still has all seven arms.
    for repeat in range(1, 4):
        order = list(ARMS[repeat-1:] + ARMS[:repeat-1])
        if repeat % 2 == 0:
            order.reverse()
        for scene in ('hang', 'fixed_bunny') if repeat % 2 else ('fixed_bunny', 'hang'):
            stages['screen'] += [task(scene, arm, repeat, 'screen', 23) for arm in order]
    plans = []
    for stage, rows in stages.items():
        path = ROOT / f'configs/active/{TAG}_{stage}.json'
        write(path, {'stage': stage, 'runs': rows, 'finite_budget': True})
        plans.append(path)
    protocol = {
        'identity': sha(ROOT / 'builds/active/Release/gipc.exe'),
        'prior_identity': '5269bb3d4668f016f845a3f48af38fef5ed1538caba5f40057e9d15d79075af4',
        'base_manifest_sha256': sha(ROOT / 'manifests/perf_v34.json'),
        'build_manifest_sha256': sha(ROOT / 'builds/active/manifest.json'),
        'old_quality_sha256': sha(ROOT / 'reports/active/ipc_revision_20261005_quality_protocol.json'),
        'plan_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in plans},
        'screen_frames': 23, 'repeats': 3, 'full_threshold_gmean': 1.05, 'min_scene_ratio': .97,
        'candidate_selection': 'Largest two-scene geometric mean of three paired medians vs combined, subject to both >=.97; ties by declared mas/spmv/bvh/fused order',
        'scope': 'Shared desktop bounded local diagnostic; no performance or whole-scene quality certification',
        'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    write(ROOT / f'reports/active/{TAG}_protocol.json', protocol)
    print(json.dumps({'prepared': {k: len(v) for k, v in stages.items()}, 'identity': protocol['identity']}))


def check(row, result):
    if result['status'] != 'completed':
        return {'passed': False, 'hard_failures': ['run_' + result['status']]}
    run = ROOT / 'runs/active' / row['name']
    m = metrics(run)
    failures = []
    if not m['finite']:
        failures.append('nonfinite_state')
    if m['pcg_failures']:
        failures.append('pcg_cap_or_breakdown')
    if any(f['abd_min_J'] is not None and f['abd_min_J'] <= 0 for f in m['frames']):
        failures.append('abd_flip')
    execution = validate(run) if row['binary'] == 'active' else {'passed': True, 'scope': 'Frozen Stiff host contract checked by runner'}
    if not execution['passed']:
        failures.append('configuration_execution')
    stats = read(run / 'output/stats.json')['frames']
    cfg = read(run / 'requested.json')['expanded_config']
    controller = audit_controller(stats, cfg) if row['binary'] == 'active' else None
    counters = {kind: {key: sum(f.get('discrete_bvh', {}).get(kind, {}).get(key, 0) for f in stats)
               for key in ('construct_calls', 'production_rebuilds', 'production_refits', 'disabled_rebuilds',
                           'swept_rebuilds', 'interval_rebuilds', 'signature_rebuilds',
                           'validation_calls', 'validation_passed', 'validation_failed',
                           'diagnostic_pairs_compared', 'storage_switches', 'ordinary_cache_restores',
                           'swept_full_builds', 'swept_refits', 'swept_refit_fallbacks')}
                for kind in ('face', 'edge')}
    for kind, c in counters.items():
        c['swept_cache_capacity_bytes_peak'] = max(
            (f.get('discrete_bvh', {}).get(kind, {}).get('swept_cache_capacity_bytes_peak', 0) for f in stats), default=0)
    if cfg['discrete_bvh_refit']:
        for kind, c in counters.items():
            if c['production_refits'] == 0:
                failures.append(kind + '_no_production_refit')
            if c['construct_calls'] != c['production_refits'] + c['production_rebuilds']:
                failures.append(kind + '_counter_mismatch')
            if c['swept_rebuilds']:
                failures.append(kind + '_swept_destroyed_ordinary_cache')
            if not c['ordinary_cache_restores'] or not c['swept_cache_capacity_bytes_peak']:
                failures.append(kind + '_split_cache_not_observed')
            if cfg['discrete_bvh_validate'] and (not c['validation_calls'] or c['validation_calls'] != c['validation_passed'] or c['validation_failed']):
                failures.append(kind + '_query_validation')
    limits = None
    if cfg['steps'] == 100 and row['scene_key'] in ('hang', 'fixed_bunny'):
        bounds = read(ROOT / 'reports/active/ipc_revision_20261005_quality_protocol.json')['scenes'][row['scene_key']]['bounds']
        limits = {k: {'value': m[k], 'bound': v, 'passed': m[k] <= v} for k, v in bounds.items()}
    return {'passed': not failures, 'hard_failures': failures, 'metrics': m, 'execution': execution,
            'controller': controller, 'bvh_counters': counters, 'material_100f_checks': limits,
            'material_100f_passed': all(v['passed'] for v in limits.values()) if limits is not None else None,
            'independent_accepted_path_CPU_CCD_this_run': False}


def run_stage(stage):
    protocol = read(ROOT / f'reports/active/{TAG}_protocol.json')
    assert protocol['identity'] == sha(ROOT / 'builds/active/Release/gipc.exe')
    fixture = read(ROOT / f'runs/active/{TAG}_fixtures/result.json')
    assert fixture['passed'] and fixture['exe_sha256'] == protocol['identity']
    path = ROOT / f'configs/active/{TAG}_{stage}.json'
    if stage == 'full':
        selection = read(ROOT / f'reports/active/{TAG}_selection.json')
        assert selection['eligible'] and sha(path) == selection['full_plan_sha256']
    else:
        assert sha(path) == protocol['plan_sha256'][path.relative_to(ROOT).as_posix()]
    output = ROOT / f'reports/active/{TAG}_{stage}_batch.json'
    assert not output.exists(), 'Immutable output names'
    blocked = set()
    if stage != 'guards':
        guards = read(ROOT / f'reports/active/{TAG}_guards_batch.json')
        blocked.update(r['scene_key'] for r in guards['runs'] if not r['checks']['passed'])
        blocked.update(r['scene_key'] for r in guards['skipped'])
    report = {'stage': stage, 'plan_sha256': sha(path), 'runs': [], 'skipped': [],
              'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    for row in read(path)['runs']:
        if row['scene_key'] in blocked:
            report['skipped'].append({'name': row['name'], 'scene_key': row['scene_key'], 'reason': 'Earlier same-scene hard/resource failure; no retry'})
        else:
            result = execute(row['config'], row['name'], row['binary'])
            checked = check(row, result)
            report['runs'].append(row | {'result': result, 'checks': checked})
            if not checked['passed']:
                blocked.add(row['scene_key'])
            print(json.dumps({'checked': row['name'], 'hard_failures': checked['hard_failures']}), flush=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'stage': stage, 'attempted': len(report['runs']), 'skipped': len(report['skipped'])}), flush=True)


def paired(batch):
    groups = defaultdict(dict)
    for row in batch['runs']:
        if row['checks']['passed']:
            groups[(row['scene_key'], row['arm'])][row['repeat']] = row
    summary = {}
    for scene in ('hang', 'fixed_bunny'):
        summary[scene] = {}
        for arm in ARMS:
            rows = list(groups[(scene, arm)].values())
            if not rows:
                continue
            ratios = {}
            for base in ('stiff', 'graph', 'combined'):
                values = [groups[(scene, base)][r['repeat']]['result']['solver_seconds'] / r['result']['solver_seconds']
                          for r in rows if r['repeat'] in groups[(scene, base)]]
                ratios[base] = {'values': values, 'median': statistics.median(values) if values else None,
                                'min': min(values) if values else None, 'max': max(values) if values else None}
            ms = [r['checks']['metrics'] for r in rows]
            summary[scene][arm] = {'completed': len(rows), 'median_seconds': statistics.median(r['result']['solver_seconds'] for r in rows),
                'paired': ratios, 'pcg_range': [min(m['pcg'] for m in ms), max(m['pcg'] for m in ms)],
                'direction_range': [min(m['directions'] for m in ms), max(m['directions'] for m in ms)],
                'max_stretch_range': [min(m['max_stretch'] for m in ms), max(m['max_stretch'] for m in ms)],
                'p99_stretch_range': [min(m['p99_stretch'] for m in ms), max(m['p99_stretch'] for m in ms)],
                'phase_ms_medians': {k: statistics.median(m['phase_ms'][k] for m in ms) for k in ms[0]['phase_ms']},
                'bvh_counters': [r['checks']['bvh_counters'] for r in rows],
                'material_100f_passed': sum(r['checks']['material_100f_passed'] is True for r in rows)}
    return summary, groups


def select():
    batch_path = ROOT / f'reports/active/{TAG}_screen_batch.json'
    batch = read(batch_path)
    summary, _ = paired(batch)
    candidates = []
    for arm in ('mas', 'spmv', 'bvh', 'fused'):
        if any(summary[s].get(arm, {}).get('completed') != 3 or
               len(summary[s][arm]['paired']['combined']['values']) != 3 for s in ('hang', 'fixed_bunny')):
            continue
        ratios = [summary[s][arm]['paired']['combined']['median'] for s in ('hang', 'fixed_bunny')]
        candidates.append({'arm': arm, 'gmean': math.sqrt(math.prod(ratios)), 'min_scene': min(ratios),
                           'ratio_by_scene': dict(zip(('hang', 'fixed_bunny'), ratios))})
    valid = [c for c in candidates if c['min_scene'] >= .97]
    best = max(valid, key=lambda c: c['gmean']) if valid else None
    eligible = bool(best and best['gmean'] >= 1.05)
    selection = {'screen_batch_sha256': sha(batch_path), 'summary': summary, 'candidates': candidates,
                 'selected': best, 'eligible': eligible, 'full_plan_sha256': None,
                 'scope': 'Predeclared diagnostic screen selection only; independent full phase required',
                 'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    if eligible:
        rows = []
        arms = ('stiff', 'graph', 'combined', best['arm'])
        for repeat in range(1, 4):
            order = list(arms[repeat-1:] + arms[:repeat-1])
            if repeat % 2 == 0:
                order.reverse()
            for scene in ('hang', 'fixed_bunny') if repeat % 2 else ('fixed_bunny', 'hang'):
                rows += [task(scene, arm, repeat, 'full', 100) for arm in order]
        plan_path = ROOT / f'configs/active/{TAG}_full.json'
        write(plan_path, {'stage': 'full', 'selected_screen_arm': best['arm'], 'runs': rows})
        selection['full_plan_sha256'] = sha(plan_path)
    write(ROOT / f'reports/active/{TAG}_selection.json', selection)
    print(json.dumps({'candidates': candidates, 'eligible': eligible, 'full_runs': 24 if eligible else 0}))


def analyze():
    selection_path = ROOT / f'reports/active/{TAG}_selection.json'
    selection = read(selection_path)
    all_rows, stage_coverage, distances = [], {}, []
    for stage in ('guards', 'screen', 'full'):
        path = ROOT / f'reports/active/{TAG}_{stage}_batch.json'
        if not path.exists():
            stage_coverage[stage] = {'attempted': 0, 'reason': 'Predeclared screen threshold not met' if not selection['eligible'] else 'Not executed'}
            continue
        batch = read(path)
        stage_coverage[stage] = {'attempted': len(batch['runs']), 'completed': sum(r['result']['status'] == 'completed' for r in batch['runs']),
                                 'failed': sum(not r['checks']['passed'] for r in batch['runs']), 'skipped': len(batch['skipped']), 'sha256': sha(path)}
        all_rows += batch['runs']
        if stage == 'guards':
            continue
        _, groups = paired(batch)
        for row in batch['runs']:
            reference = groups[(row['scene_key'], 'combined')].get(row['repeat'])
            if reference and row['checks']['passed']:
                distances.append({'run': row['name'], 'reference': reference['name'], 'stage': stage,
                                  'position_velocity': state_comparison(ROOT / 'runs/active' / row['name'], ROOT / 'runs/active' / reference['name'])})
    full_path = ROOT / f'reports/active/{TAG}_full_batch.json'
    full_summary = paired(read(full_path))[0] if full_path.exists() else {}
    report = {'protocol_sha256': sha(ROOT / f'reports/active/{TAG}_protocol.json'),
              'selection_sha256': sha(selection_path), 'coverage': stage_coverage, 'full_summary': full_summary,
              'failures': [{'name': r['name'], 'status': r['result']['status'], 'failure': r['checks']['hard_failures']} for r in all_rows if not r['checks']['passed']],
              'distances': distances, 'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    write(ROOT / f'reports/active/{TAG}_analysis.json', report)
    print(json.dumps({'coverage': stage_coverage, 'failures': report['failures']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'select', 'analyze'])
    parser.add_argument('--stage', choices=['guards', 'screen', 'full'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'run':
        if args.stage is None:
            parser.error('--stage required')
        run_stage(args.stage)
    elif args.action == 'select':
        select()
    else:
        analyze()
