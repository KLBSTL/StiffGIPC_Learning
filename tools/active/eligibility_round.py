"""Finite single-component development round; GPU launches are serial.

Validation, profiling, paired timing and physics are separate evidence. An old
failure or a shared desktop speed estimate is never converted to certification.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from config import ROOT, read, sha, expand
from ipc_benchmark import write, metrics
from run import execute
from validate_run import validate
from component_tuning import state_comparison

TAG = 'ipc_eligibility_20261006'
SCENES = {'hang': 'cloth_hang_l', 'fixed_bunny': 'cloth_fixed_bunny_l', 'mixed': 'bunny_cloth_bunny_l'}
STEPS = {'hang': 43, 'fixed_bunny': 59, 'mixed': 35}


def task(scene, arm, repeat, stage, steps=None, **changes):
    c = {'scene': SCENES[scene], 'preset': 'combined', 'steps': steps or STEPS[scene],
         'dt': .01, 'timeout_seconds': 120, 'trace_velocity': True, 'discrete_bvh_refit': True,
         'bvh_eligibility': arm == 'on'}
    if arm == 'stiff':
        c.update(preset='graph', execution='host', discrete_bvh_refit=False)
    c.update(changes)
    expand(c)
    return {'name': f'{TAG}_{stage}_{scene}_{arm}_{repeat}', 'scene_key': scene,
            'arm': arm, 'repeat': repeat, 'binary': 'base' if arm == 'stiff' else 'active', 'config': c}


def prepare():
    stages = {'guards': [task(s, 'on', 'query', 'guards', bvh_eligibility_validate=True,
                            diagnostics=['cost'], cost_frames='1-3,' + str(STEPS[s]), cost_events=False) for s in SCENES],
              'screen': [], 'profile': []}
    for repeat in range(1, 4):
        for scene in ('hang', 'fixed_bunny') if repeat % 2 else ('fixed_bunny', 'hang'):
            for arm in ('off', 'on') if repeat % 2 else ('on', 'off'):
                stages['screen'].append(task(scene, arm, repeat, 'screen'))
    for scene, frame in (('fixed_bunny', 57), ('hang', 41)):
        for arm in ('on', 'off'):
            stages['profile'].append(task(scene, arm, 'node', 'profile', steps=frame,
                                         diagnostics=['cost'], cost_frames=str(frame), cost_events=False, profile='node'))
    paths = []
    for stage, rows in stages.items():
        p = ROOT / f'configs/active/{TAG}_{stage}.json'
        write(p, {'report': f'reports/active/{TAG}_{stage}_batch.json', 'stage': stage,
                  'stop_on_failure': True, 'runs': rows})
        paths.append(p)
    protocol = {'identity': sha(ROOT / 'builds/active/Release/gipc.exe'),
                'manifest_sha256': sha(ROOT / 'builds/active/manifest.json'),
                'old_quality_sha256': sha(ROOT / 'reports/active/ipc_revision_20261005_quality_protocol.json'),
                'prior_identity': 'fb3b46246a710febecd8badb9e5b1fcf82125fc5deebefa0b9b2c19cd6017d1d',
                'plan_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                'screen_threshold_gmean': 1.05, 'min_scene_ratio': .97, 'repeats': 3,
                'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
                'scope': 'Shared desktop finite diagnostic. Separate guard and GPU timeline from paired timing.'}
    write(ROOT / f'reports/active/{TAG}_protocol.json', protocol)
    print(json.dumps({'prepared': {s: len(r) for s, r in stages.items()}, 'identity': protocol['identity']}))


def checked(task, result):
    if result['status'] != 'completed':
        return {'passed': False, 'failures': ['run_' + result['status']]}
    run = ROOT / 'runs/active' / task['name']
    m = metrics(run)
    v = validate(run) if task['binary'] == 'active' else {'passed': True, 'scope': 'Frozen Stiff host checked by runner'}
    failures = []
    if not m['finite']: failures.append('nonfinite')
    if m['pcg_failures']: failures.append('pcg_hard_failure')
    if not v['passed']: failures.append('execution_contract')
    if any(r['abd_min_J'] is not None and r['abd_min_J'] <= 0 for r in m['frames']): failures.append('ABD_flip')
    # All candidate counters stay unmodified for later independent scrutiny.
    counters = [f.get('bvh_eligibility') for f in read(run / 'output/stats.json')['frames']]
    if task['binary'] == 'active' and any(c is None for c in counters): failures.append('missing_candidate_counters')
    aggregate = {}
    if task['binary'] == 'active' and all(c is not None for c in counters):
        enabled = task['config']['bvh_eligibility']
        diagnostic = task['config'].get('bvh_eligibility_validate',False)
        if any(c['enabled']!=enabled or c['validate']!=diagnostic for c in counters): failures.append('candidate_flag_per_frame')
        for kind in ('ordinary_vf','ordinary_ee','swept_vf','swept_ee'):
            fields = counters[0][kind]
            aggregate[kind] = {k: (max(c[kind][k] for c in counters) if k in ('scratch_bytes_peak','old_peak_capacity','new_peak_capacity') else sum(c[kind][k] for c in counters)) for k in fields}
            a = aggregate[kind]
            if not a['query_calls']: failures.append(kind+'_not_exercised')
            if enabled:
                if a['enabled_queries']!=a['query_calls'] or not a['prepare_calls']: failures.append(kind+'_enabled_not_executed')
                if a['enabled_queries']!=a['prepare_calls']+a['empty_queries']: failures.append(kind+'_refresh_coverage')
                if diagnostic and (not a['validation_calls'] or a['validation_calls']!=a['prepare_calls']
                                   or a['validation_calls']!=a['validation_passed'] or a['validation_failed']):
                    failures.append(kind+'_query_validation')
            elif any(a[k] for k in ('enabled_queries','prepare_calls','validation_calls')): failures.append(kind+'_disabled_executed')
        if diagnostic and not sum(a['pairs_compared'] for a in aggregate.values()): failures.append('no_nonempty_pairs_compared')
    limits = None
    if task['config']['steps'] == 100:
        bounds = read(ROOT / 'reports/active/ipc_revision_20261005_quality_protocol.json')['scenes'][task['scene_key']]['bounds']
        limits = {k: {'value': m[k], 'bound': b, 'passed': m[k] <= b} for k, b in bounds.items()}
    return {'passed': not failures, 'failures': failures, 'metrics': m, 'execution': v,
            'eligibility_per_frame': counters, 'eligibility_counters': aggregate, 'material_checks': limits,
            'material_100f_passed': all(r['passed'] for r in limits.values()) if limits else None,
            'independent_accepted_path_CPU_CCD_this_run': False}


def run_stage(stage, scene_scope=None):
    protocol = read(ROOT / f'reports/active/{TAG}_protocol.json')
    assert sha(ROOT / 'builds/active/Release/gipc.exe') == protocol['identity']
    fixture = read(ROOT / f'runs/active/{TAG}_fixtures/result.json')
    assert fixture['passed'] and fixture['exe_sha256'] == protocol['identity']
    assert len(fixture['checks']) == 9 and all(r['passed'] for r in fixture['checks'])
    if stage != 'guards':
        guard = read(ROOT / f'reports/active/{TAG}_guards_batch.json')
        if scene_scope:
            assert scene_scope=='hang' and stage in ('screen','profile')
            covered=[r for r in guard['runs'] if r['scene_key']==scene_scope]
            assert len(covered)==1 and covered[0]['checks']['passed']
        else:
            assert len(guard['runs']) == 3 and all(r['checks']['passed'] for r in guard['runs'])
    if stage == 'full':
        selection = read(ROOT / f'reports/active/{TAG}_selection.json')
        assert selection['eligible']
    path = ROOT / f'configs/active/{TAG}_{stage}.json'
    expected = selection['full_plan_sha256'] if stage == 'full' else protocol['plan_sha256'][path.relative_to(ROOT).as_posix()]
    assert sha(path) == expected
    plan = read(path)
    if scene_scope:
        source_path=path
        path=ROOT/f'configs/active/{TAG}_{stage}_hang_subset.json'
        plan={'report':f'reports/active/{TAG}_{stage}_hang_subset_batch.json','stage':stage,
              'source_plan_sha256':sha(source_path),'guard_sha256':sha(ROOT/f'reports/active/{TAG}_guards_batch.json'),
              'scope':'Completed hang guard only; subset of original budget; no cross-scene acceptance.',
              'runs':[r for r in plan['runs'] if r['scene_key']==scene_scope]}
        write(path,plan)
    target = ROOT / plan['report']
    assert not target.exists()
    report = {'stage': stage, 'plan_sha256': sha(path), 'runs': [], 'skipped': [], 'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    stopped = False
    for row in plan['runs']:
        if stopped:
            report['skipped'].append({'name': row['name'], 'reason': 'Earlier hard/resource failure; no budget extension'})
        else:
            result = execute(row['config'], row['name'], row['binary'])
            check = checked(row, result)
            report['runs'].append(row | {'result': result, 'checks': check})
            stopped = not check['passed']
            print(json.dumps({'run': row['name'], 'passed': check['passed'], 'failures': check['failures']}), flush=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if stopped: raise SystemExit(1)


def pairs(batch):
    groups = defaultdict(dict)
    for row in batch['runs']:
        if row['checks']['passed']: groups[(row['scene_key'], row['arm'])][row['repeat']] = row
    summary = {}
    for s in ('hang', 'fixed_bunny'):
        values = [groups[(s, 'off')][i]['result']['solver_seconds'] / groups[(s, 'on')][i]['result']['solver_seconds']
                  for i in (1, 2, 3) if i in groups[(s, 'off')] and i in groups[(s, 'on')]]
        summary[s] = {'off_over_on': values, 'median': statistics.median(values) if values else None,
                      'complete_pairs': len(values)}
    return summary, groups


def select():
    path = ROOT / f'reports/active/{TAG}_screen_batch.json'
    if not path.exists():
        guards = read(ROOT / f'reports/active/{TAG}_guards_batch.json')
        assert any(not r['checks']['passed'] for r in guards['runs'])
        result = {'screen_sha256': None, 'summary': {}, 'geometric_mean': None, 'eligible': False,
                  'full_plan_sha256': None, 'disposition': 'blocked_prerequisite',
                  'reason': 'A guard did not complete; screen and profile were not launched.',
                  'guard_sha256': sha(ROOT / f'reports/active/{TAG}_guards_batch.json'),
                  'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
        write(ROOT / f'reports/active/{TAG}_selection.json', result)
        print(json.dumps(result))
        return
    summary, _ = pairs(read(path))
    complete = all(s['complete_pairs'] == 3 for s in summary.values())
    geo = math.sqrt(math.prod(s['median'] for s in summary.values())) if complete else None
    eligible = bool(complete and geo >= 1.05 and min(s['median'] for s in summary.values()) >= .97)
    result = {'screen_sha256': sha(path), 'summary': summary, 'geometric_mean': geo, 'eligible': eligible,
              'full_plan_sha256': None, 'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    if eligible:
        rows = []
        for repeat in range(1, 4):
            arms = ['stiff', 'off', 'on']
            arms = arms[repeat-1:] + arms[:repeat-1]
            if repeat % 2 == 0: arms.reverse()
            for scene in ('hang', 'fixed_bunny') if repeat % 2 else ('fixed_bunny', 'hang'):
                rows += [task(scene, arm, repeat, 'full', steps=100) for arm in arms]
        p = ROOT / f'configs/active/{TAG}_full.json'
        write(p, {'report': f'reports/active/{TAG}_full_batch.json', 'stage': 'full', 'runs': rows})
        result['full_plan_sha256'] = sha(p)
    write(ROOT / f'reports/active/{TAG}_selection.json', result)
    print(json.dumps(result))


def analyze_round(scene_scope=None):
    suffix='_hang_subset' if scene_scope else ''
    selection = read(ROOT / f'reports/active/{TAG}_selection.json')
    result = {'selection': selection, 'scene_scope':scene_scope, 'stages': {}, 'paired_states': [], 'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    for stage in ('guards', 'screen', 'profile', 'full'):
        path = ROOT / f'reports/active/{TAG}_{stage}{suffix if stage in ("screen","profile") else ""}_batch.json'
        if not path.exists():
            result['stages'][stage] = {'attempted': 0, 'reason': selection.get('reason', 'Not run; screen gate/finite budget')}
            continue
        batch = read(path)
        result['stages'][stage] = {'attempted': len(batch['runs']), 'completed': sum(r['result']['status']=='completed' for r in batch['runs']),
                                   'failed': sum(not r['checks']['passed'] for r in batch['runs']), 'skipped': batch['skipped'], 'sha256': sha(path)}
        if stage not in ('screen', 'full'): continue
        summary, groups = pairs(batch)
        result['stages'][stage]['paired_speed'] = summary
        result['stages'][stage]['arms'] = {s: {a: [{'repeat': i, 'seconds': r['result']['solver_seconds'],
                         'pcg': r['checks']['metrics']['pcg'], 'directions': r['checks']['metrics']['directions'],
                         'phase_ms': r['checks']['metrics']['phase_ms'], 'max_stretch': r['checks']['metrics']['max_stretch'],
                         'p99_stretch': r['checks']['metrics']['p99_stretch'], 'material_100f_passed': r['checks']['material_100f_passed']}
                        for i, r in groups[(s,a)].items()] for a in ('stiff', 'off', 'on')} for s in ('hang', 'fixed_bunny')}
        for scene in ('hang', 'fixed_bunny'):
            for i in (1,2,3):
                if i not in groups[(scene,'off')] or i not in groups[(scene,'on')]: continue
                a, b = groups[(scene,'off')][i], groups[(scene,'on')][i]
                result['paired_states'].append({'stage': stage, 'scene': scene, 'repeat': i,
                    'actual_state': state_comparison(ROOT/'runs/active'/a['name'], ROOT/'runs/active'/b['name'])})
    write(ROOT / f'reports/active/{TAG}{suffix}_analysis.json', result)
    print(json.dumps({'coverage': result['stages'], 'paired_state_count': len(result['paired_states'])})[:1800])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','select','analyze'])
    p.add_argument('--stage', choices=['guards','screen','profile','full'])
    p.add_argument('--scene',choices=['hang'],help='Resource-isolated diagnostic subset after that scene completes its guard')
    a = p.parse_args()
    if a.action == 'prepare': prepare()
    elif a.action == 'run':
        if not a.stage: p.error('--stage required')
        run_stage(a.stage,a.scene)
    elif a.action == 'select': select()
    else: analyze_round(a.scene)
