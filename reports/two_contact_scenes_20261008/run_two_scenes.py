"""Fresh, finite 18-run supplement using the established protected runner."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
SESSION = ROOT / 'runs/two_contact_scenes_20261008'
for folder in ('bench', 'diagnostic', 'full_eval', 'local'):
    sys.path.insert(0, str(ROOT / 'tools' / folder))
sys.path.insert(0, str(ROOT / 'reports/report56_stage1_20261008'))
from config import read, expand, digest
from linux_runner import require, write_new, verify_files
from local_identity import record, verify, tree
from windows_runner import execute, gpu_lock
from pool_metrics import metrics, export_evidence
from quality_analysis import input_comparison
from run_stage1 import config, guard_summary

spec = importlib.util.spec_from_file_location('two_scene_full_analysis', ROOT / 'tools/full_eval/analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
SCENES = ('cloth_table_l', 'cloth_stack10_l')
ARMS = ('base', 'graph', 'all')
CACHE_INPUTS = []


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tasks():
    result = []
    for repeat in range(3):
        order = ARMS[repeat:] + ARMS[:repeat]
        for scene in SCENES:
            for arm in order:
                c = config(scene, arm)
                require(c['diagnostics'] == [] and c['timeout_seconds'] == 180, 'Diagnostic or budget drift')
                result.append(dict(name=f'{scene}_{arm}_r{repeat+1}', scene=scene,
                    arm=arm, repeat=repeat+1, binary='base' if arm == 'base' else 'active',
                    config=c, config_sha256=digest(c)))
    return result


def verify_program(identity):
    verify(identity['sources'] + identity['build_evidence'] + identity.get('objects', []) +
           [identity['exe']] + identity['dlls'] + [r['binary'] for r in identity['compilers']])
    if identity['kind'] == 'active':
        # Initial imports can create these separately sealed derived inputs.
        # Compiled source identity remains unchanged; all original bytes and
        # the exact augmented inventory must still match.
        verify(CACHE_INPUTS)
        source = Path(identity['source_root'])
        current = sum([tree(source / n) for n in ('StiffGIPC','Assets','MeshProcess')], []) + [record(source / 'CMakeLists.txt')]
        require({r['path']:r for r in current} ==
                {r['path']:r for r in identity['sources'] + CACHE_INPUTS}, 'Source/input inventory changed')


def init():
    require(not SESSION.exists(), 'Fresh session required; no overwritten attempts')
    previous = ROOT / 'reports/report56_stage1_20261008'
    programs = {'active': read(previous / 'PROBE_PROGRAM_IDENTITY.json'),
                'base': read(ROOT / 'runs/report56_stage1_20261008/REFERENCE_IDENTITY.json')['programs']['base']}
    require(programs['active']['exe']['sha256'] == 'd4da6bd5ec8544b87d0f7b77f67c6bb02a95c2afa19a24a8a6013f69d89f3902',
            'Different current program')
    require(programs['base']['exe']['sha256'] == '1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205',
            'Different frozen Stiff')
    for identity in programs.values():
        verify_program(identity)
    tools = [record(ROOT / p) for p in (
        'tools/local/windows_runner.py', 'tools/local/windows_owned_job.py',
        'tools/local/local_identity.py', 'tools/bench/config.py', 'tools/bench/validate_run.py',
        'tools/bench/pool_metrics.py', 'tools/diagnostic/quality_analysis.py',
        'tools/full_eval/analysis.py', 'reports/report56_stage1_20261008/run_stage1.py')]
    write_new(SESSION / 'IDENTITY.json', dict(programs=programs, tools=tools,
        controller=record(__file__), plan=record(REPORT / 'PLAN.md'), tasks=tasks(),
        performance_certified=False, quality_certified=False))
    print(json.dumps({'status': 'frozen', 'planned_runs': len(tasks())}), flush=True)


def inspect(task, result):
    folder = SESSION / task['name']
    verify_files(folder, read(folder / 'evidence.json')['files'])
    req = read(folder / 'requested.json')
    require(req['expanded_config'] == expand(task['config']), 'Requested config differs')
    require(result['status'] == 'completed' and result['recorded_frames'] == 120 and
            result.get('finite') is True and result.get('cleanup_owned_job_empty') is True,
            'Incomplete, failed, or unclosed native run: ' + result['status'])
    require(read(folder / 'config_validation.json')['passed'], 'Native config differs')
    require(export_evidence(folder, 120, 'state')['passed'], 'State exports missing/nonfinite')
    if task['binary'] == 'active':
        require(export_evidence(folder, 120, 'velocity')['passed'], 'Actual velocity missing/nonfinite')
    summary = guard_summary(folder, result)
    require(summary['guards_passed'], 'PCG/Newton/line-search failure')
    m = metrics(folder)
    require(m['finite'] and not m['pcg_failures'], 'Nonfinite or PCG failure')
    frames = read(folder / 'output/stats.json')['frames']
    with (folder / 'trace/frames.csv').open(encoding='utf-8-sig', newline='') as stream:
        timings = list(csv.DictReader(stream))
    require(len(frames) == 120 and [int(t['frame']) for t in timings] == list(range(1, 121)),
            'Wrong continuous frame sequence')
    seconds = sum(float(t['solver_ms']) for t in timings) / 1000
    require(math.isclose(seconds, result['solver_seconds'], rel_tol=1e-10, abs_tol=1e-9), 'Timing mismatch')
    geometry = [f.get('contact_geometry', {}) for f in frames]
    return {k: task[k] for k in ('name', 'scene', 'arm', 'repeat', 'binary', 'config_sha256')} | dict(
        solver_seconds=seconds, wall_seconds=result['wall_seconds'], summary=summary,
        material=analysis.material_summary(m['frames']), material_by_frame=m['frames'],
        exits=m['exits'], actual_velocity_available=task['binary']=='active',
        native_self_pair_peak=max((g.get('native_narrow_self_pairs', 0) for g in geometry), default=0),
        native_ground_pair_peak=max((g.get('native_narrow_ground_pairs', 0) for g in geometry), default=0),
        hard_checks_passed=True, evidence=record(folder / 'evidence.json'),
        performance_certified=False, quality_certified=False)


def report(ledger):
    scenes = []
    for scene in SCENES:
        rows = [r for r in ledger['rows'] if r['scene'] == scene]
        groups = {arm: [r for r in rows if r['arm'] == arm] for arm in ARMS}
        item = dict(scene=scene, completed_per_arm={k: len(v) for k,v in groups.items()},
            median_seconds={k: statistics.median(r['solver_seconds'] for r in v) if v else None for k,v in groups.items()},
            paired_ratios=[], input_checks=[])
        for repeat in range(1, 4):
            arms = {r['arm']: r for r in rows if r['repeat'] == repeat}
            if set(arms) != set(ARMS):
                continue
            for arm in ('graph', 'all'):
                check = input_comparison(SESSION / arms['base']['name'], SESSION / arms[arm]['name'])
                require(check['passed'], 'Different physical input: ' + scene)
                item['input_checks'].append(dict(repeat=repeat, arm=arm, **check))
            t = {a: r['solver_seconds'] for a,r in arms.items()}
            item['paired_ratios'].append(dict(repeat=repeat, stiff_over_graph=t['base']/t['graph'],
                stiff_over_all=t['base']/t['all'], graph_over_all=t['graph']/t['all']))
        if len(item['paired_ratios']) == 3:
            item['paired_medians'] = {k: statistics.median(r[k] for r in item['paired_ratios'])
                for k in ('stiff_over_graph', 'stiff_over_all', 'graph_over_all')}
        item['phase_ms_medians'] = {arm: {phase: statistics.median(r['summary']['recorded_phase_ms'].get(phase, 0) for r in rr)
            for phase in ('assembly','pcg','ccd','line_search','state_update')} for arm,rr in groups.items() if rr}
        scenes.append(item)
    save(REPORT / 'RESULTS.json', dict(status=ledger['status'], scenes=scenes, rows=ledger['rows'],
        failures=ledger['failures'], performance_certified=False, quality_certified=False))
    return scenes


def resume_receipt():
    seal = read(SESSION / 'IDENTITY.json')
    old = record(REPORT / 'controller_v1.py.txt')
    require((old['bytes'], old['sha256']) ==
            (seal['controller']['bytes'], seal['controller']['sha256']), 'Original controller snapshot differs')
    ledger = read(SESSION / 'LEDGER.json')
    require(ledger['status'] == 'running' and not ledger['failures'] and
            [r['name'] for r in ledger['rows']] == [t['name'] for t in tasks()[:2]],
            'Resume is only for the verified two completed runs, not failed native attempts')
    cache = [record(ROOT / 'Assets/sorted_mesh' / name) for name in
             ('cipc_table_sorted.16.obj','cipc_table_sorted.16.part')]
    require(all(r['bytes'] > 0 for r in cache), 'Empty generated cache')
    require(input_comparison(SESSION / ledger['rows'][0]['name'],
                            SESSION / ledger['rows'][1]['name'])['passed'], 'Initial input identity differs')
    write_new(SESSION / 'RESUME_IDENTITY.json', dict(controller=record(__file__),
        original_controller=old, original_identity=record(SESSION / 'IDENTITY.json'),
        original_ledger=record(SESSION / 'LEDGER_before_cache_pause.json'), cache_inputs=cache,
        cache_producer_source=record(ROOT / 'MeshProcess/metis_partition/src/metis_sort.cpp'),
        reason='First active table import generated two sorted runtime inputs; all compiled sources, program and original inputs unchanged. No completed native run is repeated.'))
    print(json.dumps({'status':'resume_frozen','remaining_runs':16}), flush=True)


def run(resume=False):
    global CACHE_INPUTS
    seal = read(SESSION / 'IDENTITY.json')
    if resume:
        extra = read(SESSION / 'RESUME_IDENTITY.json')
        verify([extra[k] for k in ('controller','original_controller','original_identity',
                                  'original_ledger','cache_producer_source')])
        CACHE_INPUTS = extra['cache_inputs']
        verify(CACHE_INPUTS)
        ledger = read(SESSION / 'LEDGER.json')
        require(ledger['status'] == 'running' and not ledger['failures'], 'No retry of failed attempts')
    else:
        verify([seal['controller']])
        require(not (SESSION / 'LEDGER.json').exists(), 'Attempt ledger exists; no automatic retry')
        ledger = dict(status='running', planned_runs=18, rows=[], failures=[])
        write_new(SESSION / 'LEDGER.json', ledger)
    verify(seal['tools'] + [seal['plan']])
    require(seal['tasks'] == tasks(), 'Declared task list changed')
    with gpu_lock(ROOT):
        for task in seal['tasks']:
            identity = seal['programs'][task['binary']]
            verify_program(identity)
            previous = [r for r in ledger['rows'] if r['name'] == task['name']]
            if previous:
                require(len(previous) == 1 and previous[0] ==
                        inspect(task, read(SESSION / task['name'] / 'result.json')), 'Completed evidence changed')
                continue
            result = execute(SESSION, task, identity, 0)
            try:
                row = inspect(task, result)
                ledger['rows'].append(row)
                print(json.dumps({k: row[k] for k in ('name','solver_seconds')} |
                    {'pcg': row['summary']['pcg_iterations'], 'directions': row['summary']['directions']}), flush=True)
            except Exception as exc:
                ledger['status'] = 'failed'
                ledger['failures'].append(dict(name=task['name'], status=result['status'], reason=str(exc)))
                save(SESSION / 'LEDGER.json', ledger)
                report(ledger)
                raise
            save(SESSION / 'LEDGER.json', ledger)
            report(ledger)
        for identity in seal['programs'].values():
            verify_program(identity)
    verify(seal['tools'] + [seal['plan']])
    verify([read(SESSION / 'RESUME_IDENTITY.json')['controller']] if resume else [seal['controller']])
    ledger['status'] = 'completed'
    save(SESSION / 'LEDGER.json', ledger)
    scenes = report(ledger)
    print(json.dumps({'status':'completed','runs':len(ledger['rows']),
        'scenes':[{k:s[k] for k in ('scene','median_seconds','paired_medians')} for s in scenes]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init','run','resume-freeze','resume','analyze'))
    args = parser.parse_args()
    if args.action == 'init':
        init()
    elif args.action == 'run':
        run()
    elif args.action == 'resume-freeze':
        resume_receipt()
    elif args.action == 'resume':
        run(resume=True)
    else:
        print(json.dumps(report(read(SESSION / 'LEDGER.json'))), flush=True)
