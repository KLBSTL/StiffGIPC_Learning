"""Additive CPU analysis for the finite three-scene AutoDL evaluation.

Missing Stiff velocity and independent accepted-path CCD prohibit quality
certification. Material bounds are frozen from three separate calibration runs;
two held-out baseline runs cannot widen them. No GPU work is launched here.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'bench'))
from config import digest, expand, read, sha
from pool_metrics import export_evidence, metrics, state_comparison
from validate_run import validate

SCENES = ('hang', 'fixed', 'mixed')
ARMS = ('original_stiff', 'ipc_host', 'ipc_graph', 'combined_graph')
PHASES = ('assembly', 'pcg', 'ccd', 'line_search', 'state_update')
TOLERANCE = 1e-12
# +1 means a larger value is worse; -1 means a smaller value is worse.
MATERIAL_DIRECTIONS = {'max_stretch': 1, 'p99_stretch': 1, 'fixed_drift_m': 1,
                       'fem_min_J': -1, 'fem_nonpositive_peak': 1,
                       'fem_nonpositive_frame_sum': 1, 'fem_negative_volume_peak': 1,
                       'abd_min_J': -1, 'abd_nonpositive_frames': 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def input_identity(run):
    # Both frozen/active gl_main emit only abd_point_num, ground_offset and
    # ground_normal in metadata.json; no executable/source/config identity.
    names = ('trace/topology.bin', 'trace/masses.bin', 'trace/boundary_types.bin',
             'trace/body_ids.bin', 'trace/state_0000.bin', 'trace/metadata.json')
    result = {name: sha(run / name) for name in names}
    result['physical_scene_objects'] = digest(read(run / 'output/scene.json')['objects'])
    velocity = run / 'trace/velocity_0000.bin'
    result['initial_actual_velocity'] = sha(velocity) if velocity.exists() else None
    return result


def compare_inputs(left, right):
    keys = set(left) | set(right)
    required = keys - {'initial_actual_velocity'}
    checks = {key: left.get(key) is not None and left.get(key) == right.get(key) for key in required}
    actual_velocity = left.get('initial_actual_velocity') is not None and right.get('initial_actual_velocity') is not None
    if actual_velocity:
        checks['initial_actual_velocity'] = left['initial_actual_velocity'] == right['initial_actual_velocity']
    return {'passed': bool(checks) and all(checks.values()), 'checks': checks,
            'actual_initial_velocity_comparable': actual_velocity,
            'missing_velocity_reason': None if actual_velocity else 'Actual baseline velocity absent; never reconstructed from positions.'}


def baseline_configuration(run, c, frames):
    scene = read(run / 'output/scene.json')
    effective, fields = scene['effective_run'], scene['effective_scalar_fields']
    pcgs = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    checks = {'scene': scene['case_id'] == c['scene'], 'mas': fields['preconditioner_type'] == 1,
              'dt': effective['dt'] == c['dt'], 'newton_tol': effective['newton_tol'] == c['ipc_newton_tol'],
              'rho_tol': effective['pcg_tol'] == c['pcg_rho_tol'],
              'no_fabricated_resolved': not (run / 'resolved_config.json').exists(),
              'no_fabricated_velocity': not list((run / 'trace').glob('velocity_*.bin')),
              'pcg_records': bool(pcgs), 'limit_telemetry': bool(pcgs) and all('iteration_limit' in p for p in pcgs)}
    return {'passed': all(checks.values()), 'checks': checks,
            'resolved_config_available': False,
            'complete_breakdown_telemetry_available': bool(pcgs) and all('breakdown' in p for p in pcgs),
            'unobserved': ['Native resolved execution/min-updates', 'Complete PCG breakdown telemetry',
                           'Actual velocities', 'Independent accepted-path CCD']}


def read_timecost(path):
    path = Path(path)
    if not path.exists():
        return {'available': False, 'reason': 'Native timeCost.txt absent; CUDA event timing not inferred from CPU frame timing.'}
    seconds = ('time0', 'time1', 'time2', 'time3', 'time4', 'time_makePD', 'totalTime')
    counts = ('total iter', 'frames', 'totalCollisionNum', 'averageCollision', 'maxCOllisionPairNum', 'totalCgTime')
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ':' not in line:
            continue
        key, raw = (s.strip() for s in line.split(':', 1))
        if key not in seconds + counts:
            continue
        require(key not in values, 'Duplicate timeCost field: ' + key)
        value = float(raw)
        require(math.isfinite(value) and value >= 0, 'Invalid timeCost field: ' + key)
        if key in counts and key != 'averageCollision':
            require(value.is_integer(), 'Nonintegral timeCost count: ' + key)
            value = int(value)
        values[key] = value
    return {'available': True, 'source_sha256': sha(path),
            'core_cuda_event_seconds': values.get('totalTime'),
            'cumulative_phase_event_seconds': {phase: values.get(f'time{i}') for i, phase in enumerate(PHASES)},
            'make_pd_event_seconds': values.get('time_makePD'),
            'native_counters': {key: values.get(key) for key in counts},
            'missing_fields': [key for key in seconds + counts if key not in values],
            'scope': 'Core CUDA-event accumulated seconds from native timeCost.txt. totalCgTime is a PCG iteration COUNT, not seconds. Printed precision differs from per-frame JSON.'}


def timing_summary(frames, csv_rows, result, timecost=None):
    require([int(r['frame']) for r in csv_rows] == list(range(1, len(frames) + 1)), 'Wrong timing frame sequence')
    values = [float(r['solver_ms']) for r in csv_rows]
    require(values and all(math.isfinite(v) and v > 0 for v in values), 'Invalid frame solver time')
    seconds = sum(values) / 1000
    recorded = result.get('solver_seconds')
    require(finite(recorded) and math.isclose(seconds, recorded, rel_tol=1e-10, abs_tol=1e-9), 'Runner/frame solver time mismatch')
    wall = result.get('wall_seconds')
    require(finite(wall) and wall > 0, 'Missing/invalid process wall time')
    phases, missing = {}, {}
    for key in PHASES:
        available = [f['phase_ms'][key] for f in frames if key in f.get('phase_ms', {})]
        require(all(finite(v) and v >= 0 for v in available), 'Invalid phase timing: ' + key)
        missing[key] = len(frames) - len(available)
        phases[key] = sum(available) if not missing[key] else None
    terminal_values = [f['ipc_exit_assembly_ms'] for f in frames if 'ipc_exit_assembly_ms' in f]
    require(all(finite(v) and v >= 0 for v in terminal_values), 'Invalid exit assembly timing')
    terminal = sum(terminal_values) if len(terminal_values) == len(frames) else None
    return {'solver_seconds': seconds, 'process_wall_seconds': wall,
            'core_cuda_event_seconds': timecost.get('core_cuda_event_seconds') if timecost else None,
            'native_timecost': timecost if timecost is not None else {'available': False, 'reason': 'Not supplied; not reconstructed.'},
            'wall_minus_solver_seconds': wall - seconds,
            'load_seconds': None, 'export_seconds': None,
            'load_export_reason': 'Not separately instrumented; wall minus solver also includes launch, logging, other host work and measurement differences.',
            'solver_time_scope': 'CPU steady_clock around IPC_Solver plus cudaDeviceSynchronize, from frames.csv; not a CUDA-event-only measure.',
            'phase_ms': phases, 'missing_phase_frames': missing, 'exit_assembly_ms': terminal,
            'phase_sum_interpretation': 'Diagnostic phase timers; pcg includes matrix build/convert, preconditioner preparation, PCG and distribution. Do not add nested CPU waits or call phase sums an exhaustive wall decomposition.',
            'baseline_export_capability_differs': True}


def material_summary(rows):
    require(bool(rows), 'No material frames')
    require(all(v is None or isinstance(v, bool) or finite(v) for r in rows for v in r.values()), 'Nonfinite material measurement')
    def minimum(key):
        values = [r[key] for r in rows if r[key] is not None]
        return min(values) if values else None
    return {'max_stretch': max(r['max_stretch'] for r in rows),
            'p99_stretch': max(r['p99_stretch'] for r in rows),
            'fixed_drift_m': max(r['fixed_drift_m'] for r in rows),
            'fem_min_J': minimum('fem_min_J'),
            'fem_nonpositive_peak': max(r['fem_nonpositive'] for r in rows),
            'fem_nonpositive_frame_sum': sum(r['fem_nonpositive'] for r in rows),
            'fem_negative_volume_peak': max(r['fem_negative_volume'] for r in rows),
            'abd_min_J': minimum('abd_min_J'),
            'abd_nonpositive_frames': sum(r['abd_min_J'] is not None and r['abd_min_J'] <= 0 for r in rows)}


def previous_comparisons(session, task, row):
    """Persist group differences while raw exports are still present; no rerun."""
    result = []
    if task['phase'] != 'paired':
        return result
    edges = (('original_stiff', 'combined_graph'), ('ipc_host', 'ipc_graph'), ('ipc_graph', 'combined_graph'))
    arm, pair, scene = task['arm'], task['pair'], task['scene_key']
    for left, right in edges:
        if arm not in (left, right):
            continue
        other_arm = right if arm == left else left
        other_name = f'p{pair}_{scene}_{other_arm}'
        other = session / other_name
        analysis_path = session / 'analysis' / (other_name + '.json')
        if not analysis_path.exists():
            continue  # Counterpart has not yet completed; later arm handles it.
        prior = read(analysis_path)
        if not prior.get('hard_checks_passed'):
            continue
        inputs = compare_inputs(row['input_identity'], prior['input_identity'])
        require(inputs['passed'], 'Paired physical inputs differ: ' + other_name)
        a, b = (session / task['name'], other) if arm == left else (other, session / task['name'])
        record = {'left': left, 'right': right, 'pair': pair, 'scene_key': scene,
                  'initial_inputs': inputs, 'position_velocity_scope': 'Cloth, FEM and ABD separately; no reconstructed velocity.'}
        sources = {side: {'run': folder.name, 'evidence_sha256': sha(folder / 'evidence.json')}
                   for side, folder in (('left', a), ('right', b))}
        cache = session / 'pair_analysis' / f'p{pair}_{scene}_{left}__{right}.json'
        if cache.exists():
            cached = read(cache)
            require(cached['input_evidence'] == sources and cached['initial_inputs'] == inputs, 'Pair cache identity mismatch')
        else:
            cached = dict(record, input_evidence=sources, schema='full_eval_pair_states.v1')
            def complete_raw(folder, summary):
                count = summary.get('positions', {}).get('expected_frames')
                if not isinstance(count, int) or count <= 0:
                    return False
                kinds = ['state'] + (['velocity'] if summary.get('actual_velocity', {}).get('available') else [])
                return all((folder / 'trace' / f'{kind}_{frame:04d}.bin').is_file()
                           for kind in kinds for frame in range(count))
            summaries = {task['name']: row, other_name: prior}
            if complete_raw(a, summaries[a.name]) and complete_raw(b, summaries[b.name]):
                # Validate the exact raw trajectory inputs against their immutable
                # per-run inventories before caching a comparison for archival.
                for folder in (a, b):
                    inventory = read(folder / 'evidence.json')['files']
                    for item in inventory:
                        name = item['path'].replace('\\', '/')
                        if name.startswith(('trace/state_', 'trace/velocity_')) or name in ('trace/topology.bin', 'trace/metadata.json'):
                            path = folder / name
                            require(path.is_file() and path.stat().st_size == item['bytes']
                                    and sha(path) == item['sha256'], 'Raw paired input differs from inventory: ' + name)
                cached['state_difference'] = state_comparison(a, b)
            else:
                cached['state_difference'] = {'available': False, 'reason': 'Complete paired raw export window unavailable/archived; no difference reconstructed from summaries.'}
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open('x', encoding='utf-8') as stream:
                json.dump(cached, stream, indent=2, allow_nan=False)
        record['state_difference'] = cached['state_difference']
        record['pair_cache_path'] = str(cache.relative_to(session))
        record['pair_cache_sha256'] = sha(cache)
        record['input_evidence'] = sources
        result.append(record)
    return result


def analyze_run(session, task, seal):
    session = Path(session).resolve()
    run = session / task['name']
    row = {k: task.get(k) for k in ('index', 'name', 'phase', 'repeat', 'pair', 'calibration_index',
                                   'verification_index', 'scene_key', 'arm', 'variant', 'binary')}
    row.update(hard_checks_passed=False, quality_status='not_evaluated', failures=[],
               quality_certified=False, performance_certified=False, analysis_source_sha256=sha(Path(__file__)))
    try:
        request, result = read(run / 'requested.json'), read(run / 'result.json')
        c = expand(task['config']);steps = c['steps']
        row['result_status'] = result['status']
        row['result_identity'] = digest(result)
        row['requested_identity'] = sha(run / 'requested.json')
        require(request['binary'] == task['binary'] and request['expanded_config'] == c, 'Requested binary/config differs from task')
        require(task.get('expanded_config_sha256', digest(c)) == digest(c), 'Task config digest differs')
        manifest = seal['base_manifest'] if task['binary'] == 'base' else seal['active_manifest']
        require(request['source_digest'] == manifest['source_digest'] and request['exe_sha256'] == manifest['exe_sha256'], 'Source/executable identity differs')
        row['program_identity'] = {key: request[key] for key in ('source_digest', 'exe_sha256', 'config_sha256')}
        require(request.get('from_zero') is True, 'Run was not from zero')
        require(result['status'] == 'completed' and result['recorded_frames'] == steps and result.get('exit_code') == 0, 'Incomplete, resource-stopped or nonzero-exit run')
        require(not result.get('heavy_diagnostics', False) and not c['diagnostics'] and c['profile'] == 'none', 'Heavy diagnostic performance arm')
        frames = read(run / 'output/stats.json')['frames']
        require(len(frames) == steps and frames, 'Wrong stats frame count')
        row['configuration'] = baseline_configuration(run, c, frames) if task['binary'] == 'base' else validate(run)
        require(row['configuration']['passed'], 'Native configuration evidence failed')
        row['positions'] = export_evidence(run, steps, 'state')
        require(row['positions']['passed'], 'Incomplete/nonfinite position exports')
        if task['binary'] == 'active':
            row['actual_velocity'] = export_evidence(run, steps, 'velocity')
            require(row['actual_velocity']['passed'], 'Incomplete/nonfinite actual velocity exports')
        else:
            row['actual_velocity'] = {'available': False, 'passed': None, 'reason': 'Frozen Stiff has no actual velocity export; no reconstruction.'}
        row['accepted_path_ccd'] = {'available': False, 'passed': None,
                                    'reason': 'Full safety CCD is retained, but these outputs do not provide independent accepted-path CCD certification.'}
        before = request.get('gpu_before', [])
        uuids = {r.get('uuid') for r in before}
        require(before and len(uuids) == 1 and None not in uuids, 'Missing or inconsistent actual GPU UUID')
        require(all(float(r['utilization.gpu']) <= 5 for r in before), 'Pre-run GPU load was not controlled')
        row['gpu_uuid'] = next(iter(uuids))
        row['gpu_name'] = before[-1].get('name')
        pcgs = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        require(pcgs and all(finite(p['iterations']) and p['iterations'] >= 0 and int(p['iterations']) == p['iterations'] for p in pcgs), 'Invalid PCG iteration records')
        require(not any(p.get('iteration_limit') or p.get('breakdown') for p in pcgs), 'Reported PCG limit/breakdown')
        require(not any(f.get('newton_exit') == 'iteration_limit' for f in frames), 'Newton iteration limit')
        require(all(finite(n['alpha']) for f in frames for n in f['newton'] if 'alpha' in n), 'Nonfinite accepted alpha')
        row['work'] = {'newton_records': sum(len(f['newton']) for f in frames),
                       'linear_directions': len(pcgs), 'pcg_iterations': sum(int(p['iterations']) for p in pcgs),
                       'reported_pcg_failures': 0, 'complete_breakdown_telemetry': all('breakdown' in p for p in pcgs),
                       'newton_exits': dict(collections.Counter(str(f.get('newton_exit', 'unreported')) for f in frames)),
                       'missing_newton_exit_frames': sum('newton_exit' not in f for f in frames)}
        with (run / 'trace/frames.csv').open() as stream:
            timing = list(csv.DictReader(stream))
        row['timing'] = timing_summary(frames, timing, result, read_timecost(run / 'timeCost.txt'))
        observed = metrics(run)
        require(observed['finite'] and not observed['pcg_failures'], 'Nonfinite state/PCG failure')
        row['material_by_frame'] = observed['frames']
        row['material'] = material_summary(observed['frames'])
        require(row['material']['abd_nonpositive_frames'] == 0, 'ABD inversion')
        # FEM inversions are compared to frozen baseline, not newly forbidden.
        row['material_scope'] = 'Whole-run worst frame: spatial p99 stretch maximum; min J and nonpositive/negative-volume summaries. Frame-summed nonpositive counts are not distinct flip events.'
        row['input_identity'] = input_identity(run)
        calibration_path = session / 'analysis' / f"calibration1_{task['scene_key']}_original_stiff.json"
        if calibration_path.exists() and task['name'] != calibration_path.stem:
            reference = read(calibration_path)
            row['calibration_input_comparison'] = compare_inputs(row['input_identity'], reference['input_identity'])
            require(row['calibration_input_comparison']['passed'], 'Physical inputs differ from calibration baseline')
        row['paired_comparisons'] = previous_comparisons(session, task, row)
        row['hard_checks_passed'] = True
        row['quality_status'] = 'material_comparison_pending_frozen_calibration; full_quality_unverified'
    except (ValueError, KeyError, FileNotFoundError, AssertionError, IndexError, TypeError) as error:
        row['failures'].append(type(error).__name__ + ': ' + str(error))
        row['quality_status'] = 'hard_failure; no_quality_or_speed_claim'
    return row


def freeze_bounds(calibration):
    require(len(calibration) == 3 and all(r['hard_checks_passed'] for r in calibration), 'Need three complete calibration baselines')
    require(sorted(r['calibration_index'] for r in calibration) == [1, 2, 3], 'Calibration identities differ')
    bounds = {}
    for key, direction in MATERIAL_DIRECTIONS.items():
        values = [r['material'][key] for r in calibration]
        require(all(v is None for v in values) or all(finite(v) for v in values), 'Inconsistent calibration metric: ' + key)
        bounds[key] = {'applicable': values[0] is not None, 'worse_direction': direction,
                       'min': min(values) if values[0] is not None else None,
                       'max': max(values) if values[0] is not None else None}
    frozen = {'bounds': bounds, 'comparison_absolute_tolerance': TOLERANCE,
              'source_runs': [r['name'] for r in sorted(calibration, key=lambda r: r['calibration_index'])],
              'source_analysis_sha256': [r.get('ledger_analysis_sha256') for r in sorted(calibration, key=lambda r: r['calibration_index'])],
              'scope': 'First three independent Stiff whole-run material summaries only; later baseline/candidate runs never widen bounds.'}
    frozen['sha256'] = digest(frozen)
    return frozen


def compare_material(material, frozen):
    comparisons = {}
    for key, bound in frozen['bounds'].items():
        actual = material.get(key)
        if not bound['applicable']:
            comparisons[key] = {'actual': actual, 'applicable': False, 'signed_worsening_gap': None,
                                'passed': actual is None, 'reason': 'No such elements in calibration; cannot invent a numerical bound.'}
            continue
        limit = bound['max'] if bound['worse_direction'] == 1 else bound['min']
        gap = bound['worse_direction'] * (actual - limit) if finite(actual) else None
        comparisons[key] = {'actual': actual, 'applicable': True, 'bound': limit,
                            'calibration_min': bound['min'], 'calibration_max': bound['max'],
                            'signed_worsening_gap': gap, 'tolerance': TOLERANCE,
                            'passed': gap is not None and gap <= TOLERANCE}
    return {'passed': all(r['passed'] for r in comparisons.values()), 'metrics': comparisons,
            'frozen_bounds_sha256': frozen['sha256'], 'quality_certified': False}


def paired_statistics(values, required, seed=20261006):
    require(all(finite(v) and v > 0 for v in values), 'Invalid paired speed ratio')
    output = {'pairs_observed': len(values), 'pairs_required': required, 'paired_ratios': values,
              'complete': len(values) == required,
              'paired_median': statistics.median(values) if values else None,
              'one_sided_95_percent_lower': None, 'bootstrap': None,
              'quality_certified': False, 'performance_certified': False}
    if len(values) == required:
        samples = np.asarray(values)
        rng = np.random.default_rng(seed)
        medians = np.median(samples[rng.integers(0, len(values), size=(20000, len(values)))], axis=1)
        output['one_sided_95_percent_lower'] = float(np.quantile(medians, .05, method='linear'))
        output['bootstrap'] = {'method': 'Paired percentile bootstrap of the median ratio',
                               'resamples': 20000, 'seed': seed, 'lower_quantile': .05,
                               'scope': 'Nominal sample-statistical bound; not a physical-quality or workload-equivalence certificate.'}
    output['statistical_2x_threshold_met'] = bool(output['complete'] and output['paired_median'] >= 2
        and output['one_sided_95_percent_lower'] >= 2)
    return output


def component_pairs(rows, left_arm, right_arm, required):
    records, missing = [], []
    for pair in range(1, required + 1):
        a = [r for r in rows if r['phase'] == 'paired' and r['pair'] == pair and r['arm'] == left_arm]
        b = [r for r in rows if r['phase'] == 'paired' and r['pair'] == pair and r['arm'] == right_arm]
        if len(a) != 1 or len(b) != 1 or not a[0]['hard_checks_passed'] or not b[0]['hard_checks_passed']:
            missing.append({'pair': pair, 'reason': 'Missing or hard-failed paired arm'})
            continue
        a, b = a[0], b[0]
        inputs = compare_inputs(a['input_identity'], b['input_identity'])
        if a.get('gpu_uuid') != b.get('gpu_uuid') or not a.get('gpu_uuid') or not inputs['passed']:
            missing.append({'pair': pair, 'reason': 'Actual same-GPU or physical-input pairing failed'})
            continue
        candidate_comparisons = a.get('paired_comparisons', []) + b.get('paired_comparisons', [])
        differences = [c for c in candidate_comparisons if c['left'] == left_arm and c['right'] == right_arm and c['pair'] == pair]
        records.append({'pair': pair, 'reference': a['name'], 'candidate': b['name'],
                        'gpu_uuid': a['gpu_uuid'], 'initial_inputs': inputs,
                        'solver_speedup': a['timing']['solver_seconds'] / b['timing']['solver_seconds'],
                        'process_wall_speedup': a['timing']['process_wall_seconds'] / b['timing']['process_wall_seconds'],
                        'reference_newton_directions': a['work']['linear_directions'],
                        'candidate_newton_directions': b['work']['linear_directions'],
                        'reference_pcg_iterations': a['work']['pcg_iterations'],
                        'candidate_pcg_iterations': b['work']['pcg_iterations'],
                        'state_difference': differences[-1].get('state_difference') if differences else
                            {'available': False, 'reason': 'No persisted raw-state pair comparison; not reconstructed after archival.'}})
    return {'reference_arm': left_arm, 'candidate_arm': right_arm, 'pairs': records, 'missing': missing,
            'solver': paired_statistics([r['solver_speedup'] for r in records], required),
            'process_wall': paired_statistics([r['process_wall_speedup'] for r in records], required)}


def analyze_session(session, seal, plan, entries):
    """Use retained per-run CPU reports; archived raw arrays are not reread.

    The runner validates each raw evidence inventory before writing its ledger.
    This summary validates the exact retained analysis hash and embedded payload;
    it does not pretend to revalidate files moved into durable raw archives.
    """
    session = Path(session).resolve()
    tasks = plan['tasks']
    require(len(entries) <= len(tasks), 'Ledger longer than finite plan')
    rows = []
    for index, entry in enumerate(entries):
        task = entry['task']
        require(task == tasks[index], 'Ledger is not the exact planned task prefix')
        path = Path(entry.get('analysis_path', session / 'analysis' / (task['name'] + '.json')))
        if not path.is_absolute():
            path = session / path
        require(path.resolve().is_relative_to(session.resolve()), 'Analysis path outside session')
        require(sha(path) == entry['analysis_sha256'], 'Retained analysis hash changed')
        report = read(path)
        if 'analysis' in entry:
            require(report == entry['analysis'], 'Ledger embedded analysis changed')
        require(report.get('name') == task['name'], 'Analysis task identity mismatch')
        if report.get('result_identity') is not None:
            require(report['result_identity'] == digest(entry['result']), 'Ledger result changed')
        row = dict(report)
        for key in ('phase', 'scene_key', 'arm', 'pair', 'calibration_index', 'verification_index'):
            if key in row or row.get('hard_checks_passed'):
                require(row.get(key) == task.get(key), 'Analysis task field differs: ' + key)
            else:
                row[key] = task.get(key)  # Explicit skipped/analysis-failed report.
        for comparison in row.get('paired_comparisons', []):
            if 'pair_cache_path' in comparison:
                cache = session / comparison['pair_cache_path']
                require(cache.resolve().is_relative_to(session.resolve())
                        and sha(cache) == comparison['pair_cache_sha256'], 'Persisted pair cache changed')
                cached = read(cache)
                require(cached['input_evidence'] == comparison['input_evidence']
                        and cached['state_difference'] == comparison['state_difference'], 'Embedded pair comparison changed')
        row['ledger_analysis_sha256'] = entry['analysis_sha256']
        row['recorded_raw_evidence_sha256'] = entry.get('evidence_sha256')
        row['ledger_status'] = entry.get('status', entry.get('result', {}).get('status'))
        rows.append(row)
    report = {'schema': 'autodl_full_evaluation_analysis.v1', 'analysis_sha256': sha(Path(__file__)),
              'plan_sha256': digest(plan), 'seal_sha256': digest(seal), 'runs': rows,
              'planned_tasks': len(tasks), 'recorded_tasks': len(rows),
              'ledger_complete': len(rows) == len(tasks), 'scenes': {}, 'long_run_gates': {},
              'quality_certified': False, 'performance_certified': False,
              'old_guard_unchanged': True,
              'original_quality_protocol_sha256': sha(ROOT / 'tools/bench/quality_protocol.json'),
              'raw_verification_scope': 'Per-run CPU reports/ledger hashes retained; original raw inventory was validated by runner. This summary does not require archived raw arrays to remain local.',
              'limitations': ['Frozen Stiff lacks actual velocity; no reconstructed substitute.',
                              'No independent full accepted-path CCD evidence.',
                              'Material comparisons do not prove trajectory/velocity equivalence.',
                              'Sample bootstrap bounds do not certify same-quality speedup.',
                              'Wall time includes unequal export capabilities; load/export are not individually timed.']}
    for scene in SCENES:
        selected = [r for r in rows if r['scene_key'] == scene]
        hundred = [r for r in selected if r['phase'] != 'stability']
        calibration = [r for r in hundred if r['phase'] == 'calibration']
        verification = [r for r in hundred if r['phase'] == 'verification']
        result = {'frozen_material_bounds': None, 'material_comparisons': {},
                  'baseline_verification_passed': False, 'material_comparison_passed': False,
                  'completed_hard_pass_100_runs': sum(r['hard_checks_passed'] for r in hundred),
                  'required_100_runs': 25, 'quality_certified': False}
        reasons = []
        if len(calibration) == 3 and all(r['hard_checks_passed'] for r in calibration):
            frozen = freeze_bounds(calibration)
            result['frozen_material_bounds'] = frozen
            for row in hundred:
                if row['hard_checks_passed']:
                    result['material_comparisons'][row['name']] = compare_material(row['material'], frozen)
            verified = len(verification) == 2 and all(r['hard_checks_passed'] and
                result['material_comparisons'][r['name']]['passed'] for r in verification)
            result['baseline_verification_passed'] = verified
            if not verified:
                reasons.append('Two held-out baseline verifications not complete/passing the frozen range; bounds are not widened.')
            result['material_comparison_passed'] = (len(hundred) == 25 and
                len(result['material_comparisons']) == 25 and verified and
                all(c['passed'] for c in result['material_comparisons'].values()))
        else:
            reasons.append('Three independent calibration baselines not complete/hard-passing.')
        if len(hundred) != 25 or not all(r['hard_checks_passed'] for r in hundred):
            reasons.append('All 25 planned 100-frame runs are not complete/hard-passing.')
        if not result['material_comparison_passed']:
            reasons.append('Not all full-run material comparisons pass the immutable calibration range.')
        result['main_original_to_combined'] = component_pairs(hundred, 'original_stiff', 'combined_graph', 7)
        result['graph_component'] = component_pairs(hundred, 'ipc_host', 'ipc_graph', 3)
        result['combined_execution_component'] = component_pairs(hundred, 'ipc_graph', 'combined_graph', 3)
        result['stability'] = [r for r in selected if r['phase'] == 'stability']
        report['scenes'][scene] = result
        report['long_run_gates'][scene] = {'eligible': not reasons, 'reasons': reasons,
                                           'scope': '300-frame stability permission only; not quality or 2x certification.'}
    report['all_100_hard_checks_passed'] = all(v['completed_hard_pass_100_runs'] == 25 for v in report['scenes'].values())
    return report
