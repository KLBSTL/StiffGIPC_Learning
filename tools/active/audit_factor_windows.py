"""Prepare bounded diagnostic runs, then audit their accepted paths on CPU.

This helper never launches the simulator. Use batch.py for the generated plan,
which retains the shared serial-GPU lock. Audit only after timing is finished.
"""
import argparse
import collections
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from config import ROOT, digest, expand, read, sha, matches_requested
from validate_run import validate


SCENES = {'hang': ('cloth_hang_l', 21),
          'fixed_bunny': ('cloth_fixed_bunny_l', 45),
          'mixed': ('bunny_cloth_bunny_l', 35)}
ARMS = ('stiff', 'factor_inverse')
PLAN = ROOT / 'configs/active/factor_window_audit.json'
BATCH_REPORT = 'reports/active/FACTOR_WINDOW_AUDIT_BATCH.json'
CPU_TIMEOUT_SECONDS = 300
FIXED_DIAGNOSTIC_TOLERANCE_M = 1e-12


def require(condition, description):
    if not condition:
        raise ValueError(description)


def prepare(source):
    if PLAN.exists():
        raise FileExistsError(PLAN)
    timing = read(source)
    tasks = []
    for key, (scene, steps) in SCENES.items():
        for variant in ARMS:
            matches = [row for row in timing['runs'] if row['scene_key'] == key
                       and row['variant'] == variant and row['repeat'] == 1]
            require(len(matches) == 1, f'Expected exactly one source task for {key}/{variant}')
            original = matches[0]
            config = read(ROOT / original['config']) if isinstance(original['config'], str) else dict(original['config'])
            expanded = expand(config)
            require(expanded['scene'] == scene and expanded['steps'] == steps, f'Source window mismatch: {key}')
            require(expanded['diagnostics'] == [] and expanded['profile'] == 'none', 'Timing source has diagnostics')
            require(original['binary'] == ('base' if variant == 'stiff' else 'active'), 'Unexpected binary arm')
            if variant == 'factor_inverse':
                require(expanded['mas_factor_action'] == 'factor_inverse'
                        and expanded['mas_restrict'] == 'warp', 'Candidate action mismatch')
            config.update(diagnostics=['substeps'], steps=steps, timeout_seconds=120,
                          trace_velocity=False, profile='none', fixed_factor_study=False,
                          fixed_restrict_study=False)
            tasks.append({'name': f'factor_window_{key}_{variant}_audit',
                          'arm': key + '_' + variant, 'scene_key': key, 'variant': variant,
                          'repeat': 'audit', 'binary': original['binary'], 'config': config,
                          'source_timing_task': original['name']})
    payload = {'report': BATCH_REPORT, 'stop_on_failure': False, 'runs': tasks,
               'source_plan': str(source), 'source_plan_sha256': sha(source),
               'scope': 'Six separate accepted-path diagnostics; solver timings are excluded from performance comparisons.',
               'gpu_timeout_seconds_per_run': 120, 'cpu_validator_timeout_seconds_per_run': CPU_TIMEOUT_SECONDS}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({'plan': str(PLAN), 'runs': len(tasks), 'gpu_launched': False,
                      'next': 'python tools/active/batch.py --plan configs/active/factor_window_audit.json'}))


def verify_request(task, run):
    request = read(run / 'requested.json')
    config = request['expanded_config']
    require(matches_requested(config,task['config']), f'Request differs from audit plan: {run.name}')
    require(request['config_sha256'] == digest(config), f'Config digest mismatch: {run.name}')
    key = task['scene_key']
    scene_name, steps = SCENES[key]
    expected = {'scene': scene_name, 'steps': steps, 'timeout_seconds': 120, 'dt': .01,
                'ipc_newton_tol': .01, 'pcg_rho_tol': 1e-4, 'toi_remaining_fraction_tol': .01,
                'toi_trial_velocity_tol': .05, 'diagnostics': ['substeps'], 'profile': 'none',
                'trace_velocity': False, 'fixed_factor_study': False, 'fixed_restrict_study': False}
    for field, value in expected.items():
        require(config[field] == value, f'Unexpected audit {field}: {run.name}')
    expected_binary = 'base' if task['variant'] == 'stiff' else 'active'
    require(request['binary'] == task['binary'] == expected_binary, f'Wrong binary: {run.name}')
    require(sha(run / 'build_manifest.json') == request['manifest_sha256'], f'Manifest identity mismatch: {run.name}')
    manifest = read(run / 'build_manifest.json')
    require(manifest['source_digest'] == request['source_digest'], f'Source digest mismatch: {run.name}')
    require(request['exe_sha256'] in {v['sha256'] for v in manifest['binaries'].values()},
            f'Executable identity mismatch: {run.name}')
    scene = read(run / 'output/scene.json')
    require(scene['case_id'] == scene_name, f'Wrong effective scene: {run.name}')
    for field, value in [('dt', .01), ('newton_tol', .01), ('pcg_tol', 1e-4)]:
        require(scene['effective_run'][field] == value, f'Wrong effective {field}: {run.name}')
    if expected_binary == 'active':
        checked = validate(run)
        require(checked['passed'], f'Resolved configuration/execution failed: {run.name}')
        resolved = read(run / 'resolved_config.json')
        require(resolved['mas'].get('factor_action') == 'factor_inverse', 'Factor candidate was not confirmed')
        require(resolved['mas'].get('restriction_mode') == 'warp', 'Warp restriction was not confirmed')
    else:
        checked = {'passed': True, 'resolved_contract_available': False,
                   'scope': 'Frozen Stiff request, saved build identity, and effective scene values checked.'}
    return request, checked, scene


def check_coverage_and_states(task, run, scene):
    trace = run / 'trace'
    expected_frames = SCENES[task['scene_key']][1]
    metadata = read(trace / 'metadata.json')
    # The existing validator explicitly uses y+1. Do not apply it silently to
    # scenes whose plane differs from this documented executable contract.
    require(metadata['ground_normal'] == [0, 1, 0] and metadata['ground_offset'] == -1,
            f'CPU validator plane contract does not match: {run.name}')
    topology = np.fromfile(trace / 'topology.bin', dtype='<u4')
    nv, nf, nt = map(int, topology[:3])
    require(topology.size == 3 + 3 * nf + 4 * nt, f'Invalid topology: {run.name}')
    tets = topology[3 + 3 * nf:].reshape(-1, 4)
    initial_bytes = (trace / 'state_0000.bin').read_bytes()
    require(len(initial_bytes) == nv * 3 * 8, f'Invalid initial state: {run.name}')
    initial = np.frombuffer(initial_bytes, dtype='<f8').reshape(nv, 3)
    require(np.isfinite(initial).all(), f'Nonfinite initial state: {run.name}')
    fixed_boundary = np.fromfile(trace / 'boundary_types.bin', dtype='<i4') == 1
    require(len(fixed_boundary) == nv, f'Invalid boundary state: {run.name}')
    fixed_abd = np.zeros(nv, dtype=bool)
    objects_3d = [obj for obj in scene['objects'] if obj['dimension'] == 3]
    if objects_3d and all(obj['body_type'] == 'ABD' and obj.get('fixed_mode') == 'all' for obj in objects_3d):
        fixed_abd[np.unique(tets)] = True
        require(int(fixed_abd.sum()) == metadata['abd_point_num'], f'Incomplete fixed ABD map: {run.name}')
    if task['scene_key'] == 'fixed_bunny':
        require(int(fixed_abd.sum()) == 19193, 'Fixed bunny requires all 19193 ABD vertices checked')
    if task['scene_key'] == 'hang':
        require(int(fixed_boundary.sum()) == 2, 'Hanging cloth requires two fixed vertices')
    fixed = fixed_boundary | fixed_abd
    groups = collections.defaultdict(list)
    for path in sorted((trace / 'substeps').glob('safe_*.bin')):
        match = re.fullmatch(r'safe_(\d{4})_(\d{4})\.bin', path.name)
        require(match is not None, f'Unexpected safe-state filename: {path}')
        groups[int(match[1])].append((int(match[2]), path))
    require(sorted(groups) == list(range(expected_frames)), f'Incomplete physical-frame path coverage: {run.name}')
    with (trace / 'frames.csv').open() as stream:
        require(len(list(csv.DictReader(stream))) == expected_frames, f'Incomplete frame timing coverage: {run.name}')
    frame_counts = []
    max_fixed = max_fixed_abd = 0.
    state_count = 0
    previous_endpoint = None
    for frame, members in sorted(groups.items()):
        require([index for index, _ in members] == list(range(len(members))), f'Nonconsecutive safe indices: {run.name}, {frame}')
        require(len(members) >= 2, f'No exported accepted segment: {run.name}, {frame}')
        start_bytes = (trace / f'state_{frame:04d}.bin').read_bytes()
        end_bytes = (trace / f'state_{frame + 1:04d}.bin').read_bytes()
        require(members[0][1].read_bytes() == start_bytes, f'Safe start mismatch: {run.name}, {frame}')
        require(members[-1][1].read_bytes() == end_bytes, f'Safe endpoint mismatch: {run.name}, {frame}')
        if previous_endpoint is not None:
            require(previous_endpoint == start_bytes, f'Nonstationary cross-frame bridge: {run.name}, {frame}')
        previous_endpoint = end_bytes
        for _, path in members:
            data = np.fromfile(path, dtype='<f8')
            require(data.size == nv * 3 and np.isfinite(data).all(), f'Invalid/nonfinite accepted state: {path}')
            positions = data.reshape(nv, 3)
            if fixed.any():
                max_fixed = max(max_fixed, float(np.linalg.norm(positions[fixed] - initial[fixed], axis=1).max()))
            if fixed_abd.any():
                max_fixed_abd = max(max_fixed_abd, float(np.linalg.norm(positions[fixed_abd] - initial[fixed_abd], axis=1).max()))
            state_count += 1
        frame_counts.append(len(members) - 1)
    return {'coverage_passed': True, 'physical_frames': expected_frames,
            'accepted_segments': sum(frame_counts), 'stationary_bridges': expected_frames - 1,
            'expected_validator_paths': sum(frame_counts) + expected_frames - 1,
            'segments_per_frame': frame_counts, 'accepted_states_checked': state_count,
            'all_accepted_states_finite': True, 'boundary_fixed_vertices': int(fixed_boundary.sum()),
            'fixed_abd_vertices': int(fixed_abd.sum()), 'fixed_vertices': int(fixed.sum()),
            'max_fixed_drift_m': max_fixed, 'max_fixed_abd_drift_m': max_fixed_abd,
            'fixed_diagnostic_tolerance_m': FIXED_DIAGNOSTIC_TOLERANCE_M,
            'fixed_diagnostic_passed': max_fixed <= FIXED_DIAGNOSTIC_TOLERANCE_M,
            'fixed_scope': 'Boundary fixed vertices plus scene-declared all-fixed ABD geometry; zero count means no mapped fixed vertices. '
                           'The inherited 1e-12 m diagnostic bound is not a replacement for same-round Stiff repeat-envelope quality.'}


def verify(matrix_path, output):
    if output.exists():
        raise FileExistsError(output)
    require(not (ROOT / 'runs/active/.gpu.lock').exists(), 'GPU run is active; defer CPU audit until timing ends')
    matrix = read(matrix_path)
    exe = ROOT / 'builds/validator/Release/diagnose_first_path.exe'
    report = {'schema_version': 1, 'matrix': str(matrix_path), 'matrix_sha256': sha(matrix_path),
              'validator': str(exe), 'validator_sha256': sha(exe), 'analyzer_sha256': sha(Path(__file__)),
              'scope': 'Six separate from-zero accepted-path diagnostic trajectories; not the timing trajectories.',
              'physical_certified': False, 'performance_certified': False,
              'velocity_analyzed_or_reconstructed': False, 'cpu_timeout_seconds_per_run': CPU_TIMEOUT_SECONDS,
              'runs': []}
    expected = {(key, arm) for key in SCENES for arm in ARMS}
    identities = [(row['scene_key'], row['variant']) for row in matrix['runs']]
    require(len(set(identities)) == len(identities) and set(identities) <= expected, 'Unexpected/duplicate audit arms')
    report['missing_runs'] = [dict(zip(('scene_key', 'variant'), entry)) for entry in sorted(expected - set(identities))]
    scene_references = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep incremental failure evidence without ever replacing an existing report.
    with output.open('x', encoding='utf-8') as progress:
        def save():
            progress.seek(0)
            json.dump(report, progress, indent=2, allow_nan=False)
            progress.truncate()
            progress.flush()

        save()
        for task in matrix['runs']:
            name = task['name']
            require(name == Path(name).name and '/' not in name and '\\' not in name, 'Invalid run directory name')
            run = ROOT / 'runs/active' / name
            record = {'run': name, 'scene_key': task['scene_key'], 'variant': task['variant'],
                      'coverage_passed': False, 'audit_passed': False}
            report['runs'].append(record)
            try:
                result = read(run / 'result.json')
                record['result'] = {key: value for key, value in result.items() if key != 'gpu_samples'}
                require(result == task['result'], f'Batch/result mismatch: {name}')
                require(result['status'] == 'completed'
                        and result['recorded_frames'] == SCENES[task['scene_key']][1]
                        and result.get('finite') is True, f'Incomplete/nonfinite diagnostic: {name}')
                request, configuration, scene = verify_request(task, run)
                record.update(configuration=configuration, requested_sha256=sha(run / 'requested.json'),
                              exe_sha256=request['exe_sha256'])
                identity = {filename: sha(run / 'trace' / filename) for filename in
                            ('topology.bin', 'state_0000.bin', 'masses.bin', 'boundary_types.bin', 'body_ids.bin', 'metadata.json')}
                reference = scene_references.setdefault(task['scene_key'], (identity, scene))
                require(reference == (identity, scene), f'Changed scene/initial state within audit pair: {name}')
                record['initial_identity_sha256'] = identity
                record.update(check_coverage_and_states(task, run, scene))
                target = output.parent / f'CCD_{name}.json'
                log_path = target.with_suffix('.log')
                require(not target.exists() and not log_path.exists(), f'Preserve existing CCD evidence: {target}')
                command = [str(exe), str(run / 'trace'), str(target), 'substeps', '--stable-nh1']
                record['command'] = command
                record['validator_report'] = str(target)
                record['validator_log'] = str(log_path)
                with log_path.open('x', encoding='utf-8') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=CPU_TIMEOUT_SECONDS, check=False)
                record['validator_exit_code'] = process.returncode
                require(target.exists(), f'Validator did not write a report: {name}')
                ccd = read(target)
                record['ccd'] = ccd
                require(ccd['scope'] == 'accepted substeps', f'Wrong validator path scope: {name}')
                require(ccd['paths_checked'] == record['expected_validator_paths'], f'Validator path-count mismatch: {name}')
                record['validator_coverage_passed'] = True
                record['audit_passed'] = bool(process.returncode == 0 and ccd['passed']
                    and ccd['finite'] and ccd['conservative_collision_flags'] == 0
                    and ccd['abd_tet_inversions'] == 0 and record['fixed_diagnostic_passed'])
            except subprocess.TimeoutExpired:
                record['failure'] = f'CPU CCD audit exceeded predeclared {CPU_TIMEOUT_SECONDS} seconds'
            except (ValueError, KeyError, OSError) as error:
                record['failure'] = str(error)
            save()
            print(json.dumps({key: record[key] for key in ('run', 'coverage_passed', 'accepted_segments',
                              'expected_validator_paths', 'max_fixed_drift_m', 'audit_passed', 'failure') if key in record}), flush=True)
        report['all_six_audits_passed'] = not report['missing_runs'] and len(report['runs']) == 6 \
            and all(row['audit_passed'] for row in report['runs'])
        report['actual_validator_collision_flags'] = sum(row['ccd']['conservative_collision_flags']
                                                         for row in report['runs'] if 'ccd' in row)
        report['actual_validator_paths_checked'] = sum(row['ccd']['paths_checked']
                                                      for row in report['runs'] if 'ccd' in row)
        report['collision_flag_totals_cover_all_six_runs'] = len([row for row in report['runs'] if 'ccd' in row]) == 6
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--source-plan', default='configs/active/factor_windows.json')
    parser.add_argument('--matrix', default=BATCH_REPORT)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.prepare:
        prepare(ROOT / args.source_plan)
    else:
        if not args.output:
            parser.error('--output is required for CPU verification')
        report = verify(ROOT / args.matrix, ROOT / args.output)
        raise SystemExit(int(not report['all_six_audits_passed']))
