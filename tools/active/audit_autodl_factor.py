"""Bounded, serial CPU Tight-Inclusion audits of downloaded AutoDL windows.

No SSH, build, simulator, or GPU invocation. --self-test exercises only synthetic
trace inventory fixtures. A new output directory owns all per-run evidence.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from analyze_autodl_factor import child, prepare_tasks, validate_run, verify_download_provenance
from config import ROOT, digest, read, sha


DEFAULT_ROOT = ROOT / 'downloads/autodl_factor_20261004_4090'
FIXED_DIAGNOSTIC_BOUND_M = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Trace inventory exceeded its predeclared CPU budget')


def discover_validator(explicit=None):
    candidates = [explicit] if explicit else [
        ROOT / 'builds/validator/Release/diagnose_first_path.exe',
        ROOT / 'builds/validator/diagnose_first_path']
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path.is_file():
            return path
    raise FileNotFoundError('No existing independent CPU diagnose_first_path validator: '
                            + ', '.join(map(str, candidates)))


def inspect_trace(run, expected_frames, seconds=60):
    """Read every available accepted state; retain incomplete-coverage findings."""
    deadline = time.monotonic() + seconds
    trace = run / 'trace'
    scene = read(run / 'output/scene.json')
    metadata = read(trace / 'metadata.json')
    require(metadata['ground_normal'] == [0, 1, 0] and metadata['ground_offset'] == -1,
            'Existing CPU validator hardcodes y+1; exported plane must be y=-1')
    topology = np.fromfile(trace / 'topology.bin', dtype='<u4')
    require(len(topology) >= 3, 'Missing topology counts')
    nv, nf, nt = map(int, topology[:3])
    require(nv > 0 and topology.size == 3 + 3 * nf + 4 * nt, 'Malformed topology')
    require(not topology[3:].size or int(topology[3:].max()) < nv, 'Invalid topology vertex index')
    tets = topology[3 + nf * 3:].reshape(-1, 4)
    initial_bytes = (trace / 'state_0000.bin').read_bytes()
    require(len(initial_bytes) == nv * 24, 'Malformed initial state')
    initial = np.frombuffer(initial_bytes, dtype='<f8').reshape(nv, 3)
    require(np.isfinite(initial).all(), 'Nonfinite initial state')
    boundary = np.fromfile(trace / 'boundary_types.bin', dtype='<i4')
    body_ids = np.fromfile(trace / 'body_ids.bin', dtype='<i4')
    require(len(boundary) == len(body_ids) == nv, 'Incomplete fixed/body identity arrays')
    abd = int(metadata['abd_point_num'])
    require(0 <= abd <= nv, 'Invalid ABD vertex prefix')
    fixed_boundary = boundary == 1
    fixed_abd = np.zeros(nv, dtype=bool)
    objects = [obj for obj in scene['objects'] if obj['dimension'] == 3]
    if objects and all(obj['body_type'] == 'ABD' and obj.get('fixed_mode') == 'all' for obj in objects):
        fixed_abd[np.unique(tets)] = True
        require(int(fixed_abd.sum()) == abd and not fixed_abd[abd:].any(), 'Incomplete all-fixed ABD geometry mapping')
    fixed = fixed_boundary | fixed_abd
    files = sorted(path for path in (trace / 'substeps').iterdir() if path.name.startswith('safe_'))
    require(len(files) >= 2, 'At least two accepted states are required')
    groups = collections.defaultdict(list)
    issues, states = [], []
    finite = True
    max_fixed = max_fixed_abd = 0.
    for path in files:
        check_deadline(deadline)
        match = re.fullmatch(r'safe_(\d{4})_(\d{4})\.bin', path.name)
        require(path.is_file() and match is not None, 'Unexpected file read by native validator: ' + path.name)
        frame, index = int(match[1]), int(match[2])
        data = path.read_bytes()
        require(len(data) == nv * 24, 'Malformed accepted state: ' + path.name)
        positions = np.frombuffer(data, dtype='<f8').reshape(nv, 3)
        state_finite = bool(np.isfinite(positions).all())
        finite &= state_finite
        if state_finite:
            if fixed.any():
                max_fixed = max(max_fixed, float(np.linalg.norm(positions[fixed] - initial[fixed], axis=1).max()))
            if fixed_abd.any():
                max_fixed_abd = max(max_fixed_abd, float(np.linalg.norm(positions[fixed_abd] - initial[fixed_abd], axis=1).max()))
        else:
            issues.append('Nonfinite accepted state: ' + path.name)
        row = {'path': path.relative_to(run).as_posix(), 'name': path.name, 'exported_frame': frame,
               'safe_index': index, 'sha256': hashlib.sha256(data).hexdigest(),
               'bytes': len(data), 'finite': state_finite}
        states.append(row)
        groups[frame].append(row)
    if sorted(groups) != list(range(expected_frames)):
        issues.append('Exported frame groups differ from the full planned physical window')
    frame_rows, bridge_rows = [], []
    previous = None
    for frame, entries in sorted(groups.items()):
        check_deadline(deadline)
        if [row['safe_index'] for row in entries] != list(range(len(entries))) or len(entries) < 2:
            issues.append('Nonconsecutive/empty accepted-segment indices at exported frame ' + str(frame))
        endpoint = {'physical_frame': frame + 1, 'exported_frame': frame,
                    'accepted_segments': max(0, len(entries) - 1),
                    'first_safe': entries[0]['path'], 'last_safe': entries[-1]['path'],
                    'first_safe_sha256': entries[0]['sha256'], 'last_safe_sha256': entries[-1]['sha256']}
        for label, state_frame, safe in [('start', frame, entries[0]), ('end', frame + 1, entries[-1])]:
            path = trace / f'state_{state_frame:04d}.bin'
            frame_hash = sha(path) if path.is_file() else None
            matches = frame_hash is not None and frame_hash == safe['sha256']
            endpoint[label + '_physical_state_sha256'] = frame_hash
            endpoint[label + '_matches_physical_state'] = matches
            if not matches:
                issues.append(f'Accepted {label} differs from physical state at exported frame {frame}')
        if previous is not None:
            stationary = previous['sha256'] == entries[0]['sha256']
            bridge_rows.append({'from': previous['path'], 'to': entries[0]['path'], 'stationary': stationary,
                                'from_sha256': previous['sha256'], 'to_sha256': entries[0]['sha256']})
            if not stationary:
                issues.append('Nonstationary bridge between exported physical frames')
        previous = entries[-1]
        frame_rows.append(endpoint)
    files_identity = {filename: sha(trace / filename) for filename in
                      ('topology.bin', 'state_0000.bin', 'masses.bin', 'boundary_types.bin', 'body_ids.bin', 'metadata.json')}
    files_identity['scene.json'] = sha(run / 'output/scene.json')
    accepted = sum(row['accepted_segments'] for row in frame_rows)
    require(accepted + len(bridge_rows) == len(states) - 1, 'Internal path inventory count mismatch')
    return {'trace_path': str(trace), 'planned_physical_frames': expected_frames,
            'exported_physical_frames': len(groups), 'vertices': nv, 'faces': nf, 'tets': nt,
            'coverage_passed': not issues, 'issues': issues, 'all_accepted_states_finite': finite,
            'accepted_states_checked': len(states), 'accepted_segments': accepted,
            'bridge_count': len(bridge_rows), 'stationary_bridges': sum(row['stationary'] for row in bridge_rows),
            'expected_validator_paths': len(states) - 1, 'states': states, 'frame_endpoints': frame_rows,
            'bridges': bridge_rows, 'initial_identity_sha256': files_identity,
            'boundary_fixed_vertices': int(fixed_boundary.sum()), 'fixed_abd_vertices': int(fixed_abd.sum()),
            'fixed_vertices': int(fixed.sum()), 'max_fixed_drift_m': max_fixed if finite else None,
            'max_fixed_abd_drift_m': max_fixed_abd if finite else None,
            'fixed_diagnostic_bound_m': FIXED_DIAGNOSTIC_BOUND_M,
            'fixed_diagnostic_passed': finite and max_fixed <= FIXED_DIAGNOSTIC_BOUND_M,
            'fixed_scope': 'Boundary fixed plus scene-declared all-fixed ABD vertices. Inherited 1e-12 m '
                           'roundoff diagnostic does not replace the Stiff repeat-envelope quality test.'}


def audit(root, plan_path, matrix_path, output, validator, timeout, inventory_timeout):
    require(not output.exists(), 'A fresh output directory is required')
    require(1 <= timeout <= 300 and 1 <= inventory_timeout <= 120, 'Finite CPU budgets required (CCD<=300s, inventory<=120s)')
    plan, matrix, bundle = read(plan_path), read(matrix_path), read(root / 'autodl_bundle.json')
    tasks = prepare_tasks(root, plan)
    require(matrix['plan_sha256'] == sha(plan_path) and bundle['plan_digests']['window'] == digest(plan),
            'Window plan/batch/bundle mismatch')
    require(digest(bundle['files']) == bundle['source_digest'] == matrix['source_digest']
            and digest(bundle['candidate_config']) == bundle['candidate_config_sha256']
            == matrix['candidate_sha256'] == plan['candidate_sha256'], 'Candidate/source identity mismatch')
    matrix_rows = {row['name']: row for row in matrix['runs']}
    require(len(matrix_rows) == len(matrix['runs']) and set(matrix_rows) <= {task['name'] for task in tasks},
            'Duplicate or unknown matrix entries')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'download_root': str(root), 'plan_sha256': sha(plan_path),
              'matrix_sha256': sha(matrix_path), 'bundle_sha256': sha(root / 'autodl_bundle.json'),
              'wrapper_sha256': sha(Path(__file__)), 'validator': str(validator),
              'validator_sha256': sha(validator), 'validator_source_sha256': sha(ROOT / 'tools/validator/diagnose_first_path.cpp'),
              'validator_source_scope': 'Current local source hash is recorded separately; executable hash identifies the program actually run.',
              'planned_runs': len(tasks), 'recorded_runs': len(matrix['runs']), 'cpu_runs_parallel': False,
              'ccd_timeout_seconds_per_run': timeout, 'inventory_timeout_seconds_per_run': inventory_timeout,
              'max_declared_cpu_seconds': len(tasks) * (timeout + inventory_timeout),
              'budget_scope': 'Sum of per-run native CCD and trace inventory budgets; small manifest/configuration I/O is separate.',
              'physical_certified': False, 'performance_certified': False,
              'material_policy': 'Existing --stable-nh1 policy: FEM inversions are reported, not prohibited. '
                                 'Independent CCD, ABD orientation and ground checks still apply.',
              'scope': 'All available accepted paths in 54 downloaded continuous window traces, including stationary '
                       'cross-frame bridges. Full-coverage and configuration failures remain separate from CCD observations.',
              'runs': []}
    report['download_provenance'] = verify_download_provenance(root, plan, matrix, bundle, plan_path)
    references = {}
    with (output / 'audit.json').open('x', encoding='utf-8') as progress:
        def save():
            progress.seek(0)
            json.dump(report, progress, indent=2, allow_nan=False)
            progress.truncate()
            progress.flush()
        save()
        for task in tasks:
            name = task['name']
            record = {'run': name, 'scene_key': task['scene_key'], 'role': task['role'], 'repeat': task['repeat'],
                      'configuration_passed': False, 'coverage_passed': False, 'validator_invoked': False,
                      'validator_coverage_passed': False, 'accepted_path_audit_passed': False, 'failures': []}
            report['runs'].append(record)
            log_path, ccd_path = output / (name + '.stdout.log'), output / (name + '.ccd.json')
            identity_path, record_path = output / (name + '.identity.json'), output / (name + '.record.json')
            record.update(stdout=str(log_path), native_ccd_json=str(ccd_path), identity_json=str(identity_path))
            save()
            run = child(root, 'runs/active/' + name)
            with log_path.open('x', encoding='utf-8') as log:
                log.write('Offline CPU accepted-path audit. No stop-first, no simulator or GPU invocation.\n')
                log.flush()
                try:
                    require(name in matrix_rows, 'Run absent from batch')
                    record['result_status'] = matrix_rows[name]['result'].get('status')
                    try:
                        record['configuration'] = validate_run(task, matrix_rows[name], root, bundle, matrix,
                                                               report['download_provenance'])
                        record['configuration_passed'] = True
                    except (ValueError, KeyError, OSError, StopIteration, AssertionError) as error:
                        record['failures'].append('Configuration/execution: ' + str(error))
                    inventory = inspect_trace(run, task['expanded_config']['steps'], inventory_timeout)
                    write_new(identity_path, inventory)
                    record['identity_sha256'] = sha(identity_path)
                    for field in ('coverage_passed', 'all_accepted_states_finite', 'accepted_states_checked',
                                  'accepted_segments', 'bridge_count', 'stationary_bridges', 'expected_validator_paths',
                                  'fixed_vertices', 'fixed_abd_vertices', 'max_fixed_drift_m',
                                  'max_fixed_abd_drift_m', 'fixed_diagnostic_passed'):
                        record[field] = inventory[field]
                    record['failures'].extend(inventory['issues'])
                    identity = inventory['initial_identity_sha256']
                    reference = references.setdefault(task['scene_key'], identity)
                    record['same_scene_initial_identity'] = reference == identity
                    if reference != identity:
                        record['failures'].append('Scene/initial identity differs within same-scene arms')
                    if not inventory['fixed_diagnostic_passed']:
                        record['failures'].append('Fixed geometry exceeds inherited roundoff diagnostic')
                    require(inventory['all_accepted_states_finite'], 'CPU CCD not invoked on known nonfinite accepted states')
                    command = [str(validator), str(run / 'trace'), str(ccd_path), 'substeps', '--stable-nh1']
                    record.update(command=command, validator_invoked=True)
                    log.write('ARGV ' + json.dumps(command) + '\n')
                    log.flush()
                    started = time.monotonic()
                    try:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                        record['validator_exit_code'] = process.returncode
                    except subprocess.TimeoutExpired:
                        record['validator_timed_out'] = True
                        record['failures'].append(f'Native CPU validator exceeded predeclared {timeout}s; process terminated')
                    finally:
                        record['validator_wall_seconds'] = time.monotonic() - started
                    if ccd_path.exists():
                        ccd = read(ccd_path)
                        record.update(ccd=ccd, native_ccd_sha256=sha(ccd_path))
                        record['validator_coverage_passed'] = ccd.get('scope') == 'accepted substeps' \
                            and ccd.get('paths_checked') == inventory['expected_validator_paths'] \
                            and ccd.get('stopped_after_first_collision') is False
                        if not record['validator_coverage_passed']:
                            record['failures'].append('Native validator did not cover every inventoried accepted path/bridge')
                        record['ccd_observation_passed'] = bool(record.get('validator_exit_code') == 0
                            and ccd.get('passed') is True and ccd.get('finite') is True
                            and ccd.get('conservative_collision_flags') == 0 and ccd.get('abd_tet_inversions') == 0)
                        if not record['ccd_observation_passed']:
                            record['failures'].append('Native independent CCD/finite/ABD/ground observation failed')
                    else:
                        record['failures'].append('Native CPU validator did not produce JSON; stdout and wrapper record retained')
                    record['accepted_path_audit_passed'] = bool(not record['failures'] and record['configuration_passed']
                        and record['coverage_passed'] and record['validator_coverage_passed']
                        and record.get('ccd_observation_passed'))
                except (ValueError, KeyError, OSError, StopIteration, AssertionError, TimeoutError) as error:
                    record['failures'].append(str(error))
                log.write('\nWRAPPER_STATUS ' + json.dumps({key: record[key] for key in
                    ('accepted_path_audit_passed', 'validator_invoked', 'failures')}) + '\n')
            if not identity_path.exists():
                write_new(identity_path, {'available': False, 'run': name, 'failures': record['failures']})
            record['stdout_sha256'] = sha(log_path)
            write_new(record_path, record)
            save()
            print(json.dumps({key: record[key] for key in ('run', 'accepted_path_audit_passed', 'validator_invoked', 'failures')})
                  , flush=True)
        report['all_54_accepted_path_audits_passed'] = len(report['runs']) == 54 \
            and all(row['accepted_path_audit_passed'] for row in report['runs'])
        report['actual_validator_reports'] = sum('ccd' in row for row in report['runs'])
        report['actual_validator_paths_checked'] = sum(row['ccd'].get('paths_checked', 0) for row in report['runs'] if 'ccd' in row)
        report['actual_collision_flags_from_available_reports'] = sum(row['ccd'].get('conservative_collision_flags', 0)
                                                                     for row in report['runs'] if 'ccd' in row)
        report['collision_flag_total_covers_all_54_full_windows'] = len(report['runs']) == 54 \
            and all(row['coverage_passed'] and row['validator_coverage_passed'] for row in report['runs'])
        save()
    return report


def self_test():
    with tempfile.TemporaryDirectory(prefix='autodl_cpu_audit_inventory_') as temporary:
        run = Path(temporary)
        trace = run / 'trace'
        (trace / 'substeps').mkdir(parents=True)
        (run / 'output').mkdir()
        np.array([3, 1, 0, 0, 1, 2], dtype='<u4').tofile(trace / 'topology.bin')
        np.array([1, 0, 0], dtype='<i4').tofile(trace / 'boundary_types.bin')
        np.zeros(3, dtype='<i4').tofile(trace / 'body_ids.bin')
        np.ones(3, dtype='<f8').tofile(trace / 'masses.bin')
        write_new(trace / 'metadata.json', {'abd_point_num': 0, 'ground_normal': [0, 1, 0], 'ground_offset': -1})
        write_new(run / 'output/scene.json', {'objects': [{'dimension': 2, 'body_type': 'FEM'}]})
        x0 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        x1, x2 = x0.copy(), x0.copy()
        x1[1:, 1] += .1
        x2[1:, 1] += .2
        for frame, positions in enumerate((x0, x1, x2)):
            positions.tofile(trace / f'state_{frame:04d}.bin')
        for frame, values in [(0, (x0, (x0 + x1) / 2, x1)), (1, (x1, x2))]:
            for index, positions in enumerate(values):
                positions.tofile(trace / 'substeps' / f'safe_{frame:04d}_{index:04d}.bin')
        valid = inspect_trace(run, 2)
        require(valid['coverage_passed'] and valid['fixed_diagnostic_passed']
                and valid['accepted_segments'] == 3 and valid['stationary_bridges'] == 1
                and valid['expected_validator_paths'] == 4, 'Synthetic complete trace inventory failed')
        x0.tofile(trace / 'substeps/safe_0001_0000.bin')
        broken = inspect_trace(run, 2)
        require(not broken['coverage_passed'] and broken['stationary_bridges'] == 0, 'Broken bridge was accepted')
        invalid = x0.copy()
        invalid[1, 0] = np.nan
        invalid.tofile(trace / 'substeps/safe_0000_0001.bin')
        bad = inspect_trace(run, 2)
        require(not bad['all_accepted_states_finite'], 'Nonfinite accepted interior was missed')
    print(json.dumps({'synthetic_inventory': 'passed', 'accepted_segments': 3, 'stationary_bridges': 1,
                      'invalid_bridge_detected': True, 'nonfinite_interior_detected': True,
                      'native_validator_invoked': False, 'gpu_or_remote_used': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--plan', default='configs/active/autodl_window.json')
    parser.add_argument('--matrix', default='reports/active/AUTODL_WINDOW_BATCH.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--validator', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    parser.add_argument('--inventory-timeout-seconds', type=int, default=60)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required outside --self-test')
    root = args.root.resolve()
    report = audit(root, child(root, args.plan), child(root, args.matrix), args.output_dir.resolve(),
                   discover_validator(args.validator), args.timeout_seconds, args.inventory_timeout_seconds)
    return int(not report['all_54_accepted_path_audits_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
