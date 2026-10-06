"""Offline AutoDL factor-window observations; never runs a solver or CCD.

--root is the downloaded package/result root. Plan and matrix paths are relative
to that root; --output is relative to the caller's working directory. Existing
reports are never replaced. Missing/failed runs remain explicit in the output.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from analyze import analyze as analyze_geometry
from config import digest, expand, read, sha, matches_requested
from validate_run import validate


ROLES = ('stiff', 'ipc_host', 'ipc_graph', 'toi_host', 'toi_graph', 'tri_toi_graph')
REPEATS = (1, 2, 3)
SCALAR_UPPER = ('cloth_max_stretch', 'fem_nonpositive', 'fem_negative_volume', 'fixed_max_drift')
SCALAR_LOWER = ('fem_min_J',)
SCALAR_FLOOR = 1e-12  # Existing common-analyzer comparison floor, not a new tolerance.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summary(values):
    values = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in values), 'Nonfinite summary input')
    return {'count': len(values), 'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def child(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escapes downloaded root: ' + str(name))
    return path


def classify(task, config):
    """Use executed configuration, not experiment directory/name conventions."""
    if task['binary'] == 'base':
        require((config['backend'], config['execution'], config['mas']) == ('ipc', 'host', 'legacy'),
                'Official Stiff must retain native IPC/host/MAS configuration')
        return 'stiff'
    require(task['binary'] == 'active' and config['mas'] == 'cholesky'
            and config['mas_restrict'] == 'warp', 'Experimental arms require stable MAS/warp')
    if config['mas_factor_action'] == 'triangular':
        require(config['backend'] == 'toi_al' and config['execution'] == 'conditional_graph',
                'Triangular reference must be TOI Graph')
        return 'tri_toi_graph'
    require(config['mas_factor_action'] == 'factor_inverse', 'Unexpected factor action')
    backend = {'ipc': 'ipc', 'toi_al': 'toi'}[config['backend']]
    execution = {'host': 'host', 'conditional_graph': 'graph'}[config['execution']]
    return backend + '_' + execution


def prepare_tasks(root, plan):
    require(plan['stage'] == 'window', 'Only continuous diagnostic windows are supported')
    tasks = []
    names = set()
    for task in plan['runs']:
        name = task['name']
        require(name == Path(name).name and '/' not in name and '\\' not in name and name not in names,
                'Duplicate or invalid run name: ' + name)
        names.add(name)
        declared = read(child(root, task['config'])) if isinstance(task['config'], str) else task['config']
        config = expand(declared)
        require(config['diagnostics'] == ['substeps', 'physics'] and config['trace_velocity'] is True
                and config['profile'] == 'none' and not config['fixed_factor_study']
                and not config['fixed_restrict_study'], 'Window diagnostic contract changed: ' + name)
        require(config['timeout_seconds'] <= 120, 'Window exceeds the declared 120 second budget')
        tasks.append(task | {'expanded_config': config, 'role': classify(task, config)})
    scenes = sorted({task['scene_key'] for task in tasks})
    require(len(scenes) == 3 and len(tasks) == 54, 'Expected 3 scenes x 6 arms x 3 repeats')
    for key in scenes:
        rows = [task for task in tasks if task['scene_key'] == key]
        require(collections.Counter((row['role'], row['repeat']) for row in rows)
                == collections.Counter((role, repeat) for role in ROLES for repeat in REPEATS),
                'Each scene needs exactly three paired repeats of all six roles: ' + key)
        configs = [row['expanded_config'] for row in rows]
        for field in ('scene', 'steps', 'dt', 'ipc_newton_tol', 'pcg_rho_tol',
                      'toi_remaining_fraction_tol', 'toi_trial_velocity_tol'):
            require(len({config[field] for config in configs}) == 1, 'Unequal scene/tolerance field: ' + field)
        active = [row['expanded_config'] for row in rows if row['role'] != 'stiff']
        ignore = {'backend', 'execution', 'mas_factor_action'}
        common = [{key: value for key, value in config.items() if key not in ignore} for config in active]
        require(all(value == common[0] for value in common), 'Experimental arms change extra components: ' + key)
    return tasks


def verify_download_provenance(root, plan, matrix, bundle, plan_path):
    """Verify actual downloaded bytes and the original driver's ordered ledger.

    Downloaded Python is hashed, never imported or executed. The local reviewed
    parser is used only when its source hash matches the downloaded driver.
    """
    result = {'passed': False, 'source_files': [], 'binaries': [], 'additive_tools': [], 'failures': [],
              'source_files_verified': False, 'actual_binaries_verified': False,
              'canonical_manifest_bound': False, 'ordered_driver_ledger_verified': False,
              'identity_scope': 'Actual downloaded frozen inputs and executable bytes, canonical build manifest, '
                                'saved run manifests/requests, and original ordered chunk-driver ledger. '
                                'Compiler objects need not be downloaded and are not claimed byte-verified here.'}
    for entry in bundle['files']:
        row = {'path': entry['path'], 'expected_sha256': entry['sha256'], 'verified': False}
        try:
            path = child(root, entry['path'])
            row.update(actual_sha256=sha(path), bytes=path.stat().st_size)
            row['verified'] = row['actual_sha256'] == entry['sha256'] and row['bytes'] == entry['bytes']
            if not row['verified']:
                row['failure'] = 'Frozen source/input bytes differ from bundle'
        except (ValueError, OSError) as error:
            row['failure'] = str(error)
        result['source_files'].append(row)
    result['source_files_verified'] = bool(result['source_files']) and all(row['verified'] for row in result['source_files'])
    if not result['source_files_verified']:
        result['failures'].append('One or more actual frozen inputs are missing or differ')
    canonical = None
    try:
        canonical_path = root / 'builds/autodl-active/manifest.json'
        canonical = read(canonical_path)
        result['canonical_build_manifest_sha256'] = sha(canonical_path)
        require(canonical['files'] == bundle['files'] and canonical['source_digest'] == bundle['source_digest']
                == matrix['source_digest'] and canonical['candidate_sha256'] == bundle['candidate_config_sha256']
                == plan['candidate_sha256'] == matrix['candidate_sha256'], 'Canonical build/candidate/source mismatch')
        result['canonical_manifest_bound'] = True
        require(set(canonical['binaries']) == {'base', 'active'}, 'Missing canonical simulator binary records')
        require(set(canonical.get('validator_binaries', {})) == {'validate_path', 'diagnose_first_path'},
                'Missing canonical independent validator binary records')
        for group in ('binaries', 'validator_binaries'):
            for name, entry in canonical[group].items():
                row = {'group': group, 'name': name, 'path': entry['path'],
                       'expected_sha256': entry['sha256'], 'verified': False}
                try:
                    path = child(root, entry['path'])
                    row.update(actual_sha256=sha(path), bytes=path.stat().st_size)
                    row['verified'] = row['actual_sha256'] == entry['sha256']
                    if not row['verified']:
                        row['failure'] = 'Actual executable bytes differ from canonical build'
                except (ValueError, OSError) as error:
                    row['failure'] = str(error)
                result['binaries'].append(row)
        result['actual_binaries_verified'] = len(result['binaries']) == 4 and all(row['verified'] for row in result['binaries'])
        if not result['actual_binaries_verified']:
            result['failures'].append('One or more actual Linux executables are missing or differ')
    except (ValueError, KeyError, OSError) as error:
        result['failures'].append('Canonical build identity: ' + str(error))
    for name in ('autodl_job.py', 'autodl_window_chunks.py', 'autodl_transfer.py'):
        path = root / 'tools/active' / name
        row = {'path': str(path.relative_to(root)), 'bytes_available': path.is_file()}
        if path.is_file():
            row['sha256'] = sha(path)
        else:
            result['failures'].append('Missing actual additive orchestration source: ' + name)
        result['additive_tools'].append(row)
    try:
        require(canonical is not None, 'Ordered ledger requires the canonical build manifest')
        import autodl_window_chunks as reviewed_driver
        actual_driver = root / 'tools/active/autodl_window_chunks.py'
        actual_hash = sha(actual_driver)
        local_hash = sha(Path(reviewed_driver.__file__))
        require(actual_hash == local_hash, 'Downloaded chunk driver differs from the local reviewed ledger parser')
        require(matrix.get('driver_sha256') == actual_hash, 'Cumulative ledger driver SHA does not match downloaded bytes')
        reviewed_driver.check_window_plan(plan)
        smoke_path = root / 'reports/active/AUTODL_SMOKE_BATCH.json'
        smoke_plan_path = root / 'configs/active/autodl_smoke.json'
        smoke, smoke_plan = read(smoke_path), read(smoke_plan_path)
        require(digest(smoke_plan) == bundle['plan_digests']['smoke'], 'Smoke plan differs from frozen bundle')
        initial_failed = reviewed_driver.smoke_failures(smoke, smoke_plan, sha(smoke_plan_path), plan, canonical)
        for row in smoke['runs']:
            name = row['name']
            require(name == Path(name).name and '/' not in name and '\\' not in name, 'Unsafe smoke run name')
            require(read(child(root, 'runs/active/' + name + '/result.json')) == row['result'],
                    'Smoke ledger differs from preserved run result: ' + name)
        require(matrix.get('skipped_failed_smoke_arms') == sorted(initial_failed), 'Smoke-failure skip list differs')
        cursor, failed = reviewed_driver.validate_prefix(matrix, plan, canonical, sha(plan_path), actual_hash,
            initial_failed, lambda name: read(child(root, 'runs/active/' + name + '/result.json')))
        result.update(ordered_driver_ledger_verified=True,
                      ordered_ledger={'next_plan_index': cursor, 'completed_repeats': matrix['completed_repeats'],
                          'full_plan_prefix_processed': cursor == len(plan['runs']), 'failed_arms': sorted(failed),
                          'initial_failed_smoke_arms': sorted(initial_failed), 'driver_sha256': actual_hash,
                          'reviewed_parser_sha256': local_hash, 'smoke_report_sha256': sha(smoke_path),
                          'smoke_plan_sha256': sha(smoke_plan_path),
                          'scope': 'Original order, cumulative cursor, repeat completion, failure skips, and every '
                                   'preserved smoke/window result checked; absent trailing runs remain pending.'})
    except (ValueError, KeyError, OSError, ImportError) as error:
        result['failures'].append('Ordered execution ledger: ' + str(error))
    result['passed'] = not result['failures'] and result['source_files_verified'] and result['actual_binaries_verified'] \
        and result['canonical_manifest_bound'] and result['ordered_driver_ledger_verified']
    return result


def validate_run(task, batch, root, bundle, matrix, provenance=None):
    run = child(root, 'runs/active/' + task['name'])
    request, result = read(run / 'requested.json'), read(run / 'result.json')
    config = task['expanded_config']
    require({key: value for key, value in batch.items() if key != 'result'}
            == {key: value for key, value in task.items() if key not in ('expanded_config', 'role')},
            'Batch task differs from plan: ' + run.name)
    require(result == batch['result'], 'Saved result differs from batch: ' + run.name)
    require(request['expanded_config'] == config and request['config_sha256'] == digest(config),
            'Requested configuration/digest mismatch: ' + run.name)
    require(matches_requested(config,request['requested_config']) and request['binary'] == task['binary'],
            'Original request/binary mismatch: ' + run.name)
    require(result['status'] == 'completed' and result['recorded_frames'] == config['steps']
            and result.get('finite') is True, 'Incomplete/nonfinite run: ' + run.name)
    require(result.get('heavy_diagnostics') is True and result.get('timing_is_diagnostic') is True,
            'Missing heavy diagnostic timing declaration: ' + run.name)
    manifest = read(run / 'build_manifest.json')
    if provenance is not None:
        require(provenance['passed'], 'Downloaded source/executable/ordered-ledger verification has not passed')
        require(sha(run / 'build_manifest.json') == provenance['canonical_build_manifest_sha256'],
                'Saved run manifest differs from the canonical remote build: ' + run.name)
    require(manifest['source_digest'] == request['source_digest'] == matrix['source_digest'] == bundle['source_digest']
            and manifest['files'] == bundle['files'], 'Source/bundle identity mismatch: ' + run.name)
    require(manifest['candidate_sha256'] == matrix['candidate_sha256'] == bundle['candidate_config_sha256'],
            'Candidate identity mismatch: ' + run.name)
    require(manifest['binaries'][task['binary']]['sha256'] == request['exe_sha256'],
            'Executable identity mismatch: ' + run.name)
    runner = next(row for row in bundle['files'] if row['path'] == 'tools/active/autodl_linux.py')
    require(request['runner_sha256'] == runner['sha256'], 'Runner identity mismatch: ' + run.name)
    require(all(gpu['uuid'] == manifest['gpu']['uuid'] for gpu in request['gpu_before']),
            'Selected GPU identity mismatch: ' + run.name)
    saved_validation = read(run / 'config_validation.json')
    require(saved_validation['passed'] is True, 'Remote configuration validation failed: ' + run.name)
    scene = read(run / 'output/scene.json')
    require(scene['case_id'] == config['scene'] and scene['effective_scalar_fields']['preconditioner_type'] == 1,
            'Effective scene or preconditioner mismatch: ' + run.name)
    for field, option in [('dt', 'dt'), ('newton_tol', 'ipc_newton_tol'), ('pcg_tol', 'pcg_rho_tol')]:
        require(scene['effective_run'][field] == config[option], 'Effective stop/time-step mismatch: ' + field)
    checked = validate(run) if task['binary'] == 'active' else saved_validation
    require(checked['passed'] is True, 'Offline requested/resolved/observed check failed: ' + run.name)
    if task['binary'] == 'active':
        resolved = read(run / 'resolved_config.json')
        require(resolved['mas'].get('factor_action') == config['mas_factor_action']
                and resolved['mas'].get('restriction_mode') == 'warp'
                and resolved.get('effective_preconditioner') == 'mas' and resolved['mas'].get('active') is True,
                'Unconfirmed MAS action/restriction: ' + run.name)
    with (run / 'trace/frames.csv').open() as stream:
        frame_times = list(csv.DictReader(stream))
    stats_frames = read(run / 'output/stats.json')['frames']
    require(len(frame_times) == len(stats_frames) == config['steps'],
            'Frame/timing/stats coverage mismatch: ' + run.name)
    pcg_records = [newton['pcg'] for frame in stats_frames for newton in frame['newton'] if 'pcg' in newton]
    require(not any(pcg.get('iteration_limit') or pcg.get('breakdown') for pcg in pcg_records),
            'PCG limit/breakdown observed (including official Stiff): ' + run.name)
    topology = np.fromfile(run / 'trace/topology.bin', dtype='<u4', count=3)
    require(len(topology) == 3, 'Missing topology counts: ' + run.name)
    nv = int(topology[0])
    velocity_frames = []
    for frame in range(config['steps'] + 1):
        require((run / 'trace' / f'state_{frame:04d}.bin').stat().st_size == nv * 24,
                'Missing/malformed continuous state: ' + run.name)
        velocity_path = run / 'trace' / f'velocity_{frame:04d}.bin'
        if frame and velocity_path.exists():
            velocity = np.fromfile(velocity_path, dtype='<f8')
            require(velocity.size == nv * 3 and np.isfinite(velocity).all(),
                    'Malformed/nonfinite exported velocity: ' + run.name)
            velocity_frames.append(frame)
    complete_velocity = velocity_frames == list(range(1, config['steps'] + 1))
    require(complete_velocity or (task['binary'] == 'base' and not velocity_frames),
            'Missing/partial velocity export from a supported arm: ' + run.name)
    samples = result.get('gpu_samples', [])
    return {'name': run.name, 'path': str(run), 'role': task['role'], 'variant': task['variant'],
            'repeat': task['repeat'], 'configuration_checks': checked, 'requested_sha256': sha(run / 'requested.json'),
            'exe_sha256': request['exe_sha256'], 'build_manifest_sha256': sha(run / 'build_manifest.json'),
            'body_ids_sha256': sha(run / 'trace/body_ids.bin'), 'gpu_uuid': manifest['gpu']['uuid'],
            'identity_scope': provenance['identity_scope'] if provenance is not None else
                'Saved request/manifest/bundle consistency only; actual downloaded source/executable/ledger verification not supplied.',
            'actual_binary_verified': provenance['actual_binaries_verified'] if provenance is not None else False,
            'source_files_verified': provenance['source_files_verified'] if provenance is not None else False,
            'velocity_export_available': complete_velocity,
            'finite_exported_velocities': True if complete_velocity else None,
            'velocity_unavailable_reason': None if complete_velocity else
                'Frozen Stiff does not implement GIPC_TRACE_VELOCITY; requested flag is not evidence of export.',
            'gpu_sample_count': len(samples),
            'foreign_compute_sample_count': sum(bool(sample.get('foreign_compute_pids')) for sample in samples),
            'controlled_load_requested': request.get('controlled_load_requested', False)}


def path_inventory(run, frames):
    """Coverage interface for an independent CPU validator; no CCD inference."""
    trace = run / 'trace'
    groups = collections.defaultdict(list)
    for path in sorted((trace / 'substeps').glob('safe_*.bin')):
        match = re.fullmatch(r'safe_(\d{4})_(\d{4})\.bin', path.name)
        require(match is not None, 'Unexpected safe-state filename: ' + path.name)
        groups[int(match[1])].append((int(match[2]), path))
    require(sorted(groups) == list(range(frames)), 'Incomplete accepted-path physical-frame coverage')
    size = (trace / 'state_0000.bin').stat().st_size
    hashes = {frame: sha(trace / f'state_{frame:04d}.bin') for frame in range(frames + 1)}
    rows = []
    for frame, entries in sorted(groups.items()):
        require([index for index, _ in entries] == list(range(len(entries))) and len(entries) >= 2,
                'Nonconsecutive/missing accepted segment')
        require(all(path.stat().st_size == size for _, path in entries), 'Malformed accepted state size')
        require(sha(entries[0][1]) == hashes[frame] and sha(entries[-1][1]) == hashes[frame + 1],
                'Accepted path endpoints differ from physical-frame states')
        rows.append({'physical_frame': frame + 1, 'exported_frame_index': frame,
                     'accepted_segments': len(entries) - 1,
                     'safe_files': [path.relative_to(run).as_posix() for _, path in entries],
                     'frame_start_sha256': hashes[frame], 'frame_end_sha256': hashes[frame + 1]})
    segments = sum(row['accepted_segments'] for row in rows)
    return {'coverage_checked': True, 'physical_frames': frames, 'accepted_segments': segments,
            'stationary_bridges': frames - 1, 'expected_validator_paths': segments + frames - 1,
            'frames': rows, 'validator_scope_expected': 'accepted substeps',
            'ccd_checked': False, 'accepted_interior_finiteness_checked_here': False,
            'note': 'Ordered endpoint/size coverage only. CPU CCD must independently inspect every accepted state/path; '
                    'no collision flag count or physical pass is inferred from this inventory.'}


def compare_reference(report, references, candidates):
    own = [pair for pair in report['comparisons'] if pair['a'] in references and pair['b'] in references]
    require(len(own) == 3, 'Reference needs three repeat pairs')
    metrics = [key for key in own[0]['frames'][0] if key.startswith(('position_', 'velocity_'))]
    reference_velocity_available = any(key.startswith('velocity_') for key in metrics)
    count = len(report['baseline_frame_envelope'])
    repeat_frames = [{'frame': i + 1, **{key: max(pair['frames'][i][key] for pair in own) for key in metrics}}
                     for i in range(count)]
    observations = []
    for candidate in candidates:
        pairs = [pair for pair in report['comparisons'] if
                 (pair['a'] == candidate and pair['b'] in references) or
                 (pair['b'] == candidate and pair['a'] in references)]
        require(len(pairs) == 3, 'Missing candidate/reference state comparison: ' + candidate)
        available_metrics = [key for key in metrics if all(key in frame for pair in pairs for frame in pair['frames'])]
        frames = [{'frame': i + 1, **{key: {
            'range_against_three_references': [min(pair['frames'][i][key] for pair in pairs),
                                               max(pair['frames'][i][key] for pair in pairs)],
            'reference_repeat_max': repeat_frames[i][key],
            'outside_reference_repeat': max(pair['frames'][i][key] for pair in pairs) > repeat_frames[i][key]}
            for key in available_metrics}} for i in range(count)]
        observations.append({'run': candidate, 'frames': frames,
            'available_metrics': available_metrics, 'unavailable_metrics': [key for key in metrics if key not in available_metrics],
            'velocity_comparison_available': any(key.startswith('velocity_') for key in available_metrics),
            'max_rms_against_references': {key: max(pair['maxima'][key] for pair in pairs) for key in available_metrics},
            'outside_reference_repeat_frames': {key: [f['frame'] for f in frames if f[key]['outside_reference_repeat']]
                                                for key in available_metrics}, 'quality_certified': False})
    return {'reference_names': references, 'baseline_frame_envelope': report['baseline_frame_envelope'],
            'candidate_gates': [gate for gate in report['candidate_gates'] if gate['run'] in candidates],
            'state_repeat_comparison': {'metrics': metrics, 'reference_repeat_frames': repeat_frames,
                'reference_velocity_available': reference_velocity_available,
                'unavailable_reference_velocity_metrics': [] if reference_velocity_available else
                    [key.replace('position_', 'velocity_', 1) for key in metrics if key.startswith('position_')],
                'reference_repeat_global_maxima': {key: max(frame[key] for frame in repeat_frames) for key in metrics},
                'candidate_observations': observations, 'position_units': 'm', 'velocity_units': 'm/s',
                'scope': 'Mass-weighted cloth/FEM/ABD groups; fixed geometry has a separate drift metric. '
                         'Velocities are compared only when both sides exported them; frozen Stiff velocity is unavailable. '
                         'No reconstruction. Trajectory difference is not truth error. '
                         'Strict observed repeat envelopes, no added state tolerance.'}}


def summarize_scene(tasks, identities):
    tasks = sorted(tasks, key=lambda task: (ROLES.index(task['role']), task['repeat']))
    identities = {row['name']: row for row in identities}
    paths = [Path(identities[task['name']]['path']) for task in tasks]
    require(len({identities[task['name']]['body_ids_sha256'] for task in tasks}) == 1, 'Body IDs differ within scene')
    require(len({identities[task['name']]['gpu_uuid'] for task in tasks}) == 1, 'Hardware differs within scene')
    require(len({identities[task['name']]['build_manifest_sha256'] for task in tasks}) == 1,
            'Build provenance differs within scene')
    report = analyze_geometry(paths, 3)
    configs = {task['name']: task['expanded_config'] for task in tasks}
    steps = tasks[0]['expanded_config']['steps']
    runs = []
    for row in report['runs']:
        frames = row['frames']
        require(len(frames) == steps, 'Analyzer has incomplete physical frames')
        seconds = sum(frame['seconds'] for frame in frames)
        require(math.isclose(seconds, row['result']['solver_seconds'], rel_tol=1e-12, abs_tol=1e-12),
                'Solver time sum mismatch: ' + row['name'])
        runs.append(row | identities[row['name']] | {
            'solver_seconds': seconds, 'mean_frame_ms': seconds * 1000 / steps,
            'direction_count': sum(frame['pcg_calls'] for frame in frames),
            'pcg_iterations': sum(frame['pcg_iterations'] for frame in frames),
            'pcg_limit_or_breakdown_hits': sum(frame['pcg_limit_hits'] for frame in frames),
            'outer_count': sum(frame['outers'] for frame in frames),
            'linear_stage_seconds': sum(frame['linear_stage_seconds'] for frame in frames)
                if all(frame['linear_stage_seconds'] is not None for frame in frames) else None,
            'max_cloth_stretch': max(frame['cloth_max_stretch'] for frame in frames),
            'min_fem_J': min(frame['fem_min_J'] for frame in frames),
            'max_fem_nonpositive': max(frame['fem_nonpositive'] for frame in frames),
            'max_fem_negative_volume': max(frame['fem_negative_volume'] for frame in frames),
            'min_abd_J': min(frame['abd_min_J'] for frame in frames),
            'max_fixed_drift_m': max(frame['fixed_max_drift'] for frame in frames)})
    metrics = ('solver_seconds', 'mean_frame_ms', 'direction_count', 'pcg_iterations', 'outer_count',
               'linear_stage_seconds', 'max_cloth_stretch', 'min_fem_J', 'max_fem_nonpositive',
               'max_fem_negative_volume', 'min_abd_J', 'max_fixed_drift_m')
    summaries = [{'role': role, **{metric: summary(row[metric] for row in runs if row['role'] == role)
                                  for metric in metrics}} for role in ROLES]
    lookup = {(row['role'], row['repeat']): row for row in runs}
    pairs = []
    for label, numerator, denominator in [
        ('ipc_graph_gain', 'ipc_host', 'ipc_graph'), ('toi_graph_gain', 'toi_host', 'toi_graph'),
        ('toi_gain_host', 'ipc_host', 'toi_host'), ('toi_gain_graph', 'ipc_graph', 'toi_graph'),
        ('factor_action_gain_toi_graph', 'tri_toi_graph', 'toi_graph'),
        ('fused_vs_stiff', 'stiff', 'toi_graph'), ('optimized_ipc_vs_stiff', 'stiff', 'ipc_graph')]:
        observations = []
        for repeat in REPEATS:
            old, new = lookup[numerator, repeat], lookup[denominator, repeat]
            require(old['solver_seconds'] > 0 and new['solver_seconds'] > 0, 'Nonpositive solver time')
            observations.append({'repeat': repeat, 'numerator_run': old['name'], 'denominator_run': new['name'],
                'numerator_seconds': old['solver_seconds'], 'denominator_seconds': new['solver_seconds'],
                'ratio': old['solver_seconds'] / new['solver_seconds']})
        pairs.append({'component': label, 'numerator_role': numerator, 'denominator_role': denominator,
                      'pairs': observations, 'summary': summary(row['ratio'] for row in observations)})
    # Reuse the already-computed geometric observations for the second envelope;
    # the scalar comparisons below preserve the public analyzer's exact rules.
    tri_names = [row['name'] for row in runs if row['role'] == 'tri_toi_graph']
    tri_rows = [row for row in runs if row['name'] in tri_names]
    tri_envelope = [{'frame': i + 1, **{key: [min(row['frames'][i][key] for row in tri_rows),
                                            max(row['frames'][i][key] for row in tri_rows)]
                     for key in (*SCALAR_UPPER, 'fem_min_J', 'abd_min_J')}} for i in range(steps)]
    tri_gates = []
    for row in runs:
        if row['name'] in tri_names:
            continue
        violations = []
        for frame, bounds in zip(row['frames'], tri_envelope):
            for metric in SCALAR_UPPER:
                if frame[metric] > bounds[metric][1] + SCALAR_FLOOR:
                    violations.append({'frame': frame['frame'], 'metric': metric,
                                       'value': frame[metric], 'bound': bounds[metric][1]})
            for metric in SCALAR_LOWER:
                if frame[metric] < bounds[metric][0] - SCALAR_FLOOR:
                    violations.append({'frame': frame['frame'], 'metric': metric,
                                       'value': frame[metric], 'bound': bounds[metric][0]})
            if frame['abd_min_J'] <= 0:
                violations.append({'frame': frame['frame'], 'metric': 'abd_inversion'})
        tri_gates.append({'run': row['name'], 'endpoint_envelope_passes': not violations,
                          'violation_count': len(violations), 'violations': violations, 'quality_certified': False})
    tri_report = report | {'baseline_frame_envelope': tri_envelope, 'candidate_gates': tri_gates}
    names = [row['name'] for row in runs]
    stiff_names = [row['name'] for row in runs if row['role'] == 'stiff']
    return {'scene_key': tasks[0]['scene_key'], 'scene': configs[names[0]]['scene'], 'frames': steps,
            'dt': configs[names[0]]['dt'], 'common_complete_frame_count_asserted': True,
            'same_initial_state_scene_materials_topology': report['same_initial_state_scene_topology'],
            'scales_m': report['scales_m'], 'summary': summaries, 'paired_time_ratios': pairs, 'runs': runs,
            'quality_against_reference': {
                'stiff': compare_reference(report, stiff_names, [name for name in names if name not in stiff_names]),
                'tri_toi_graph': compare_reference(tri_report, tri_names, [name for name in names if name not in tri_names])},
            'absent_group_policy': 'FEM/ABD J=1 with an absent group is a placeholder, not a material-quality claim.'}


def analyze(root, plan_path, matrix_path, output):
    require(not output.exists(), 'Refusing to replace an existing output: ' + str(output))
    plan, matrix, bundle = read(plan_path), read(matrix_path), read(root / 'autodl_bundle.json')
    tasks = prepare_tasks(root, plan)
    require(matrix['plan_sha256'] == sha(plan_path) and bundle['plan_digests']['window'] == digest(plan),
            'Downloaded window plan/batch/bundle identity mismatch')
    require(digest(bundle['files']) == bundle['source_digest'] and digest(bundle['candidate_config'])
            == bundle['candidate_config_sha256'] == matrix['candidate_sha256'] == plan['candidate_sha256'],
            'Bundle/candidate digest mismatch')
    batch_by_name = {row['name']: row for row in matrix['runs']}
    require(len(batch_by_name) == len(matrix['runs']) and set(batch_by_name) <= {task['name'] for task in tasks},
            'Unexpected or duplicate batch run')
    report = {'schema_version': 1, 'root': str(root), 'plan_sha256': sha(plan_path), 'matrix_sha256': sha(matrix_path),
        'bundle_sha256': sha(root / 'autodl_bundle.json'), 'analyzer_sha256': sha(Path(__file__)),
        'common_analyzer_sha256': sha(Path(__file__).with_name('analyze.py')),
        'physical_certified': False, 'performance_certified': False, 'timing_is_heavy_diagnostic': True,
        'timing_scope': 'Heavy diagnostic configuration requests substeps, physics and velocity exports. Active arms '
                        'export velocity; frozen Stiff does not, so diagnostic overhead is not identical. All ratios are '
                        'diagnostic only; three repeats do not constitute the seven-pair performance acceptance test.',
        'velocity_scope': 'Actual exported active-arm velocities in m/s. Frozen Stiff velocity is unavailable despite '
                          'the requested export flag; no reconstructed baseline, no tolerance relaxation, no velocity certification.',
        'linear_stage_scope': 'Whole ordinary PCG/linear stage where available, not an individual kernel. '
                              'All-zero TOI phase placeholders remain unavailable/null.',
        'quality_scope': 'Stiff and triangular TOI Graph repeat envelopes are formed separately. '
                         'Existing scalar comparison floor is 1e-12; state comparisons add no tolerance. '
                         'CCD is deliberately left to a separate CPU validator.',
        'planned_runs': len(tasks), 'recorded_runs': len(matrix['runs']), 'run_checks': [],
        'accepted_path_inventory': [], 'scenes': []}
    report['download_provenance'] = verify_download_provenance(root, plan, matrix, bundle, plan_path)
    identities = []
    for task in tasks:
        check = {'run': task['name'], 'scene_key': task['scene_key'], 'role': task['role'],
                 'repeat': task['repeat'], 'passed': False}
        report['run_checks'].append(check)
        try:
            require(task['name'] in batch_by_name, 'Run absent from batch (not treated as zero cost or a pass)')
            partial = batch_by_name[task['name']]['result']
            check['result_status'] = partial.get('status')
            check['available_result_observation'] = {key: partial[key] for key in
                ('status', 'recorded_frames', 'solver_seconds', 'wall_seconds', 'exit_code', 'finite',
                 'heavy_diagnostics', 'timing_is_diagnostic', 'error') if key in partial}
            identity = validate_run(task, batch_by_name[task['name']], root, bundle, matrix, report['download_provenance'])
            identities.append(identity)
            check['passed'] = True
            record = {'run': task['name'], 'trace_path': str(Path(identity['path']) / 'trace'),
                      'coverage_checked': False, 'ccd_checked': False}
            report['accepted_path_inventory'].append(record)
            try:
                record.update(path_inventory(Path(identity['path']), task['expanded_config']['steps']))
            except (ValueError, KeyError, OSError) as error:
                record['failure'] = str(error)
        except (ValueError, KeyError, OSError, StopIteration, AssertionError) as error:
            check['failure'] = str(error)
    valid_names = {row['name'] for row in identities}
    for key in sorted({task['scene_key'] for task in tasks}):
        selected = [task for task in tasks if task['scene_key'] == key]
        scene = {'scene_key': key, 'analysis_completed': False}
        if all(task['name'] in valid_names for task in selected):
            try:
                scene.update(summarize_scene(selected, [row for row in identities if row['name'] in {t['name'] for t in selected}]))
                scene['analysis_completed'] = True
            except (ValueError, KeyError, OSError, AssertionError) as error:
                scene['failure'] = str(error)
        else:
            scene['failure'] = 'Missing or invalid runs: no complete same-round six-arm quality/time comparison.'
        report['scenes'].append(scene)
    report['all_planned_runs_validated'] = all(check['passed'] for check in report['run_checks'])
    report['all_scenes_analyzed'] = all(scene['analysis_completed'] for scene in report['scenes'])
    report['all_accepted_path_inventories_complete'] = len(report['accepted_path_inventory']) == len(tasks) \
        and all(row['coverage_checked'] for row in report['accepted_path_inventory'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan', default='configs/active/autodl_window.json')
    parser.add_argument('--matrix', default='reports/active/AUTODL_WINDOW_BATCH.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = analyze(root, child(root, args.plan), child(root, args.matrix), args.output.resolve())
    print(json.dumps({'output': str(args.output.resolve()), 'planned_runs': report['planned_runs'],
        'recorded_runs': report['recorded_runs'], 'all_planned_runs_validated': report['all_planned_runs_validated'],
        'all_scenes_analyzed': report['all_scenes_analyzed'],
        'all_accepted_path_inventories_complete': report['all_accepted_path_inventories_complete'],
        'physical_certified': False, 'performance_certified': False}))
    return int(not (report['all_planned_runs_validated'] and report['all_scenes_analyzed']
                    and report['all_accepted_path_inventories_complete']))


if __name__ == '__main__':
    raise SystemExit(main())
