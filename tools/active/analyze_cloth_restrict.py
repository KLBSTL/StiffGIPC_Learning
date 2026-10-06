"""CPU-only analysis of the declared cloth restriction benchmark.

No velocity reconstruction, timing certification, or accepted-path CCD is done here.
Run after GPU timing completes: processing all trajectories can affect host load.
"""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from config import ROOT, digest, expand, read, sha, matches_requested


SCENES = {'hang': 'cloth_hang_l', 'sphere': 'cloth_sphere7_l',
          'bunny': 'cloth_fixed_bunny_l'}
VARIANTS = ('stiff', 'toi_serial', 'toi_warp', 'ipc_warp')
REPEATS = ('r1', 'r2', 'r3')
SCALARS = ('cloth_max_stretch', 'cloth_p95_stretch', 'cloth_min_area_ratio',
           'cloth_max_area_ratio', 'fixed_max_drift_m', 'fixed_abd_max_drift_m')


def require(condition, description):
    if not condition:
        raise ValueError(description)


def load_positions(directory, frame, nv):
    data = np.fromfile(directory / f'state_{frame:04d}.bin', dtype='<f8')
    require(data.size == nv * 3, f'Wrong state size: {directory}, frame {frame}')
    return data.reshape(nv, 3)


def range_summary(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return {'count': len(values), 'median': float(np.median(values)) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def check_configuration(task, run, reference_scene):
    request = read(run / 'requested.json')
    config = request['expanded_config']
    declared = read(ROOT / task['config']) if isinstance(task['config'], str) else task['config']
    require(matches_requested(config,declared), f'Plan/request mismatch: {run.name}')
    require(request['config_sha256'] == digest(config), f'Config digest mismatch: {run.name}')
    require(request['binary'] == task['binary'], f'Binary kind mismatch: {run.name}')
    require(config['scene'] == SCENES[task['scene_key']], f'Scene mismatch: {run.name}')
    for key, value in {'steps': 100, 'dt': .01, 'ipc_newton_tol': .01,
                       'pcg_rho_tol': 1e-4, 'toi_remaining_fraction_tol': .01,
                       'toi_trial_velocity_tol': .05, 'trace_velocity': False,
                       'diagnostics': [], 'profile': 'none'}.items():
        require(config[key] == value, f'Unexpected benchmark {key}: {run.name}')
    variant = task['variant']
    require(config['backend'] == ('toi_al' if variant.startswith('toi_') else 'ipc'),
            f'Backend mismatch: {run.name}')
    require(config['mas_restrict'] == ('warp' if variant.endswith('_warp') else 'serial'),
            f'Restriction mismatch: {run.name}')
    require(config['mas'] == ('legacy' if variant == 'stiff' else 'cholesky'),
            f'MAS mismatch: {run.name}')
    scene = read(run / 'output/scene.json')
    require(scene == reference_scene, f'Materials/scene mismatch: {run.name}')
    effective = scene['effective_run']
    for scene_key, key in [('dt', 'dt'), ('newton_tol', 'ipc_newton_tol'),
                           ('pcg_tol', 'pcg_rho_tol')]:
        require(effective[scene_key] == config[key], f'Effective {key} mismatch: {run.name}')
    require(scene['effective_scalar_fields']['preconditioner_type'] == 1,
            f'Scene is not MAS: {run.name}')
    manifest = read(run / 'build_manifest.json')
    require(sha(run / 'build_manifest.json') == request['manifest_sha256'],
            f'Build manifest mismatch: {run.name}')
    require(manifest['source_digest'] == request['source_digest'],
            f'Source digest mismatch: {run.name}')
    resolved = None
    if request['binary'] == 'active':
        resolved = read(run / 'resolved_config.json')
        mapping = {'contact_backend': 'backend', 'dt': 'dt',
                   'ipc_newton_tol': 'ipc_newton_tol', 'pcg_rho_tol': 'pcg_rho_tol',
                   'configured_pcg_execution': 'execution'}
        for key, wanted in mapping.items():
            require(resolved[key] == config[wanted], f'Resolved {key} mismatch: {run.name}')
        require(resolved['requested_preconditioner'] == 'mas', f'Resolved MAS mismatch: {run.name}')
        require(resolved['mas']['cholesky'] and resolved['mas']['wide_apply']
                and not resolved['mas']['inverse64'], f'Resolved stable MAS mismatch: {run.name}')
        require(resolved['mas']['restriction_mode'] == config['mas_restrict'],
                f'Resolved restriction mismatch: {run.name}')
        for flag, key in [('GIPC_CCD_BVH_REFIT', 'refit'), ('GIPC_BATCHED_ENERGY', 'batch'),
                          ('GIPC_ENERGY_REUSE', 'reuse')]:
            require(resolved['acceleration_features'][flag] == config[key],
                    f'Resolved acceleration mismatch: {run.name}: {flag}')
        if config['backend'] == 'toi_al':
            require(resolved['effective_preconditioner'] == 'mas' and resolved['mas']['active'],
                    f'Inactive MAS: {run.name}')
            for key, wanted in [('remaining_fraction_tol', 'toi_remaining_fraction_tol'),
                                ('trial_velocity_tol_m_s', 'toi_trial_velocity_tol'),
                                ('choose_start', 'choose_start'),
                                ('restart_full_step_guard', 'restart_guard')]:
                require(resolved['toi'][key] == config[wanted], f'Resolved TOI {key}: {run.name}')
    return request, resolved


def pair_positions(left, right, weights, scale):
    count = min(len(left['_cloth_positions']), len(right['_cloth_positions']))
    frames = []
    for i in range(count):
        delta = left['_cloth_positions'][i] - right['_cloth_positions'][i]
        if not np.isfinite(delta).all():
            frames.append({'frame': i + 1, 'cloth_rms_m': None,
                           'cloth_rms_percent_scale': None, 'cloth_max_vertex_m': None})
            continue
        square = np.einsum('ij,ij->i', delta, delta)
        rms = float(np.sqrt(np.average(square, weights=weights)))
        frames.append({'frame': i + 1, 'cloth_rms_m': rms,
                       'cloth_rms_percent_scale': 100 * rms / scale,
                       'cloth_max_vertex_m': float(np.sqrt(square.max()))})
    return {'a': left['name'], 'b': right['name'], 'a_variant': left['variant'],
            'b_variant': right['variant'], 'same_variant': left['variant'] == right['variant'],
            'same_repeat': left['repeat'] == right['repeat'], 'frames': frames,
            'max_cloth_rms_m': max((f['cloth_rms_m'] for f in frames if f['cloth_rms_m'] is not None), default=None),
            'max_cloth_rms_percent_scale': max((f['cloth_rms_percent_scale'] for f in frames
                                               if f['cloth_rms_percent_scale'] is not None), default=None)}


def reference_envelope(runs, pairs, variant):
    references = [r for r in runs if r['variant'] == variant and r['complete_finite']]
    own_pairs = [p for p in pairs if p['a'] in {r['name'] for r in references}
                 and p['b'] in {r['name'] for r in references}]
    frames = []
    if references:
        for i in range(100):
            observations = [r['frames'][i] for r in references]
            frames.append({'frame': i + 1,
                           **{key: [min(f[key] for f in observations), max(f[key] for f in observations)]
                              for key in SCALARS},
                           'own_repeat_cloth_rms_m_max': max((p['frames'][i]['cloth_rms_m']
                                                              for p in own_pairs), default=None)})
    return {'reference_variant': variant, 'reference_names': [r['name'] for r in references],
            'complete_three_repeat_envelope': len(references) == 3,
            'frames': frames, 'own_repeat_pair_names': [[p['a'], p['b']] for p in own_pairs],
            'own_repeat_max_cloth_rms_m': max((p['max_cloth_rms_m'] for p in own_pairs), default=None),
            'scalar_repeat_ranges': {key: range_summary(r[key] for r in references)
                                     for key in ('max_cloth_stretch', 'max_cloth_p95_stretch',
                                                 'min_cloth_area_ratio', 'max_fixed_drift_m',
                                                 'max_fixed_abd_drift_m')}}


def analyze_scene(key, tasks):
    tasks = sorted(tasks, key=lambda t: (VARIANTS.index(t['variant']), REPEATS.index(t['repeat'])))
    reference_task = next((t for t in tasks if t['variant'] == 'stiff'), None)
    require(reference_task is not None, f'No Stiff reference for {key}')
    reference_run = ROOT / 'runs/active' / reference_task['name']
    directory = reference_run / 'trace'
    scene = read(reference_run / 'output/scene.json')
    raw = np.fromfile(directory / 'topology.bin', dtype='<u4')
    nv, nf, nt = map(int, raw[:3])
    require(raw.size == 3 + 3 * nf + 4 * nt, f'Invalid topology: {key}')
    faces = raw[3:3 + nf * 3].reshape(-1, 3)
    tets = raw[3 + nf * 3:].reshape(-1, 4)
    in_tet = np.zeros(nv, dtype=bool)
    in_tet[tets.ravel()] = True
    cloth_faces = faces[~in_tet[faces].any(axis=1)]
    cloth = np.zeros(nv, dtype=bool)
    cloth[cloth_faces.ravel()] = True
    boundary_fixed = np.fromfile(directory / 'boundary_types.bin', dtype='<i4') == 1
    abd_count = int(read(directory / 'metadata.json')['abd_point_num'])
    objects_3d = [o for o in scene['objects'] if o['dimension'] == 3]
    require(all(o['body_type'] == 'ABD' and o['fixed_mode'] == 'all' for o in objects_3d),
            f'Cannot infer fixed ABD mapping for {key}')
    require(abd_count == {'hang': 0, 'sphere': 483, 'bunny': 19193}[key], f'Unexpected ABD count: {key}')
    fixed_abd = in_tet & (np.arange(nv) < abd_count)
    require(int(fixed_abd.sum()) == abd_count, f'Incomplete fixed ABD tet mapping: {key}')
    fixed = boundary_fixed | fixed_abd
    free_cloth = cloth & ~fixed
    mass = np.fromfile(directory / 'masses.bin', dtype='<f8')
    initial = load_positions(directory, 0, nv)
    require(np.isfinite(initial).all() and mass.size == nv and np.all(mass[free_cloth] > 0),
            f'Invalid initial state/mass: {key}')
    scale = float(np.linalg.norm(np.ptp(initial[cloth], axis=0)))
    edges = np.unique(np.sort(np.concatenate([cloth_faces[:, [0, 1]], cloth_faces[:, [1, 2]],
                                              cloth_faces[:, [2, 0]]]), axis=1), axis=0)
    rest_length = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)

    def areas(x):
        tri = x[cloth_faces]
        return .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    rest_area = areas(initial)
    require(scale > 0 and np.all(rest_length > 0) and np.all(rest_area > 0), f'Degenerate cloth: {key}')
    identity_files = ('topology.bin', 'state_0000.bin', 'masses.bin', 'boundary_types.bin',
                      'body_ids.bin', 'metadata.json')
    identity = {name: sha(directory / name) for name in identity_files}
    runs = []
    for task in tasks:
        run = ROOT / 'runs/active' / task['name']
        request, resolved = check_configuration(task, run, scene)
        for filename, expected_sha in identity.items():
            require(sha(run / 'trace' / filename) == expected_sha, f'Initial identity mismatch: {run.name}/{filename}')
        result = read(run / 'result.json')
        require(result == task['result'], f'Batch/result mismatch: {run.name}')
        stats = read(run / 'output/stats.json')['frames']
        with (run / 'trace/frames.csv').open() as stream:
            times = list(csv.DictReader(stream))
        count = result['recorded_frames']
        require(len(stats) == count and len(times) == count, f'Frame coverage mismatch: {run.name}')
        observations, positions = [], []
        for i in range(count):
            x = load_positions(run / 'trace', i + 1, nv)
            finite = bool(np.isfinite(x).all())
            positions.append(x[free_cloth].copy())
            pcg = [n['pcg'] for n in stats[i].get('newton', []) if 'pcg' in n]
            phases = stats[i].get('phase_ms', {})
            # TOI's ordinary stats currently publish an all-zero phase object
            # even while real linear solves ran. These are placeholders, not
            # measured zero-cost phases. Do not use them in a cost breakdown.
            phase_available = bool(phases) and any(value > 0 for value in phases.values())
            if resolved:
                require(all(p.get('execution') == request['expanded_config']['execution']
                            for p in pcg if p.get('iterations', 0) > 0),
                        f'PCG execution mismatch: {run.name}, frame {i + 1}')
            observation = {'frame': i + 1, 'solver_seconds': float(times[i]['solver_ms']) / 1000,
                           'finite': finite, 'direction_count': len(pcg),
                           'pcg_iterations': sum(p['iterations'] for p in pcg),
                           'max_direction_pcg_iterations': max((p['iterations'] for p in pcg), default=0),
                           'pcg_iteration_limit_hits': sum(bool(p.get('iteration_limit')) for p in pcg),
                           'pcg_breakdowns': sum(bool(p.get('breakdown')) for p in pcg),
                           'toi_outer_count': len(stats[i].get('toi', [])),
                           'linear_stage_seconds': phases['pcg'] / 1000 if phase_available and 'pcg' in phases else None,
                           'phase_timing_available': phase_available,
                           'phase_timing_unavailable_reason': None if phase_available else
                               ('all_zero_placeholder' if phases else 'not_recorded'),
                           'phase_seconds': {name: value / 1000 if phase_available else None
                                             for name, value in phases.items()},
                           **dict.fromkeys(SCALARS, None)}
            if finite:
                stretch = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1) / rest_length
                area_ratio = areas(x) / rest_area
                peak = int(np.argmax(stretch))
                observation.update(cloth_max_stretch=float(stretch[peak]),
                                   cloth_p95_stretch=float(np.percentile(stretch, 95)),
                                   cloth_min_area_ratio=float(area_ratio.min()),
                                   cloth_max_area_ratio=float(area_ratio.max()),
                                   fixed_max_drift_m=float(np.linalg.norm(x[fixed] - initial[fixed], axis=1).max()) if fixed.any() else 0.,
                                   fixed_abd_max_drift_m=float(np.linalg.norm(x[fixed_abd] - initial[fixed_abd], axis=1).max()) if fixed_abd.any() else 0.,
                                   peak_edge_global_indices=edges[peak].tolist(),
                                   peak_edge_rest_length_m=float(rest_length[peak]))
            observations.append(observation)
        finite_frames = [f for f in observations if f['finite']]
        all_finite = bool(observations) and len(finite_frames) == len(observations)
        total = sum(f['solver_seconds'] for f in observations)
        require(math.isclose(total, result['solver_seconds'], rel_tol=1e-12, abs_tol=1e-12),
                f'Timer sum mismatch: {run.name}')
        peak_frame = max(finite_frames, key=lambda f: f['cloth_max_stretch']) if finite_frames else None
        linear = (sum(f['linear_stage_seconds'] for f in observations)
                  if all(f['linear_stage_seconds'] is not None for f in observations) else None)
        directions = sum(f['direction_count'] for f in observations)
        iterations = sum(f['pcg_iterations'] for f in observations)
        phase_names = set().union(*(f['phase_seconds'] for f in observations))
        runs.append({'name': run.name, 'variant': task['variant'], 'repeat': task['repeat'],
                     'status': result['status'], 'recorded_frames': count, 'finite': all_finite,
                     'complete_finite': result['status'] == 'completed' and count == 100 and all_finite,
                     'requested_sha256': sha(run / 'requested.json'), 'exe_sha256': request['exe_sha256'],
                     'source_digest': request['source_digest'], 'configuration_verified': True,
                     'resolved_configuration_verified': resolved is not None,
                     'solver_seconds': total, 'mean_frame_ms': total * 1000 / count if count else None,
                     **{key: sum(f[key] for f in observations) for key in
                        ('direction_count', 'pcg_iterations', 'pcg_iteration_limit_hits', 'pcg_breakdowns',
                         'toi_outer_count')},
                     'mean_pcg_iterations_per_direction': iterations / directions if directions else None,
                     'max_direction_pcg_iterations': max((f['max_direction_pcg_iterations'] for f in observations), default=0),
                     'linear_stage_seconds': linear,
                     'linear_stage_fraction_of_solver': linear / total if linear is not None and total > 0 else None,
                     'phase_timing_available_frames': sum(f['phase_timing_available'] for f in observations),
                     'phase_seconds': {name: (sum(f['phase_seconds'][name] for f in observations)
                                              if all(f['phase_seconds'].get(name) is not None for f in observations) else None)
                                       for name in sorted(phase_names)},
                     'max_cloth_stretch': peak_frame['cloth_max_stretch'] if peak_frame else None,
                     'peak_stretch_frame': peak_frame['frame'] if peak_frame else None,
                     'peak_stretch_edge_global_indices': peak_frame['peak_edge_global_indices'] if peak_frame else None,
                     'max_cloth_p95_stretch': max((f['cloth_p95_stretch'] for f in finite_frames), default=None),
                     'min_cloth_area_ratio': min((f['cloth_min_area_ratio'] for f in finite_frames), default=None),
                     'max_fixed_drift_m': max((f['fixed_max_drift_m'] for f in finite_frames), default=None),
                     'max_fixed_abd_drift_m': max((f['fixed_abd_max_drift_m'] for f in finite_frames), default=None),
                     'frames': observations, '_cloth_positions': positions})
    pairs = [pair_positions(a, b, mass[free_cloth], scale) for a, b in itertools.combinations(runs, 2)]
    envelopes = {variant: reference_envelope(runs, pairs, variant) for variant in ('stiff', 'toi_serial')}
    gates = []
    for variant, envelope in envelopes.items():
        for run in runs:
            if run['variant'] == variant:
                continue
            violations = []
            for frame, bound in zip(run['frames'], envelope['frames']):
                if not frame['finite']:
                    violations.append({'frame': frame['frame'], 'metric': 'nonfinite'})
                    continue
                for metric in ('cloth_max_stretch', 'cloth_p95_stretch', 'fixed_max_drift_m',
                               'fixed_abd_max_drift_m'):
                    if frame[metric] > bound[metric][1]:
                        violations.append({'frame': frame['frame'], 'metric': metric, 'value': frame[metric],
                                           'bound': bound[metric][1], 'excess': frame[metric] - bound[metric][1]})
                if frame['cloth_min_area_ratio'] < bound['cloth_min_area_ratio'][0]:
                    violations.append({'frame': frame['frame'], 'metric': 'cloth_min_area_ratio',
                                       'value': frame['cloth_min_area_ratio'], 'bound': bound['cloth_min_area_ratio'][0],
                                       'excess': bound['cloth_min_area_ratio'][0] - frame['cloth_min_area_ratio']})
            relevant = [p for p in pairs if (p['a'] == run['name'] and p['b'] in envelope['reference_names'])
                        or (p['b'] == run['name'] and p['a'] in envelope['reference_names'])]
            divergent = set()
            for pair in relevant:
                for observed, bound in zip(pair['frames'], envelope['frames']):
                    if observed['cloth_rms_m'] is not None and bound['own_repeat_cloth_rms_m_max'] is not None:
                        if observed['cloth_rms_m'] > bound['own_repeat_cloth_rms_m_max']:
                            divergent.add(observed['frame'])
            complete = envelope['complete_three_repeat_envelope'] and run['complete_finite']
            gates.append({'run': run['name'], 'reference_variant': variant,
                          'complete_comparison': complete,
                          'stretch_exceedance_frames': sum(v['metric'] == 'cloth_max_stretch' for v in violations),
                          'p95_stretch_exceedance_frames': sum(v['metric'] == 'cloth_p95_stretch' for v in violations),
                          'trajectory_outside_reference_repeat_frames': sorted(divergent),
                          'observed_endpoint_metrics_within_repeat_envelope': complete and not violations,
                          'violations': violations, 'quality_certified': False})
    lookup = {(r['variant'], r['repeat']): r for r in runs}
    ratios = []
    for numerator in ('stiff', 'toi_serial', 'ipc_warp'):
        records = []
        for repeat in REPEATS:
            left, right = lookup.get((numerator, repeat)), lookup.get(('toi_warp', repeat))
            if left and right and left['complete_finite'] and right['complete_finite']:
                records.append({'repeat': repeat, 'numerator_run': left['name'], 'denominator_run': right['name'],
                                'numerator_seconds': left['solver_seconds'], 'denominator_seconds': right['solver_seconds'],
                                'ratio': left['solver_seconds'] / right['solver_seconds']})
        ratios.append({'numerator_variant': numerator, 'denominator_variant': 'toi_warp',
                       'ratio_meaning': 'numerator solver time / candidate solver time; >1 means candidate faster',
                       'pairs': records, 'summary': range_summary(r['ratio'] for r in records)})
    for run in runs:
        del run['_cloth_positions']
    summary = []
    for variant in VARIANTS:
        arm_runs = [r for r in runs if r['variant'] == variant]
        valid = [r for r in arm_runs if r['complete_finite']]
        summary.append({'variant': variant, 'runs_observed': len(arm_runs),
                        'complete_finite_runs': len(valid),
                        'solver_seconds': range_summary(r['solver_seconds'] for r in valid),
                        'mean_frame_ms': range_summary(r['mean_frame_ms'] for r in valid),
                        'direction_count': range_summary(r['direction_count'] for r in valid),
                        'pcg_iterations': range_summary(r['pcg_iterations'] for r in valid),
                        'mean_pcg_iterations_per_direction': range_summary(r['mean_pcg_iterations_per_direction'] for r in valid),
                        'max_direction_pcg_iterations': range_summary(r['max_direction_pcg_iterations'] for r in valid),
                        'linear_stage_seconds': range_summary(r['linear_stage_seconds'] for r in valid),
                        'phase_timing_available_frames': range_summary(r['phase_timing_available_frames'] for r in valid),
                        'linear_stage_fraction_of_solver': range_summary(r['linear_stage_fraction_of_solver'] for r in valid),
                        'phase_seconds': {name: range_summary(r['phase_seconds'].get(name) for r in valid)
                                          for name in sorted(set().union(*(r['phase_seconds'] for r in valid)))},
                        'max_cloth_stretch': range_summary(r['max_cloth_stretch'] for r in valid),
                        'max_cloth_p95_stretch': range_summary(r['max_cloth_p95_stretch'] for r in valid),
                        'min_cloth_area_ratio': range_summary(r['min_cloth_area_ratio'] for r in valid),
                        'max_fixed_drift_m': range_summary(r['max_fixed_drift_m'] for r in valid),
                        'pcg_iteration_limit_hits': sum(r['pcg_iteration_limit_hits'] for r in arm_runs),
                        'pcg_breakdowns': sum(r['pcg_breakdowns'] for r in arm_runs)})
    return {'scene_key': key, 'scene': SCENES[key], 'dt': .01, 'expected_frames': 100,
            'same_initial_state_materials_stopping_parameters': True, 'identity_sha256': identity,
            'scene_sha256': sha(reference_run / 'output/scene.json'), 'effective_run': scene['effective_run'],
            'vertex_count': nv, 'cloth_vertex_count': int(cloth.sum()), 'free_cloth_vertex_count': int(free_cloth.sum()),
            'boundary_fixed_vertex_count': int(boundary_fixed.sum()), 'fixed_abd_vertex_count': int(fixed_abd.sum()),
            'fixed_vertex_count': int(fixed.sum()), 'cloth_initial_bbox_diagonal_m': scale,
            'fixed_abd_scope': 'All 3D scene objects verified ABD and fixed_mode=all; all ABD tet vertices checked.',
            'summary': summary, 'runs': runs, 'reference_envelopes': envelopes, 'position_comparisons': pairs,
            'candidate_observations': gates, 'paired_time_ratios': ratios,
            'variant_time_summary': {variant: range_summary(r['solver_seconds'] for r in runs
                                                             if r['variant'] == variant and r['complete_finite'])
                                     for variant in VARIANTS}}


def analyze(matrix_path):
    matrix = read(matrix_path)
    grouped = collections.defaultdict(list)
    identities = set()
    for task in matrix['runs']:
        identity = (task['scene_key'], task['variant'], task['repeat'])
        require(identity[0] in SCENES and identity[1] in VARIANTS and identity[2] in REPEATS,
                f'Unexpected matrix entry: {identity}')
        require(identity not in identities, f'Duplicate matrix entry: {identity}')
        identities.add(identity)
        grouped[identity[0]].append(task)
    expected = set(itertools.product(SCENES, VARIANTS, REPEATS))
    scenes = [analyze_scene(key, grouped[key]) for key in SCENES if grouped[key]]
    return {'schema_version': 1, 'matrix': str(matrix_path), 'matrix_sha256': sha(matrix_path),
            'analyzer_sha256': sha(Path(__file__)), 'expected_runs': 36, 'observed_runs': len(identities),
            'missing_runs': [dict(zip(('scene_key', 'variant', 'repeat'), row)) for row in sorted(expected - identities)],
            'all_36_runs_complete_finite': identities == expected
                and all(r['complete_finite'] for s in scenes for r in s['runs']),
            'physical_certified': False, 'performance_certified': False,
            'accepted_path_ccd': {'checked_by_this_tool': False, 'status': 'separate diagnostic required'},
            'velocity': {'analyzed': False, 'reconstructed': False},
            'comparison_policy': 'Reference envelopes are formed solely from each reference arm, without candidates. '
                'Pointwise observed bounds are exact: no tolerance is added. Tiny excesses remain observations. '
                'Position differences describe trajectories, not errors against physical truth. '
                'Shared-desktop times are diagnostic; ratios do not certify equal physical quality or exclusive GPU load.',
            'linear_stage_scope': 'Existing stats phase_ms.pcg is the inclusive whole linear stage: matrix conversion, '
                'preconditioner preparation, PCG and solution dispatch; it is not an individual kernel or pure PCG-loop time. '
                'No new tracing was enabled. Missing data and all-zero phase placeholders are unavailable/null, not measured zero cost. '
                'Phase sums need not exactly equal solver wall time.',
            'scenes': scenes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='reports/active/CLOTH_RESTRICT_TIMING_BATCH.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    report = analyze(ROOT / args.matrix)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(target), 'observed_runs': report['observed_runs'],
                      'all_36_runs_complete_finite': report['all_36_runs_complete_finite'],
                      'scenes': [{'scene': s['scene'], 'time_ratios': [
                          {'numerator': r['numerator_variant'], **r['summary']} for r in s['paired_time_ratios']]}
                          for s in report['scenes']]}), flush=True)
