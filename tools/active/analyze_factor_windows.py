"""CPU-only analysis of 27 continuous factor-action window runs.

Reuses the common geometry analyzer. Run only after GPU timing completes.
The original Stiff and triangular-TOI envelopes are formed independently.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from analyze import analyze as analyze_geometry
from config import ROOT, digest, expand, read, sha, matches_requested
from validate_run import validate


SCENES = {'hang': ('cloth_hang_l', 21),
          'fixed_bunny': ('cloth_fixed_bunny_l', 45),
          'mixed': ('bunny_cloth_bunny_l', 35)}
VARIANTS = ('stiff', 'triangular', 'factor_inverse')
REPEATS = (1, 2, 3)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def summary_range(values):
    values = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), 'Nonfinite summary input')
    return {'count': len(values), 'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def check_run(task):
    run = ROOT / 'runs/active' / task['name']
    request = read(run / 'requested.json')
    config = request['expanded_config']
    result = read(run / 'result.json')
    declared = read(ROOT / task['config']) if isinstance(task['config'], str) else task['config']
    require(matches_requested(config,declared), f'Plan/request configuration mismatch: {run.name}')
    require(request['config_sha256'] == digest(config), f'Request digest mismatch: {run.name}')
    require(result == task['result'], f'Batch/result mismatch: {run.name}')
    scene_name, expected_frames = SCENES[task['scene_key']]
    constants = {'scene': scene_name, 'steps': expected_frames, 'dt': .01,
                 'ipc_newton_tol': .01, 'pcg_rho_tol': 1e-4,
                 'toi_remaining_fraction_tol': .01, 'toi_trial_velocity_tol': .05,
                 'diagnostics': [], 'profile': 'none', 'trace_velocity': False,
                 'fixed_factor_study': False, 'fixed_restrict_study': False}
    if task['variant'] == 'stiff':
        constants.update(backend='ipc', execution='host', mas='legacy', mas_restrict='serial',
                         mas_factor_action='triangular', refit=False, batch=False, reuse=False)
        expected_binary = 'base'
    else:
        constants.update(backend='toi_al', execution='conditional_graph', mas='cholesky',
                         mas_restrict='warp', mas_factor_action=task['variant'],
                         refit=True, batch=True, reuse=True, choose_start=True,
                         restart_guard=True, mu_mode='diagonal')
        expected_binary = 'active'
    for key, expected in constants.items():
        require(config[key] == expected, f'Unexpected {key} for {run.name}: {config[key]}')
    require(request['binary'] == task['binary'] == expected_binary, f'Binary mismatch: {run.name}')
    require(result['status'] == 'completed' and result['recorded_frames'] == expected_frames
            and result.get('finite') is True, f'Incomplete/nonfinite run: {run.name}')
    frames = read(run / 'output/stats.json')['frames']
    with (run / 'trace/frames.csv').open() as stream:
        timing = list(csv.DictReader(stream))
    require(len(frames) == len(timing) == expected_frames, f'Common complete frame count failed: {run.name}')
    require(all((run / 'trace' / f'state_{i:04d}.bin').exists() for i in range(expected_frames + 1)),
            f'Missing continuous trace frame: {run.name}')
    require(not any((run / 'trace').glob('velocity_*.bin')),
            f'Unexpected velocity exports in timing run: {run.name}; no velocity analysis was declared')
    scene = read(run / 'output/scene.json')
    for field, wanted in [('dt', 'dt'), ('newton_tol', 'ipc_newton_tol'), ('pcg_tol', 'pcg_rho_tol')]:
        require(scene['effective_run'][field] == config[wanted], f'Effective {field} mismatch: {run.name}')
    require(scene['effective_scalar_fields']['preconditioner_type'] == 1, f'Not MAS: {run.name}')
    manifest_path = run / 'build_manifest.json'
    require(sha(manifest_path) == request['manifest_sha256'], f'Build manifest mismatch: {run.name}')
    manifest = read(manifest_path)
    require(manifest['source_digest'] == request['source_digest'], f'Source identity mismatch: {run.name}')
    require(request['exe_sha256'] in {item['sha256'] for item in manifest['binaries'].values()},
            f'Executable identity not in saved build manifest: {run.name}')
    if expected_binary == 'active':
        resolved_checks = validate(run)
        require(resolved_checks['passed'], f'Requested/resolved/observed mismatch: {run.name}')
        resolved = read(run / 'resolved_config.json')
        require(resolved['mas'].get('factor_action') == task['variant'], f'Unconfirmed factor action: {run.name}')
        require(resolved.get('effective_preconditioner') == 'mas' and resolved['mas'].get('active') is True,
                f'Unconfirmed active MAS: {run.name}')
    else:
        resolved_checks = {'passed': True, 'resolved_contract_available': False,
                           'scope': 'Frozen Stiff: saved manifest, request, scene effective values and complete trace; no resolved interface.'}
    return {'name': run.name, 'path': str(run), 'variant': task['variant'], 'repeat': task['repeat'],
            'requested_sha256': sha(run / 'requested.json'), 'exe_sha256': request['exe_sha256'],
            'build_manifest_sha256': sha(manifest_path), 'configuration_checks': resolved_checks}


def group_position_gate(report, references, candidates):
    own = [p for p in report['comparisons'] if p['a'] in references and p['b'] in references]
    require(len(own) == 3, 'Three reference runs must produce three repeat pairs')
    keys = [key for key in own[0]['frames'][0] if key.startswith('position_')]
    count = len(report['baseline_frame_envelope'])
    envelope = [{'frame': i + 1, **{key: max(pair['frames'][i][key] for pair in own) for key in keys}}
                for i in range(count)]
    observations = []
    for candidate in candidates:
        pairs = [p for p in report['comparisons'] if
                 (p['a'] == candidate and p['b'] in references) or
                 (p['b'] == candidate and p['a'] in references)]
        require(len(pairs) == 3, f'Incomplete candidate/reference pairs: {candidate}')
        frames = []
        for i, bounds in enumerate(envelope):
            entry = {'frame': i + 1}
            for key in keys:
                values = [pair['frames'][i][key] for pair in pairs]
                entry[key] = {'range_against_three_references': [min(values), max(values)],
                              'reference_repeat_max': bounds[key],
                              'outside_reference_repeat': max(values) > bounds[key]}
            frames.append(entry)
        observations.append({'run': candidate,
                             'outside_reference_repeat_frames': {
                                 key: [f['frame'] for f in frames if f[key]['outside_reference_repeat']] for key in keys},
                             'max_rms_against_references_m': {
                                 key: max(pair['maxima'][key] for pair in pairs) for key in keys},
                             'frames': frames, 'quality_certified': False})
    return {'reference_names': list(references), 'group_metrics': keys,
            'reference_repeat_frames': envelope,
            'reference_repeat_global_maxima_m': {key: max(e[key] for e in envelope) for key in keys},
            'candidate_observations': observations,
            'scope': 'Mass-weighted positions by cloth/FEM/ABD group. Trajectory differences are not physical truth errors. '
                     'Strict observed repeat ranges; no added position tolerance and no reconstructed velocities.'}


def summarize_scene(key, tasks):
    scene, expected_frames = SCENES[key]
    tasks = sorted(tasks, key=lambda task: (VARIANTS.index(task['variant']), task['repeat']))
    identities = [check_run(task) for task in tasks]
    by_name = {r['name']: r for r in identities}
    grouped = {variant: [r for r in identities if r['variant'] == variant] for variant in VARIANTS}
    paths = [Path(r['path']) for r in identities]
    # The common analyzer checks initial coordinates, topology, masses, boundary
    # types and the full scene. Include body ids as an additional identity check.
    body_hash = sha(paths[0] / 'trace/body_ids.bin')
    require(all(sha(p / 'trace/body_ids.bin') == body_hash for p in paths), f'Body mapping mismatch: {key}')
    against_stiff = analyze_geometry(paths, 3)
    triangular_paths = [Path(r['path']) for variant in ('triangular', 'factor_inverse') for r in grouped[variant]]
    against_triangular = analyze_geometry(triangular_paths, 3)
    require(all(len(row['frames']) == expected_frames for row in against_stiff['runs']),
            f'Unexpected analyzed frame coverage: {key}')
    require(all(len(row['frames']) == expected_frames for row in against_triangular['runs']),
            f'Unexpected paired frame coverage: {key}')
    runs = []
    for row in against_stiff['runs']:
        frames = row['frames']
        seconds = sum(f['seconds'] for f in frames)
        require(math.isclose(seconds, row['result']['solver_seconds'], rel_tol=1e-12, abs_tol=1e-12),
                f'Timing sum mismatch: {row["name"]}')
        runs.append(row | by_name[row['name']] | {
            'solver_seconds': seconds, 'mean_frame_ms': seconds * 1000 / expected_frames,
            'direction_count': sum(f['pcg_calls'] for f in frames),
            'pcg_iterations': sum(f['pcg_iterations'] for f in frames),
            'pcg_limit_or_breakdown_hits': sum(f['pcg_limit_hits'] for f in frames),
            'outer_count': sum(f['outers'] for f in frames),
            'linear_stage_seconds': sum(f['linear_stage_seconds'] for f in frames)
                if all(f['linear_stage_seconds'] is not None for f in frames) else None,
            'max_cloth_stretch': max(f['cloth_max_stretch'] for f in frames),
            'peak_cloth_stretch_frame': max(frames, key=lambda f: f['cloth_max_stretch'])['frame'],
            'min_fem_J': min(f['fem_min_J'] for f in frames),
            'max_fem_nonpositive': max(f['fem_nonpositive'] for f in frames),
            'max_fem_negative_volume': max(f['fem_negative_volume'] for f in frames),
            'min_abd_J': min(f['abd_min_J'] for f in frames),
            'max_fixed_drift_m': max(f['fixed_max_drift'] for f in frames)})
    summary_keys = ('solver_seconds', 'mean_frame_ms', 'direction_count', 'pcg_iterations',
                    'outer_count', 'linear_stage_seconds', 'max_cloth_stretch', 'min_fem_J',
                    'max_fem_nonpositive', 'max_fem_negative_volume', 'min_abd_J', 'max_fixed_drift_m')
    summaries = [{'variant': variant,
                  **{metric: summary_range(r[metric] for r in runs if r['variant'] == variant) for metric in summary_keys},
                  'pcg_limit_or_breakdown_hits': sum(r['pcg_limit_or_breakdown_hits'] for r in runs if r['variant'] == variant)}
                 for variant in VARIANTS]
    lookup = {(r['variant'], r['repeat']): r for r in runs}
    ratios = []
    for numerator in ('stiff', 'triangular'):
        pairs = []
        for repeat in REPEATS:
            old, new = lookup[numerator, repeat], lookup['factor_inverse', repeat]
            pairs.append({'repeat': repeat, 'numerator_run': old['name'], 'denominator_run': new['name'],
                          'numerator_seconds': old['solver_seconds'], 'denominator_seconds': new['solver_seconds'],
                          'ratio': old['solver_seconds'] / new['solver_seconds']})
        ratios.append({'numerator_variant': numerator, 'denominator_variant': 'factor_inverse',
                       'pairs': pairs, 'summary': summary_range(p['ratio'] for p in pairs)})
    comparisons = {}
    for reference, report in [('stiff', against_stiff), ('triangular', against_triangular)]:
        references = [r['name'] for r in grouped[reference]]
        candidate_variants = ('triangular', 'factor_inverse') if reference == 'stiff' else ('factor_inverse',)
        candidates = [r['name'] for variant in candidate_variants for r in grouped[variant]]
        comparisons[reference] = {
            'reference_names': references, 'baseline_frame_envelope': report['baseline_frame_envelope'],
            'candidate_gates': [g for g in report['candidate_gates'] if g['run'] in candidates],
            'position_repeat_comparison': group_position_gate(report, references, candidates)}
    return {'scene_key': key, 'scene': scene, 'dt': .01, 'frames': expected_frames,
            'common_complete_frame_count_asserted': True,
            'same_initial_state_scene_materials_topology': against_stiff['same_initial_state_scene_topology'],
            'scales_m': against_stiff['scales_m'], 'body_ids_sha256': body_hash,
            'summary': summaries, 'paired_time_ratios': ratios, 'runs': runs,
            'quality_against_reference': comparisons,
            'absent_group_policy': 'For pure cloth, generic FEM/ABD J=1 is an absent-group placeholder; '
                                   'only groups present in scales_m have position comparisons. No FEM-material claim is inferred.'}


def analyze_matrix(matrix_path):
    matrix = read(matrix_path)
    expected = set(itertools.product(SCENES, VARIANTS, REPEATS))
    actual, grouped = set(), collections.defaultdict(list)
    for task in matrix['runs']:
        identity = (task['scene_key'], task['variant'], task['repeat'])
        require(identity in expected and identity not in actual, f'Unexpected/duplicate matrix entry: {identity}')
        actual.add(identity)
        grouped[task['scene_key']].append(task)
    require(actual == expected, f'Expected 27 runs; missing entries: {sorted(expected - actual)}')
    scenes = [summarize_scene(key, grouped[key]) for key in SCENES]
    return {'schema_version': 1, 'matrix': str(matrix_path), 'matrix_sha256': sha(matrix_path),
            'analyzer_sha256': sha(Path(__file__)), 'common_analyzer_sha256': sha(ROOT / 'tools/active/analyze.py'),
            'runs_checked': 27, 'physical_certified': False, 'performance_certified': False,
            'velocity_analyzed_or_reconstructed': False, 'accepted_path_ccd_checked_here': False,
            'timing_scope': 'Shared desktop diagnostic solver time. Same-repeat ratios >1 mean factor_inverse is faster. '
                            'TOI all-zero ordinary phase timings are unavailable; no per-kernel claim is made.',
            'quality_scope': 'Separate three-run Stiff and triangular-TOI envelopes. Common analyzer scalar checks retain '
                'their existing 1e-12 numerical comparison floor; no threshold is expanded here. Position repeat ranges '
                'have no added tolerance. Neither endpoint observations nor trajectory comparisons certify full physical quality.',
            'scenes': scenes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='reports/active/FACTOR_WINDOW_BATCH.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    report = analyze_matrix(ROOT / args.matrix)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(target), 'runs_checked': report['runs_checked'],
                      'scenes': [{'scene': s['scene'], 'ratios': [
                          {'numerator': r['numerator_variant'], **r['summary']} for r in s['paired_time_ratios']]}
                          for s in report['scenes']]}), flush=True)
