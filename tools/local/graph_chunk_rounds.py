"""One reviewed fixed-system Graph probe at a time; no automatic continuation."""
import argparse
import json
import math
import statistics
from pathlib import Path
import itertools
import numpy as np
import bvh_rounds
from local_plan import ROOT
from config import expand, read
from linux_runner import child, require, sha, write_new
from local_identity import verify_seal
from windows_runner import execute, gpu_lock
from validate_run import validate

STAGES = {'short': ('fixed', 2), 'long': ('fixed', 57), 'hang': ('hang', 41),
          'long_retry': ('fixed', 57)}
RESTORED = ('full_system_restored', 'primary_restored_bitwise', 'workspace_restored',
            'production_graph_restored', 'operator_signature_restored',
            'production_info_restored', 'solve_context_restored', 'config_restored',
            'cost_trace_restored')


def task(stage):
    scene, frame = STAGES[stage]
    c = bvh_rounds.tasks(scene, 1)[0]['config'].copy()
    c.update(steps=frame, diagnostics=['fixed'], fixed_frames=str(frame),
             fixed_directions='1', fixed_graph_chunk_study=True, pcg_graph_chunk=1)
    return {'name': stage + '_probe', 'binary': 'active', 'variant': 'K1_private_K4_probe',
            'scene_key': scene, 'repeat': 1, 'config': expand(c)}


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def analyze(study):
    require(study['schema'] == 'fixed_graph_chunk_v1' and study['passed'], 'Native probe failed')
    require(all(study[k] is True for k in RESTORED), 'Production restoration failed')
    require(study['restoration_error'] == '' and study['warmup_pairs'] == 2 and
            study['measured_pairs'] == 7, 'Probe budget/restore contract failed')
    runs = study['runs']
    require(len(runs) == 18, 'Missing or additional solves')
    expected = [(warm, i, j + 1, k) for warm, count in ((True, 2), (False, 7))
                for i in range(1, count + 1)
                for j, k in enumerate((1, 4) if i % 2 else (4, 1))]
    actual = [(r['warmup'], r['pair'], r['order'], r['chunk']) for r in runs]
    require(actual == expected, 'Probe order is not the declared paired order')
    require(all(r['passed'] and r['finite'] and not r['pcg'].get('iteration_limit', False)
                and not r['pcg'].get('breakdown', False) for r in runs), 'Invalid solve')
    require(all(finite_positive(r[k]) for r in runs for k in ('solve_host_ms', 'solve_event_ms')),
            'Invalid complete-solve time')
    require(all(r['cache_hit'] == (r['arm_use'] != 1) for r in runs), 'Unexpected recapture')
    pairs = []
    measured = [r for r in runs if not r['warmup']]
    for i in range(1, 8):
        arms = {r['chunk']: r for r in measured if r['pair'] == i}
        a, b = arms[1], arms[4]
        pairs.append({'pair': i, 'host_speedup': a['solve_host_ms'] / b['solve_host_ms'],
                      'event_speedup': a['solve_event_ms'] / b['solve_event_ms'],
                      'K1_iterations': a['iterations'], 'K4_iterations': b['iterations'],
                      'K4_tail_steps': b['pcg']['graph_inactive_tail_steps'],
                      'K1_residual': a['true_relative_residual'],
                      'K4_residual': b['true_relative_residual']})
    host = statistics.median(r['host_speedup'] for r in pairs)
    event = statistics.median(r['event_speedup'] for r in pairs)
    matched = all(r['K1_iterations'] == r['K4_iterations'] for r in pairs)
    saving = 1 - 1 / host
    return {'correctness_contract_passed': True, 'pairs': pairs,
            'median_complete_solve_host_speedup': host,
            'median_complete_solve_event_speedup': event,
            'median_complete_solve_host_time_saving': saving,
            'all_paired_iterations_equal': matched,
            'local_performance_gate': matched and saving >= .15 and event > 1,
            'quality_certified': False, 'performance_certified': False,
            'shared_preparation_cost_remeasured': False,
            'scope': 'Frozen A/b/M PCG replay. Matrix conversion and M preparation are shared prior work; final readback includes GPU waiting, not extra transfer-only cost.'}


def code_identity():
    return {n: sha(Path(__file__).parent / n) for n in ('graph_chunk_rounds.py', 'graph_chunk_rounds_test.py')}


def review_resource_failure(session, reviewed):
    failed=session/'long'/'analysis.json'
    require(reviewed == sha(failed) and not read(failed)['diagnosis_complete'],
            'Explicit resource-failure review required')
    require(read(session/'long'/'long_probe'/'result.json')['status']=='memory_budget',
            'Only the one reviewed memory-blocked probe can use this recovery')


def numerical_comparison(study, out):
    """Observed K1 repeat variation is diagnostic, never a relaxed quality gate."""
    solutions = {1: {}, 4: {}}
    for r in study['runs']:
        if r['warmup']: continue
        p = Path(r['solution_file']).resolve()
        require(p.is_relative_to(out.resolve()) and not p.is_symlink(), 'Solution outside run')
        raw = p.read_bytes()
        ident = r['solution_identity']
        require(len(raw) == ident['bytes'] == study['system']['dofs'] * 8, 'Wrong solution size')
        value = 14695981039346656037
        for byte in raw: value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
        require(value == ident['fnv1a64'], 'Solution content changed')
        v = np.frombuffer(raw, dtype='<f8')
        require(np.isfinite(v).all(), 'Nonfinite saved solution')
        solutions[r['chunk']][r['pair']] = v
    def relative(a, b):
        norm = np.linalg.norm(b)
        return float(np.linalg.norm(a-b) / norm if norm else np.linalg.norm(a-b))
    k1 = list(solutions[1].values())
    self_range = max(relative(a, b) for a, b in itertools.combinations(k1, 2))
    pairs = [relative(solutions[4][i], solutions[1][i]) for i in range(1, 8)]
    return {'K1_observed_max_relative_repeat_difference': self_range,
            'paired_K4_K1_relative_solution_difference': pairs,
            'within_observed_K1_range_plus_roundoff': max(pairs) <= self_range + 1e-12,
            'scope': 'Frozen-system numerical comparison only; observed repeat range is not a material/trajectory quality certificate.'}


def run(seal_path, session, stage, reviewed, resource_review=None):
    predecessor = {'short': None, 'long': 'short', 'hang': 'long', 'long_retry': 'short'}[stage]
    tools = code_identity()
    with gpu_lock(ROOT):
        seal = verify_seal(ROOT, seal_path)
        if stage == 'long_retry':
            review_resource_failure(session,resource_review)
        else:
            require(resource_review is None,'Resource review belongs only to long_retry')
        if predecessor:
            p = session / predecessor / 'analysis.json'
            require(reviewed == sha(p), 'Explicit previous analysis SHA required')
            previous = read(p)
            require(previous['diagnosis_complete'], 'Previous correctness/resource failure blocks continuation')
            if stage == 'hang':
                require(previous['probe']['local_performance_gate'], 'Long-system performance gate did not pass')
        else:
            require(reviewed is None, 'First probe has no predecessor')
        folder = session / stage
        folder.mkdir(parents=True, exist_ok=False)
        t = task(stage)
        write_new(folder / 'started.json', {'task': t, 'seal_sha256': sha(seal_path),
                  'tools': tools, 'reviewed_previous_sha256': reviewed,
                  'reviewed_resource_failure_sha256': resource_review})
        result = execute(folder, t, seal['programs']['active'], 0)
        analysis = {'stage': stage, 'diagnosis_complete': False, 'errors': [],
                    'quality_certified': False, 'performance_certified': False,
                    'automatic_next_round': False, 'whole_scene_timing_usable': False}
        try:
            require(result['status'] == 'completed' and result['recorded_frames'] == t['config']['steps']
                    and bvh_rounds.within_time_budget(result), 'Incomplete/resource/deadline failure')
            out = folder / t['name']
            require(validate(out)['passed'], 'Resolved configuration contract failed')
            frame = t['config']['steps']
            study = read(out / 'fixed' / f'f{frame}_n1_graph_chunk_study.json')
            analysis['probe'] = analyze(study)
            analysis['numerical'] = numerical_comparison(study, out)
            analysis['probe']['local_performance_gate'] &= analysis['numerical']['within_observed_K1_range_plus_roundoff']
            analysis['diagnosis_complete'] = True
        except (ValueError, KeyError, FileNotFoundError) as error:
            analysis['errors'].append(type(error).__name__ + ': ' + str(error))
        verify_seal(ROOT, seal_path)
        require(tools == code_identity(), 'Probe tools changed during execution')
        write_new(folder / 'analysis.json', analysis)
        write_new(folder / 'receipt.json', {'seal_sha256': sha(seal_path), 'tools': tools,
                  'analysis_sha256': sha(folder / 'analysis.json'),
                  'evidence_sha256': sha(folder / t['name'] / 'evidence.json')})
        print(json.dumps(analysis | {'analysis_sha256': sha(folder / 'analysis.json')}, ensure_ascii=False))
        return int(not analysis['diagnosis_complete'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--reviewed-previous-sha')
    p.add_argument('--reviewed-resource-sha')
    a = p.parse_args()
    session = child(ROOT, a.session)
    require(session.is_relative_to(ROOT / 'runs') and session != ROOT / 'runs', 'Session below runs/ required')
    return run(child(ROOT, a.seal), session, a.stage, a.reviewed_previous_sha, a.reviewed_resource_sha)


if __name__ == '__main__':
    raise SystemExit(main())
