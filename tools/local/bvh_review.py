"""Read-only review of exactly six sealed BVH stages; never launches a solver."""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
import bvh_rounds as controller
from config import read
from linux_runner import require, sha
from pool_metrics import state_comparison
from quality_analysis import input_comparison


def review(session, seal_path):
    folders = [session / f'{scene}_r{r}' for scene in ('fixed', 'hang') for r in (1, 2, 3)]
    require(set(session.iterdir()) == set(folders), 'Expected exactly six completed stages')
    records = {}
    for f in folders:
        controller.verify_stage(f, sha(seal_path), controller.code_identity())
        records[f.name] = read(f / 'analysis.json')
    scenes = {}
    for scene in ('fixed', 'hang'):
        stages = [records[f'{scene}_r{r}'] for r in (1, 2, 3)]
        refs = [next(x for x in s['runs'] if x['variant'] == 'D1S1') for s in stages]
        arms = {}
        for arm in ('D1S1', 'D0S1', 'D0S0', 'D1S0'):
            rows = [next(x for x in s['runs'] if x['variant'] == arm) for s in stages]
            times = [x['timing']['prefix']['solver_ms'] for x in rows]
            ratios = [ref['timing']['prefix']['solver_ms'] / t for ref, t in zip(refs, times)]
            phase_delta = {}
            for key in ('assembly', 'pcg', 'ccd', 'line_search', 'state_update', 'unclassified'):
                def value(row):
                    p = row['timing']['prefix']
                    return p['unclassified_ms'] if key == 'unclassified' else p['phase_ms'][key]
                phase_delta[key] = [value(x) - value(ref) for x, ref in zip(rows, refs)]
            cross = []
            if arm != 'D1S1':
                for s, row, ref in zip(stages, rows, refs):
                    cross.append(next(p['state_difference'] for p in s['comparisons']
                                      if {p['left'], p['right']} == {row['name'], ref['name']}))
            median = statistics.median(ratios)
            eligible = (arm != 'D1S1' and 1 - 1 / median >= .05 and
                        sum(r > 1 for r in ratios) >= 2 and
                        all(x['within_original_bounds'] for x in rows + refs))
            arms[arm] = {'solver_ms': times, 'paired_speedups': ratios,
                         'median_speedup': median, 'median_net_time_saving': 1 - 1 / median,
                         'faster_pairs': sum(r > 1 for r in ratios),
                         'directions': [x['timing']['prefix']['directions'] for x in rows],
                         'pcg_iterations': [x['timing']['prefix']['pcg_iterations'] for x in rows],
                         'material_passes': [x['within_original_bounds'] for x in rows],
                         'phase_delta_ms_candidate_minus_reference': phase_delta,
                         'cross_arm_state_difference': cross,
                         'eligible_for_further_quality_checks': eligible,
                         'retained_as_optimization': False}
        repeats = []
        for a, b in itertools.combinations((1, 2, 3), 2):
            left = session / f'{scene}_r{a}' / f'{scene}_D1S1_r{a}'
            right = session / f'{scene}_r{b}' / f'{scene}_D1S1_r{b}'
            inputs = input_comparison(left, right)
            require(inputs['passed'], 'Reference repeats have different initial inputs')
            repeats.append({'rounds': [a, b], 'initial_inputs_equal': True,
                            'state_difference': state_comparison(left, right)})
        scenes[scene] = {'frames': controller.plan()['scenes'][scene], 'arms': arms,
                         'reference_self_repeat_differences': repeats}
    eligible = [arm for arm in ('D0S1', 'D0S0', 'D1S0') if all(
        scenes[s]['arms'][arm]['eligible_for_further_quality_checks'] for s in scenes)]
    return {'schema': 'bvh_four_arm_review.v1', 'complete_runs': 24,
            'stage_analysis_sha256': {f.name: sha(f / 'analysis.json') for f in folders},
            'scenes': scenes,
            'two_scene_median_speedup_geometric_mean': {
                a: math.sqrt(scenes['fixed']['arms'][a]['median_speedup'] *
                             scenes['hang']['arms'][a]['median_speedup'])
                for a in ('D0S1', 'D0S0', 'D1S0')},
            'decision': ('Further independent quality checks needed: ' + ','.join(eligible) if eligible else
                         'Keep D1S1; stop BVH combination branch without interval grid'),
            'quality_certified': False, 'performance_certified': False,
            'scope': 'Shared WDDM, 59/51 frames from zero, no new Stiff denominator. '
                     'Old material bounds unchanged; repeat ranges are observations, not certification. '
                     'Phase counters omit separately labelled exit assembly; unclassified includes it.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--seal', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = review(a.session.resolve(), a.seal.resolve())
    with a.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({'complete_runs': result['complete_runs'],
                      'scenes': {s: {arm: {k: row[k] for k in
                         ('median_speedup', 'median_net_time_saving', 'material_passes')}
                         for arm, row in data['arms'].items()} for s, data in result['scenes'].items()},
                      'decision': result['decision']}, ensure_ascii=False))


if __name__ == '__main__': main()
