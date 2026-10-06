"""Offline paired uncertainty and state coverage; no additional simulation."""
import itertools
import json
import math
import numpy as np

from config import ROOT, read, sha
from component_tuning import TAG, state_comparison
from ipc_benchmark import write


def main():
    analysis_path = ROOT / f'reports/active/{TAG}_analysis.json'
    analysis = read(analysis_path)
    summary = analysis['full_summary']
    speed = {}
    for base in ('stiff', 'graph', 'combined'):
        values = {s: summary[s]['bvh']['paired'][base]['values'] for s in ('hang', 'fixed_bunny')}
        assert all(len(v) == 3 for v in values.values())
        medians = {s: float(np.median(v)) for s, v in values.items()}
        samples = [math.sqrt(math.prod(float(np.median([v[i] for i in indices])) for v in values.values()))
                   for indices in itertools.product(range(3), repeat=3)]
        speed[base] = {'ratios': values, 'paired_median_by_scene': medians,
                       'geometric_mean': math.sqrt(math.prod(medians.values())),
                       'one_sided_95_bootstrap_lower': float(np.quantile(samples, .05)),
                       'bootstrap_scope': 'Exact 27 resamplings of three whole paired rounds, both scenes clustered by round; diagnostic small sample only'}
    batch = read(ROOT / f'reports/active/{TAG}_full_batch.json')
    repeats = []
    for scene in ('hang', 'fixed_bunny'):
        for arm in ('stiff', 'graph', 'combined', 'bvh'):
            rows = [r for r in batch['runs'] if r['scene_key'] == scene and r['arm'] == arm]
            for a, b in itertools.combinations(rows, 2):
                repeats.append({'scene': scene, 'arm': arm, 'run': a['name'], 'reference': b['name'],
                                'state': state_comparison(ROOT / 'runs/active' / a['name'], ROOT / 'runs/active' / b['name'])})
    guard = read(ROOT / f'reports/active/{TAG}_guards_batch.json')
    # independently recompute ordinary periodic counters from chronological stats
    periodic = []
    for stage in ('guards', 'screen', 'full'):
        source = read(ROOT / f'reports/active/{TAG}_{stage}_batch.json')
        for row in source['runs']:
            cfg = read(ROOT / 'runs/active' / row['name'] / 'requested.json')['expanded_config']
            if not cfg['discrete_bvh_refit'] or cfg['discrete_bvh_validate']:
                continue
            frames = read(ROOT / 'runs/active' / row['name'] / 'output/stats.json')['frames']
            for kind in ('face', 'edge'):
                total = 0
                for frame in frames:
                    c = frame['discrete_bvh'][kind]
                    assert c['interval_rebuilds'] == (total+c['construct_calls'])//8-total//8
                    assert c['production_rebuilds'] == c['interval_rebuilds']
                    assert c['signature_rebuilds'] == c['swept_rebuilds'] == 0
                    total += c['construct_calls']
                periodic.append({'run': row['name'], 'kind': kind, 'construct_calls': total, 'passed': True})
    report = {'analysis_sha256': sha(analysis_path), 'speed': speed, 'within_arm_repeated_state': repeats,
              'periodic_checks': periodic,
              'material_pass_counts': {s: {a: r['material_100f_passed'] for a, r in arms.items()} for s, arms in summary.items()},
              'guard_capacity_bytes': {r['scene_key']+'_'+r['repeat']: sum(c['swept_cache_capacity_bytes_peak'] for c in r['checks']['bvh_counters'].values()) for r in guard['runs']},
              'speed_only_stage_geomean_over_1_10': speed['graph']['geometric_mean'] >= 1.10,
              'speed_only_stage_bootstrap_lower_over_1': speed['graph']['one_sided_95_bootstrap_lower'] > 1,
              'baseline_material_protocol_available': all(r['material_100f_passed'] == 3 for arms in summary.values() for a,r in arms.items() if a=='stiff'),
              'stiff_actual_velocity_export_available': False, 'new_acceptance_CPU_CCD_this_round': False,
              'dynamic_topology_mapping_address_invalidation_GPU_covered': False,
              'performance_certified': False, 'quality_certified': False, 'default_promoted': False}
    write(ROOT / f'reports/active/{TAG}_evidence.json', report)
    print(json.dumps({'speed': speed, 'material_pass_counts': report['material_pass_counts'],
                      'periodic_run_tree_checks': len(periodic), 'baseline_material_protocol_available': report['baseline_material_protocol_available']}))


if __name__ == '__main__':
    main()
