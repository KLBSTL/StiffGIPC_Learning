"""Summarize sampled AutoDL TOI full-step energy components."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(rows):
    names = ('kinetic', 'abd_shape', 'fem', 'cloth', 'bending',
             'constraint', 'al_contact', 'friction')
    deformation = names[1:6]
    result = {'count': len(rows)}
    if not rows:
        return result
    result['median_delta'] = {
        name: statistics.median(n['full_step_component_delta'][name] for _, n in rows)
        for name in names
    }
    result['positive_count'] = {
        name: sum(n['full_step_component_delta'][name] > 0 for _, n in rows)
        for name in names
    }
    result['largest_positive_deformation'] = dict(collections.Counter(
        max(deformation, key=lambda name: n['full_step_component_delta'][name])
        for _, n in rows
    ))
    result['largest_positive_all'] = dict(collections.Counter(
        max(names, key=lambda name: n['full_step_component_delta'][name])
        for _, n in rows
    ))
    result['full_step_energy_positive'] = sum(
        n['finite_step_energy_audit'][-1]['total_delta'] > 0 for _, n in rows
    )
    return result


def summarize_triangles(rows):
    pairs = [(n, n['triangle_energy_audit']) for _, n in rows
             if 'triangle_energy_audit' in n]
    audits = [audit for _, audit in pairs]
    result = {'count': len(audits)}
    if not audits:
        return result
    for key in ('positive_triangles', 'top1_positive_fraction',
                'top10_positive_fraction', 'half_positive_count',
                'ninety_positive_count', 'max_tensile_axis_strain_before',
                'max_tensile_axis_strain_full'):
        result[f'median_{key}'] = statistics.median(a[key] for a in audits)
    result['max_triangle_repeats'] = collections.Counter(
        a['max_delta_triangle'] for a in audits).most_common(10)
    result['max_net_sum_error'] = max(abs(a['net_delta'] - n['full_step_component_delta']['cloth'])
                                  for n, a in pairs)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stats', type=Path)
    parser.add_argument('--hot-from', type=int, default=70)
    args = parser.parse_args()
    frames = json.loads(args.stats.read_text(encoding='utf-8'))['frames']
    all_newton = [n for frame in frames for n in frame['newton']]
    sampled = [(i + 1, n) for i, frame in enumerate(frames)
               for n in frame['newton'] if 'full_step_component_delta' in n]
    backtracked = [(i, n) for i, n in sampled if n['line_search_r'] < 1]
    full_step = [(i, n) for i, n in sampled if n['line_search_r'] == 1]
    cloth_largest = [(i, n) for i, n in backtracked
                     if max(n['full_step_component_delta'],
                            key=n['full_step_component_delta'].get) == 'cloth']
    result = {
        'frames': len(frames),
        'all_newton': len(all_newton),
        'all_backtracked': sum(n.get('line_search_r', 1) < 1 for n in all_newton),
        'sampled': summarize(sampled),
        'backtracked': summarize(backtracked),
        'backtracked_hot': summarize([(i, n) for i, n in backtracked if i >= args.hot_from]),
        'triangle_backtracked': summarize_triangles(backtracked),
        'triangle_full_step_accepted': summarize_triangles(full_step),
        'triangle_cloth_largest': summarize_triangles(cloth_largest),
        'backtracked_frame_counts': dict(sorted(collections.Counter(i for i, _ in backtracked).items())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
