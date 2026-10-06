"""Inspect an identified PT/EE singularity along exported accepted safe states."""
import argparse
import json
from pathlib import Path

import numpy as np


def closest_pt(p, a, b, c):
    ab, ac = b - a, c - a
    d1, d2 = np.dot(ab, p - a), np.dot(ac, p - a)
    if d1 <= 0 and d2 <= 0:
        return a, (0.0, 0.0)
    d3, d4 = np.dot(ab, p - b), np.dot(ac, p - b)
    if d3 >= 0 and d4 <= d3:
        return b, (1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        u = d1 / (d1 - d3)
        return a + u * ab, (u, 0.0)
    d5, d6 = np.dot(ab, p - c), np.dot(ac, p - c)
    if d6 >= 0 and d5 <= d6:
        return c, (0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        v = d2 / (d2 - d6)
        return a + v * ac, (0.0, v)
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + v * (c - b), (1 - v, v)
    denom = va + vb + vc
    u, v = vb / denom, vc / denom
    return a + u * ab + v * ac, (u, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lines = (args.run / 'run.log').read_text(errors='replace').splitlines()
    marker = 'TOI distance linearization is singular at a safe contact: '
    failures = [json.loads(line.split(marker, 1)[1]) for line in lines if marker in line]
    if len(failures) != 1:
        raise ValueError(f'Expected one indexed singularity, found {len(failures)}')
    failure = failures[0]
    if failure['kind'] != 0:
        raise NotImplementedError('The current path audit supports PT contacts')
    ids = failure['ids']
    frame = failure['frame']
    states = sorted((args.run / 'trace/substeps').glob(f'safe_{frame:04d}_*.bin'))
    if not states:
        raise FileNotFoundError('Accepted safe substep traces are missing')
    bodies = np.fromfile(args.run / 'trace/body_ids.bin', dtype='<i4')
    meta = json.loads((args.run / 'trace/metadata.json').read_text())
    trace = []
    for path in states:
        x = np.fromfile(path, dtype='<f8').reshape(-1, 3)
        p, a, b, c = x[ids]
        q, uv = closest_pt(p, a, b, c)
        n = np.cross(b - a, c - a)
        signed = float(np.dot(p - a, n) / np.linalg.norm(n))
        trace.append({'substep': int(path.stem[-4:]), 'distance_m': float(np.linalg.norm(p - q)),
                      'signed_plane_distance_m': signed, 'closest_coordinates': list(uv),
                      'triangle_double_area_m2': float(np.linalg.norm(n))})
    accepted = []
    with (args.run / 'toi_diagnostic.jsonl').open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['frame'] == frame and 'accepted' in item:
                a = item['accepted']
                accepted.append({'outer': item['outer'], 'alpha': a['alpha'],
                                 'full_ccd_alpha': a['full_ccd_alpha'],
                                 'active_added': a['active_added'],
                                 'safe_ccd_broad_pairs': a['safe_ccd_broad_pairs']})
    result = {'failure': failure, 'vertex_class': ['ABD' if i < meta['abd_point_num'] else 'FEM' for i in ids],
              'body_ids': [int(bodies[i]) for i in ids], 'accepted_safe_states': len(trace),
              'minimum_recorded_distance_m': min(r['distance_m'] for r in trace),
              'first_distance_below_delta_substep': next((r['substep'] for r in trace if r['distance_m'] < failure['delta']), None),
              'last_safe_states': trace[-12:], 'last_accepted_steps': accepted[-12:],
              'all_safe_states': trace}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['vertex_class', 'body_ids', 'accepted_safe_states',
        'minimum_recorded_distance_m', 'first_distance_below_delta_substep', 'last_safe_states', 'last_accepted_steps']}))


if __name__ == '__main__':
    main()
