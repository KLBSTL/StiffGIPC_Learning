"""Small, dependency-free trajectory gate for the AutoDL image without NumPy."""
import argparse
import array
import json
import math
import sys
from pathlib import Path


def doubles(path):
    data = array.array('d')
    with path.open('rb') as stream:
        data.frombytes(stream.read())
    if sys.byteorder != 'little':
        data.byteswap()
    return data


def solver_info(run):
    stats = json.loads((run / 'output/stats.json').read_text())
    frames = stats['frames']
    if isinstance(frames, dict):
        frames = list(frames.values())
    pcg = [n['pcg'] for frame in frames for n in frame.get('newton', []) if 'pcg' in n]
    return {
        'pcg_iterations': [p['iterations'] for p in pcg],
        'pcg_limit_hits': sum(bool(p.get('iteration_limit', False)) for p in pcg),
        'outer_limit_hits': sum(frame.get('newton_exit') == 'iteration_limit' or
                                frame.get('toi_exit') == 'iteration_limit' for frame in frames),
        'graph_captures': max((p.get('graph_captures_total', 0) for p in pcg), default=0),
        'graph_cache_hits': sum(bool(p.get('graph_cache_hit', False)) for p in pcg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    xpaths = {int(p.stem[-4:]): p for p in (args.reference / 'trace').glob('state_*.bin')}
    ypaths = {int(p.stem[-4:]): p for p in (args.candidate / 'trace').glob('state_*.bin')}
    indices = sorted(xpaths.keys() & ypaths.keys())
    if len(indices) < 2:
        raise ValueError('Need at least two common state samples')
    masses = doubles(args.reference / 'trace/masses.bin')
    total_mass = sum(masses)
    max_rms = max_diff = scene_scale = 0.0
    initial_identical = False
    first_position_difference = first_gate_exceedance = None
    for frame in indices:
        x, y = doubles(xpaths[frame]), doubles(ypaths[frame])
        if len(x) != len(y) or len(x) != 3 * len(masses):
            raise ValueError(f'Frame {frame}: state/mass dimensions differ')
        if frame == indices[0]:
            initial_identical = x == y
            minima = [min(x[axis::3]) for axis in range(3)]
            maxima = [max(x[axis::3]) for axis in range(3)]
            scene_scale = math.sqrt(sum((hi - lo) ** 2 for lo, hi in zip(minima, maxima)))
        weighted_squared = 0.0
        for vertex, mass in enumerate(masses):
            offset = 3 * vertex
            squared = sum((y[offset + axis] - x[offset + axis]) ** 2 for axis in range(3))
            weighted_squared += mass * squared
            max_diff = max(max_diff, math.sqrt(squared))
        max_rms = max(max_rms, math.sqrt(weighted_squared / total_mass))
        if x != y and first_position_difference is None:
            first_position_difference = frame
        if math.sqrt(weighted_squared / total_mass) / scene_scale > 1e-6 and first_gate_exceedance is None:
            first_gate_exceedance = frame
    ref, candidate = solver_info(args.reference), solver_info(args.candidate)
    normalized = max_rms / scene_scale
    report = {
        'initial_state_identical': initial_identical,
        'frames': indices[-1],
        'state_samples_compared': len(indices),
        'trajectory_scope': 'all_frames' if indices == list(range(indices[-1] + 1)) else 'sampled_states',
        'max_position_difference_m': max_diff,
        'max_mass_weighted_rms_m': max_rms,
        'scene_scale_m': scene_scale,
        'normalized_max_mass_rms': normalized,
        'first_position_difference_frame': first_position_difference,
        'first_position_gate_exceedance_frame': first_gate_exceedance,
        'pcg_iteration_sequence_identical': ref['pcg_iterations'] == candidate['pcg_iterations'],
        'reference': ref,
        'candidate': candidate,
        'passed_position_gate': initial_identical and normalized <= 1e-6 and
                                candidate['pcg_limit_hits'] == 0 and candidate['outer_limit_hits'] == 0,
    }
    report['passed'] = report['passed_position_gate'] and report['trajectory_scope'] == 'all_frames'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('reference', 'candidate')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
