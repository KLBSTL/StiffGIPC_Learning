"""Report the unchanged-state assembly and energy-probe restoration gate."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stats', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args()
    frames = json.loads(args.stats.read_text(encoding='utf-8'))['frames']
    events = [n['state_snapshot_audit'] for f in frames for n in f.get('newton', [])
              if 'state_snapshot_audit' in n]
    if not events:
        raise ValueError('No state audit events')
    structures = all(e['assembly']['structure_identical'] for e in events)
    norms = {}
    finite = True
    for kind in ('matrix', 'rhs', 'solution_buffer'):
        values = [e['assembly'].get(kind, {}).get('relative_difference') for e in events]
        valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        finite &= valid
        norms[kind] = max(values) if valid else None
    restored = all(e['protected_state_restored_bitwise'] for e in events)
    passed = structures and finite and restored and all(v <= args.tolerance for v in norms.values())
    report = {'events': len(events), 'frames': sorted({e['frame'] for e in events}),
              'contact_systems': sum(e['active_contacts'] > 0 for e in events),
              'structure_identical': structures, 'protected_state_restored_bitwise': restored,
              'maximum_relative_assembly_differences': norms,
              'alpha_zero_maximum_vertex_absolute_difference_m': max(
                  e['alpha_zero_vertex_max_abs_difference'] for e in events),
              'alpha_zero_maximum_bitwise_different_vertices': max(
                  e['alpha_zero_bitwise_different_vertices'] for e in events),
              'finite': finite, 'tolerance': args.tolerance, 'passed': passed,
              'scope': 'Recorded unchanged-state converted A/b and protected state; contact coverage is reported separately, not full nonlinear equivalence'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
