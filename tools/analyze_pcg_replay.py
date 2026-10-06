"""Evaluate fixed-system PCG repeats without comparing separate trajectories."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stats', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat-tol', type=float, default=1e-12)
    parser.add_argument('--graph-tol', type=float, default=1e-6)
    args = parser.parse_args()
    frames = json.loads(args.stats.read_text(encoding='utf-8'))['frames']
    events = [n['pcg']['fixed_system_replay'] for f in frames for n in f.get('newton', [])
              if 'fixed_system_replay' in n.get('pcg', {})]
    if not events:
        raise ValueError('No fixed-system PCG replay events recorded')
    repeats = [(e, r) for e in events for r in e['same_execution_runs']]
    host_refs = [r for e in events for r in e['host_reference_runs']]
    finite = all(isinstance(v, (int, float)) and math.isfinite(v)
                 for e in events for v in [e['primary_true_relative_residual']])
    finite &= all(isinstance(r[k], (int, float)) and math.isfinite(r[k])
                  for r in [r for _, r in repeats] + host_refs
                  for k in ('relative_solution_difference', 'true_relative_residual'))
    repeat_difference = max(r['relative_solution_difference'] or 0 for _, r in repeats)
    graph_difference = max((r['relative_solution_difference'] or 0 for r in host_refs), default=0)
    host_repeat_difference = max((r['relative_to_first_host'] or 0 for r in host_refs), default=0)
    rhs_unchanged = all(e['rhs_unchanged_bitwise'] for e in events)
    solution_restored = all(e['primary_solution_restored_bitwise'] for e in events)
    limit_hits = (sum(e['primary_iteration_limit'] for e in events)
                  + sum(r['iteration_limit'] for _, r in repeats) + sum(r['iteration_limit'] for r in host_refs))
    passed = (finite and rhs_unchanged and solution_restored and limit_hits == 0
              and repeat_difference <= args.repeat_tol and host_repeat_difference <= args.repeat_tol
              and graph_difference <= args.graph_tol)
    report = {
        'scope': 'Repeated zero-initialized solves of the same assembled system and preconditioner; assembly and nonlinear trajectory are outside scope',
        'stats': str(args.stats), 'events': len(events),
        'frames': [e['frame'] for e in events],
        'executions': sorted({e['primary_execution'] for e in events}),
        'same_execution_repeats': len(repeats), 'host_reference_solves': len(host_refs),
        'maximum_same_execution_relative_difference': repeat_difference,
        'maximum_graph_host_relative_difference': graph_difference,
        'maximum_host_reference_repeat_relative_difference': host_repeat_difference,
        'maximum_bitwise_different_dofs': max(r['bitwise_different_dofs'] for _, r in repeats),
        'same_execution_iteration_mismatches': sum(r['iterations'] != e['primary_iterations'] for e, r in repeats),
        'primary_residuals': [e['primary_true_relative_residual'] for e in events],
        'finite': finite, 'rhs_unchanged_bitwise': rhs_unchanged,
        'primary_solution_restored_bitwise': solution_restored,
        'replay_iteration_limit_hits': limit_hits,
        'repeat_tolerance': args.repeat_tol, 'graph_host_tolerance': args.graph_tol, 'passed': passed,
    }
    for operator in ('spmv', 'preconditioner'):
        records = [r for e in events for r in e.get('operator_replays', {}).get(operator, [])]
        if records:
            report[f'maximum_{operator}_same_input_relative_difference'] = max(
                r['relative_solution_difference'] for r in records)
            report[f'maximum_{operator}_bitwise_different_dofs'] = max(
                r['bitwise_different_dofs'] for r in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('primary_residuals', 'scope', 'stats')}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
