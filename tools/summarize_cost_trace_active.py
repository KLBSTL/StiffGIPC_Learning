"""Summarize selective diagnostic scopes without adding nested inclusive times.

CUDA event intervals include host submission gaps and instrumentation overhead.
Probe costs are kept separate; they are not production iteration measurements.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scopes = {row['scope_id']: row for row in rows}
    if len(scopes) != len(rows):
        raise ValueError('Duplicate scope IDs; do not append multiple processes into one trace')
    for row in rows:
        if row.get('schema') != 'gipc.cost.v1' or not row.get('inclusive'):
            raise ValueError('Unknown cost trace schema or interval convention')
        if row['parent_scope_id'] and row['parent_scope_id'] not in scopes:
            raise ValueError('Missing parent; trace may be truncated')
        if row['gpu_interval_ms'] is not None and row['gpu_interval_ms'] < 0:
            raise ValueError('Negative CUDA interval')
    groups = collections.defaultdict(list)
    children = collections.defaultdict(list)
    for row in rows:
        groups[row['sample_kind'], row['stage']].append(row)
        children[row['parent_scope_id']].append(row)

    def first_gpu_descendants(row):
        # CPU-only scopes may contain GPU children; those remain included in the
        # nearest GPU ancestor and must be subtracted there, exactly once.
        for child in children[row['scope_id']]:
            if child['gpu_interval_ms'] is not None:
                yield child
            else:
                yield from first_gpu_descendants(child)

    table = []
    nesting_failures = []
    for (kind, stage), values in sorted(groups.items()):
        measured = [v for v in values if v['gpu_interval_ms'] is not None]
        intervals = [v['gpu_interval_ms'] for v in measured]
        exclusive = []
        for row in measured:
            remainder = row['gpu_interval_ms'] - sum(
                child['gpu_interval_ms'] for child in first_gpu_descendants(row))
            if remainder < -0.05:
                nesting_failures.append({'scope_id': row['scope_id'], 'remainder_ms': remainder})
            exclusive.append(remainder)
        table.append({
            'sample_kind': kind, 'stage': stage, 'calls': len(values),
            'cpu_submit_inclusive_ms': sum(v['cpu_submit_ms'] for v in values),
            'gpu_interval_inclusive_ms': sum(intervals) if intervals else None,
            'gpu_interval_median_us': statistics.median(intervals) * 1000 if intervals else None,
            'gpu_interval_exclusive_ms': sum(exclusive) if intervals else None,
        })
    return {
        'source': str(path.resolve()), 'records': len(rows),
        'linear_systems': len({v['linear_system_id'] for v in rows}),
        'frames': sorted({v['frame'] for v in rows}),
        'diagnostic_only': True, 'performance_certified': False,
        'interpretation': (
            'Never sum the inclusive stage table. Exclusive interval remainders include '
            'host submission gaps and tracing overhead, not just device computation. '
            'frozen_operator_probe uses fixed b and must not be counted as production work.'),
        'nesting_failures': nesting_failures, 'stages': table,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.trace)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x') as target:
            target.write(text + '\n')
    else:
        print(text)
    if result['nesting_failures']:
        raise SystemExit('Cost interval nesting failed; do not use this attribution')


if __name__ == '__main__':
    main()
