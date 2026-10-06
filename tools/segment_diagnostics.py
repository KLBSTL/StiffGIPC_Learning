"""Aggregate one-run timing by base geometry contact, without quality claims."""
import argparse
import csv
import json
from pathlib import Path


def read_run(path):
    requested = json.loads((path / 'requested.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    if result['status'] != 'completed' or result.get('recorded_frames') != requested['steps']:
        raise ValueError(f'Incomplete run: {path}')
    with (path / 'trace/frames.csv').open() as source:
        times = {int(row['frame']): float(row['solver_ms']) / 1000 for row in csv.DictReader(source)}
    frames = json.loads((path / 'output/stats.json').read_text())['frames']
    if isinstance(frames, dict):
        frames = list(frames.values())
    if len(times) != len(frames):
        raise ValueError(f'Timing/stats length mismatch: {path}')
    return requested, result, times, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = [read_run(path) for path in args.runs]
    base = data[0]
    scenes = {(item[0]['scene'], item[0]['steps'], item[0].get('dt')) for item in data}
    if len(scenes) != 1:
        raise ValueError('Runs must share scene, frame count, and requested dt')
    if base[0]['arm'] != 'base':
        raise ValueError('First run must be the official base')
    contact = [i + 1 for i, frame in enumerate(base[3]) if frame['contact_geometry']['geometric_contact']]
    no_contact = [i for i in base[2] if i not in set(contact)]
    results = []
    for path, (_, run, times, _) in zip(args.runs, data):
        if set(times) != set(base[2]):
            raise ValueError(f'Frame indices differ: {path}')
        results.append({'run': path.name, 'arm': json.loads((path / 'requested.json').read_text())['arm'],
                        'source_digest': json.loads((path / 'requested.json').read_text())['source_digest'],
                        'exe_sha256': json.loads((path / 'requested.json').read_text())['exe_sha256'],
                        'timing_is_diagnostic': run['timing_is_diagnostic'],
                        'total_solver_seconds': sum(times.values()),
                        'contact_solver_seconds': sum(times[i] for i in contact),
                        'noncontact_solver_seconds': sum(times[i] for i in no_contact)})
    report = {'scope': 'single-run diagnostic timings; not quality-matched speedups',
              'classification': 'official base contact_geometry.geometric_contact, native narrow phase distance <= 1e-4 initial bbox diagonal',
              'contact_frames_1_based': contact, 'contact_frame_count': len(contact),
              'noncontact_frame_count': len(no_contact), 'runs': results}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'contact_frame_count': len(contact), 'noncontact_frame_count': len(no_contact),
                      'runs': [{k: r[k] for k in ['arm', 'total_solver_seconds', 'contact_solver_seconds',
                              'noncontact_solver_seconds']} for r in results]}))


if __name__ == '__main__':
    main()
