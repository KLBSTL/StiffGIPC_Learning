"""External additive driver: run one original AutoDL window repeat at a time.

The frozen package and original plans are unchanged. This script performs no
transfer or cleanup. Downloaded trace cleanup must retain run metadata/results.
"""
import argparse
import json
import os
from pathlib import Path
import platform

from config import ROOT, read, sha
from autodl_linux import execute, verify_bundle, write_new

DRIVER_SCHEMA = 'autodl.window_chunks.v1'


def check_window_plan(plan):
    if plan.get('stage') != 'window' or len(plan['runs']) != 54:
        raise ValueError('This driver requires the original 54-run window plan')
    rows = plan['runs']
    if [r['repeat'] for r in rows] != [1] * 18 + [2] * 18 + [3] * 18:
        raise ValueError('Expected three ordered contiguous 18-run repeats')
    if len({r['name'] for r in rows}) != 54:
        raise ValueError('Run names must be unique')
    for row in rows:
        expected = {'mixed': 35, 'hang': 21, 'fixed_bunny': 45}.get(row['scene_key'])
        if row['config']['steps'] != expected or row['config']['timeout_seconds'] != 120:
            raise ValueError('Window frame/time budget differs from frozen protocol')


def smoke_failures(smoke, smoke_plan, smoke_plan_sha, plan, manifest):
    if (smoke.get('plan_sha256') != smoke_plan_sha or
            smoke.get('candidate_sha256') != plan['candidate_sha256'] or
            smoke.get('source_digest') != manifest['source_digest']):
        raise ValueError('Matching original smoke report is required')
    if len(smoke['runs']) != len(smoke_plan['runs']):
        raise ValueError('Smoke batch is incomplete; do not infer passes for absent arms')
    for actual, expected in zip(smoke['runs'], smoke_plan['runs']):
        if {k: actual.get(k) for k in expected} != expected:
            raise ValueError('Smoke report task/order differs from original plan')
    passed = {r['arm'] for r in smoke['runs'] if r['result']['status'] == 'completed'}
    return {r['arm'] for r in plan['runs']} - passed


def validate_prefix(report, plan, manifest, plan_sha, driver_sha, initial_failed, result_loader=None):
    required = {'driver_schema': DRIVER_SCHEMA, 'driver_sha256': driver_sha,
                'plan_sha256': plan_sha, 'candidate_sha256': plan['candidate_sha256'],
                'source_digest': manifest['source_digest']}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError('Existing window report identity/driver differs; refusing to append')
    cursor = report.get('next_plan_index')
    if not isinstance(cursor, int) or not 0 <= cursor <= len(plan['runs']):
        raise ValueError('Invalid stored plan cursor')
    completed = [r for r in (1, 2, 3) if cursor >= r * 18]
    if report.get('completed_repeats') != completed:
        raise ValueError('Completed repeats do not match the stored plan prefix')
    failed = set(initial_failed); recorded = iter(report['runs'])
    for task in plan['runs'][:cursor]:
        if task['arm'] in failed:
            continue
        actual = next(recorded, None)
        if actual is None or {k: actual.get(k) for k in task} != task:
            raise ValueError('Existing results are not the expected ordered plan prefix')
        if result_loader is not None and result_loader(task['name']) != actual['result']:
            raise ValueError('Cumulative report differs from preserved run result: ' + task['name'])
        if actual['result']['status'] != 'completed':
            failed.add(task['arm'])
    if next(recorded, None) is not None or sorted(failed) != report.get('failed_arms'):
        raise ValueError('Unexpected trailing records or failed-arm bookkeeping')
    return cursor, failed


def check_requested_repeat(cursor, repeat):
    start, end = (repeat - 1) * 18, repeat * 18
    if not start <= cursor < end:
        raise ValueError('Requested repeat is already complete or its predecessor is incomplete')
    return end


def save_report(path, report):
    # Same-directory atomic replacement; never delete old run results or traces.
    temporary = path.with_name(path.name + f'.partial.{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def run_repeat(repeat, gpu):
    verify_bundle()
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('GLUT needs DISPLAY; launch this driver through xvfb-run')
    plan_path = ROOT / 'configs/active/autodl_window.json'
    plan = read(plan_path); check_window_plan(plan)
    manifest = read(ROOT / 'builds/autodl-active/manifest.json')
    if plan['candidate_sha256'] != manifest['candidate_sha256']:
        raise ValueError('Window candidate differs from compiled candidate')
    smoke_path = ROOT / 'reports/active/AUTODL_SMOKE_BATCH.json'
    smoke_plan_path = ROOT / 'configs/active/autodl_smoke.json'
    initial_failed = smoke_failures(read(smoke_path), read(smoke_plan_path), sha(smoke_plan_path), plan, manifest)
    target = ROOT / 'reports/active/AUTODL_WINDOW_BATCH.json'
    driver_hash = sha(__file__)
    lock = ROOT / 'runs/active/.window_chunks.lock'
    write_new(lock, {'pid': os.getpid(), 'repeat': repeat, 'driver_sha256': driver_hash})
    try:
        if not target.exists():
            if repeat != 1:
                raise ValueError('Repeat 1 must run before creating the cumulative window report')
            report = {'driver_schema': DRIVER_SCHEMA, 'driver_sha256': driver_hash,
                      'plan_sha256': sha(plan_path), 'candidate_sha256': plan['candidate_sha256'],
                      'source_digest': manifest['source_digest'], 'runs': [],
                      'next_plan_index': 0, 'completed_repeats': [],
                      'failed_arms': sorted(initial_failed), 'skipped_failed_smoke_arms': sorted(initial_failed),
                      'performance_certified': False, 'excluded_by_gate': []}
            write_new(target, report)
        else:
            report = read(target)
        cursor, failed = validate_prefix(report, plan, manifest, sha(plan_path), driver_hash, initial_failed,
                                        lambda name: read(ROOT / 'runs/active' / name / 'result.json'))
        end = check_requested_repeat(cursor, repeat)
        for index in range(cursor, end):
            task = plan['runs'][index]
            if task['arm'] not in failed:
                # A process interrupted between result write and report append
                # needs explicit reconciliation, never an automatic rerun.
                if (ROOT / 'runs/active' / task['name']).exists():
                    raise FileExistsError('Unreported existing run; reconcile its evidence before continuing: ' + task['name'])
                result = execute(task, gpu, manifest, controlled_load=False)
                report['runs'].append(task | {'result': result})
                if result['status'] != 'completed':
                    failed.add(task['arm'])
            report.update(next_plan_index=index + 1, failed_arms=sorted(failed),
                          completed_repeats=[r for r in (1, 2, 3) if index + 1 >= r * 18])
            save_report(target, report)
        print(json.dumps({'repeat_completed': repeat, 'completed_repeats': report['completed_repeats'],
                          'cumulative_recorded_runs': len(report['runs']), 'failed_arms': sorted(failed),
                          'driver_sha256': driver_hash, 'report': str(target)}), flush=True)
        return int(bool(failed))
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', required=True, type=int, choices=[1, 2, 3])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if platform.system() != 'Linux' or args.gpu < 0:
        parser.error('Linux and a nonnegative physical GPU index are required')
    return run_repeat(args.repeat, args.gpu)


if __name__ == '__main__':
    raise SystemExit(main())
