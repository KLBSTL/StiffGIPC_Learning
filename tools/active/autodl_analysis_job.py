"""Run the three CPU-only AutoDL analyses after all 54 window runs terminate.

Exclusive jobs/analysis.json and analysis.log preserve every stage outcome.
No simulator, build, source/config edit, remote connection, or GPU lock acquisition.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from config import ROOT, read, sha
import autodl_window_chunks as driver

TERMINAL = {'completed', 'failed', 'launcher_failed', 'timeout', 'disk_reserve',
            'memory_budget', 'foreign_gpu_load', 'incomplete_or_nonfinite',
            'binary_changed_during_run', 'configuration_failed'}
DEPENDENCIES = ('autodl_analysis_job.py', 'analyze_autodl_factor.py', 'audit_autodl_factor.py',
                'render_autodl_factor.py', 'analyze.py', 'config.py', 'validate_run.py',
                'autodl_window_chunks.py', 'autodl_linux.py', 'autodl_job.py',
                'autodl_transfer.py', 'render_factor_windows.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def alive(pid):
    require(isinstance(pid, int) and pid > 1, 'Invalid saved job PID')
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def idle(root):
    for name in ('.gpu.lock', '.window_chunks.lock'):
        require(not (root / 'runs/active' / name).exists(), 'GPU/window lock still exists: ' + name)
    jobs = {}
    for name in ('build', 'smoke', 'window1', 'window2', 'window3'):
        path = root / 'jobs' / (name + '.json')
        job = read(path)
        require(job.get('status') in ('completed', 'failed') and job.get('ended_utc'), 'Job is not terminal: ' + name)
        require(not alive(job['pid']), 'Recorded job process is still alive: ' + name)
        jobs[name] = {'status': job['status'], 'pid': job['pid'], 'sha256': sha(path)}
    command = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
    query = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    require(not query.stdout.strip(), 'GPU compute processes remain active: ' + query.stdout.strip())
    return {'jobs': jobs, 'gpu_query_command': command, 'active_compute_pids': [], 'checked_utc': now()}


def preflight(root, validator):
    plan_path = root / 'configs/active/autodl_window.json'
    matrix_path = root / 'reports/active/AUTODL_WINDOW_BATCH.json'
    plan, matrix = read(plan_path), read(matrix_path)
    driver.check_window_plan(plan)
    manifest = read(root / 'builds/autodl-active/manifest.json')
    smoke_plan_path = root / 'configs/active/autodl_smoke.json'
    initial = driver.smoke_failures(read(root / 'reports/active/AUTODL_SMOKE_BATCH.json'),
        read(smoke_plan_path), sha(smoke_plan_path), plan, manifest)
    cursor, failed = driver.validate_prefix(matrix, plan, manifest, sha(plan_path),
        sha(root / 'tools/active/autodl_window_chunks.py'), initial,
        lambda name: read(root / 'runs/active' / name / 'result.json'))
    require(cursor == 54 and matrix['completed_repeats'] == [1, 2, 3] and len(matrix['runs']) == 54,
            'All 54 planned window runs must have terminal preserved results; absent/skipped runs are not complete')
    require(all(row['result'].get('status') in TERMINAL for row in matrix['runs']), 'Nonterminal window result')
    identities = {name: sha(root / 'tools/active' / name) for name in DEPENDENCIES}
    require(sha(Path(driver.__file__)) == identities['autodl_window_chunks.py'], 'Imported ledger parser differs from target root')
    require(validator.is_file() and os.access(validator, os.X_OK), 'Native CPU validator is missing or not executable')
    return {'plan_sha256': sha(plan_path), 'matrix_sha256': sha(matrix_path), 'terminal_runs': 54,
            'failed_arms': sorted(failed), 'dependencies_sha256': identities,
            'validator_sha256': sha(validator),
            'validator_source_sha256': sha(root / 'tools/validator/diagnose_first_path.cpp'),
            'idle': idle(root)}


def stages(root, validator):
    active = root / 'reports/active'
    specs = [('analyze', 'analyze_autodl_factor.py', '--output', active / 'AUTODL_FACTOR_WINDOW_ANALYSIS.json', 1800),
             ('audit', 'audit_autodl_factor.py', '--output-dir', active / 'AUTODL_FACTOR_CCD', 21600),
             ('render', 'render_autodl_factor.py', '--output-dir', active / 'AUTODL_FACTOR_FIGURES', 1800)]
    rows = []
    for name, script, option, destination, budget in specs:
        command = [sys.executable, str(root / 'tools/active' / script), '--root', str(root), option, str(destination)]
        report = destination
        if name == 'audit':
            command += ['--validator', str(validator), '--timeout-seconds', '300', '--inventory-timeout-seconds', '60']
            report = destination / 'audit.json'
        elif name == 'render':
            report = destination / 'render_manifest.json'
        rows.append({'name': name, 'command': command, 'destination': str(destination),
                     'report': str(report), 'timeout_seconds': budget})
    return rows


def outcome(name, code, path):
    """Exit 1 plus a finalized report is evidence, not an unhandled script crash."""
    report = read(path)
    if name == 'analyze':
        keys = ('all_planned_runs_validated', 'all_scenes_analyzed', 'all_accepted_path_inventories_complete')
        finalized = report.get('planned_runs') == 54 and len(report.get('run_checks', [])) == 54 and all(k in report for k in keys)
        passed = all(report.get(k) is True for k in keys)
        detail = {k: report.get(k) for k in keys}
        failure_status = 'observations_invalid'
    elif name == 'audit':
        finalized = len(report.get('runs', [])) == 54 and 'all_54_accepted_path_audits_passed' in report
        passed = report.get('all_54_accepted_path_audits_passed') is True
        detail = {k: report.get(k) for k in ('all_54_accepted_path_audits_passed', 'actual_validator_reports',
                  'actual_collision_flags_from_available_reports', 'collision_flag_total_covers_all_54_full_windows')}
        numerical = report.get('collision_flag_total_covers_all_54_full_windows') is True and all(
            row.get('validator_exit_code') in (0, 1) and not row.get('validator_timed_out') for row in report['runs'])
        failure_status = 'numerical_observations_failed' if numerical else 'audit_incomplete'
    else:
        finalized = 'all_three_scenes_rendered' in report and 'failures' in report
        passed = report.get('all_three_scenes_rendered') is True
        detail = {'all_three_scenes_rendered': passed, 'failures': report.get('failures')}
        failure_status = 'render_incomplete'
    require(finalized and code in (0, 1) and (code == 0) == passed, 'Missing/incomplete report or inconsistent script exit status')
    return {'status': 'completed' if passed else failure_status, 'observations': detail, 'report_sha256': sha(path)}


def run(root, validator):
    require(sys.platform == 'linux', 'Detached analysis job is Linux-only')
    jobs = root / 'jobs'; jobs.mkdir(exist_ok=True)
    status, log_path = jobs / 'analysis.json', jobs / 'analysis.log'
    require(not status.exists() and not log_path.exists(), 'Analysis job files already exist; never overwrite evidence')
    record = {'schema': 'autodl.analysis_job.v1', 'status': 'running', 'pid': os.getpid(),
              'started_utc': now(), 'root': str(root), 'python': sys.executable,
              'runner_sha256': sha(Path(__file__)), 'physical_certified': False,
              'performance_certified': False, 'stages': []}
    with status.open('x', encoding='utf-8') as state, log_path.open('x', encoding='utf-8') as log:
        def save():
            state.seek(0); json.dump(record, state, indent=2, allow_nan=False)
            state.truncate(); state.flush(); os.fsync(state.fileno())
        save()
        try:
            planned = stages(root, validator)
            require(all(not Path(row['destination']).exists() for row in planned), 'One or more analysis outputs already exist')
            record['preflight'] = preflight(root, validator); save()
            env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for row in planned:
                row.update(status='running', started_utc=now())
                record['stages'].append(row); save()
                process = None; started = time.monotonic()
                try:
                    row['idle_before_stage'] = idle(root)
                    log.write('\nSTAGE ' + json.dumps(row) + '\n'); log.flush()
                    process = subprocess.Popen(row['command'], cwd=root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    row['child_pid'] = process.pid; save()
                    row['exit_code'] = process.wait(timeout=row['timeout_seconds'])
                    row.update(outcome(row['name'], row['exit_code'], Path(row['report'])))
                except Exception as error:
                    row.update(status='script_failed', error=type(error).__name__ + ': ' + str(error))
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                finally:
                    row.update(ended_utc=now(), wall_seconds=time.monotonic() - started)
                    log.write('\nSTAGE_RESULT ' + json.dumps(row) + '\n'); log.flush(); save()
            statuses = [row['status'] for row in record['stages']]
            record['status'] = ('failed' if 'script_failed' in statuses else
                                'completed' if all(s == 'completed' for s in statuses) else 'completed_with_findings')
        except BaseException as error:
            record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            log.write('\nJOB_FAILURE ' + record['error'] + '\n'); log.flush()
        finally:
            record['ended_utc'] = now(); save()
    return 2 if record['status'] == 'failed' else int(record['status'] != 'completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--validator', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    validator = args.validator.resolve() if args.validator else root / 'builds/autodl-active/validator/diagnose_first_path'
    return run(root, validator)


if __name__ == '__main__':
    raise SystemExit(main())
