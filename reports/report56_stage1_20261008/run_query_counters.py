"""One bounded NCU query capture; reuse the verified Job/lock/resource guards."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys
import time
from run_stage1 import ROOT, REPORT, SESSION, guard_summary
for folder in ('bench', 'diagnostic', 'full_eval', 'local'):
    sys.path.insert(0, str(ROOT / 'tools' / folder))
from config import expand, digest, environment
from linux_runner import require, read, write_new, inventory, gpu_query, verify_files
from local_identity import verify, record
from windows_runner import gpu_lock, driver_model, prepare_run_output
from windows_owned_job import OwnedJob
from profiler_windows import processes, remaining, resource_reason, solver_succeeded
from validate_run import validate

NCU = Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.3.1/target/windows-desktop-win7-x64/ncu.exe')
METRICS = ('gpu__time_duration.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct')

def metric_rows(path):
    """NCU raw CSV is wide in this installed version; retain units explicitly."""
    rows = list(csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()))
    units = [row for row in rows if row.get('ID') == '' and row.get('Kernel Name') == '']
    kernels = [row for row in rows if row.get('ID', '').isdigit()]
    require(len(units) == len(kernels) == 1, 'One units row and one query expected')
    require(kernels[0]['Kernel Name'].startswith('_selfQuery_ee('), 'Wrong query kernel')
    return [{'Metric Name': name, 'Metric Value': kernels[0][name],
             'Metric Unit': units[0][name]} for name in METRICS]

def main():
    identity = read(SESSION / 'PROBE_IDENTITY.json')
    reference = SESSION / 'observer_off/cloth_sphere7_l'
    reference_files = inventory(reference, [reference / n for n in
        ('requested.json', 'result.json', 'evidence.json', 'config_validation.json')])
    verify_files(reference, read(reference / 'evidence.json')['files'])
    require(read(reference / 'result.json')['status'] == 'completed' and
            read(reference / 'config_validation.json')['passed'], 'Completed reference required')
    request = read(reference / 'requested.json')
    require(request['exe_sha256'] == identity['exe']['sha256'] and
            request['source_digest'] == identity['source_digest'], 'Reference identity differs')
    files = identity['sources'] + identity['build_evidence'] + identity['objects'] + [identity['exe']] + identity['dlls']
    verify(files)
    original = request['expanded_config']
    require(original['diagnostics'] == [] and original['steps'] == 120 and
            original['timeout_seconds'] == 180, 'Frozen reference scope differs')
    config = expand(dict(original, diagnostics=['cost'], cost_frames='49', cost_events=False))
    require(all(config[k] == original[k] for k in original
                if k not in {'diagnostics', 'cost_frames', 'cost_events'}), 'Numerical configuration changed')
    out = SESSION / 'query_counters_f49'
    # Exactly one fixed output and one plan: no retries or new-directory fallback.
    require(not out.exists(), 'Single attempt already reserved; preserve failure evidence')
    prepare_run_output(out, config)
    command = [str(NCU), '--target-processes', 'all', '--replay-mode', 'kernel',
        '--nvtx', '--nvtx-include', 'ipc.physical_frame/', '--kernel-name-base', 'function',
        '--kernel-name', 'regex:^_selfQuery_ee$', '--launch-count', '1',
        '--metrics', ','.join(METRICS), '--cache-control', 'none', '--clock-control', 'none',
        '--export', str(out / 'query'), str(Path(identity['exe']['path']))]
    own = record(__file__)
    profiler = record(NCU)
    plan = {'schema': 'gipc.one_query_counter_check.v1', 'max_attempts': 1,
        'purpose': 'Test dynamic local-memory traffic/occupancy on one actual DCD EE query; not speed certification',
        'selected_frame': 49, 'max_profiled_kernels': 1, 'timeout_seconds': 180,
        'reference': reference_files, 'program_sha256': identity['exe']['sha256'],
        'metrics': METRICS, 'command': command, 'script': own, 'profiler': profiler,
        'no_driver_or_clock_settings_changed': True, 'nsight_systems_budget_unchanged': True}
    write_new(REPORT / 'NCU_QUERY_CHECK_PLAN.json', plan)
    result = {'status': 'not_launched', 'recorded_frames': 0, 'heavy_diagnostics': True,
        'performance_certified': False, 'physical_quality_certified': False,
        'timing_is_diagnostic': True, 'automatic_retry': False, 'timeout_seconds': 180}
    job = None
    samples = []
    started = None
    with gpu_lock(ROOT):
        try:
            mode = driver_model(0)
            before = [gpu_query(0)]; time.sleep(.25); before.append(gpu_query(0))
            require(before[0]['uuid'] == before[1]['uuid'], 'GPU identity changed')
            foreign = processes(0, mode, lambda pid: False, 10)
            require(not foreign['blocking_foreign_pids'], 'Foreign compute load')
            require(mode == 'WDDM' or all(x['utilization.gpu'] <= 5 for x in before), 'Busy GPU')
            free = before[-1]['memory.free']; budget = min(.75 * free, free - 1536)
            require(budget >= 1024 and shutil.disk_usage(out).free >= 4 * 1024**3, 'Resource reserve')
            env = environment(config, out); env['CUDA_VISIBLE_DEVICES'] = before[-1]['uuid']
            write_new(out / 'requested.json', {'requested_config': config, 'expanded_config': config,
                'config_sha256': digest(config), 'binary': 'active', 'exe_sha256': identity['exe']['sha256'],
                'source_digest': identity['source_digest'], 'from_zero': True,
                'external_profiler': plan, 'environment': {k: v for k, v in env.items()
                    if k.startswith('GIPC_') or k == 'CUDA_VISIBLE_DEVICES'}})
            write_new(out / 'build_manifest.json', identity)
            result.update(gpu_before=before, gpu_memory_budget_mib=budget,
                gpu_driver_model=mode, desktop_load_uncontrolled=mode == 'WDDM')
            started = time.monotonic(); deadline = started + 180
            job = OwnedJob()
            with (out / 'run.log').open('xb') as log:
                job.launch(command, out, env, log)
                write_new(out / 'process.json', {'pid': job.pid, 'command': command,
                    'ownership': 'Windows Job, suspended assignment, kill-on-close, no breakaway'})
                status = 'running'
                while not job.finished():
                    job.observe(); sample = gpu_query(0, timeout=remaining(deadline, 180))
                    require(sample['uuid'] == before[-1]['uuid'], 'GPU identity changed')
                    current = processes(0, mode, job.owns_pid, remaining(deadline, 180))
                    samples.append(sample | {'elapsed_seconds': time.monotonic() - started})
                    reason = resource_reason(sample, free, budget, shutil.disk_usage(out).free,
                                             current['blocking_foreign_pids'])
                    if reason: status = reason; job.terminate(); break
                    time.sleep(max(0, min(.5, deadline - time.monotonic())))
                process_log = job.observe(); exit_code = job.poll()
                if status == 'running':
                    status = ('timeout' if time.monotonic() - started > 180 else
                              'completed' if exit_code == 0 else 'profiler_failed')
                if status == 'completed' and not solver_succeeded(process_log, Path(identity['exe']['path'])):
                    status = 'solver_not_observed_or_failed'
            result.update(status=status, exit_code=exit_code, owned_processes=process_log)
            verify(files + [own, profiler]); verify_files(reference, reference_files)
            if status == 'completed':
                result.update(recorded_frames=120, finite=True)
                write_new(out / 'result.json', result)
                check = validate(out); write_new(out / 'config_validation.json', check)
                require(check['passed'], 'Requested/resolved configuration mismatch')
                summary = guard_summary(out, result)
                require(summary['completed'] and summary['guards_passed'], 'Simulation guard failed')
                scopes = [json.loads(line) for line in (out / 'cost.jsonl').read_text().splitlines() if line]
                require(scopes and all(x.get('frame') == 49 and x.get('nvtx_available') is True and
                    x.get('gpu_events_enabled') is False and x.get('gpu_interval_ms') is None and
                    x.get('sample_kind') == 'production' for x in scopes), 'Wrong capture scope')
                require(sum(x['stage'] == 'ipc.physical_frame' for x in scopes) == 1,
                        'One completed NVTX physical-frame scope required')
                report_file = out / 'query.ncu-rep'
                require(report_file.is_file() and report_file.stat().st_size > 0, 'Missing NCU report')
                evidence = {'passed': True, 'simulation_summary': summary,
                    'profiler_report': record(report_file), 'cost_trace': record(out / 'cost.jsonl'),
                    'scope': 'NCU report and selected completed NVTX frame; counters validated separately'}
                write_new(out / 'capture_validation.json', evidence)
                with (out / 'counter.csv').open('wb') as stream:
                    subprocess.run([str(NCU), '--import', str(out / 'query.ncu-rep'),
                        '--csv', '--page', 'raw', '--print-units', 'base'],
                        stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=10)
                counters = metric_rows(out / 'counter.csv')
                require({r['Metric Name'] for r in counters}.issuperset(METRICS), 'Missing dynamic counters')
                result['counters'] = counters
        except BaseException as error:
            result.update(status='counter_check_failed', error=type(error).__name__ + ': ' + str(error))
        finally:
            if job is not None:
                try:
                    if not job.finished(): job.terminate()
                    result.update(cleanup_owned_job_empty=job.finished(), owned_processes=job.observe())
                finally: job.close()
            result['gpu_samples'] = samples
            if started is not None: result['wall_seconds'] = time.monotonic() - started
            (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            write_new(out / 'evidence.json', {'files': inventory(out, [p for p in out.rglob('*') if p.is_file()])})
    print(json.dumps({'status': result['status'], 'output': str(out), 'automatic_retry': False}))
    return 0 if result['status'] == 'completed' else 2

if __name__ == '__main__': raise SystemExit(main())
