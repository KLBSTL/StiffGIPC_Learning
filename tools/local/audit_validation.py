"""Bounded, serial CTest and MAS synchronization validation on Windows.

Uses the existing owned Job and GPU/resource policy; no implicit retry, build,
cleanup, arbitrary shell command, or performance certification.
"""
from pathlib import Path
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for folder in ('bench', 'diagnostic', 'local'):
    sys.path.insert(0, str(ROOT/'tools'/folder))
from config import read
from linux_runner import require, write_new, gpu_query
from local_identity import verify, verify_active_sources, record
from windows_runner import gpu_lock, driver_model, gpu_processes, remaining, final_status
from windows_owned_job import OwnedJob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('ctest', 'racecheck', 'synccheck', 'matrix'))
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--matrix-input', type=Path)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    require((args.kind=='matrix') == (args.matrix_input is not None), 'Matrix input is required only for matrix validation')
    require(re.fullmatch(r'[A-Za-z0-9_]+', args.session), 'Invalid session path')
    out = ROOT/'runs'/args.session
    require(not out.exists(), 'Fresh output required; preserve all prior attempts')
    out.mkdir(parents=True)
    identity = read(args.identity)
    evidence = identity['sources']+identity['build_evidence']+identity.get('objects', [])+[identity['exe']]+identity['dlls']
    if args.matrix_input:
        evidence.append(record(args.matrix_input))
    verify_active_sources(identity)
    verify(evidence)
    exe = Path(identity['exe']['path'])
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith('GIPC_')}
    (out/'tmp').mkdir()
    env['TMP'] = env['TEMP'] = str(out/'tmp')
    if args.kind == 'ctest':
        command = [r'D:\computer\cmake\bin\ctest.exe',
            '--test-dir', str(exe.parent.parent), '-C', 'Release', '--output-on-failure', '-j', '1']
        timeout = 300
    elif args.kind == 'matrix':
        command = [str(exe)]
        env['GIPC_MAS_PIVOT_SYNC_FIXTURE'] = str(out/'fixture.json')
        env['GIPC_MAS_PIVOT_SYNC_INPUT'] = str(args.matrix_input.resolve())
        timeout = 120
    else:
        sanitizer = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\compute-sanitizer\compute-sanitizer.exe')
        require(sanitizer.is_file(), 'Compute Sanitizer unavailable')
        # Each fixture batch has 50 matching launches. Check both production
        # instantiations with two matrices per block plus an odd tail.
        skip = 50 if args.kind == 'racecheck' else 150
        command = [str(sanitizer), '--tool', args.kind, '--error-exitcode', '99',
            '--kernel-name', 'kns=__inverse6_P96x96', '--launch-skip', str(skip),
            '--launch-count', '2']
        if args.kind == 'racecheck':
            command += ['--racecheck-report', 'analysis', '--racecheck-num-workers', '2']
        command += [str(exe)]
        env['GIPC_MAS_PIVOT_SYNC_FIXTURE'] = str(out/'fixture.json')
        timeout = 180
    result = dict(kind=args.kind, status='launcher_failed', performance_certified=False,
        command=command, timeout_seconds=timeout, identity=record(args.identity),
        controller=record(__file__), executable=record(command[0]))
    job = None
    samples = []
    try:
        with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
            mode = driver_model(0)
            before = gpu_query(0)
            processes = gpu_processes(0, None, mode)
            require(not processes['blocking_foreign_pids'], 'Foreign compute gate failed')
            free = before['memory.free']
            budget = min(.75*free, free-1536)
            require(budget >= 1024 and shutil.disk_usage(out).free >= 4*1024**3, 'Resource reserve insufficient')
            env['CUDA_VISIBLE_DEVICES'] = before['uuid']
            result.update(gpu_before=before, process_before=processes, memory_budget_mib=budget)
            start = time.monotonic()
            deadline = start+timeout
            status = 'running'
            with (out/'validation.log').open('xb') as log:
                job = OwnedJob()
                job.launch(command, out, env, log)
                while not job.finished():
                    job.observe()
                    try:
                        sample = gpu_query(0, timeout=remaining(deadline, timeout))
                        processes = gpu_processes(0, None, mode, timeout=remaining(deadline, timeout), ownership=job.owns_pid)
                        remaining(deadline, timeout)
                    except subprocess.TimeoutExpired:
                        status = 'timeout' if time.monotonic() >= deadline else 'monitor_timeout'
                        job.terminate()
                        break
                    require(sample['uuid'] == before['uuid'], 'GPU identity changed')
                    samples.append(sample|{'processes': processes})
                    status = ('foreign_gpu_load' if processes['blocking_foreign_pids'] else
                        'disk_reserve' if shutil.disk_usage(out).free < 1024**3 else
                        'memory_budget' if sample['memory.free'] < 768 or free-sample['memory.free'] > budget else 'running')
                    if status != 'running':
                        job.terminate()
                        break
                    if job.poll() not in (None, 0):
                        status = 'failed'
                        job.terminate()
                        break
                    time.sleep(max(0, min(.5, deadline-time.monotonic())))
                elapsed = time.monotonic()-start
                result.update(status=final_status(status, job.poll(), elapsed, timeout),
                    exit_code=job.poll(), wall_seconds=elapsed)
            verify(evidence)
            if result['status'] == 'completed':
                if args.kind != 'ctest':
                    require(read(out/'fixture.json')['passed'] is True, 'Native fixture failed')
                    if args.kind != 'matrix':
                        log = (out/'validation.log').read_text(errors='replace')
                        require('SUMMARY: 0' in log and '========= Warning:' not in log,
                            'Sanitizer did not confirm a clean summary')
                    result['fixture'] = record(out/'fixture.json')
                result['passed'] = True
    except Exception as error:
        result.update(status='launcher_or_validation_failed', error=type(error).__name__+': '+str(error), passed=False)
    finally:
        if job is not None:
            try:
                if not job.finished():
                    job.terminate()
                result.update(owned_processes=job.observe(), cleanup_owned_job_empty=job.finished())
            finally:
                job.close()
        result['gpu_samples'] = samples
        write_new(out/'result.json', result)
    print({k:v for k,v in result.items() if k in ('kind','status','passed','wall_seconds','exit_code','error')}, flush=True)
    return 0 if result.get('passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
