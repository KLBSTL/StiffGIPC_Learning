"""One explicit, bounded Windows Nsight diagnostic; never builds or auto-reruns.

Inherit an already completed sealed Step3 fixed-cloth run. Only four diagnostic
fields change. The existing experiment plan, quality gate and runner stay frozen.
At most two launch attempts per original seal, under the same local GPU lock.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from local_plan import ROOT
from config import read, expand, digest, environment
from linux_runner import require, child, sha, write_new, inventory, verify_files, gpu_query
from local_identity import verify_seal, verify, record
from windows_runner import gpu_lock, driver_model, process_rows
from validate_run import validate
from windows_owned_job import OwnedJob

NSYS = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe')
TIMEOUT = 120
DIAGNOSTIC_KEYS = {'diagnostics', 'cost_frames', 'cost_events', 'profile'}


def derive_config(request, result, mode):
    require(mode in ('node', 'graph'), 'Only node or graph profiling')
    c = request['expanded_config']
    require(expand(c) == c and request['config_sha256'] == digest(c), 'Reference configuration identity differs')
    require(request['binary'] == 'active' and result['status'] == 'completed' and
            result['recorded_frames'] == 59 and result.get('finite') is True, 'Completed finite active 59-frame reference required')
    require(c['scene'] == 'cloth_fixed_bunny_l' and c['steps'] == 59 and c['timeout_seconds'] == 120 and
            c['backend'] == 'ipc' and c['execution'] == 'conditional_graph', 'Unsupported reference scope')
    require(c['diagnostics'] == [] and c['profile'] == 'none' and not c['contact_pool_validate'], 'Reference is instrumented')
    require(c['ipc_termination'] == 'legacy' and c['ipc_min_updates'] == 6 and
            c['ipc_newton_tol'] == .01 and c['ipc_cumulative_tol'] == .01 and
            c['pcg_rho_tol'] == 1e-4 and c['dt'] == .01, 'Frozen stopping/material time step differs')
    out = dict(c)
    out.update(diagnostics=['cost'], cost_frames='57', cost_events=False, profile=mode)
    resolved = expand(out)
    require(all(resolved[k] == c[k] for k in c if k not in DIAGNOSTIC_KEYS), 'Non-diagnostic configuration changed')
    return resolved


def nsys_command(nsys, exe, out, mode):
    require(mode in ('node', 'graph'), 'Invalid graph trace mode')
    return [str(nsys), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
            '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0', '--capture-range=nvtx',
            '--nvtx-capture=ipc.physical_frame', '--capture-range-end=stop',
            '--cuda-graph-trace=' + mode, '--output=' + str(out / 'nsight'), str(exe)]


def remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0: raise subprocess.TimeoutExpired('owned profiler tree', TIMEOUT)
    return min(10, left)


def owned_process_rows(stdout, mode, ownership):
    # Reuse unchanged WDDM/TCC foreign-load classification, then promote only
    # PIDs whose current membership in this exact kernel Job is verified.
    result = process_rows(stdout, None, mode)
    for row in result['rows']:
        if ownership(row['pid']):
            row.update(owned=True, blocking_foreign_compute=False, classification='owned_job_member')
    result['blocking_foreign_pids'] = [r['pid'] for r in result['rows'] if r['blocking_foreign_compute']]
    result['unknown_load'] = any(r['classification'] == 'unknown_WDDM_desktop_or_compute' for r in result['rows'])
    return result


def processes(gpu, mode, ownership, timeout):
    raw = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=timeout)
    return owned_process_rows(raw.stdout, mode, ownership)


def resource_reason(sample, before_free, budget, disk_free, foreign):
    if foreign: return 'foreign_gpu_load'
    if disk_free < 1024**3: return 'disk_reserve'
    if sample['memory.free'] < 768 or before_free - sample['memory.free'] > budget: return 'memory_budget'
    return None


def reserve_attempt(root, seal_sha, output, mode):
    # Called with the existing GPU lock held. Slots are immutable and failures
    # consume their slot; no hidden retry or change to the original round plan.
    folder = child(root, 'runs/local_profile_attempts/' + seal_sha)
    folder.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        path = folder / f'attempt{i}.json'
        if not path.exists():
            write_new(path, {'seal_sha256': seal_sha, 'output': str(output), 'profile': mode,
                             'attempt': i, 'max_attempts': 2, 'owner_pid': os.getpid()})
            return record(path)
    raise ValueError('Two profiling launch attempts already reserved for this seal; no automatic budget extension')


def prepare(root, seal_path, reference, mode, nsys):
    seal = verify_seal(root, seal_path)
    identity = seal['programs']['active']
    require(Path(identity['exe']['path']).resolve().is_relative_to((root / 'build/local_step3_20261006').resolve()),
            'Only the sealed Step3 standalone program is allowed')
    req, res = read(reference/'requested.json'), read(reference/'result.json')
    verify_files(reference, read(reference/'evidence.json')['files'])
    require(read(reference/'config_validation.json')['passed'] is True, 'Reference native configuration failed')
    require(req['source_digest'] == identity['source_digest'] and req['exe_sha256'] == identity['exe']['sha256'],
            'Reference does not match sealed source/executable')
    c = derive_config(req, res, mode)
    profiler = record(nsys)
    version = subprocess.run([str(nsys), '--version'], capture_output=True, text=True, check=True, timeout=10)
    profiler['version'] = version.stdout.strip()
    require('Nsight Systems' in profiler['version'], 'Unexpected profiler identity')
    own = [record(Path(__file__).parent/n) for n in ('profiler_windows.py', 'windows_owned_job.py', 'profiler_windows_contracts_test.py')]
    reference_ids = [record(reference/n) for n in ('requested.json', 'result.json', 'evidence.json', 'config_validation.json')]
    return seal, identity, c, profiler, own, reference_ids


def capture_evidence(out):
    with (out/'trace/frames.csv').open(encoding='utf-8-sig', newline='') as f:
        frames = list(csv.DictReader(f))
    require([int(r['frame']) for r in frames] == list(range(1, 60)), 'Complete ordered 59-frame output required')
    require(all(math.isfinite(float(r['solver_ms'])) and float(r['solver_ms']) > 0 for r in frames), 'Invalid frame timing')
    stats = read(out/'output/stats.json')['frames']
    require(len(stats) == 59 and not any(f.get('newton_exit') == 'iteration_limit' for f in stats), 'Incomplete/capped solver stats')
    raw = (out/'final.bin').read_bytes()
    require(bool(raw) and len(raw) % 8 == 0 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<d', raw)), 'Nonfinite final state')
    with (out/'cost.jsonl').open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    require(rows and all(r.get('schema') == 'gipc.cost.v1' and r.get('frame') == 57 for r in rows), 'Missing/wrong cost frame')
    require(all(r.get('nvtx_available') is True and r.get('gpu_events_enabled') is False and
                r.get('gpu_interval_ms') is None and r.get('operator_probe_enabled') is False and
                r.get('sample_kind') == 'production' and r.get('measurement_mode') == 'nvtx_cpu_only' for r in rows),
            'NVTX absent or unexpected heavy instrumentation')
    require(all(isinstance(r.get('cpu_submit_ms'), (int, float)) and math.isfinite(r['cpu_submit_ms']) and
                r['cpu_submit_ms'] >= 0 for r in rows), 'Invalid CPU scope time')
    ids = [r['scope_id'] for r in rows]
    require(len(ids) == len(set(ids)) and all(r['parent_scope_id'] == 0 or r['parent_scope_id'] in ids for r in rows),
            'Invalid scope ancestry')
    outer = [r for r in rows if r['stage'] == 'ipc.physical_frame']
    require(len(outer) == 1 and outer[0]['parent_scope_id'] == 0, 'One completed physical-frame scope required')
    reports = list(out.glob('*.nsys-rep'))
    require(len(reports) == 1 and reports[0].stat().st_size > 0, 'Missing/ambiguous/empty Nsight report')
    return {'passed': True, 'recorded_frames': 59, 'selected_frame': 57, 'cost_scope_count': len(rows),
            'nvtx_available': True, 'cost_trace': record(out/'cost.jsonl'), 'profiler_report': record(reports[0]),
            'scope': 'Native completed NVTX marker plus Nsight report presence; SQLite GPU activity inspection is a separate offline step.'}


def execute(root, seal_path, reference, out, mode, nsys, gpu):
    require(os.name == 'nt' and gpu >= 0, 'Windows and nonnegative GPU index required')
    require(not out.exists(), 'Output already exists; preserve evidence')
    result = {'status': 'launcher_failed', 'raw_status': 'not_launched', 'recorded_frames': 0,
              'heavy_diagnostics': True, 'diagnostic_only': True, 'timing_is_diagnostic': True,
              'performance_certified': False, 'physical_quality_certified': False,
              'capture_verified': False, 'automatic_next_round': False, 'timeout_seconds': TIMEOUT,
              'timing_scope': 'Instrumented WDDM desktop diagnostic, never paired speed/quality certification.'}
    samples = []; process_log = []; job = None; started = None
    with gpu_lock(root):
        # No profiler target exists during preflight. Failed identity checks do
        # not consume a launch slot; the independent output still records failure.
        (out/'output').mkdir(parents=True)
        try:
            seal, identity, c, profiler, own, reference_ids = prepare(root, seal_path, reference, mode, nsys)
            model = driver_model(gpu)
            before = [gpu_query(gpu)]; time.sleep(.25); before.append(gpu_query(gpu))
            pre = processes(gpu, model, lambda pid: False, 10)
            require(not pre['blocking_foreign_pids'] and (model == 'WDDM' or all(g['utilization.gpu'] <= 5 for g in before)),
                    'Foreign compute/load preflight failed; no process stopped')
            free = before[-1]['memory.free']; budget = min(.75*free, free-1536)
            require(budget >= 1024 and shutil.disk_usage(out).free >= 4*1024**3, 'Memory/disk reserve insufficient')
            result.update(gpu_before=before, gpu_memory_budget_mib=budget, gpu_driver_model=model,
                          desktop_load_uncontrolled=model == 'WDDM', prelaunch_processes=pre,
                          process_telemetry_capability_gap='WDDM N/A process memory cannot exclude other desktop/compute load.')
            exe = Path(identity['exe']['path']); env = environment(c, out)
            env['CUDA_VISIBLE_DEVICES'] = before[-1]['uuid']
            command = nsys_command(nsys, exe, out, mode)
            attempt = reserve_attempt(root, sha(seal_path), out, mode)
            write_new(out/'requested.json', {'requested_config': c, 'expanded_config': c, 'config_sha256': digest(c),
                'binary': 'active', 'source_digest': identity['source_digest'], 'exe_sha256': identity['exe']['sha256'],
                'reference_run': str(reference), 'reference_evidence': reference_ids, 'seal': record(seal_path),
                'diagnostic_delta_keys': sorted(DIAGNOSTIC_KEYS), 'from_zero': True, 'command': command,
                'profiler': profiler, 'new_tools': own, 'attempt': attempt, 'gpu_index': gpu,
                'environment': {k:v for k,v in env.items() if k.startswith('GIPC_') or k == 'CUDA_VISIBLE_DEVICES'},
                'resource_policy': {'absolute_timeout_seconds': TIMEOUT, 'memory_budget_mib': budget,
                    'memory_budget_formula': 'min(.75*free_before,free_before-1536) >=1024 MiB',
                    'memory_free_min_mib': 768, 'disk_before_bytes': 4*1024**3, 'disk_during_bytes': 1024**3,
                    'sample_interval_seconds': .5, 'cleanup_grace_seconds': 5, 'max_launch_attempts_per_seal': 2}})
            write_new(out/'build_manifest.json', identity)
            started = time.monotonic(); deadline = started + TIMEOUT
            job = OwnedJob()
            status = 'running'
            with (out/'run.log').open('xb') as log:
                job.launch(command, out, env, log)
                result['raw_status'] = 'launched'
                write_new(out/'process.json', {'pid': job.pid, 'command': command, 'lifecycle': job.lifecycle,
                                             'ownership': 'Windows Job; suspended assignment; kill-on-close; no breakaway'})
                while not job.finished():
                    try:
                        remaining(deadline)
                        process_log = job.observe()
                        sample = gpu_query(gpu, timeout=remaining(deadline))
                        require(sample['uuid'] == before[-1]['uuid'], 'GPU identity changed during profiling')
                        current = processes(gpu, model, job.owns_pid, remaining(deadline))
                        remaining(deadline)
                        samples.append(sample | {'elapsed_seconds': time.monotonic()-started, 'gpu_processes': current})
                        reason = resource_reason(sample, free, budget, shutil.disk_usage(out).free, current['blocking_foreign_pids'])
                        if reason:
                            status = reason; break
                        if job.poll() not in (None, 0):
                            status = 'profiler_failed'; break
                    except subprocess.TimeoutExpired:
                        status = 'timeout' if time.monotonic() >= deadline else 'monitor_timeout'; break
                    time.sleep(max(0, min(.5, deadline-time.monotonic())))
                if status != 'running': job.terminate()
                process_log = job.observe()
                exit_code = job.poll()
                if status == 'running':
                    status = 'completed' if exit_code == 0 else 'profiler_failed'
                    solver = [p for p in process_log if Path(p['image']).resolve() == exe.resolve()]
                    if not solver or any(p['exit_code'] != 0 for p in solver): status = 'solver_not_observed_or_failed'
            result.update(raw_status=status, status=status, exit_code=exit_code,
                          wall_seconds=time.monotonic()-started, owned_processes=process_log)
            verify_seal(root, seal_path); verify(own + reference_ids)
            verify([{k:v for k,v in profiler.items() if k != 'version'}])
        except BaseException as error:
            result.update(status='launcher_or_monitor_failed', error=type(error).__name__+': '+str(error))
        finally:
            if job is not None:
                try:
                    if not job.finished(): job.terminate()
                    result['owned_processes'] = job.observe()
                    result['cleanup_owned_job_empty'] = job.finished()
                    result['root_exit_code_after_cleanup'] = job.poll()
                except BaseException as error:
                    result.update(status='cleanup_failed', cleanup_error=type(error).__name__+': '+str(error))
                finally:
                    try:
                        job.close()
                    except BaseException as error:
                        result.update(status='cleanup_failed', close_error=type(error).__name__+': '+str(error))
            result['gpu_samples'] = samples
            if started is not None: result['wall_seconds'] = time.monotonic()-started
            write_new(out/'raw_result.json', result)
        if result['status'] == 'completed':
            try:
                evidence = capture_evidence(out)
                result.update(recorded_frames=59, finite=True)
                # Existing config validator reads result.json. This provisional
                # state is private to the new output; raw_result is immutable.
                write_new(out/'result.json', result)
                check = validate(out); write_new(out/'config_validation.json', check)
                require(check['passed'], 'Native requested/resolved execution mismatch')
                write_new(out/'capture_validation.json', evidence)
                result['capture_verified'] = True
            except Exception as error:
                result.update(status='capture_validation_failed', capture_error=type(error).__name__+': '+str(error))
        path = out/'result.json'
        if path.exists(): path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        else: write_new(path, result)
        write_new(out/'evidence.json', {'files': inventory(out, [p for p in out.rglob('*') if p.is_file()])})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'run'))
    p.add_argument('--seal', default='build/local_step3_20261006/local_diagnostic_seal.json')
    p.add_argument('--reference-run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=('node', 'graph'), default='node')
    p.add_argument('--nsys', type=Path, default=NSYS)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args(); root = ROOT.resolve()
    seal, reference, out = (child(root, x) for x in (a.seal, a.reference_run, a.output))
    require(out.is_relative_to(root/'runs') and out != root/'runs', 'New output must be below runs/')
    require(not out.exists(), 'Output exists')
    if a.action == 'plan':
        _, identity, c, profiler, _, _ = prepare(root, seal, reference, a.mode, a.nsys)
        print(json.dumps({'config': c, 'command': nsys_command(a.nsys, Path(identity['exe']['path']), out, a.mode),
                          'profiler': profiler, 'gpu_started': False, 'timeout_seconds': TIMEOUT}, indent=2))
        return 0
    result = execute(root, seal, reference, out, a.mode, a.nsys, a.gpu)
    print(json.dumps({'status': result['status'], 'capture_verified': result['capture_verified'],
                      'output': str(out), 'automatic_next_round': False}))
    return 0 if result['status'] == 'completed' and result['capture_verified'] else 2


if __name__ == '__main__': raise SystemExit(main())
