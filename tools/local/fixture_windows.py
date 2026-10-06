"""Finite Windows fixture launcher. No build, cleanup, retry, or speed certification.

Historical mas_cholesky inputs replay M/probes, not complete A/b/M PCG solves.
Use fixed for the native two-frame current-system host/Graph diagnostic instead.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from config import sha, digest

TIMEOUT = 120.0
SAMPLE_INTERVAL = .5
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
KINDS = ('component', 'pcg_guard', 'pcg_chunk', 'contact_pool', 'mas_cholesky', 'fixed')


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe_path(path):
    p = Path(path).absolute()
    for ancestor in (p, *p.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, 'is_junction', lambda: False)(),
                'Linked path rejected: ' + str(ancestor))
    return p.resolve()


def record(path):
    p = safe_path(path)
    a = p.stat()
    value = sha(p)
    b = p.stat()
    require((a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns), 'Identity changed while hashing')
    return {'path': str(p), 'bytes': b.st_size, 'sha256': value}


def identity(exe, kind, prefix):
    sources = [ROOT / 'CMakeLists.txt'] + sorted(p for p in (ROOT / 'StiffGIPC').rglob('*') if p.is_file())
    inputs = sorted((ROOT / 'Assets').rglob('*')) if kind == 'fixed' else []
    if prefix:
        require(Path(str(prefix) + '_meta.json').is_file(), 'Fixture prefix has no _meta.json')
        inputs = sorted(prefix.parent.glob(prefix.name + '_*'))
    return {'sources': [record(p) for p in sources], 'exe': record(exe),
            'dlls': [record(p) for p in sorted(exe.parent.glob('*.dll'))],
            'inputs': [record(p) for p in inputs if p.is_file()],
            'scope': 'Current source and adjacent DLL hashes; not proof of clean compilation or complete OS/driver DLL closure.'}


def fixture_environment(kind, out, prefix=None, scene=None, mas='legacy', restrict='serial', action=None):
    require(kind in KINDS, 'Unknown fixture kind')
    require((prefix is not None) == (kind == 'mas_cholesky'), '--prefix is required only for mas_cholesky')
    require((scene is not None) == (kind == 'fixed'), '--scene is required only for fixed')
    require(scene is None or scene in ('cloth_fixed_bunny_l', 'bunny_cloth_bunny_l'), 'Unsupported fixed scene')
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIPC_')}
    report = out / 'fixture.json'
    if kind in ('component', 'pcg_guard', 'pcg_chunk', 'contact_pool'):
        key = {'component': 'GIPC_VALIDATE_COMPONENTS', 'pcg_guard': 'GIPC_PCG_GUARD_FIXTURE',
               'pcg_chunk': 'GIPC_PCG_CHUNK_GUARD_FIXTURE',
               'contact_pool': 'GIPC_CONTACT_POOL_FIXTURE'}[kind]
        env[key] = str(report)
    elif kind == 'mas_cholesky':
        env.update(GIPC_MAS_CHOLESKY_FIXTURE=str(prefix), GIPC_MAS_FIXTURE_OUTPUT=str(out / 'mas'),
                   GIPC_MAS_RESTRICT_MODE=restrict, GIPC_MAS_FACTOR_ACTION=action or 'factor_inverse')
        report = out / 'mas/fixture.json'  # Native creates the fresh mas subdirectory.
    else:
        env.update(GIPC_SCENE=scene, GIPC_CONTACT_BACKEND='ipc', GIPC_STEPS='2', GIPC_DT='0.01',
                   GIPC_NEWTON_TOL='0.01', GIPC_PCG_TOL='0.0001', GIPC_IPC_TERMINATION='legacy',
                   GIPC_IPC_CUMULATIVE_TOL='0.01', GIPC_IPC_MIN_UPDATES='6',
                   GIPC_PCG_EXECUTION='conditional_graph', GIPC_PCG_PRECONDITIONER='mas',
                   GIPC_MAS_CHOLESKY=str(int(mas == 'cholesky')), GIPC_MAS_WIDE_APPLY='0',
                   GIPC_MAS_INVERSE64='0', GIPC_PCG_FUSED_DIAG_UPDATE='0', GIPC_LEGACY_RESTRICT='atomic',
                   GIPC_MAS_RESTRICT_MODE=restrict,
                   GIPC_MAS_FACTOR_ACTION=action or ('factor_inverse' if mas == 'cholesky' else 'triangular'),
                   GIPC_FIXED_STUDY_DIR=str(out / 'fixed'), GIPC_FIXED_STUDY_FRAMES='2',
                   GIPC_FIXED_STUDY_DIRECTIONS='1', GIPC_FIXED_STUDY_COMPACT='1', GIPC_MAS_SNAPSHOT='1',
                   GIPC_RESOLVED_CONFIG=str(out / 'resolved_config.json'), GIPC_DUMP_STATE=str(out / 'final.bin'),
                   GIPC_OUTPUT_PATH=str(out / 'output'))
        report = out / 'fixed/f2_n1_study.json'
    return env, report


def gpu_query(gpu, timeout=5):
    fields = ['uuid', 'name', 'driver_version', 'utilization.gpu', 'memory.free', 'memory.total',
              'driver_model.current']
    p = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=' + ','.join(fields),
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True,
                       timeout=timeout, creationflags=NO_WINDOW)
    rows = list(csv.reader(p.stdout.splitlines(), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == len(fields), 'Expected one complete GPU row')
    result = dict(zip(fields, rows[0]))
    for key in ('utilization.gpu', 'memory.free', 'memory.total'):
        result[key] = float(result[key])
        require(math.isfinite(result[key]) and result[key] >= 0, 'Invalid GPU telemetry')
    result['driver_model.current'] = result['driver_model.current'].strip().upper()
    require(result['driver_model.current'] in ('WDDM', 'TCC'), 'Unrecognized Windows GPU driver model')
    return result


def process_rows(stdout, own_pid, driver_model):
    """Same evidence boundary as local windows_runner; N/A is not CUDA proof."""
    rows, foreign = [], []
    for values in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if not values:
            continue
        require(len(values) == 3, 'Incomplete GPU process query')
        pid, name, raw = int(values[0]), values[1].strip(), values[2].strip()
        try:
            used = float(raw)
        except ValueError:
            used = None
        require(used is None or (math.isfinite(used) and used >= 0), 'Invalid process GPU memory')
        solver = name.replace('\\', '/').split('/')[-1].lower() in ('gipc.exe', 'gipc')
        blocking = pid != own_pid and (driver_model != 'WDDM' or solver or (used is not None and used > 0))
        if blocking:
            foreign.append(pid)
        rows.append({'pid': pid, 'process_name': name, 'used_gpu_memory_mib': used, 'raw_memory': raw,
                     'owned': pid == own_pid, 'blocking_foreign_compute': blocking,
                     'classification': 'owned' if pid == own_pid else 'foreign_compute_evidence' if blocking
                     else 'unknown_WDDM_desktop_or_compute'})
    return {'rows': rows, 'blocking_foreign_pids': foreign,
            'unknown_load': any(r['classification'] == 'unknown_WDDM_desktop_or_compute' for r in rows),
            'capability_gap': ('WDDM process memory may be unavailable; N/A does not exclude concurrent compute. '
                               'No exclusive-load or speed certification.' if driver_model == 'WDDM' else None)}


def gpu_processes(gpu, own_pid, driver_model, timeout=5):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu),
                             '--query-compute-apps=pid,process_name,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True,
                            timeout=timeout, creationflags=NO_WINDOW)
    return process_rows(result.stdout, own_pid, driver_model)


def initial_load_gate(before, diagnostic_under_desktop_load):
    require(len(before) == 2 and before[0]['uuid'] == before[1]['uuid'], 'GPU identity changed')
    modes = [x['driver_model.current'] for x in before]
    require(modes[0] == modes[1] and modes[0] in ('WDDM', 'TCC'), 'GPU driver model changed or unavailable')
    permitted = bool(diagnostic_under_desktop_load and modes == ['WDDM', 'WDDM'])
    high_load = any(x['utilization.gpu'] > 5 for x in before)
    require(not high_load or permitted,
            'GPU load gate failed; high load requires explicit WDDM diagnostic option')
    return {'diagnostic_under_desktop_load': bool(diagnostic_under_desktop_load),
            'desktop_load_gate_override': permitted and high_load,
            'load_controlled': False,
            'diagnostic_only': True}


def stop_owned(process):
    """Windows terminate/kill address this Popen handle only, never foreign PIDs."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def monitor(process, out, gpu, before, budget, start, samples):
    deadline = start + TIMEOUT
    next_sample = start
    while process.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            return 'timeout'
        if now < next_sample:
            time.sleep(min(next_sample - now, deadline - now))
            continue
        try:
            sample = gpu_query(gpu, timeout=min(5, deadline - now))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 'timeout'
            sample['processes'] = gpu_processes(gpu, process.pid, sample['driver_model.current'],
                                               timeout=min(5, remaining))
        except subprocess.TimeoutExpired:
            return 'timeout' if time.monotonic() >= deadline else 'monitor_timeout'
        now = time.monotonic()
        sample.update(elapsed_seconds=now - start, disk_free_bytes=shutil.disk_usage(out).free)
        samples.append(sample)
        if now >= deadline:
            return 'timeout'
        if sample['uuid'] != before['uuid'] or sample['driver_model.current'] != before['driver_model.current']:
            return 'gpu_identity_changed'
        if sample['processes']['blocking_foreign_pids']:
            return 'foreign_gpu_load'
        if sample['disk_free_bytes'] < 1024**3:
            return 'disk_reserve'
        if sample['memory.free'] < 768 or before['memory.free'] - sample['memory.free'] > budget:
            return 'memory_budget'
        next_sample = max(next_sample + SAMPLE_INTERVAL, now)
    if time.monotonic() >= deadline:
        return 'timeout'
    return 'completed' if process.returncode == 0 else 'failed'


def report_evidence(kind, path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=invalid_constant)
    if kind != 'fixed':
        key = 'pass' if kind == 'contact_pool' else 'passed'
        return {'native_passed': value.get(key) is True, 'report': record(path),
                'scope': 'M action and saved probes only; not full PCG replay' if kind == 'mas_cholesky' else 'Native fixture contract'}
    runs = value.get('runs', [])
    wanted = {(mode, tol, repeat) for mode in ('host', 'graph') for tol in (1e-4, 1e-16) for repeat in (1, 2)}
    actual = {(r.get('mode'), r.get('rho_tolerance'), r.get('repeat')) for r in runs}
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    clean = bool(runs) and all(r.get('error') == '' and r.get('limit') is False and
                             finite(r.get('true_relative_residual')) for r in runs)
    preserved = value.get('system_unchanged') is True and value.get('primary_restored_bitwise') is True
    return {'native_passed': preserved and clean and len(runs) == 8 and actual == wanted,
            'report': record(path), 'system_and_primary_preserved': preserved,
            'runs': runs, 'expected_modes_and_repeats': len(runs) == 8 and actual == wanted,
            'numerical_acceptance_certified': False,
            'scope': 'Current frozen A/b/M diagnostic; all run errors/limits and true residuals retained. No new residual threshold or whole-trajectory acceptance inferred.'}


@contextmanager
def gpu_lock():
    import msvcrt
    # Share the same byte lock with windows_runner and future profiling tasks.
    path = ROOT / 'runs/.local_gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def execute(args):
    exe, out = safe_path(args.exe), safe_path(args.output)
    prefix = safe_path(args.prefix) if args.prefix else None
    env, report = fixture_environment(args.kind, out, prefix, args.scene, args.mas, args.restrict, args.factor_action)
    require(exe.is_file(), 'Executable missing')
    require(not out.exists(), 'Output must be fresh; no overwrite or resume')
    out.mkdir(parents=True, exist_ok=False)
    process = None
    samples = []
    result = {'status': 'launcher_failed', 'fixture_kind': args.kind, 'performance_certified': False,
              'physical_quality_certified': False, 'timeout_seconds': TIMEOUT,
              'diagnostic_only': True, 'load_controlled': False}
    start = None
    try:
        ident = identity(exe, args.kind, prefix)
        request = {'kind': args.kind, 'command': [str(exe)], 'cwd': str(ROOT), 'timeout_seconds': TIMEOUT,
                   'diagnostic_under_desktop_load': bool(getattr(args, 'diagnostic_under_desktop_load', False)),
                   'gpu_index': args.gpu, 'source_digest': digest(ident['sources']),
                   'exe_sha256': ident['exe']['sha256'], 'runner_sha256': sha(__file__),
                   'environment': {k: v for k, v in env.items() if k.upper().startswith('GIPC_')},
                   'resource_policy': {'disk_before_bytes': 4*1024**3, 'disk_during_bytes': 1024**3,
                                       'memory_budget_formula': 'min(.75*free,free-1536) MiB >=1024',
                                       'memory_free_min_mib': 768, 'sample_interval_seconds': SAMPLE_INTERVAL}}
        write_new(out / 'build_manifest.json', ident)
        write_new(out / 'requested.json', request)
        before = [gpu_query(args.gpu)]
        time.sleep(.25)
        before.append(gpu_query(args.gpu))
        budget = min(.75 * before[-1]['memory.free'], before[-1]['memory.free'] - 1536)
        result.update(gpu_before=before, gpu_memory_budget_mib=budget)
        result.update(initial_load_gate(before, getattr(args, 'diagnostic_under_desktop_load', False)))
        processes = gpu_processes(args.gpu, None, before[-1]['driver_model.current'])
        result['gpu_processes_before'] = processes
        result['process_telemetry_capability_gap'] = processes['capability_gap']
        require(not processes['blocking_foreign_pids'], 'Foreign GPU compute evidence; no foreign process stopped')
        require(budget >= 1024 and shutil.disk_usage(out).free >= 4*1024**3, 'Initial GPU memory/disk reserve failed')
        env['CUDA_VISIBLE_DEVICES'] = before[-1]['uuid']
        result['cuda_visible_devices'] = env['CUDA_VISIBLE_DEVICES']
        if args.kind == 'fixed':
            (out / 'fixed').mkdir()
            (out / 'output').mkdir()
        with (out / 'run.log').open('xb') as log:
            start = time.monotonic()
            process = subprocess.Popen([str(exe)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
            write_new(out / 'process.json', {'pid': process.pid, 'exe': str(exe)})
            result['status'] = monitor(process, out, args.gpu, before[-1], budget, start, samples)
            stop_owned(process)
            result['exit_code'] = process.wait(timeout=2)
        require(identity(exe, args.kind, prefix) == ident, 'Source/binary/DLL/input changed during fixture')
        result['identity_unchanged'] = True
        if report.is_file():
            result['fixture'] = report_evidence(args.kind, report)
        if result['status'] == 'completed' and not result.get('fixture', {}).get('native_passed'):
            result['status'] = 'fixture_failed_or_incomplete'
    except Exception as error:
        result.update(status='launcher_or_monitor_failed', error=type(error).__name__ + ': ' + str(error))
    finally:
        stop_owned(process)
        result['wall_seconds'] = time.monotonic() - start if start is not None else None
        result['gpu_samples'] = samples
        write_new(out / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--kind', required=True, choices=KINDS)
    p.add_argument('--prefix', type=Path)
    p.add_argument('--scene', choices=('cloth_fixed_bunny_l', 'bunny_cloth_bunny_l'))
    p.add_argument('--mas', choices=('legacy', 'cholesky'), default='legacy')
    p.add_argument('--restrict', choices=('serial', 'warp'), default='serial')
    p.add_argument('--factor-action', choices=('triangular', 'factor_inverse'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--diagnostic-under-desktop-load', action='store_true',
                   help='Permit high utilization only under twice-observed WDDM; diagnostic only, no speed certification')
    args = p.parse_args()
    require(os.name == 'nt', 'This launcher requires Windows')
    with gpu_lock():
        result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}, ensure_ascii=False))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
