"""One bounded serial experiment; old evidence is never overwritten."""
import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import time
import psutil
from config import ROOT, expand, environment, read, sha, digest

def gpu():
    q = 'utilization.gpu,memory.free,temperature.gpu,power.draw,clocks.sm'
    r = subprocess.run(['nvidia-smi', '--query-gpu=' + q, '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, timeout=10)
    return dict(zip(['utilization', 'free_mib', 'temperature_c', 'power_w', 'sm_mhz'],
                    map(float, r.stdout.strip().split(','))))

def execute(config, name, binary='active', program_defaults=False):
    if not name or any(x in name for x in ('/', '\\', ':')) or name in ('.', '..'):
        raise ValueError('name must be a single directory name')
    c = expand(config)
    out = ROOT / 'runs/active' / name
    if out.exists():
        raise FileExistsError(out)
    manifests = {'active': 'builds/active/manifest.json', 'frozen': 'manifests/perf_v54_local.json',
                 'base': 'manifests/perf_v34.json','reference':'manifests/ipc_reference_20261005.json',
                 'base_observed':'manifests/base_observed_20261005.json'}
    builds = {'active': 'active', 'frozen': 'local-v54', 'base': 'local-base',
              'reference':'ipc-reference','base_observed':'base-observed'}
    exe = ROOT / 'builds' / builds[binary] / 'Release/gipc.exe'
    manifest_path = ROOT / manifests[binary]
    manifest = read(manifest_path)
    for entry in manifest['files']:
        if binary in ('base','base_observed') and not entry['path'].startswith(('sources/stiff_base/','sources/stiff_base_observed/')):
            continue
        if binary=='reference' and entry['path'].startswith(('sources/stiff_active/','tools/active/')):continue
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('Source identity mismatch: ' + entry['path'])
    if binary=='reference':
        archive=manifest['archive']['overlay']
        if sha(ROOT/archive['path'])!=archive['sha256']:raise RuntimeError('Reference source archive changed')
        for runtime in read(ROOT/'manifests/ipc_reference_runtime_20261005.json'):
            if sha(ROOT/runtime['path'])!=runtime['sha256']:
                raise RuntimeError('Reference runtime identity mismatch: '+runtime['path'])
    exe_hash = sha(exe)
    if exe_hash != manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']:
        raise RuntimeError('Binary identity mismatch')
    if binary in ('base','base_observed') and any((c['backend'] != 'ipc', c['execution'] != 'host', c['mas'] != 'legacy',
                                 c['refit'], c['batch'], c['reuse'])):
        raise ValueError('Official base cannot represent requested features')
    if binary!='active' and (c['ipc_termination']!='legacy' or c['ipc_cumulative_tol']!=c['ipc_newton_tol'] or c['ipc_min_updates']!=6 or c['ipc_residual_shadow'] or c['ipc_terminal_audit_frame']):
        raise ValueError('Frozen programs cannot execute new IPC stopping modes')
    if binary != 'active' and c['toi_remaining_fraction_tol'] != c['ipc_newton_tol']:
        raise ValueError('Frozen binaries cannot separate stopping tolerances')
    if binary!='active' and (c['mas_restrict']!='serial' or c['fixed_restrict_study']):
        raise ValueError('Frozen binaries cannot execute active restriction candidates')
    if binary!='active' and c['fixed_mas_stage_study']:
        raise ValueError('Frozen binaries cannot execute active MAS stage probes')
    if binary!='active' and (c['legacy_restrict']!='atomic' or c['fixed_legacy_restrict_study']):
        raise ValueError('Frozen binaries cannot execute ordered legacy restriction')
    if binary!='active' and not c['cost_events']:
        raise ValueError('Frozen binaries cannot execute NVTX-only cost mode')
    if binary!='active' and (c['mas_fused_dot'] or c['fixed_mas_dot_study'] or c['discrete_bvh_refit']
        or c['discrete_bvh_validate'] or c['discrete_bvh_rebuild_interval']!=8
        or c['spmv_fused_quadratic'] or c['fixed_spmv_quadratic_study']):
        raise ValueError('Frozen binaries cannot execute report component candidates')
    if binary!='active' and (c['bvh_eligibility'] or c['bvh_eligibility_validate']):
        raise ValueError('Frozen binaries cannot execute BVH eligibility')
    if binary!='active' and (c['bounded_ccd'] or c['bounded_ccd_validate']):
        raise ValueError('Frozen binaries cannot execute bounded CCD')
    if binary!='active' and (c['contact_pool'] or c['contact_pool_validate']):
        raise ValueError('Frozen binaries cannot execute contact pool')
    if binary!='active' and c['mu_coordinates']!='generalized':
        raise ValueError('Frozen binaries cannot execute the world penalty candidate')
    if binary not in ('active','reference') and c['mas_factor_action']!='triangular':
        raise ValueError('Frozen binaries cannot execute the factor inverse action')
    if binary!='active' and c['full_step_exit_probe'] is not None:
        raise ValueError('Frozen binaries cannot execute the selected exit probe')
    if program_defaults and (binary!='active' or c['backend']!='toi_al' or c['mas']!='cholesky'):
        raise ValueError('Native default test requires active TOI with stable MAS')
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError('Disk reserve <4 GiB')
    lock = ROOT / 'runs/active/.gpu.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Lock is exclusive across all common-runner processes. Do not auto-delete a stale lock.
    with lock.open('x') as f:
        f.write(json.dumps({'name': name, 'pid': __import__('os').getpid()}))
    proc = None
    try:
        before = [gpu()]
        time.sleep(1)
        before.append(gpu())
        budget = min(.75 * before[-1]['free_mib'], before[-1]['free_mib'] - 1536)
        if budget < 1024:
            raise RuntimeError('GPU memory reserve insufficient')
        (out / 'output').mkdir(parents=True)
        for directory in ('fixed','state_window'):
            if directory in c['diagnostics']:(out/directory).mkdir()
        env = environment(c, out)
        native_unset=['GIPC_MAS_FACTOR_ACTION','GIPC_TOI_MU_COORDINATES'] if program_defaults else []
        for key in native_unset:env.pop(key,None)
        command=[str(exe)]
        if c['profile']!='none':
            nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe'
            command=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                     '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
                     '--capture-range=nvtx','--nvtx-capture='+('ipc.physical_frame' if c['backend']=='ipc' else 'linear.total_including_diagnostics'),'--capture-range-end=stop',
                     '--cuda-graph-trace='+c['profile'],'--output='+str(out/'nsight'),str(exe)]
        req = {'requested_config': config, 'expanded_config': c, 'config_sha256': digest(c),
               'native_program_defaults_unset':native_unset,
               'environment': {k:v for k,v in env.items() if k.startswith('GIPC_')},
               'binary': binary, 'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
               'source_digest': manifest['source_digest'], 'exe_sha256': exe_hash,
               'runner_sha256': sha(__file__), 'gpu_before': before,
               'command':command,
               'timing_scope': 'shared desktop diagnostic; never certifies exclusive load',
               'memory_budget_mib': budget}
        (out / 'requested.json').write_text(json.dumps(req, indent=2))
        # Freeze a self-contained reference to the exact build before future active edits.
        shutil.copyfile(manifest_path, out / 'build_manifest.json')
        status = 'running'
        start = time.monotonic()
        samples = []
        with (out / 'run.log').open('wb') as log:
            proc = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            (out / 'process.json').write_text(json.dumps({'pid': proc.pid, 'exe': str(exe)}))
            while proc.poll() is None:
                g = gpu()
                samples.append(g)
                reason = ('timeout' if time.monotonic() - start > c['timeout_seconds'] else
                          'disk_reserve' if shutil.disk_usage(out).free < 1024**3 else
                          'memory_budget' if g['free_mib'] < 768 or before[-1]['free_mib']-g['free_mib'] > budget else None)
                if reason:
                    status = reason
                    terminate_owned(proc)
                    break
                time.sleep(.5)
            proc.wait(timeout=15)
        if status == 'running':
            status = 'completed' if proc.returncode == 0 else 'failed'
        result = {'status': status, 'exit_code': proc.returncode, 'wall_seconds': time.monotonic()-start,
                  'recorded_frames': 0, 'gpu_samples': samples, 'timing_is_diagnostic': True,
                  'heavy_diagnostics': bool(c['diagnostics'] or c['ipc_residual_shadow'] or
                      c['ipc_residual_cpu_audit'] or c['ipc_terminal_audit_frame'] or c['discrete_bvh_validate'] or
                      c['bvh_eligibility_validate'] or c['bounded_ccd_validate'] or c['contact_pool_validate']),
                  'performance_certified': False}
        if (out / 'trace/frames.csv').exists():
            with (out / 'trace/frames.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            result.update(recorded_frames=len(frames), solver_seconds=sum(float(f['solver_ms']) for f in frames)/1000)
        if (out / 'final.bin').exists():
            result['finite'] = all(math.isfinite(x[0]) for x in struct.iter_unpack('<d', (out/'final.bin').read_bytes()))
        result['resolved_config_written'] = (out/'resolved_config.json').exists()
        if sha(exe) != exe_hash:
            result['status'] = 'binary_changed_during_run'
        if status == 'completed' and result['recorded_frames'] != c['steps']:
            result['status'] = 'incomplete'
        if c['profile']!='none':
            reports=list(out.glob('*.nsys-rep'))
            result['profiler_reports']=[{'path':str(p),'sha256':sha(p)} for p in reports]
            if result['status']=='completed' and not reports:
                result['status']='profiling_missing_report'
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'run': name, **{k:v for k,v in result.items() if k != 'gpu_samples'}}), flush=True)
        return result
    except BaseException as error:
        if out.exists() and not (out/'result.json').exists():
            (out/'result.json').write_text(json.dumps({'status':'launcher_failed','recorded_frames':0,
                'error':str(error),'timing_is_diagnostic':True,'performance_certified':False},indent=2))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            terminate_owned(proc)
            proc.wait(timeout=15)
        lock.unlink(missing_ok=True)

def terminate_owned(proc):
    # A profiler owns a child simulator. Only this launch's descendants are stopped.
    try:
        children=psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children=[]
    for child in reversed(children):
        try:
            # Windows crash-reporting helpers can be protected children. They
            # are not simulator work and must not prevent stopping our handle.
            if __import__('pathlib').Path(child.exe()).is_relative_to(ROOT):child.terminate()
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if proc.poll() is None:proc.terminate()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--binary', choices=['active', 'frozen', 'base','reference','base_observed'], default='active')
    p.add_argument('--program-defaults',action='store_true',help='Omit only the two improved feature overrides and verify native defaults')
    a = p.parse_args()
    r = execute(read(a.config), a.name, a.binary,a.program_defaults)
    raise SystemExit(int(r['status'] != 'completed'))
