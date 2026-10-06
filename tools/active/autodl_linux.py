"""Linux-only, bounded AutoDL build/run adapter using active config contracts.

Does not install packages, connect SSH, infer quality approval, or certify speed.
Every invocation uses one explicit physical GPU index; no automatic stale-lock cleanup.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import struct
import subprocess
import time

from config import ROOT, digest, environment, expand, read, sha
from validate_run import validate


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def checked(command, log, timeout=1800, env=None, cwd=ROOT):
    with log.open('x') as stream:
        stream.write('ARGV ' + json.dumps(command) + '\n'); stream.flush()
        p = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=timeout)
        finally:
            stop_owned(p)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}); see {log}')


def verify_bundle():
    bundle = read(ROOT / 'autodl_bundle.json')
    if digest(bundle['files']) != bundle['source_digest']:
        raise RuntimeError('Bundle manifest digest mismatch')
    for row in bundle['files']:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != row['sha256']:
            raise RuntimeError('Bundle input mismatch: ' + row['path'])
    for stage, expected in bundle['plan_digests'].items():
        if stage not in ('smoke', 'window', 'pilot', 'audit', 'paired') or digest(read(ROOT / f'configs/active/autodl_{stage}.json')) != expected:
            raise RuntimeError('Packaged plan changed: ' + stage)
    return bundle


def gpu_query(index):
    keys = ['uuid', 'name', 'driver_version', 'utilization.gpu', 'memory.free',
            'temperature.gpu', 'power.draw', 'clocks.sm', 'compute_cap']
    p = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=' + ','.join(keys),
                        '--format=csv,noheader,nounits'], capture_output=True,
                       text=True, check=True, timeout=10)
    rows = list(csv.reader(p.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1:
        raise RuntimeError('Exactly one selected physical GPU is required')
    result = dict(zip(keys, rows[0]))
    for key in ('utilization.gpu', 'memory.free', 'temperature.gpu'):
        result[key] = float(result[key])
    return result


def compute_pids(index):
    p = subprocess.run(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'], capture_output=True,
                       text=True, check=True, timeout=10)
    return [int(line.strip()) for line in p.stdout.splitlines() if line.strip()]


def gpu_lock(name):
    path = ROOT / 'runs/active/.gpu.lock'
    write_new(path, {'name': name, 'pid': os.getpid(), 'platform': platform.platform()})
    return path


def build(gpu, arch, jobs):
    bundle = verify_bundle()
    root = ROOT / 'builds/autodl-active'
    if root.exists():
        raise FileExistsError('Build requires a fresh package and isolated build directory: ' + str(root))
    selected = gpu_query(gpu)
    detected = selected['compute_cap'].replace('.', '')
    if arch != detected:
        raise ValueError(f'Explicit arch {arch} differs from selected GPU capability {detected}')
    root.mkdir(parents=True)
    commands = []
    lock = gpu_lock('autodl-build-fixtures')
    try:
        for kind, source in [('base', 'stiff_base'), ('active', 'stiff_active'), ('validator', None)]:
            source_path = ROOT / 'sources' / source if source else ROOT / 'tools/validator'
            directory = root / kind
            configure = ['cmake', '-S', str(source_path), '-B', str(directory), '-G', 'Unix Makefiles',
                         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
            if source:
                configure.append('-DCMAKE_CUDA_ARCHITECTURES=' + arch)
            command = ['cmake', '--build', str(directory), '--parallel', str(jobs), '--verbose']
            if source:
                command.extend(['--target', 'gipc'])
            checked(configure, root / f'{kind}_configure.log')
            checked(command, root / f'{kind}_build.log')
            targets = ['gipc'] if source else ['validate_path', 'diagnose_first_path']
            for target in targets:
                if not (directory / target).is_file():
                    raise RuntimeError('Expected Linux target output missing: ' + str(directory / target))
            commands.extend([configure, command])
        # No preexisting Windows/old Linux binary is reused.
        binaries = {kind: root / kind / 'gipc' for kind in ('base', 'active')}
        units = []
        links = []
        for kind in ('base', 'active', 'validator'):
            directory = root / kind
            for row in read(directory / 'compile_commands.json'):
                argv = row.get('arguments') or shlex.split(row['command'])
                if '-o' not in argv:
                    raise RuntimeError('Actual object mapping unavailable')
                obj = (Path(row['directory']) / argv[argv.index('-o') + 1]).resolve()
                if not obj.is_relative_to(directory):
                    raise RuntimeError('Expected fresh object missing: ' + str(obj))
                # CMake also exports targets deliberately omitted from this build.
                if not obj.is_file() and 'CMakeFiles/gipc.dir' not in obj.as_posix():
                    continue
                if not obj.is_file():
                    raise RuntimeError('Expected gipc object missing: ' + str(obj))
                if 'CMakeFiles/gipc.dir' in obj.as_posix():
                    asset_tree = 'stiff_base' if kind == 'base' else 'stiff_perf_v50'
                    expected_assets = (ROOT / 'sources' / asset_tree / 'Assets').as_posix() + '/'
                    definitions = [arg.split('=', 1)[1].replace('"', '').replace('\\', '')
                                   for arg in argv if arg.startswith('-DGIPC_ASSETS_DIR=')]
                    if definitions != [expected_assets]:
                        raise RuntimeError('Unexpected compiled asset root: ' + repr(definitions))
                units.append(row | {'kind': kind, 'object': str(obj.relative_to(ROOT)), 'object_sha256': sha(obj)})
            for path in directory.rglob('link.txt'):
                links.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                              'command': path.read_text(),
                              'response_files': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'text': p.read_text()}
                                                 for p in path.parent.glob('*.rsp')]})
        if len({r['object'] for r in units}) != len(units) or not links:
            raise RuntimeError('Compile/link provenance incomplete or objects collide')
        verify_bundle()
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIPC_')}
        env['CUDA_VISIBLE_DEVICES'] = selected['uuid']
        if compute_pids(gpu):
            raise RuntimeError('Another compute process is active before GPU fixtures')
        for variable, target in [('GIPC_VALIDATE_COMPONENTS', 'components.json'),
                                 ('GIPC_PCG_GUARD_FIXTURE', 'guards.json')]:
            checked([str(binaries['active'])], root / (target + '.log'), timeout=60,
                    env=env | {variable: str(root / target)})
            result = read(root / target)
            if not result.get('passed', False):
                raise RuntimeError('GPU fixture did not explicitly pass: ' + target)
        checked([str(root / 'validator/validate_path'), '--self-test'], root / 'validator_selftest.log', timeout=60)
        tools = {}
        for name in ('cmake', 'nvcc', 'c++'):
            tools[name] = subprocess.run([name, '--version'], capture_output=True, text=True,
                                         timeout=10, check=True).stdout
        manifest = {'schema': 1, 'files': bundle['files'], 'source_digest': bundle['source_digest'],
                    'candidate_sha256': bundle['candidate_config_sha256'], 'gpu': selected,
                    'toolchain': tools, 'platform': platform.platform(), 'build_commands': commands,
                    'compiler_inputs': units, 'link_evidence': links,
                    'link_input_objects_and_archives': [
                        {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
                        for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.o', '.a')],
                    'binaries': {key: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
                                 for key, path in binaries.items()},
                    'validator_binaries': {name: {'path': str((root / 'validator' / name).relative_to(ROOT)),
                                                  'sha256': sha(root / 'validator' / name)}
                                           for name in ('validate_path', 'diagnose_first_path')}}
        write_new(root / 'manifest.json', manifest)
        print(json.dumps({'build': 'completed', 'objects': len(units), 'manifest': str(root / 'manifest.json')}))
    finally:
        lock.unlink(missing_ok=True)


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def execute(task, gpu, manifest, controlled_load):
    config = expand(task['config'])
    out = ROOT / 'runs/active' / task['name']
    if Path(task['name']).name != task['name'] or out.exists():
        raise ValueError('Run name must be fresh and a single component')
    binary = task['binary']; record = manifest['binaries'][binary]; exe = ROOT / record['path']
    if sha(exe) != record['sha256']:
        raise RuntimeError('Built executable identity changed')
    if config['profile'] != 'none':
        raise ValueError('Remote timing adapter does not enable Nsight')
    lock = gpu_lock(task['name']); process = None
    out.mkdir(parents=True); (out / 'output').mkdir()
    result = {'status': 'launcher_failed', 'recorded_frames': 0, 'performance_certified': False,
              'physical_quality_certified': False, 'timing_is_diagnostic': True}
    try:
        before = [gpu_query(gpu)]; time.sleep(1); before.append(gpu_query(gpu))
        foreign = compute_pids(gpu)
        if foreign or (controlled_load and any(g['utilization.gpu'] > 5 for g in before)):
            raise RuntimeError('GPU load gate failed before launch; no process was stopped')
        budget = min(.75 * before[-1]['memory.free'], before[-1]['memory.free'] - 1536)
        if budget < 1024 or shutil.disk_usage(ROOT).free < 4 * 1024**3:
            raise RuntimeError('Insufficient GPU memory or disk reserve')
        env = environment(config, out); env['CUDA_VISIBLE_DEVICES'] = before[-1]['uuid']
        request = {'requested_config': task['config'], 'expanded_config': config, 'config_sha256': digest(config),
                   'binary': binary, 'source_digest': manifest['source_digest'], 'exe_sha256': record['sha256'],
                   'runner_sha256': sha(__file__), 'environment': {k: v for k, v in env.items() if k.startswith('GIPC_')},
                   'command': [str(exe)], 'gpu_before': before, 'gpu_index': gpu, 'controlled_load_requested': controlled_load}
        write_new(out / 'requested.json', request); write_new(out / 'build_manifest.json', manifest)
        start = time.monotonic(); samples = []; status = 'running'
        with (out / 'run.log').open('xb') as log:
            process = subprocess.Popen([str(exe)], cwd=out, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write_new(out / 'process.json', {'pid': process.pid, 'exe': str(exe)})
            while process.poll() is None:
                sample = gpu_query(gpu); others = [p for p in compute_pids(gpu) if p != process.pid]
                samples.append(sample | {'foreign_compute_pids': others})
                status = ('timeout' if time.monotonic() - start > config['timeout_seconds'] else
                          'disk_reserve' if shutil.disk_usage(out).free < 1024**3 else
                          'memory_budget' if sample['memory.free'] < 768 or before[-1]['memory.free'] - sample['memory.free'] > budget else
                          'foreign_gpu_load' if others else 'running')
                if status != 'running':
                    stop_owned(process); break
                time.sleep(.5)
            process.wait(timeout=10)
        result.update(status=('completed' if process.returncode == 0 else 'failed') if status == 'running' else status,
                      exit_code=process.returncode, wall_seconds=time.monotonic() - start, gpu_samples=samples,
                      heavy_diagnostics=bool(config['diagnostics']))
        if (out / 'trace/frames.csv').exists():
            with (out / 'trace/frames.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            result.update(recorded_frames=len(frames), solver_seconds=sum(float(r['solver_ms']) for r in frames) / 1000)
        if (out / 'final.bin').exists():
            values = (out / 'final.bin').read_bytes()
            result['finite'] = bool(values) and len(values) % 8 == 0 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<d', values))
        if result['status'] == 'completed' and (result['recorded_frames'] != config['steps'] or not result.get('finite')):
            result['status'] = 'incomplete_or_nonfinite'
        if sha(exe) != record['sha256']:
            result['status'] = 'binary_changed_during_run'
    except Exception as error:
        result['error'] = str(error)
    finally:
        if process is not None:
            stop_owned(process)
        lock.unlink(missing_ok=True)
        write_new(out / 'result.json', result)
    if binary == 'active' and result['status'] == 'completed':
        validation = validate(out); write_new(out / 'config_validation.json', validation)
        if not validation['passed']:
            result['status'] = 'configuration_failed'
            (out / 'result.json').write_text(json.dumps(result, indent=2))
    elif binary == 'base' and result['status'] == 'completed':
        scene = read(out / 'output/scene.json')
        effective = scene['effective_run']
        checks = {field: effective[field] == config[key] for field, key in
                  [('dt', 'dt'), ('newton_tol', 'ipc_newton_tol'), ('pcg_tol', 'pcg_rho_tol')]}
        checks['native_mas'] = scene['effective_scalar_fields']['preconditioner_type'] == 1
        write_new(out / 'config_validation.json', {'passed': all(checks.values()), 'checks': checks,
                                                  'scope': 'Official base runtime scalars; no active resolved-config contract'})
        if not all(checks.values()):
            result['status'] = 'configuration_failed'
            (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'run': task['name'], 'status': result['status'], 'frames': result['recorded_frames'],
                      'solver_seconds': result.get('solver_seconds')}), flush=True)
    return result


def accepted_gate(path, plan, manifest):
    gate = read(path)
    if gate.get('source_digest') != manifest['source_digest'] or gate.get('candidate_sha256') != plan['candidate_sha256']:
        raise ValueError('Gate belongs to another candidate/build')
    accepted = set()
    required = ['quality_passed', 'baseline_repeat_range_frozen']
    if plan['stage'] == 'paired':
        required.append('load_controlled')
    for scene, record in gate.get('scenes', {}).items():
        allowed_review = ('pilot_audit',) if plan['stage'] == 'paired' else ('window', 'pilot_audit')
        if all(record.get(key) is True for key in required) and record.get('reviewed_stage') in allowed_review:
            evidence = record.get('evidence', [])
            if not evidence:
                raise ValueError('Gate needs actual reviewed evidence')
            for row in evidence:
                p = (ROOT / row['path']).resolve()
                if not p.is_relative_to(ROOT) or sha(p) != row['sha256']:
                    raise ValueError('Gate evidence mismatch')
            accepted.add(scene)
    return accepted


def gated_scenes(stage, gate_path, plan, manifest):
    scenes = {r['scene_key'] for r in plan['runs']}
    if stage in ('pilot', 'audit', 'paired'):
        if not gate_path:
            raise ValueError('Every 100-frame stage requires --gate with actual reviewed quality evidence')
        scenes &= accepted_gate(gate_path, plan, manifest)
        if not scenes:
            raise ValueError('No scene passed the required quality/baseline gates')
    return scenes


def run(stage, gpu, gate_path):
    verify_bundle()
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('GLUT needs DISPLAY; use xvfb-run -a python3 tools/active/autodl_linux.py run ...')
    plan_path = ROOT / f'configs/active/autodl_{stage}.json'; plan = read(plan_path)
    manifest = read(ROOT / 'builds/autodl-active/manifest.json')
    if plan['candidate_sha256'] != manifest['candidate_sha256']:
        raise ValueError('Plan and build candidate differ')
    scenes = gated_scenes(stage, gate_path, plan, manifest)
    report_path = ROOT / f'reports/active/AUTODL_{stage.upper()}_BATCH.json'
    report = {'plan_sha256': sha(plan_path), 'candidate_sha256': plan['candidate_sha256'],
              'source_digest': manifest['source_digest'], 'runs': [], 'performance_certified': False,
              'excluded_by_gate': sorted({r['scene_key'] for r in plan['runs']} - scenes)}
    failed = set()
    if stage != 'smoke':
        previous = read(ROOT / 'reports/active/AUTODL_SMOKE_BATCH.json')
        if previous.get('candidate_sha256') != plan['candidate_sha256'] or previous.get('source_digest') != manifest['source_digest']:
            raise ValueError('A matching remote smoke batch is required before longer runs')
        completed = {r['arm'] for r in previous['runs'] if r['result']['status'] == 'completed'}
        failed = {r['arm'] for r in plan['runs']} - completed
        report['skipped_failed_smoke_arms'] = sorted(failed)
    if stage in ('pilot', 'audit', 'paired'):
        windows = read(ROOT / 'reports/active/AUTODL_WINDOW_BATCH.json')
        if windows.get('candidate_sha256') != plan['candidate_sha256'] or windows.get('source_digest') != manifest['source_digest']:
            raise ValueError('A matching remote window batch is required before 100-frame runs')
        completed = {}
        for row in windows['runs']:
            if row['result']['status'] == 'completed':
                completed.setdefault(row['arm'], set()).add(row['repeat'])
        failed |= {r['arm'] for r in plan['runs'] if completed.get(r['arm'], set()) != {1, 2, 3}}
        report['skipped_incomplete_window_arms'] = sorted(failed)
    write_new(report_path, report)
    for task in plan['runs']:
        if task['scene_key'] not in scenes or task['arm'] in failed:
            continue
        result = execute(task, gpu, manifest, stage == 'paired')
        report['runs'].append(task | {'result': result})
        report_path.write_text(json.dumps(report, indent=2))
        if result['status'] != 'completed':
            failed.add(task['arm'])
    return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--gpu', type=int, default=0)
    b.add_argument('--arch', required=True); b.add_argument('--jobs', type=int, default=2)
    r = sub.add_parser('run'); r.add_argument('--gpu', type=int, default=0)
    r.add_argument('--stage', choices=['smoke', 'window', 'pilot', 'audit', 'paired'], required=True)
    r.add_argument('--gate', type=Path)
    sub.add_parser('verify')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        parser.error('Execution is Linux-only; preparation and selftest work locally')
    if args.command == 'verify':
        b = verify_bundle(); print(json.dumps({'verified_files': len(b['files'])})); return 0
    if args.gpu < 0:
        parser.error('--gpu must be a physical nonnegative index')
    if args.command == 'build':
        if not 1 <= args.jobs <= 8 or not args.arch.isdigit():
            parser.error('--jobs must be 1..8 and --arch numeric')
        build(args.gpu, args.arch, args.jobs); return 0
    return run(args.stage, args.gpu, args.gate)


if __name__ == '__main__':
    raise SystemExit(main())
