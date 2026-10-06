"""Bounded serial scene matrix, with strict identity checks on resume."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'base': ('base', None, 'paper'),
    'graph': ('base_graph', None, 'paper'),
    'toi005': ('base_toi', .05, 'robust'),
    'toi005_graph': ('base_toi_graph', .05, 'robust'),
    'toi1': ('base_toi', 1., 'robust'),
    'toi1_graph': ('base_toi_graph', 1., 'robust'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', default='cloth_hang_l,cloth_sphere7_l,bunny_cloth_bunny_l')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--budget', type=int, default=1800)
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    mpath = ROOT / 'manifests/robust_port_v32_autodl.json'
    manifest = json.loads(mpath.read_text())
    runner_sha = hashlib.sha256((ROOT / 'tools/run_robust_port.py').read_bytes()).hexdigest()
    records, skipped = [], []
    start = time.monotonic()
    report = ROOT / f'reports/AUTODL_V32_MATRIX_{a.steps}.json'
    failed, slow = set(), set()
    for repeat in range(1, a.repeats + 1):
        for scene in a.scenes.split(','):
            order = a.arms.split(',')
            if repeat % 2 == 0:
                order = order[::-1]
            for label in order:
                key = (scene, label)
                if key in failed or (repeat > 1 and key in slow) or time.monotonic() - start > a.budget:
                    skipped.append({'scene': scene, 'arm': label, 'repeat': repeat,
                        'reason': 'prior_failure' if key in failed else 'slow_first_repeat' if key in slow else 'matrix_budget'})
                    continue
                arm, tol, policy = ARMS[label]
                name = f'v32_{scene}_{a.steps}_{label}_r{repeat:02d}'
                run = ROOT / 'runs/autodl' / name
                command = [sys.executable, str(ROOT / 'tools/run_robust_port.py'),
                    '--platform', 'autodl', '--manifest', str(mpath), '--arm', arm,
                    '--toi-policy', policy, '--scene', scene, '--steps', str(a.steps),
                    '--name', name, '--trace', '--trace-stride', '1', '--suite', '0',
                    '--dt', '.01', '--tol', '.01', '--pcg-tol', '.0001', '--timeout', str(a.timeout)]
                if tol is not None:
                    command += ['--robust-velocity-tol', str(tol)]
                if run.exists():
                    req = json.loads((run / 'requested.json').read_text())
                    expected = {'scene': scene, 'steps': a.steps, 'arm': arm, 'toi_policy': policy,
                        'robust_velocity_tol': tol, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                        'suite': '0', 'trace': True, 'trace_stride': 1, 'platform': 'autodl',
                        'substeps': False, 'profile': False, 'audit_friction_snapshot': False,
                        'source_digest': manifest['source_digest'], 'runner_sha256': runner_sha,
                        'mu_scope': None, 'quality_only': False}
                    for k, value in expected.items():
                        assert req[k] == value, (run, k)
                    binary = 'stiff_base' if arm == 'base' else 'stiff_robust_port'
                    assert req['exe_sha256'] == manifest['binaries'][f'builds/autodl-{binary}/gipc']['sha256']
                    result = json.loads((run / 'result.json').read_text())
                    code = 0 if result['status'] == 'completed' else 1
                else:
                    with (ROOT / 'runs/autodl' / (name + '.launcher.log')).open('wb') as stream:
                        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                    result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
                row = {'scene': scene, 'arm': label, 'repeat': repeat,
                    'run': run.relative_to(ROOT).as_posix(), 'launcher_exit_code': code, **result}
                records.append(row)
                if code or result['status'] != 'completed' or result.get('recorded_frames') != a.steps or not result.get('finite'):
                    failed.add(key)
                if result.get('wall_seconds', 0) > 60:
                    slow.add(key)
                report.write_text(json.dumps({'protocol': vars(a), 'source_digest': manifest['source_digest'],
                    'binaries': manifest['binaries'], 'runner_sha256': runner_sha,
                    'runs': records, 'skipped': skipped}, indent=2))
                print(json.dumps(row), flush=True)
    report.write_text(json.dumps({'protocol': vars(a), 'source_digest': manifest['source_digest'],
        'binaries': manifest['binaries'], 'runner_sha256': runner_sha,
        'runs': records, 'skipped': skipped}, indent=2))
    print(json.dumps({'finished': True, 'runs': len(records), 'failed_groups': len(failed), 'skipped': len(skipped)}), flush=True)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
