"""Bounded, serial v36 diagnostics and validation; preserve all completed runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmark_local_preconditioners_v33 import idle, gpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'study', 'profile', 'bunny', 'matrix', 'paths', 'reference'], required=True)
    args = parser.parse_args()
    tasks = []
    if args.phase == 'smoke':
        tasks = [('sphere_smoke', 'cloth_sphere7_l', 'base_toi_graph', 'diag', 2, ['--fixed-study', '--fixed-study-frames', '1,2', '--fixed-study-directions', '1', '--audit-pcg'])]
    elif args.phase == 'study':
        tasks = [('sphere_fixed', 'cloth_sphere7_l', 'base_toi_graph', 'diag', 100, ['--fixed-study']),
                 ('hang_fixed', 'cloth_hang_l', 'base_toi_graph', 'mas', 100, ['--fixed-study'])]
    elif args.phase == 'profile':
        for scene, pc in [('cloth_sphere7_l', 'diag'), ('cloth_hang_l', 'mas')]:
            for suite in ['0', '1']:
                tasks.append((f'{scene}_suite{suite}', scene, 'base_toi_graph', pc, 100, ['--suite', suite, '--profile']))
    elif args.phase == 'bunny':
        tasks = [('bunny_mas', 'bunny_cloth_bunny_l', 'base_toi_graph', 'mas', 100,
                  ['--failure-system', '--audit-pcg', '--diagnostic-toi', '--timeout', '180'])]
    elif args.phase == 'paths':
        tasks = [(f'{s}_paths', s, 'base_toi_graph', pc, 100, ['--substeps', '--audit-suite'])
                 for s, pc in [('cloth_sphere7_l', 'diag'), ('cloth_hang_l', 'mas')]]
    elif args.phase == 'reference':
        for div in [1, 2, 4]:
            tasks.append((f'sphere_base_strict_dt{div}', 'cloth_sphere7_l', 'base', None, 100 * div,
                          ['--dt', str(.01 / div), '--tol', '.001', '--pcg-tol', '1e-8']))
        tasks.append(('sphere_toi_strict', 'cloth_sphere7_l', 'base_toi_graph', 'diag', 100,
                      ['--pcg-tol', '1e-8', '--audit-pcg']))
    else:
        for scene, pc in [('cloth_sphere7_l', 'diag'), ('cloth_hang_l', 'mas')]:
            groups = [('base', 'base', None), ('toi_suite', 'base_toi_graph', pc)]
            for repeat in range(1, 4):
                for label, arm, preconditioner in groups if repeat % 2 else list(reversed(groups)):
                    tasks.append((f'{scene}_{label}_r{repeat}', scene, arm, preconditioner, 100, []))
    target = ROOT / f'reports/AUTODL_PERF_V36_{args.phase.upper()}.json'
    assert not target.exists(), 'Use a new phase/report identity for reruns'
    rows = []
    for label, scene, arm, pc, steps, extra in tasks:
        samples = idle()
        name = f'autodl_perf_v36_{args.phase}_{label}'
        runner = ROOT / 'tools/run_perf_v36.py'
        command = [sys.executable, str(runner), '--platform', 'autodl', '--arm', arm, '--scene', scene,
                   '--name', name, '--steps', str(steps), '--trace', '--dt', '.01', '--tol', '.01',
                   '--pcg-tol', '1e-4', '--suite', '1' if arm != 'base' else '0', '--timeout', '180']
        if pc:
            command += ['--preconditioner', pc, '--robust-velocity-tol', '.05', '--fused-diag-update', '0']
        if args.phase != 'matrix':
            command += ['--quality-only']
        command += extra
        with (ROOT / 'builds' / (name + '.log')).open('wb') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        run = ROOT / 'runs/autodl' / name
        result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
        row = {'label': label, 'scene': scene, 'run': run.relative_to(ROOT).as_posix(), 'command': command,
               'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(), 'pre_gpu_samples': samples,
               'post_gpu': gpu(), 'launcher_exit_code': code, **result}
        rows.append(row)
        target.write_text(json.dumps({'phase': args.phase, 'runs': rows}, indent=2))
        print(json.dumps({k: row[k] for k in ['label', 'status', 'recorded_frames', 'solver_seconds'] if k in row}), flush=True)
        if code and args.phase not in ['bunny', 'reference']:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
