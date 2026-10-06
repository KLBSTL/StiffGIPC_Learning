"""Validate and time existing batching/refit features on the frozen v34 binary."""
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
    parser.add_argument('--phase', choices=['smoke', 'matrix', 'paths'], required=True)
    args = parser.parse_args()
    if args.phase == 'smoke':
        tasks = [('suite_audit', 1, '1', ['--audit-suite', '--quality-only'])]
    elif args.phase == 'paths':
        tasks = [('suite_paths', 1, '1', ['--substeps', '--quality-only'])]
    else:
        groups = [('suite_off', '0'), ('suite_on', '1')]
        tasks = [(label, repeat, suite, []) for repeat in range(1, 4)
                 for label, suite in (groups if repeat % 2 else list(reversed(groups)))]
    target = ROOT / f'reports/AUTODL_PERF_V35_{args.phase.upper()}.json'
    assert not target.exists(), 'Preserve previous records'
    rows = []
    for label, repeat, suite, extra in tasks:
        samples = idle() if args.phase == 'matrix' else [gpu()]
        name = f'autodl_perf_v35_{args.phase}_cloth_sphere7_l_{label}_r{repeat:02d}'
        run = ROOT / 'runs/autodl' / name
        assert not run.exists()
        runner = ROOT / 'tools/run_perf_v34.py'
        command = [sys.executable, str(runner), '--platform', 'autodl',
                   '--manifest', str(ROOT / 'manifests/perf_v34_cuda128_autodl.json'),
                   '--arm', 'base_toi_graph', '--scene', 'cloth_sphere7_l', '--name', name,
                   '--steps', '100', '--trace', '--dt', '.01', '--tol', '.01',
                   '--pcg-tol', '.0001', '--suite', suite, '--timeout', '180',
                   '--preconditioner', 'diag', '--robust-velocity-tol', '.05',
                   '--fused-diag-update', '0', *extra]
        with (ROOT / 'runs/autodl' / (name + '.launcher.log')).open('wb') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
        row = {'scene': 'cloth_sphere7_l', 'label': label, 'repeat': repeat,
               'run': run.relative_to(ROOT).as_posix(), 'command': command,
               'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
               'pre_gpu_samples': samples, 'post_gpu': gpu(), 'launcher_exit_code': code, **result}
        rows.append(row)
        target.write_text(json.dumps({'phase': args.phase, 'protocol': {'dt': .01, 'tol': .01,
            'pcg_tol': .0001, 'velocity_tol': .05, 'fused_diag_update': '0',
            'gpu_gate_percent': 30, 'platform': 'autodl', 'interleaved': True}, 'runs': rows}, indent=2))
        print(json.dumps({k: row[k] for k in ['label', 'repeat', 'status', 'recorded_frames', 'solver_seconds'] if k in row}), flush=True)
        if code or result['status'] != 'completed' or result['recorded_frames'] != 100:
            return 1
        if args.phase == 'smoke':
            frames = json.loads((run / 'output/stats.json').read_text())['frames']
            for frame in frames:
                assert frame['bvh_refit_audit']['identical']
                assert frame['energy_batch_audit']['max_relative_error'] <= 1e-10
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
