"""Serial local v34 smoke, fixed-tolerance timing, and accepted-path export."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from benchmark_local_preconditioners_v33 import idle, gpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'matrix', 'paths'], required=True)
    parser.add_argument('--background-timing', action='store_true', help='Diagnostic shared-load measurements, never exclusive GPU timing')
    parser.add_argument('--platform', choices=['local','autodl'], default='local')
    args = parser.parse_args()
    tasks = []
    if args.phase == 'smoke':
        tasks = [('cloth_sphere7_l', 'audit', 1, 'run_perf_v34.py', 'base_toi_graph', 'diag', '1', 30, ['--audit-graph', '--audit-pcg']),
                 ('cloth_hang_l', 'mas_fallback', 1, 'run_perf_v34.py', 'base_toi_graph', 'mas', '1', 10, [])]
    elif args.phase == 'paths':
        tasks = [('cloth_sphere7_l', 'fused_paths', 1, 'run_perf_v34.py', 'base_toi_graph', 'diag', '1', 100, ['--substeps'])]
    else:
        groups = [('base', 'run_robust_port.py', 'base', None, None),
                  ('v32_diag', 'run_robust_port.py', 'base_toi_graph', 'diag', None),
                  ('v34_off', 'run_perf_v34.py', 'base_toi_graph', 'diag', '0'),
                  ('v34_on', 'run_perf_v34.py', 'base_toi_graph', 'diag', '1')]
        for repeat in range(1, 4):
            order = groups if repeat == 1 else list(reversed(groups)) if repeat == 2 else groups[2:] + groups[:2]
            tasks.extend(('cloth_sphere7_l', label, repeat, runner, arm, pc, fused, 100, []) for label, runner, arm, pc, fused in order)
    prefix = 'AUTODL' if args.platform == 'autodl' else 'LOCAL'
    target = ROOT / f'reports/{prefix}_PERF_V34_{args.phase.upper()}.json'
    assert not target.exists(), 'Preserve previous records'
    rows = []
    for scene, label, repeat, script, arm, pc, fused, steps, extra in tasks:
        samples = idle() if args.phase == 'matrix' and not args.background_timing else [gpu()]
        if args.background_timing:
            # Record a second sample without pretending a shared GPU is idle.
            time.sleep(1)
            samples.append(gpu())
        name = f'{args.platform}_perf_v34_{args.phase}_{scene}_{label}_r{repeat:02d}'
        run = ROOT / 'runs' / args.platform / name
        assert not run.exists()
        runner = ROOT / 'tools' / script
        command = [sys.executable, str(runner), '--arm', arm, '--scene', scene,
                   '--name', name, '--steps', str(steps), '--trace', '--dt', '.01', '--tol', '.01',
                   '--pcg-tol', '.0001', '--suite', '0', '--timeout', '180', *extra]
        if args.platform == 'autodl':
            manifest = 'perf_v34_cuda128_autodl.json' if script == 'run_perf_v34.py' else 'robust_port_v32_autodl.json'
            command += ['--platform', 'autodl', '--manifest', str(ROOT / 'manifests' / manifest)]
        if pc:
            command += ['--preconditioner', pc, '--robust-velocity-tol', '.05']
        if fused is not None:
            command += ['--fused-diag-update', fused]
        if args.phase != 'matrix' or args.background_timing:
            command.append('--quality-only')
        with (ROOT / 'runs' / args.platform / (name + '.launcher.log')).open('wb') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
        row = {'scene': scene, 'label': label, 'repeat': repeat, 'run': run.relative_to(ROOT).as_posix(),
               'command': command, 'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
               'pre_gpu_samples': samples, 'post_gpu': gpu(), 'launcher_exit_code': code, **result}
        rows.append(row)
        target.write_text(json.dumps({'phase': args.phase, 'protocol': {'dt': .01, 'tol': .01, 'pcg_tol': .0001,
            'velocity_tol': .05, 'suite': '0', 'gpu_gate_percent': None if args.background_timing else 30,
            'background_timing': args.background_timing, 'platform': args.platform,
            'interleaved': True}, 'runs': rows}, indent=2))
        print(json.dumps({k: row[k] for k in ['label', 'repeat', 'status', 'recorded_frames', 'solver_seconds'] if k in row}), flush=True)
        if code or result['status'] != 'completed' or result['recorded_frames'] != steps:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
