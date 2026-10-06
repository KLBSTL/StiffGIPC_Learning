"""Serial bounded v32 profiling and preconditioner diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['profile', 'bunny'], required=True)
    p.add_argument('--quality-only', action='store_true')
    a = p.parse_args()
    tasks = []
    if a.phase == 'profile':
        for scene in ['cloth_hang_l', 'cloth_sphere7_l']:
            for label, arm, pc in [('graph', 'base_graph', None),
                                    ('toi_mas', 'base_toi_graph', 'mas'),
                                    ('toi_diag', 'base_toi_graph', 'diag')]:
                tasks.append((scene, label, arm, pc, 60))
    else:
        for pc in ['mas', 'diag']:
            tasks.append(('bunny_cloth_bunny_l', 'toi_' + pc, 'base_toi_graph', pc, 40))
    rows = []
    report = ROOT / f'reports/LOCAL_PERF_V33_{a.phase.upper()}.json'
    assert not report.exists(), 'Preserve existing diagnostic report'
    runner = ROOT / 'tools/run_robust_port.py'
    for scene, label, arm, pc, steps in tasks:
        name = f'local_perf_v33_{a.phase}_{scene}_{label}'
        command = [sys.executable, str(runner), '--platform', 'local', '--arm', arm,
                   '--scene', scene, '--steps', str(steps), '--name', name,
                   '--trace', '--profile', '--audit-pcg', '--dt', '.01', '--tol', '.01',
                   '--pcg-tol', '.0001', '--suite', '0', '--timeout', '120']
        if pc:
            command += ['--preconditioner', pc, '--robust-velocity-tol', '.05']
        if a.quality_only:
            command += ['--quality-only']
        run = ROOT / 'runs/local' / name
        assert not run.exists(), run
        launcher = ROOT / 'runs/local' / (name + '.launcher.log')
        with launcher.open('wb') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
        rows.append({'scene': scene, 'label': label, 'arm': arm, 'preconditioner': pc,
                     'run': run.relative_to(ROOT).as_posix(), 'steps': steps,
                     'command': command, 'launcher_exit_code': code, **result})
        report.write_text(json.dumps({'protocol': vars(a), 'runs': rows,
            'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest()}, indent=2))
        print(json.dumps(rows[-1]), flush=True)
    return bool(any(r['status'] != 'completed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
