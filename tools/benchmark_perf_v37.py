"""Serial, bounded v37 diagnostic experiments. No timing claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from benchmark_local_preconditioners_v33 import idle

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--phase', required=True, choices=['smoke', 'window', 'bunny'])
a = p.parse_args()
if a.phase == 'smoke':
    tasks = [('diag_smoke', 'cloth_sphere7_l', 'diag', 'base_toi_graph', 2, []),
             ('mas_smoke', 'cloth_hang_l', 'mas', 'base_toi_graph', 2, ['--mas-audit'])]
elif a.phase == 'window':
    tasks = [(f'host_r{i}', 'cloth_sphere7_l', 'diag', 'base_toi', 25,
              ['--pcg-tol', '1e-16', '--state-window']) for i in range(1, 4)]
    tasks += [('host_reassembly', 'cloth_sphere7_l', 'diag', 'base_toi', 25,
               ['--pcg-tol', '1e-16', '--state-window', '--state-audit-from-frame', '21', '--pcg-replay-from-frame', '21'])]
else:
    tasks = [('bunny_mas', 'bunny_cloth_bunny_l', 'mas', 'base_toi_graph', 100,
              ['--mas-audit', '--failure-system'])]
target = ROOT / f'reports/AUTODL_PERF_V37_{a.phase.upper()}.json'
assert not target.exists()
records = []
for label, scene, preconditioner, arm, steps, extra in tasks:
    samples = idle()
    name = f'autodl_perf_v37_{label}'
    runner = ROOT / 'tools/run_perf_v37.py'
    command = [sys.executable, str(runner), '--platform', 'autodl', '--arm', arm,
               '--scene', scene, '--name', name, '--steps', str(steps), '--trace',
               '--dt', '.01', '--tol', '.01', '--pcg-tol', '1e-4', '--suite', '1',
               '--preconditioner', preconditioner, '--robust-velocity-tol', '.05',
               '--fused-diag-update', '0', '--quality-only', '--audit-pcg',
               '--diagnostic-toi', '--timeout', '180'] + extra
    with (ROOT / f'builds/{name}.log').open('wb') as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    run = ROOT / f'runs/autodl/{name}'
    result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
    records.append({'command': command, 'run': str(run.relative_to(ROOT)), 'pre_gpu': samples,
                    'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(), 'exit_code': code, **result})
    target.write_text(json.dumps(records, indent=2))
    print(json.dumps({k: result[k] for k in ['status', 'recorded_frames', 'solver_seconds'] if k in result}), flush=True)
    if code and a.phase != 'bunny':
        raise SystemExit(code)
