"""Two diagnostic timing breakdowns, separate from the formal matrix."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from benchmark_local_preconditioners_v33 import idle, gpu

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'reports/AUTODL_PERF_V35_PROFILE.json'
archive = ROOT / 'autodl_perf_v35_profile_20261003.tar.gz'
assert not target.exists() and not archive.exists()
rows, files = [], [target, Path(__file__)]
for suite in ['0', '1']:
    samples = idle()
    name = f'autodl_perf_v35_profile_cloth_sphere7_l_suite{suite}'
    run = ROOT / 'runs/autodl' / name
    runner = ROOT / 'tools/run_perf_v34.py'
    command = [sys.executable, str(runner), '--platform', 'autodl',
               '--manifest', str(ROOT / 'manifests/perf_v34_cuda128_autodl.json'),
               '--arm', 'base_toi_graph', '--scene', 'cloth_sphere7_l', '--name', name,
               '--steps', '100', '--trace', '--dt', '.01', '--tol', '.01',
               '--pcg-tol', '.0001', '--suite', suite, '--timeout', '180',
               '--preconditioner', 'diag', '--robust-velocity-tol', '.05',
               '--fused-diag-update', '0', '--profile', '--quality-only']
    with (ROOT / 'builds' / (name + '.log')).open('wb') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    req = json.loads((run / 'requested.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    frames = json.loads((run / 'output/stats.json').read_text())['frames']
    assert result['status'] == 'completed' and result['finite'] and result['recorded_frames'] == 100
    assert req['profile'] and result['timing_is_diagnostic'] and result['robust_frame_protocol_verified']
    directions = [n for f in frames for n in f['newton'] if 'pcg' in n]
    outer = [t for f in frames for t in f['toi'] if 'alpha' in t]
    assert len(frames) == 100 and all('profile_ms' in n for n in directions)
    assert all('profile_ms' in t and t['safe_state_verified'] for t in outer)
    assert not any(n['pcg'].get('iteration_limit', False) for n in directions)
    assert all(f['toi_exit'] != 'iteration_limit' for f in frames)
    phases = {k: sum(n['profile_ms'][k] for n in directions) / 1000
              for k in ['assembly', 'pcg', 'line_search']}
    phases.update({k: sum(t['profile_ms'][k] for t in outer) / 1000
                   for k in ['active_update', 'safe_ccd', 'safe_update']})
    row = {'suite': suite, 'run': run.relative_to(ROOT).as_posix(), 'command': command,
           'requested': req, 'result': result, 'pre_gpu_samples': samples, 'post_gpu': gpu(),
           'directions': len(directions), 'cg_iterations': sum(n['pcg']['iterations'] for n in directions),
           'accepted_outer': len(outer), 'profile_seconds': phases,
           'profiled_directions': len(directions)}
    rows.append(row)
    target.write_text(json.dumps({'note': 'Diagnostic synchronized timers; not formal speed measurements. Single run per configuration.',
                                 'runs': rows}, indent=2))
    files += [run / p for p in ['requested.json', 'result.json', 'output/stats.json', 'trace/frames.csv']]
    print(json.dumps({k: row[k] for k in ['suite', 'directions', 'cg_iterations', 'profile_seconds']}), flush=True)
files += [ROOT / 'tools/run_perf_v34.py', ROOT / 'manifests/perf_v34_cuda128_autodl.json']
with tarfile.open(archive, 'w:gz') as tar:
    for path in files:
        tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(json.dumps({'archive': str(archive), 'files': len(files), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
