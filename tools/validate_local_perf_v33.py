"""Export all accepted paths serially, then independently audit two cloth cases."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from report_v32_autodl import quality

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate(row):
    run = ROOT / row['run']
    exe = ROOT / 'builds/validator/Release/validate_path.exe'
    output = ROOT / f"reports/local_perf_v33_{row['scene']}_accepted_ccd.json"
    assert not output.exists(), 'Preserve existing validation'
    command = [str(exe), str(run / 'trace'), str(output), 'substeps', '--stable-nh1']
    with (ROOT / f"builds/local_perf_v33_{row['scene']}_ccd.log").open('wb') as stream:
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    validation = read(output) if output.exists() else None
    return {**row, 'validator_command': command, 'validator_exit_code': code,
            'validator_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
            'validation': validation}


def main():
    target = ROOT / 'reports/LOCAL_PERF_V33_ACCEPTED_PATHS.json'
    assert not target.exists()
    tasks = []
    for scene in ['cloth_hang_l', 'cloth_sphere7_l']:
        name = f'local_perf_v33_paths_{scene}_toi_diag'
        run = ROOT / 'runs/local' / name
        assert not run.exists()
        command = [sys.executable, str(ROOT / 'tools/run_robust_port.py'), '--arm', 'base_toi_graph',
                   '--scene', scene, '--name', name, '--steps', '100', '--trace', '--substeps',
                   '--preconditioner', 'diag', '--robust-velocity-tol', '.05', '--dt', '.01',
                   '--tol', '.01', '--pcg-tol', '.0001', '--suite', '0', '--timeout', '180']
        with (ROOT / 'runs/local' / (name + '.launcher.log')).open('wb') as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result = read(run / 'result.json')
        assert code == 0 and result['status'] == 'completed' and result['recorded_frames'] == 100
        stats = read(run / 'output/stats.json')['frames']
        accepted = [t for f in stats for t in f['toi'] if 'alpha' in t]
        paths = sorted((run / 'trace/substeps').glob('safe_*.bin'))
        # Export includes solve-start bridge plus every accepted update for each frame.
        assert len(paths) == len(accepted) + 100
        assert all(t['safe_state_verified'] for t in accepted)
        tasks.append({'scene': scene, 'run': run.relative_to(ROOT).as_posix(),
                      'command': command, 'result': result, 'accepted_steps': len(accepted),
                      'saved_path_states': len(paths),
                      'quality_vs_formal_diag_r01': quality(run, ROOT / f'runs/local/local_perf_v33_formal_{scene}_toi_diag_r01')})
        print(json.dumps({'scene': scene, 'accepted_steps': len(accepted), 'states': len(paths)}), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(validate, row) for row in tasks]):
            row = future.result()
            rows.append(row)
            v = row['validation'] or {}
            print(json.dumps({'scene': row['scene'], 'passed': v.get('passed'),
                              'paths': v.get('paths_checked'), 'flags': v.get('conservative_collision_flags')}), flush=True)
    report = {'protocol': '100 physical frames, .05 m/s block diagonal + Graph, all accepted states including frame bridges; CPU Tight-Inclusion, Stable NH1',
              'all_passed': all(r['validator_exit_code'] == 0 and r['validation']['passed'] for r in rows),
              'runs': rows, 'timing_excluded': True}
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
