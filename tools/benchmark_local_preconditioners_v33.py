"""Interleaved 100-frame local measurements with unchanged v32 executables."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'base': ('base', None), 'graph': ('base_graph', None),
        'toi_mas': ('base_toi_graph', 'mas'), 'toi_diag': ('base_toi_graph', 'diag')}


def gpu():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    return dict(zip(['utilization', 'free_mib', 'temperature_c', 'power_w', 'sm_mhz'], map(float, line)))


def idle():
    samples, quiet = [], 0
    for _ in range(15):
        snapshot = gpu()
        samples.append(snapshot)
        quiet = quiet + 1 if snapshot['utilization'] <= 30 else 0
        if quiet >= 2:
            return samples
        time.sleep(1)
    raise RuntimeError('GPU background activity prevents formal timing: ' + json.dumps(samples))


def main():
    runner = ROOT / 'tools/run_robust_port.py'
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / 'manifests/robust_port_v32.json').read_text())
    report = ROOT / 'reports/LOCAL_PERF_V33_MATRIX_100.json'
    assert not report.exists(), 'Preserve the existing matrix'
    records, failures = [], set()
    for repeat in range(1, 4):
        for scene in ['cloth_hang_l', 'cloth_sphere7_l']:
            labels = list(ARMS)
            if repeat % 2 == 0:
                labels.reverse()
            for label in labels:
                if (scene, label) in failures:
                    continue
                samples = idle()
                arm, pc = ARMS[label]
                name = f'local_perf_v33_formal_{scene}_{label}_r{repeat:02d}'
                run = ROOT / 'runs/local' / name
                command = [sys.executable, str(runner), '--arm', arm, '--scene', scene,
                    '--steps', '100', '--name', name, '--trace', '--dt', '.01',
                    '--tol', '.01', '--pcg-tol', '.0001', '--suite', '0', '--timeout', '180']
                if pc:
                    command += ['--preconditioner', pc, '--robust-velocity-tol', '.05']
                assert not run.exists(), run
                with (ROOT / 'runs/local' / (name + '.launcher.log')).open('wb') as stream:
                    code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {'status': 'launcher_failed'}
                row = {'scene': scene, 'label': label, 'repeat': repeat, 'run': run.relative_to(ROOT).as_posix(),
                    'command': command, 'idle_gpu_samples': samples, 'gpu_after': gpu(),
                    'launcher_exit_code': code, **result}
                records.append(row)
                if code or result['status'] != 'completed' or result.get('recorded_frames') != 100:
                    failures.add((scene, label))
                report.write_text(json.dumps({'source_digest': manifest['source_digest'],
                    'binaries': manifest['binaries'], 'runner_sha256': runner_sha,
                    'protocol': {'frames': 100, 'repeats': 3, 'dt': .01, 'tol': .01, 'pcg_tol': .0001,
                        'velocity_tol': .05, 'suite': '0', 'graph_execution': 'conditional_graph',
                        'profiling': False, 'audit': False, 'idle_gate_percent': 30,
                        'background_note': 'User paused GPU apps; persistent desktop load about 20-23%. Interleaved repeats; shared-load measurements, not exclusive GPU timing.'},
                    'runs': records}, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k not in ['command', 'idle_gpu_samples']}), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
