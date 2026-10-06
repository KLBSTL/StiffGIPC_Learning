"""Launch the measured local scene profile using the preserved v32 solver."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest_path = ROOT / 'manifests/local_perf_v33.json'
    config = json.loads(manifest_path.read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=config['scene_preconditioners'], required=True)
    parser.add_argument('--name', required=True, help='Fresh output name; existing runs are preserved')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--substeps', action='store_true', help='Export accepted paths; timing becomes diagnostic')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps <= 0 or args.timeout <= 0:
        parser.error('steps and timeout must be positive')
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..') or '/' in args.name or '\\' in args.name:
        parser.error('name must be a single output directory name')
    runner = ROOT / 'tools/run_robust_port.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == config['runner_sha256']
    frozen = json.loads((ROOT / config['execution_manifest']).read_text(encoding='utf-8-sig'))
    assert frozen['source_digest'] == config['source_digest']
    assert frozen['binaries']['builds/local-fused-v32/Release/gipc.exe']['sha256'] == config['exe_sha256']
    preconditioner = config['scene_preconditioners'][args.scene]
    command = [sys.executable, str(runner), '--platform', 'local', '--arm', config['arm'],
               '--scene', args.scene, '--name', args.name, '--steps', str(args.steps),
               '--timeout', str(args.timeout), '--trace', '--preconditioner', preconditioner,
               '--manifest', str(ROOT / config['execution_manifest'])]
    for flag in ['dt', 'tol', 'pcg_tol', 'robust_velocity_tol', 'suite']:
        command.extend(['--' + flag.replace('_', '-'), str(config[flag])])
    if args.substeps:
        command.append('--substeps')
    metadata = {'configuration_version': config['implementation_version'],
                'configuration_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'resolved_preconditioner': preconditioner, 'command': command,
                'quality_matched_speed_qualified': False}
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    run = ROOT / 'runs/local' / args.name
    assert not run.exists(), f'Preserve existing run: {run}'
    code = subprocess.run(command, cwd=ROOT).returncode
    if run.is_dir():
        metadata['launcher_exit_code'] = code
        (run / 'v33_configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
