"""Export complete runs and audit metadata without build objects or old history."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v34', 'v35'], required=True)
    args = parser.parse_args()
    output = ROOT / f'autodl_perf_{args.version}_results_20261003.tar.gz'
    assert not output.exists()
    selected = set()
    for phase in ['SMOKE', 'MATRIX', 'PATHS']:
        report = ROOT / f'reports/AUTODL_PERF_{args.version.upper()}_{phase}.json'
        assert report.is_file()
        selected.add(report)
        for record in json.loads(report.read_text())['runs']:
            run = ROOT / record['run']
            selected.update(p for p in run.rglob('*') if p.is_file())
            selected.add(run.parent / (run.name + '.launcher.log'))
    selected.update(p for p in (ROOT / 'manifests').glob('*autodl.json'))
    selected.update(p for p in (ROOT / 'builds').glob('autodl*.json'))
    selected.update(p for p in (ROOT / 'builds').glob(f'autodl_{args.version}_*.log'))
    for name in ['run_perf_v34.py', 'run_robust_port.py', 'benchmark_perf_v34.py',
                 'benchmark_perf_v35.py', 'benchmark_local_preconditioners_v33.py',
                 'autodl_v34_build.sh', 'export_perf_v34_autodl.py']:
        path = ROOT / 'tools' / name
        if path.is_file():
            selected.add(path)
    with tarfile.open(output, 'w:gz') as tar:
        for path in sorted(selected):
            assert path.resolve().is_relative_to(ROOT.resolve())
            tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps({'archive': str(output), 'files': len(selected), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
