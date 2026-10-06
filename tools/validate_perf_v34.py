"""Independent CPU validation of v34's saved accepted paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=ROOT)
parser.add_argument('--platform', choices=['local', 'autodl'], default='local')
parser.add_argument('--version', choices=['v34', 'v35'], default='v34')
args = parser.parse_args()
data = args.data.resolve()
prefix = args.platform.upper()
matrix = json.loads((data / f'reports/{prefix}_PERF_{args.version.upper()}_PATHS.json').read_text())
row = matrix['runs'][0]
assert row['status'] == 'completed' and row['recorded_frames'] == 100
run = data / row['run']
report = ROOT / f'reports/{args.platform}_perf_{args.version}_accepted_ccd.json'
target = ROOT / f'reports/{prefix}_PERF_{args.version.upper()}_PATH_VALIDATION.json'
assert not report.exists() and not target.exists(), 'Preserve previous validation'
exe = ROOT / 'builds/validator/Release/validate_path.exe'
command = [str(exe), str(run / 'trace'), str(report), 'substeps', '--stable-nh1']
with (ROOT / f'builds/{args.platform}_perf_{args.version}_ccd.log').open('wb') as stream:
    code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
validation = json.loads(report.read_text()) if report.exists() else None
result = {'run': row['run'], 'command': command, 'exit_code': code,
          'validator_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'validation': validation}
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'passed': bool(validation and validation['passed']), 'exit_code': code,
                  'paths': validation['paths_checked'] if validation else None}))
raise SystemExit(code)
