"""Bounded five-snapshot study, preserving every result and threshold failure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifests/mas_replay_v42a.json').read_text())
for f in manifest['files']:assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256']
exe=ROOT/'builds/replay/mas_replay';binary_sha=hashlib.sha256(exe.read_bytes()).hexdigest()
report=ROOT/'reports/replay_matrix.json';assert not report.exists();records=[]
cases=[('v39_default',39,'wide_bunny'),('v39_strict',39,'wide_bunny_strict'),
       ('v41_graph',41,'double_bunny'),('v41_host',41,'double_bunny_host'),('v41_wide',41,'wide_guard')]
for case,version,name in cases:
    prefix=ROOT.parent/f'v{version}_20261003/runs/autodl'/f'autodl_perf_v{version}_{name}'/'mas_audit_failure'
    for mode in [5,6,7]:
        assert shutil.disk_usage(ROOT).free>768*1024**2,'disk reserve'
        out=ROOT/'runs'/f'{case}_m{mode}';assert not out.exists()
        command=[str(exe),str(prefix),str(out),str(mode)]
        with (ROOT/'reports'/f'{case}_m{mode}.log').open('w') as log:
            try:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240);code=proc.returncode
            except subprocess.TimeoutExpired:code='timeout'
        row={'case':case,'mode':mode,'command':command,'exit_code':code,'binary_sha256':binary_sha}
        records.append(row);report.write_text(json.dumps(records,indent=2))
        if (out/'result.json').exists():
            raw=json.loads((out/'result.json').read_text());row.update(operator_repeat_passed=raw['operator_repeat_passed'],
                solves=[{k:s[k] for k in ['tolerance','repeat','iterations','reason','true_relative_residual']} for s in raw['solves']])
            report.write_text(json.dumps(records,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
        if code!=0:raise SystemExit(f'Execution failure retained: {case} mode {mode}: {code}')
