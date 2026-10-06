"""Predeclared supplemental precision check: same binary/budget, all five cases, mode 7."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1];exe=ROOT/'builds/replay/mas_replay'
report=ROOT/'reports/strict16_matrix.json';assert not report.exists();rows=[]
identity=hashlib.sha256(exe.read_bytes()).hexdigest()
for case,version,name in [('v39_default',39,'wide_bunny'),('v39_strict',39,'wide_bunny_strict'),('v41_graph',41,'double_bunny'),('v41_host',41,'double_bunny_host'),('v41_wide',41,'wide_guard')]:
    assert shutil.disk_usage(ROOT).free>768*1024**2
    prefix=ROOT.parent/f'v{version}_20261003/runs/autodl'/f'autodl_perf_v{version}_{name}'/'mas_audit_failure'
    out=ROOT/'strict16'/case;assert not out.exists()
    cmd=[str(exe),str(prefix),str(out),'7','1e-16','.3','2']
    with (ROOT/'reports'/f'strict16_{case}.log').open('w') as log:
        try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        except subprocess.TimeoutExpired:code='timeout'
    row={'case':case,'command':cmd,'exit_code':code,'binary_sha256':identity};rows.append(row)
    if (out/'result.json').exists():row['result']=json.loads((out/'result.json').read_text())
    report.write_text(json.dumps(rows,indent=2));print(json.dumps({'case':case,'code':code,'solves':[{k:s[k] for k in ['iterations','reason','true_relative_residual']} for s in row.get('result',{}).get('solves',[])]}),flush=True)
    if code!=0:raise SystemExit('Supplemental execution failure preserved')
