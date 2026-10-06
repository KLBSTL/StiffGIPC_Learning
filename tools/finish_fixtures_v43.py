"""Correct two historical path labels; preserve first three passes and failed lookup."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];rows=[];report=ROOT/'reports/V43_fixtures_corrected.json';assert not report.exists()
for case,name in [('v41_graph','double_bunny'),('v41_host','double_bunny_host'),('v41_wide','wide_guard')]:
    source=ROOT.parent/'v41_20261003/runs/autodl'/f'autodl_perf_v41_{name}'/'mas_audit_failure'
    assert Path(str(source)+'_meta.json').exists()
    out=ROOT/'runs/fixtures_corrected'/case;assert not out.exists()
    env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')};env.update(GIPC_MAS_CHOLESKY_FIXTURE=str(source),GIPC_MAS_FIXTURE_OUTPUT=str(out))
    command=[str(ROOT/'builds/autodl-stiff_perf_v43/gipc')]
    with (ROOT/'reports'/f'fixture_corrected_{case}.log').open('w') as log:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    rows.append({'case':case,'source':str(source),'output':str(out),'command':command,'code':code});report.write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
    if code:raise SystemExit(code)
