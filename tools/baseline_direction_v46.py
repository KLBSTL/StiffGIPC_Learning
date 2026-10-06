"""Bounded uninterrupted controls after the v46 direction review; frozen v45 solver."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
report=ROOT/'reports/DIRECTION_V46_CONTROLS_20261004.json';assert not report.exists()
rows=[]
for mode,arm in [('graph','base_toi_graph'),('host','base_toi')]:
    name='v46_continuous_'+mode
    command=[sys.executable,str(ROOT/'tools/run_perf_v45.py'),'--platform','local','--arm',arm,
        '--scene','bunny_cloth_bunny_l','--steps','40','--name',name,'--timeout','180',
        '--preconditioner','mas','--mas-cholesky','1','--suite','1','--dt','.01','--tol','.01',
        '--pcg-tol','1e-4','--robust-velocity-tol','.05','--inner-exit','native',
        '--trace','--substeps','--physics','--audit-pcg','--quality-only',
        '--failure-system','--stage-from-frame','30','--ccd-pair-limit','10000000']
    with (ROOT/'reports'/f'DIRECTION_V46_{mode}_runner.log').open('x') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=270)
    run=ROOT/'runs/local'/name
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
    rows.append({'mode':mode,'command':command,'returncode':p.returncode,'result':result})
    report.write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
