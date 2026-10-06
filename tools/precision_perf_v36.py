"""Follow up the fixed-system precision result without changing production defaults."""
import json
from pathlib import Path
import subprocess
import sys
from benchmark_local_preconditioners_v33 import idle, gpu

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'reports/AUTODL_PERF_V36_PRECISION.json'
assert not target.exists()
tasks=[]
for repeat in range(1,4):
    tolerances=['1e-8','1e-12'] if repeat%2 else ['1e-12','1e-8']
    for tol in tolerances:
        tasks.append((f'toi_{tol}_r{repeat}','base_toi_graph',100,['--pcg-tol',tol]))
tasks += [('toi_tight_audit','base_toi_graph',100,['--pcg-tol','1e-12','--audit-pcg','--audit-graph']),
          ('toi_nonlinear_tight','base_toi_graph',100,['--pcg-tol','1e-12','--tol','.001','--robust-velocity-tol','.005','--audit-pcg']),
          ('base_dt4_repeat','base',400,['--dt','.0025','--pcg-tol','1e-8','--tol','.001']),
          ('base_dt8','base',800,['--dt','.00125','--pcg-tol','1e-8','--tol','.001'])]
rows=[]
for label,arm,steps,extra in tasks:
    samples=idle();name='autodl_perf_v36_precision_'+label
    command=[sys.executable,str(ROOT/'tools/run_perf_v36.py'),'--platform','autodl','--arm',arm,
             '--scene','cloth_sphere7_l','--name',name,'--steps',str(steps),'--trace','--dt','.01','--tol','.01',
             '--suite','0' if arm=='base' else '1','--timeout','180']
    if arm!='base':command+=['--preconditioner','diag','--robust-velocity-tol','.05']
    if label not in [f'toi_{t}_r{r}' for t in ['1e-8','1e-12'] for r in range(1,4)]:command+=['--quality-only']
    command+=extra
    with (ROOT/'builds'/f'{name}.log').open('wb') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    run=ROOT/'runs/autodl'/name
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
    rows.append({'label':label,'run':run.relative_to(ROOT).as_posix(),'command':command,'pre_gpu_samples':samples,'post_gpu':gpu(),'launcher_exit_code':code,**result})
    target.write_text(json.dumps({'runs':rows,'scope':'Precision and reference probes, no default changes'},indent=2))
    print(json.dumps({k:rows[-1][k] for k in ['label','status','recorded_frames','solver_seconds'] if k in rows[-1]}),flush=True)
