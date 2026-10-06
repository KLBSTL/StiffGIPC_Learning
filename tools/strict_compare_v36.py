"""Precision-controlled Graph comparison and tighter official-base reference probes."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import tarfile
from benchmark_local_preconditioners_v33 import idle, gpu

ROOT=Path(__file__).resolve().parents[1]
report=ROOT/'reports/AUTODL_PERF_V36_STRICT_COMPARE.json';assert not report.exists()
tasks=[]
for repeat in range(1,4):
    arms=['base_toi','base_toi_graph'] if repeat%2 else ['base_toi_graph','base_toi']
    for arm in arms:tasks.append((f'{arm}_r{repeat}',arm,100,.01,[],True))
for div in [1,2,4,8]:tasks.append((f'base_dt{div}','base',100*div,.01/div,['--tol','.001','--quality-only'],False))
tasks.append(('base_dt4_repeat','base',400,.0025,['--tol','.001','--quality-only'],False))
tasks.append(('graph_paths','base_toi_graph',100,.01,['--substeps','--quality-only'],False))
rows=[]
for label,arm,steps,dt,extra,timing in tasks:
    samples=idle();name='autodl_perf_v36_strict_'+label
    command=[sys.executable,str(ROOT/'tools/run_perf_v36.py'),'--platform','autodl','--arm',arm,
             '--scene','cloth_sphere7_l','--name',name,'--steps',str(steps),'--trace','--dt',str(dt),
             '--tol','.01','--pcg-tol','1e-16','--suite','0' if arm=='base' else '1','--timeout','180']
    if arm!='base':command+=['--preconditioner','diag','--robust-velocity-tol','.05']
    command+=extra
    with (ROOT/'builds'/f'{name}.log').open('wb') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    run=ROOT/'runs/autodl'/name
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
    rows.append({'label':label,'run':run.relative_to(ROOT).as_posix(),'command':command,'pre_gpu_samples':samples,'post_gpu':gpu(),'launcher_exit_code':code,'formal_timing':timing,**result})
    report.write_text(json.dumps({'runs':rows,'scope':'Same precision Graph comparison and tighter reference; no default changes'},indent=2))
    print(json.dumps({k:rows[-1][k] for k in ['label','status','recorded_frames','solver_seconds'] if k in rows[-1]}),flush=True)
paths=[report,Path(__file__)]
for row in rows:
    paths.extend(p for p in (ROOT/row['run']).rglob('*') if p.is_file())
index=ROOT/'reports/AUTODL_PERF_V36_STRICT_FILES.json'
index.write_text(json.dumps([{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],indent=2));paths.append(index)
archive=ROOT/'strict_compare_v36.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
    for p in paths:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size}),flush=True)
