"""Bounded diagnostics with an immutable v44 binary; not a performance benchmark."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','graph','host','strict'],required=True);a=p.parse_args()
report=ROOT/'reports'/f'V44_{a.phase}.json';assert not report.exists()
if a.phase=='smoke':
    tasks=[('off','cloth_hang_l','base_toi_graph',2,0,'1e-4'),('on','cloth_hang_l','base_toi_graph',2,1,'1e-4')]
else:
    tasks=[(a.phase,'bunny_cloth_bunny_l','base_toi' if a.phase=='host' else 'base_toi_graph',40,30,'1e-16' if a.phase=='strict' else '1e-4')]
records=[]
for name,scene,arm,steps,stage,rho in tasks:
    command=[sys.executable,str(ROOT/'tools/run_perf_v44.py'),'--platform','autodl','--arm',arm,'--scene',scene,
        '--preconditioner','mas','--mas-cholesky','1','--steps',str(steps),'--name','autodl_perf_v44_'+name,
        '--suite','1','--dt','.01','--robust-velocity-tol','.05','--trace','--audit-pcg','--failure-system',
        '--diagnostic-toi','--quality-only','--timeout','240','--stage-from-frame',str(stage),'--pcg-tol',rho]
    with (ROOT/'reports'/f'runner_{name}.log').open('w') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=280)
    run=ROOT/'runs/autodl'/('autodl_perf_v44_'+name)
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
    records.append({'name':name,'command':command,'code':proc.returncode,**result})
    report.write_text(json.dumps(records,indent=2));print(json.dumps(records[-1]),flush=True)
    if a.phase=='smoke' and proc.returncode:raise SystemExit(proc.returncode)
if a.phase=='smoke':
    off=ROOT/'runs/autodl/autodl_perf_v44_off';on=ROOT/'runs/autodl/autodl_perf_v44_on'
    assert (off/'final.bin').read_bytes()==(on/'final.bin').read_bytes()
    assert not (off/'stage.jsonl').exists()
    rows=[json.loads(line) for line in (on/'stage.jsonl').read_text().splitlines()]
    assert rows and all(x['sequence']==i+1 for i,x in enumerate(rows))
    assert {'pcg_end','ccd_query_end','outer_end'} <= {x['stage'] for x in rows}
    print(json.dumps({'stage_events':len(rows),'smoke_final_bitwise_equal':True,'default_off_no_stage_file':True}),flush=True)
