"""Bounded staged integration checks; every simulation uses the frozen v43 runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['fixtures','smoke','smoke_corrected','hang','guard','bunny','host','repeat','strict']);a=p.parse_args()
report=ROOT/'reports'/f'V43_{a.phase}.json';assert not report.exists();records=[]
if a.phase=='fixtures':
    for case,old,run,prefix in [('smoke','v37','autodl_perf_v37_mas_smoke','mas_audit_initial'),
                              ('default','v39','autodl_perf_v39_wide_bunny','mas_audit_failure'),
                              ('strict','v39','autodl_perf_v39_wide_bunny_strict','mas_audit_failure'),
                              ('v41_graph','v41','autodl_perf_v41_double_bunny','mas_audit_failure'),
                              ('v41_host','v41','autodl_perf_v41_double_bunny_host','mas_audit_failure'),
                              ('v41_wide','v41','autodl_perf_v41_wide_guard','mas_audit_failure')]:
        source=ROOT.parent/(old+'_20261003')/'runs/autodl'/run/prefix
        out=ROOT/'runs/fixtures'/case;assert not out.exists()
        env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
        env.update(GIPC_MAS_CHOLESKY_FIXTURE=str(source),GIPC_MAS_FIXTURE_OUTPUT=str(out))
        command=[str(ROOT/'builds/autodl-stiff_perf_v43/gipc')]
        with (ROOT/'reports'/f'fixture_{case}.log').open('w') as log:
            proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        records.append({'case':case,'source':str(source),'command':command,'code':proc.returncode})
        report.write_text(json.dumps(records,indent=2));print(json.dumps(records[-1]),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
else:
    if a.phase in ('smoke','smoke_corrected'):
        tasks=[('native_smoke','cloth_hang_l','base_toi_graph','mas',0,0,2,[]),
               ('chol_smoke','cloth_hang_l','base_toi_graph','mas',0,1,2,['--mas-audit']),
               ('chol_host_smoke','cloth_hang_l','base_toi','mas',0,1,2,[]),
               ('diag_control','cloth_sphere7_l','base_toi_graph','diag',0,1,2,[])]
    elif a.phase=='hang':tasks=[('chol_hang','cloth_hang_l','base_toi_graph','mas',0,1,100,['--substeps','--physics'])]
    elif a.phase=='guard':tasks=[('wide_guard','bunny_cloth_bunny_l','base_toi_graph','mas',1,0,100,['--mas-audit'])]
    elif a.phase=='bunny':tasks=[('chol_bunny','bunny_cloth_bunny_l','base_toi_graph','mas',0,1,100,['--mas-audit','--substeps','--physics'])]
    elif a.phase=='host':tasks=[('chol_bunny_host','bunny_cloth_bunny_l','base_toi','mas',0,1,100,['--mas-audit','--physics'])]
    elif a.phase=='repeat':tasks=[('chol_bunny_repeat','bunny_cloth_bunny_l','base_toi_graph','mas',0,1,100,['--mas-audit','--physics'])]
    else:tasks=[('chol_bunny_strict','bunny_cloth_bunny_l','base_toi_graph','mas',0,1,100,['--pcg-tol','1e-16','--mas-audit','--physics'])]
    for name,scene,arm,prec,wide,inverse,steps,extra in tasks:
        command=[sys.executable,str(ROOT/'tools/run_perf_v43.py'),'--platform','autodl','--arm',arm,
                 '--scene',scene,'--preconditioner',prec,'--mas-wide',str(wide),'--mas-inverse64','0','--mas-cholesky',str(inverse),
                 '--steps',str(steps),'--name','autodl_perf_v43_'+name,'--suite','1','--dt','.01',
                 '--robust-velocity-tol','.05','--trace','--audit-pcg','--failure-system',
                 '--diagnostic-toi','--quality-only','--timeout','240']+extra
        with (ROOT/'reports'/f'runner_{name}.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=280)
        run=ROOT/'runs/autodl'/('autodl_perf_v43_'+name)
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
        record={'name':name,'command':command,'code':proc.returncode,**result}
        if (run/'output/stats.json').exists():
            stats=json.loads((run/'output/stats.json').read_text())
            solves=[dict(frame=i+1,**n['pcg']) for i,f in enumerate(stats['frames']) for n in f.get('newton',[]) if 'pcg' in n]
            record.update(pcg_solves=len(solves),limits=[s for s in solves if s.get('iteration_limit')],
                          breakdowns=[s for s in solves if s.get('breakdown')],
                          max_iterations=max((s.get('iterations',0) for s in solves),default=0))
        records.append(record);report.write_text(json.dumps(records,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='command'}),flush=True)
        if proc.returncode and a.phase in ('smoke','smoke_corrected','hang'):raise SystemExit(proc.returncode)
