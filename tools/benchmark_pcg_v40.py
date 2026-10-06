"""Frozen StiffGIPC / TOI controls for PCG cap diagnosis, not performance scoring."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
report=ROOT/'reports/PCG_V40_CONTROLS.json';assert not report.exists()
tasks=[(f'base_r{i}','base',1e-4) for i in range(1,4)]
tasks += [('base_strict','base',1e-14),('toi_wide_host','base_toi',1e-4),('toi_wide_graph_repeat','base_toi_graph',1e-4)]
records=[]
for name,arm,tol in tasks:
    command=[sys.executable,str(ROOT/'tools/run_perf_v39.py'),'--platform','autodl','--arm',arm,
             '--scene','bunny_cloth_bunny_l','--steps','100','--name','autodl_pcg_v40_'+name,
             '--dt','.01','--tol','.01','--pcg-tol',str(tol),'--trace','--audit-pcg',
             '--quality-only','--timeout','240']
    if arm!='base':
        command += ['--preconditioner','mas','--mas-wide','1','--suite','1',
                    '--robust-velocity-tol','.05','--failure-system','--mas-audit']
    with (ROOT/'reports'/('PCG_V40_'+name+'.log')).open('w') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=280)
    run=ROOT/'runs/autodl'/('autodl_pcg_v40_'+name)
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launcher_failed'}
    stats=json.loads((run/'output/stats.json').read_text()) if (run/'output/stats.json').exists() else {'frames':[]}
    solves=[dict(frame=i+1,direction=k+1,**item['pcg']) for i,f in enumerate(stats['frames'])
            for k,item in enumerate(f.get('newton',[])) if 'pcg' in item]
    limits=[x for x in solves if x.get('iteration_limit')]
    row={'name':name,'command':command,'exit_code':proc.returncode,**result,
         'stats_frames':len(stats['frames']),'pcg_solves':len(solves),'limit_events':limits,
         'max_iterations':max((x.get('iterations',0) for x in solves),default=0),
         'max_true_relative_residual':max((x.get('true_relative_residual',0) for x in solves),default=0),
         'newton_limit_frames':[i+1 for i,f in enumerate(stats['frames']) if f.get('newton_exit')=='iteration_limit']}
    records.append(row);report.write_text(json.dumps(records,indent=2))
    print(json.dumps({k:row[k] for k in ['name','status','recorded_frames','pcg_solves','max_iterations','limit_events','max_true_relative_residual'] if k in row}),flush=True)
    if result['status'] in ('launcher_failed','memory_budget','disk_reserve','timeout'):break
