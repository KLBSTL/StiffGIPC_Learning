"""Run the bounded repair gates serially. Failed runs retain their evidence."""
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(name,scene,steps,reduced='1',arm='base_toi_graph',timeout=180):
    command=[sys.executable,str(ROOT/'tools/run_perf_v53.py'),'--platform','local','--arm',arm,
        '--scene',scene,'--steps',str(steps),'--name','v53_'+name,'--timeout',str(timeout),
        '--trace','--substeps','--physics','--audit-pcg','--quality-only','--mas-cholesky','1',
        '--preconditioner','mas','--suite','1','--dt','.01','--pcg-tol','1e-4',
        '--robust-velocity-tol','.05','--inner-exit','native','--ccd-pair-limit','10000000',
        '--choose-start',reduced,'--reduced-slack','0']
    with (ROOT/'reports'/f'V53_{name}_runner.log').open('x') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+90)
    result=json.loads((ROOT/'runs/local'/('v53_'+name)/'result.json').read_text())
    print(json.dumps({'name':name,'returncode':proc.returncode,'result':result}),flush=True)
    return result['status']=='completed'

if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='smoke':
        assert run('hang_off','cloth_hang_l',3,'0',timeout=40)
        assert run('hang_on','cloth_hang_l',3,timeout=40)
    elif phase=='bunny':
        assert run('bunny40','bunny_cloth_bunny_l',40)
    elif phase=='long':
        assert run('bunny100','bunny_cloth_bunny_l',100,timeout=450)
    elif phase=='host':
        assert run('bunny100_host','bunny_cloth_bunny_l',100,arm='base_toi',timeout=450)
    elif phase=='control':
        assert run('bunny40_off','bunny_cloth_bunny_l',40,'0')
    else:raise ValueError(phase)
