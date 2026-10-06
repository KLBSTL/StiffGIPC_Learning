"""Finite same-binary controls for the safe-restart full-step guard."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(label,scene,variant,steps=100,repeat=1,timeout=180):
    switch,guard={'off':('0','0'),'legacy':('1','0'),'guard':('1','1')}[variant]
    spec=json.loads((ROOT/f'sources/stiff_base/Assets/benchmark_scenes/{scene}.json').read_text())
    dt=spec['effective_scalar_fields']['dt'];prec='mas' if scene in ['cloth_hang_l','bunny_cloth_bunny_l'] else 'diag'
    name=f'v54_{label}_{variant}_r{repeat}'
    cmd=[sys.executable,str(ROOT/'tools/run_perf_v54.py'),'--platform','local','--arm','base_toi_graph',
         '--scene',scene,'--steps',str(steps),'--name',name,'--timeout',str(timeout),'--dt',str(dt),
         '--trace','--substeps','--physics','--audit-pcg','--quality-only','--mas-cholesky','1',
         '--preconditioner',prec,'--suite','1','--pcg-tol','1e-4','--robust-velocity-tol','.05',
         '--inner-exit','native','--ccd-pair-limit','10000000','--choose-start',switch,'--restart-guard',guard,'--reduced-slack','0']
    with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout+90)
    result=json.loads((ROOT/'runs/local'/name/'result.json').read_text())
    row={'name':name,'scene':scene,'variant':variant,'choose_start':switch,'restart_guard':guard,'repeat':repeat,
         'dt':dt,'preconditioner':prec,'result':result,'command':cmd,'returncode':p.returncode}
    print(json.dumps(row),flush=True);return row
def main():
    phase=sys.argv[1];target=ROOT/'reports'/f'V54_{phase.upper()}_MATRIX.json';assert not target.exists();rows=[]
    if phase=='smoke':cases=[('smoke','cloth_hang_l',v,3,1,40) for v in ['legacy','guard']]
    elif phase=='screen':cases=[('screen','cloth_fixed_bunny_l',v,30,1,90) for v in ['off','legacy','guard']]
    elif phase=='cloth':
        cases=[]
        for label,scene in [('hang','cloth_hang_l'),('sphere','cloth_sphere7_l'),('fixed_bunny','cloth_fixed_bunny_l')]:
            for v,r in [('off',1),('legacy',1),('guard',1),('guard',2),('legacy',2),('off',2)]:cases.append((label,scene,v,100,r,180))
    else:raise ValueError(phase)
    for case in cases:
        row=run(*case);rows.append(row);target.write_text(json.dumps(rows,indent=2))
        if row['result']['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
