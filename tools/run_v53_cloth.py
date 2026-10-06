"""Serial OFF/ON/ON/OFF cloth tests, preserving failures and the v53 runner."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=[('hang','cloth_hang_l','mas'),('sphere','cloth_sphere7_l','diag'),('fixed_bunny','cloth_fixed_bunny_l','diag')]
def main():
    target=ROOT/'reports/V53_CLOTH_MATRIX.json';assert not target.exists();rows=[]
    for label,scene,prec in CASES:
        s=json.loads((ROOT/f'sources/stiff_base/Assets/benchmark_scenes/{scene}.json').read_text())
        dt=s['effective_scalar_fields']['dt']
        for switch,repeat in [('0',1),('1',1),('1',2),('0',2)]:
            name=f'v53_cloth_{label}_{"on" if switch=="1" else "off"}_r{repeat}'
            cmd=[sys.executable,str(ROOT/'tools/run_perf_v53.py'),'--platform','local','--arm','base_toi_graph',
                '--scene',scene,'--steps','100','--name',name,'--timeout','180','--dt',str(dt),
                '--trace','--substeps','--physics','--audit-pcg','--quality-only','--mas-cholesky','1',
                '--preconditioner',prec,'--suite','1','--pcg-tol','1e-4','--robust-velocity-tol','.05',
                '--inner-exit','native','--ccd-pair-limit','10000000','--choose-start',switch,'--reduced-slack','0']
            with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=270)
            result=json.loads((ROOT/'runs/local'/name/'result.json').read_text())
            row={'name':name,'scene':scene,'choose_start':switch,'repeat':repeat,'dt':dt,
                'preconditioner':prec,'returncode':proc.returncode,'result':result,'command':cmd}
            rows.append(row);target.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
            if result['status']!='completed':break
if __name__=='__main__':main()
