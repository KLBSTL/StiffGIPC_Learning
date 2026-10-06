"""Predeclared frozen-v54 dt refinement, 16 finite serial runs."""
import json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=ROOT/'reports/V54_DT_MATRIX.json';assert not target.exists()
    m=json.loads((ROOT/'manifests/perf_v54_local.json').read_text())
    for f in m['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
    for p,e in m['binaries'].items():assert sha(ROOT/p)==e['sha256']
    old=json.loads((ROOT/'reports/V54_CLOTH_MATRIX.json').read_text());rows=[]
    for scene in ['cloth_sphere7_l','cloth_fixed_bunny_l']:
        for div in [2,4]:
            for variant,repeat in [('off',1),('guard',1),('guard',2),('off',2)]:
                template=next(r for r in old if r['scene']==scene and r['variant']==variant and r['repeat']==repeat)
                name=f"v54_dt{div}_{scene}_{variant}_r{repeat}";dt=.01/div;steps=50*div
                cmd=template['command'].copy()
                for key,val in [('--name',name),('--steps',str(steps)),('--dt',str(dt)),('--timeout','180')]:cmd[cmd.index(key)+1]=val
                with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:
                    p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=270)
                result=json.loads((ROOT/'runs/local'/name/'result.json').read_text())
                row={**template,'name':name,'dt':dt,'refinement':div,'physical_seconds':.5,'command':cmd,'result':result,'returncode':p.returncode}
                rows.append(row);target.write_text(json.dumps(rows,indent=2));print(json.dumps({'run':name,'result':result}),flush=True)
                if p.returncode or result['status']!='completed':raise SystemExit(1)
    for div in [2,4]:
        (ROOT/'reports'/f'V54_DT{div}_MATRIX.json').write_text(json.dumps([r for r in rows if r['refinement']==div],indent=2))
if __name__=='__main__':main()
