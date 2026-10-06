"""Isolate first-frame state perturbation under a common subsequent native rule."""
import json,subprocess
from run_v54_exit import ROOT,read
target=ROOT/'reports/V54_PERTURB_MATRIX.json';assert not target.exists();rows=[]
template=next(r for r in read(ROOT/'reports/V54_BATCH_MATRIX.json') if r['name']=='v54_batch_fixed_window_native_r1')
for origin,rep in [('native',1),('velocity',1),('velocity',2),('native',2)]:
    name=f'v54_perturb_{origin}_r{rep}';cmd=template['command'].copy()
    cp=ROOT/'runs/local'/f'v54_exit_fixed_{"resume" if origin=="native" else "velocity"}_r1/final_checkpoint'
    assert read(cp/'checkpoint.json')['completed_physical_frames']==24
    for key,val in [('--name',name),('--steps','6'),('--checkpoint-load',str(cp)),('--fixed-study-frames','25'),('--stage-from-frame','25')]:cmd[cmd.index(key)+1]=val
    with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    resultpath=ROOT/'runs/local'/name/'result.json';assert resultpath.exists(),'Preflight failure: retain and stop without simulation-budget change'
    result=read(resultpath);r={**template,'name':name,'kind':'perturb','origin':origin,'repeat':rep,'target_frame':25,'end_frame':30,'command':cmd,'result':result,'returncode':p.returncode}
    rows.append(r);target.write_text(json.dumps(rows,indent=2));print(json.dumps({'name':name,'result':result}),flush=True)
    if result['status']!='completed':break
