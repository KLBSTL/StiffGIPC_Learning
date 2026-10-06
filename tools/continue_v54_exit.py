"""Resume only the unlaunched sphere jobs; preserve original preflight failure."""
import json,subprocess
from pathlib import Path
from run_v54_exit import ROOT,read,gate
target=ROOT/'reports/V54_EXIT_MATRIX.json';rows=read(target)
assert len(rows)==7 and rows[-1]['name']=='v54_exit_sphere_resume_r1'
failure=ROOT/'reports/V54_EXIT_LAUNCH_FAILURE.json';assert not failure.exists()
failure.write_text(json.dumps({'attempt':'v54_exit_sphere_resume_r2','stage':'launcher GPU free-memory baseline',
    'simulation_started':False,'error':'GPU free-memory baseline did not stabilize','retry':'v54_exit_sphere_resume_r2_retry'},indent=2))
base=rows[-1]
def run(name,kind,repeat,cp=True):
    cmd=base['command'].copy();mode='native' if kind=='resume' else 'velocity_only'
    for key,val in [('--name',name),('--inner-exit',mode)]:cmd[cmd.index(key)+1]=val
    if not cp:
        at=cmd.index('--checkpoint-load');del cmd[at:at+2];cmd[cmd.index('--steps')+1]='25'
    with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    path=ROOT/'runs/local'/name/'result.json'
    assert path.exists(),'Preflight failed again; preserve logs and stop without more retries'
    result=read(path);row={**base,'name':name,'kind':kind,'repeat':repeat,'command':cmd,'inner_exit':mode,'result':result,'returncode':p.returncode}
    rows.append(row);target.write_text(json.dumps(rows,indent=2));print(json.dumps({'run':name,'result':result}),flush=True);return row
r=run('v54_exit_sphere_resume_r2_retry','resume',2)
assert r['result']['status']=='completed'
g=gate(['v54_exit_sphere_seed_r1','v54_exit_sphere_resume_r1',r['name']],25)
(ROOT/'reports/V54_EXIT_GATE_sphere.json').write_text(json.dumps(g,indent=2));print(json.dumps({'gate':'sphere','passed':g['passed']}),flush=True)
for rep in [1,2]:
    kind='velocity' if g['passed'] else 'fromzero';r=run(f'v54_exit_sphere_{kind}_r{rep}',kind,rep,g['passed'])
    if r['result']['status']!='completed':break
