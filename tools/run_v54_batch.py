"""Execute the bounded combined v54 study, preserving preflight and run failures."""
import json,subprocess,sys
from pathlib import Path
from run_v54_exit import ROOT,read,sha,gate
def main():
    target=ROOT/'reports/V54_BATCH_MATRIX.json';assert not target.exists();rows=[];launch_failures=[]
    manifest=read(ROOT/'manifests/perf_v54_local.json')
    for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
    for p,e in manifest['binaries'].items():assert sha(ROOT/p)==e['sha256']
    old=read(ROOT/'reports/V54_CLOTH_MATRIX.json')
    def run(label,scene,kind,mode,repeat,steps,start,cp=None):
        template=next(r for r in old if r['scene']==scene and r['variant']=='guard' and r['repeat']==1)
        original=f'v54_batch_{label}_{kind}_{mode}_r{repeat}';cmd=template['command'].copy()
        for key,val in [('--name',original),('--steps',str(steps)),('--timeout','90'),('--inner-exit',mode)]:cmd[cmd.index(key)+1]=val
        cmd+=['--checkpoint-final','--fixed-study','--compact-study','--fixed-study-frames',str(start),'--fixed-study-directions','1','--stage-from-frame',str(start),'--linear-stages']
        if cp:cmd+=['--checkpoint-load',str(cp)]
        elif kind=='seed':cmd+=['--checkpoint-frame',str(start)]
        for attempt in [0,1]:
            name=original if attempt==0 else original+'_retry';cmd[cmd.index('--name')+1]=name
            logpath=ROOT/'reports'/f'{name}_runner.log'
            with logpath.open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            resultpath=ROOT/'runs/local'/name/'result.json'
            if resultpath.exists():break
            error=logpath.read_text(errors='replace');assert 'GPU free-memory baseline did not stabilize' in error,error[-1000:]
            launch_failures.append({'run':name,'simulation_started':False,'reason':'GPU free-memory baseline did not stabilize'})
            (ROOT/'reports/V54_BATCH_LAUNCH_FAILURES.json').write_text(json.dumps(launch_failures,indent=2))
        assert resultpath.exists(),'Two preflight failures; no further retry'
        result=read(resultpath);row={**template,'name':name,'kind':kind,'inner_exit':mode,'repeat':repeat,'target_frame':start,
            'end_frame':start+steps-1 if cp else steps,'command':cmd.copy(),'result':result,'returncode':p.returncode}
        rows.append(row);target.write_text(json.dumps(rows,indent=2));print(json.dumps({'name':name,'result':result}),flush=True);return row
    def window(label,scene,start,cp):
        failed=set()
        for mode,rep in [('native',1),('velocity_only',1),('velocity_only',2),('native',2)]:
            if mode in failed:continue
            row=run(label,scene,'window',mode,rep,7,start,cp)
            if row['result']['status']!='completed':failed.add(mode)
    fixed=ROOT/'runs/local/v54_exit_fixed_seed_r1/checkpoint'
    assert read(ROOT/'reports/V54_EXIT_GATE_fixed.json')['passed']
    window('fixed','cloth_fixed_bunny_l',24,fixed)
    seed=run('sphere','cloth_sphere7_l','seed','native',1,30,30)
    if seed['result']['status']!='completed':return
    cp=ROOT/'runs/local'/seed['name']/'checkpoint';names=[seed['name']]
    for rep in [1,2]:
        row=run('sphere','cloth_sphere7_l','resume','native',rep,1,30,cp)
        if row['result']['status']!='completed':return
        names.append(row['name'])
    g=gate(names,30);(ROOT/'reports/V54_BATCH_SPHERE_GATE.json').write_text(json.dumps(g,indent=2))
    print(json.dumps({'sphere_gate_passed':g['passed']}),flush=True)
    if not g['passed']:return
    for rep in [1,2]:
        row=run('sphere','cloth_sphere7_l','single','velocity_only',rep,1,30,cp)
        if row['result']['status']!='completed':return
    window('sphere','cloth_sphere7_l',30,cp)
if __name__=='__main__':main()
