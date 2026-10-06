"""Interleaved component matrix; retain failures and skip their repeats."""
import argparse,json,subprocess,sys
from pathlib import Path
from run_bunny_components import ROOT,PRESETS,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','matrix','toi','quality'],required=True);a=p.parse_args()
    target=ROOT/f'reports/BUNNY_COMPONENTS_{a.phase.upper()}.json';assert not target.exists()
    labels=['base','host','graph','refit','batch','reuse','combined','cholesky']
    if a.phase=='matrix':tasks=[(k,r,100,False) for r in range(1,4) for k in (labels if r==1 else list(reversed(labels)) if r==2 else labels[4:]+labels[:4])]
    elif a.phase=='smoke':tasks=[(k,0,3,False) for k in ['base','host','combined']]
    elif a.phase=='toi':tasks=[('toi',1,100,False)]
    else:tasks=[(k,1,100,True) for k in ['base','combined']]
    rows=[];failed=set();runner=ROOT/'tools/run_bunny_components.py'
    for label,repeat,steps,audit in tasks:
        if label in failed:continue
        name=f'bunny_components_{a.phase}_{label}_r{repeat}'
        cmd=[sys.executable,str(runner),'--preset',label,'--name',name,'--steps',str(steps),'--timeout','120']+(['--audit'] if audit else [])
        with (ROOT/f'reports/{name}.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=160)
        path=ROOT/'runs/local'/name/'result.json';result=json.loads(path.read_text()) if path.exists() else {'status':'launcher_failed'}
        rows.append({'label':label,'repeat':repeat,'name':name,'command':cmd,'launcher_code':r.returncode,'result':result})
        target.write_text(json.dumps({'phase':a.phase,'scene':'bunny_cloth_bunny_l','presets':PRESETS,'runner_sha256':sha(runner),'runs':rows},indent=2))
        print(json.dumps({'name':name,**{k:v for k,v in result.items() if k!='gpu_samples'}}),flush=True)
        if r.returncode:failed.add(label)
    return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
