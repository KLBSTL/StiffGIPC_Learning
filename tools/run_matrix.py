"""Sequential, resumable experiments. Never launches concurrent GPU runs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    cfg=json.loads(a.config.read_text());platform=cfg.get('platform','local')
    matrix=ROOT/'runs'/platform/cfg['name'];matrix.mkdir(parents=True,exist_ok=True)
    summary={'config':cfg,'runs':[],'started_unix':time.time(),'status':'running'}
    def save():
        (matrix/'matrix_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def launch(scene,arm,steps,rep,stage):
        name=f"{cfg['name']}_{scene}_{arm}_{stage}_r{rep:02d}"
        run=ROOT/'runs'/platform/name
        if (run/'result.json').exists():result=json.loads((run/'result.json').read_text())
        else:
            if run.exists():
                result={'status':'incomplete_previous_launch','reason':'Inspect existing run before retry'}
                summary['runs'].append({'scene':scene,'arm':arm,'stage':stage,'name':name,'result':result});save();return result
            args=[sys.executable,str(ROOT/'tools/run_local.py'),'--platform',platform,'--arm',arm,'--scene',scene,
                  '--name',name,'--steps',str(steps),'--timeout',str(cfg.get('pilot_timeout',120) if stage=='pilot' else cfg.get('timeout',600)),
                  '--suite',str(cfg.get('suite',{}).get(arm,0 if arm=='base' else 1)),'--trace','--trace-stride',str(1 if rep==1 else cfg.get('trace_stride',100)),
                  '--tol',str(cfg.get('tol',.01)),'--pcg-tol',str(cfg.get('pcg_tol',1e-4))]
            if cfg.get('build_tag'):args.extend(['--build-tag',cfg['build_tag']])
            if cfg.get('manifest'):args.extend(['--manifest',str(ROOT/cfg['manifest'])])
            with (matrix/(name+'.launcher.log')).open('wb') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'status':'launch_failed','launcher_log':name+'.launcher.log'}
        summary['runs'].append({'scene':scene,'arm':arm,'stage':stage,'name':name,'result':result});save()
        print(json.dumps({'scene':scene,'arm':arm,'stage':stage,'result':result}),flush=True)
        return result
    save()
    for scene in cfg['scenes']:
        eligible=[]
        for arm in cfg['arms']:
            pilot=launch(scene,arm,cfg.get('pilot_steps',2),1,'pilot')
            if pilot['status']=='completed' and pilot.get('finite',False):eligible.append(arm)
        failed_measurements=set()
        for rep in range(1,cfg.get('repeats',3)+1):
            rotation=(rep-1)%len(eligible) if eligible else 0
            for arm in eligible[rotation:]+eligible[:rotation]:
                if arm in failed_measurements:continue
                measured=launch(scene,arm,cfg.get('steps',100),rep,'measure')
                if measured['status']!='completed':failed_measurements.add(arm)
        for arm in set(cfg['arms'])-set(eligible):
            summary['runs'].append({'scene':scene,'arm':arm,'stage':'measure','result':{'status':'skipped_failed_pilot'}});save()
    summary['status']='completed';summary['finished_unix']=time.time();save()

if __name__=='__main__':main()
