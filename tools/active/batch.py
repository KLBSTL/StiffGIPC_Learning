"""Sequential finite batches; stop a failed arm rather than extending its budget."""
import argparse
import json
from config import ROOT, read, sha
from run import execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan_path=ROOT/a.plan;plan=read(plan_path);target=ROOT/plan['report']
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    report={'plan_sha256':sha(plan_path),'runs':[]};failed=set()
    for task in plan['runs']:
        arm=task.get('arm',task['name'])
        if arm in failed:continue
        config=read(ROOT/task['config']) if isinstance(task['config'],str) else task['config']
        result=execute(config,task['name'],task.get('binary','active'))
        report['runs'].append(task|{'result':result})
        target.write_text(json.dumps(report,indent=2))
        if result['status']!='completed':
            failed.add(arm)
            if plan.get('stop_on_failure',True):break
    raise SystemExit(int(bool(failed)))
