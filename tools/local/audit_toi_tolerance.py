"""Bounded algorithm experiment: stricter existing TOI inner tolerance.

The completed default 120-frame run exited 650/775 inner problems via full-step
with raw direction speed > .05 m/s. The native full-step guard uses 100*tol.
Test the existing parameter only; keep material, full CCD and PCG rho unchanged.
This changes nonlinear stopping and is NOT an execution-equivalence experiment.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
for part in ('bench','local','diagnostic','full_eval'):
    sys.path.insert(0,str(ROOT/'tools'/part))
from config import read,expand
from linux_runner import require,write_new
from local_identity import verify,verify_active_sources,record
from windows_runner import execute,gpu_lock
from toi_pcg_audit import summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--tolerance',type=float,required=True)
    args=p.parse_args()
    require(args.session.replace('_','').isalnum(),'Invalid session path')
    require(0<args.tolerance<.05,'Require stricter explicit tolerance')
    out=ROOT/'runs'/args.session
    require(not out.exists(),'Fresh session required')
    out.mkdir(parents=True)
    identity=read(args.identity)
    verify_active_sources(identity)
    verify(identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    task=dict(name='mixed_toi_stricter',binary='active',config=expand(dict(
        preset='toi',scene='bunny_cloth_bunny_l',steps=120,timeout_seconds=180,
        mas='legacy',fem_direct_pullback=True,mas_pivot_only_sync=True,
        toi_trial_velocity_tol=args.tolerance)))
    controller=record(__file__)
    with (out/'controller_at_run.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    write_new(out/'identity.json',dict(program=identity,task=task,controller=controller,
        retained_controller=record(out/'controller_at_run.py'),
        summary_controller=record(ROOT/'tools/local/toi_pcg_audit.py'),
        scope=__doc__))
    with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        result=execute(out,task,identity,0)
    row=dict(result=result,nonlinear_stopping_changed=True,performance_certified=False,quality_certified=False)
    if result['status']=='completed':
        try:
            row['summary']=summary(out/task['name'],result)
        except Exception as error:row['analysis_error']=str(error)
    verify([controller])
    write_new(out/'analysis.json',row)
    print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
    return 0 if 'summary' in row else 2


if __name__=='__main__':raise SystemExit(main())
