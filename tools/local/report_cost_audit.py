"""Two bounded CPU/NVTX cost runs, then read-only whole-scene accounting.

Diagnostic scopes are inclusive CPU envelopes, never extra GPU time. They
can rule out a small candidate but cannot prove that waiting is removable.
Only diagnostic fields change from a completed sealed screening run.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('tools/local','tools/bench','tools/diagnostic')]
from config import read,expand,digest
from linux_runner import require,write_new,verify_files
from local_identity import record,verify,verify_seal
from windows_runner import execute,gpu_lock
from validate_run import validate
from pool_metrics import export_evidence

SESSION=ROOT/'runs/report_execution_20261007'
REPORT=ROOT/'reports/report_execution_20261007'
SEAL=ROOT/'build/report_observer_20261007_v2/report_seal_final_capability_v3.json'

def derive(reference):
    request=read(reference/'requested.json');result=read(reference/'result.json')
    require(result['status']=='completed' and result['recorded_frames']==100 and result['finite'],
            'Completed finite 100-frame reference required')
    c=request['expanded_config'];require(c==expand(c) and digest(c)==request['config_sha256'],'Config identity')
    require(c['diagnostics']==[] and c['profile']=='none','Uninstrumented reference required')
    observed=expand(c|dict(diagnostics=['cost'],cost_frames='1-100',cost_events=False))
    require({k for k in c if c[k]!=observed[k]}=={'diagnostics','cost_frames','cost_events'},'Non-diagnostic change')
    return request,observed

def summarize(cost):
    rows=[json.loads(s) for s in cost.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    require(rows and all(r['measurement_mode']=='nvtx_cpu_only' and r['gpu_events_enabled'] is False
                        and r['sample_kind']=='production' for r in rows),'CPU-only native production scopes required')
    roots=[r for r in rows if r['stage']=='ipc.physical_frame' and r['parent_scope_id']==0]
    require(sorted(r['frame'] for r in roots)==list(range(1,101)),'100 complete frame scopes required')
    sums=defaultdict(float);counts=defaultdict(int);per_frame=defaultdict(lambda:defaultdict(float))
    ids={r['scope_id'] for r in rows};require(len(ids)==len(rows),'Duplicate scope IDs')
    for r in rows:
        require(r['parent_scope_id']==0 or r['parent_scope_id'] in ids,'Broken ancestry')
        sums[r['stage']]+=r['cpu_submit_ms'];counts[r['stage']]+=1
        per_frame[r['frame']][r['stage']]+=r['cpu_submit_ms']
    whole=sums['ipc.physical_frame']
    def windows(a,b):
        return {k:sum(per_frame[f].get(k,0.) for f in range(a,b+1)) for k in sorted(sums)}
    stages={k:dict(cpu_inclusive_ms=v,calls=counts[k],fraction_of_physical_cpu=v/whole)
            for k,v in sorted(sums.items())}
    return dict(schema='report.cost.v1',source=record(cost),stages=stages,per_frame=dict(per_frame),
        initial_1_3=windows(1,3),contact_22_24=windows(22,24),
        timing_scope='Inclusive CPU submission/wait envelopes. Nested scopes are not additive. No CUDA events; no inferred GPU kernel duration.',
        performance_certified=False)

def run(scene):
    reference=SESSION/f'screen_{scene}_combined_1';out=SESSION/f'cost_{scene}_combined'
    require(not out.exists(),'One attempt only; no retry')
    with gpu_lock(ROOT):
        seal=verify_seal(ROOT,SEAL);req,c=derive(reference)
        verify_files(reference,read(reference/'evidence.json')['files'])
        identity=seal['programs']['active']
        require(req['exe_sha256']==identity['exe']['sha256'] and req['source_digest']==identity['source_digest'],
                'Different sealed native program')
        t=dict(name=out.name,binary='active',config=c,scene_key=scene,phase='cost',repeat=1)
        own=record(Path(__file__));write_new(SESSION/f'cost_plan_{scene}.json',dict(task=t,controller=own,
            seal=record(SEAL),reference=record(reference/'evidence.json'),maximum_attempts=1,
            solver_timeout_seconds=120,quality_certified=False,performance_certified=False))
        result=execute(SESSION,t,identity,0)
        verify([own]);verify_seal(ROOT,SEAL)
        require(result['status']=='completed' and validate(out)['passed'],'Diagnostic incomplete; no retry')
        require(export_evidence(out,100,'state')['passed'] and export_evidence(out,100,'velocity')['passed'],
                'Nonfinite or missing exports')
    data=summarize(out/'cost.jsonl');write_new(REPORT/f'COST_{scene}.json',data)
    print(json.dumps(dict(scene=scene,physical_ms=data['stages']['ipc.physical_frame']['cpu_inclusive_ms'],
        energy_reduction=data['stages'].get('ipc.energy_batch_reduce'),
        scalar_reduction=data['stages'].get('ipc.energy_scalar_reduce_and_readback'),
        scope='CPU envelopes only; not a formal speed measurement')),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','summarize'))
    p.add_argument('--scene',choices=('sphere','fixed'),required=True);a=p.parse_args()
    if a.action=='run':run(a.scene)
    else:
        data=summarize(SESSION/f'cost_{a.scene}_combined/cost.jsonl')
        write_new(REPORT/f'COST_{a.scene}.json',data)

if __name__=='__main__':main()
