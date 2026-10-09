"""One complete mixed-scene pool validation; not a performance comparison.

The new full-run cost evidence identifies collision/line search as a major
remaining cost. Check the already-present opt-in pool on this different workload
before considering any timed round. All historical rejected results are retained.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for name in ('bench', 'diagnostic', 'local'):
    sys.path.insert(0, str(ROOT/'tools'/name))
from config import read, expand
from linux_runner import require, write_new
from local_identity import verify, verify_active_sources, record
from windows_runner import execute, gpu_lock
from pool_metrics import pool_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True)
    p.add_argument('--identity', type=Path, required=True)
    args = p.parse_args()
    require(args.session.replace('_','').isalnum(), 'Invalid session path')
    out = ROOT/'runs'/args.session
    require(not out.exists(), 'Fresh validation required')
    out.mkdir(parents=True)
    identity = read(args.identity)
    verify_active_sources(identity)
    verify(identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    task = dict(name='mixed_pool_validation', binary='active', config=expand(dict(
        scene='bunny_cloth_bunny_l',steps=120,timeout_seconds=180,preset='combined',
        discrete_bvh_refit=True,fem_direct_pullback=True,mas_pivot_only_sync=True,
        contact_pool=True,contact_pool_validate=True)))
    write_new(out/'identity.json',dict(program=identity,task=task,controller=record(__file__)))
    with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        result = execute(out,task,identity,0)
    evidence = dict(passed=False)
    if result['status']=='completed':
        frames = read(out/task['name']/'output/stats.json')['frames']
        evidence = pool_evidence(frames,True,True)
    write_new(out/'analysis.json',dict(result=result,pool=evidence,performance_certified=False,quality_certified=False))
    print(json.dumps(dict(status=result['status'],pool_passed=evidence['passed'],
        complete_contract=evidence.get('complete_contract'),
        energy_equivalence_passed=evidence.get('same_state_energy_alpha_support',{}).get('passed'))),flush=True)
    return 0 if result['status']=='completed' and evidence['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
