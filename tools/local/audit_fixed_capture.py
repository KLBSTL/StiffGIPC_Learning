"""Capture adjacent real Newton systems for bounded linear-solver diagnosis."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    require(args.session.replace('_','').isalnum(), 'Invalid session path')
    out = ROOT/'runs'/args.session
    require(not out.exists(), 'Fresh capture required')
    out.mkdir(parents=True)
    identity = read(args.identity)
    verify_active_sources(identity)
    verify(identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    tasks = []
    for name, scene, frame, backend in (
        ('fixed_ipc', 'cloth_fixed_bunny_l', 57, 'ipc'),
        ('mixed_ipc', 'bunny_cloth_bunny_l', 34, 'ipc'),
        ('mixed_toi', 'bunny_cloth_bunny_l', 34, 'toi_al')):
        cfg = expand(dict(scene=scene, steps=frame, timeout_seconds=180,
            preset='toi' if backend=='toi_al' else 'combined', mas='legacy',
            fem_direct_pullback=True, mas_pivot_only_sync=True,
            discrete_bvh_refit=backend=='ipc', diagnostics=['fixed'],
            fixed_frames=str(frame), fixed_directions='1,2,3'))
        tasks.append(dict(name=name, binary='active', config=cfg))
    write_new(out/'identity.json', dict(program=identity, controller=record(__file__), tasks=tasks))
    ledger = []
    with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            result = execute(out, task, identity, 0)
            ledger.append(dict(task=task, result=result))
            if result['status'] != 'completed':
                break
    write_new(out/'ledger.json', dict(runs=ledger, skipped=tasks[len(ledger):]))
    print(json.dumps(dict(completed=sum(r['result']['status']=='completed' for r in ledger),
        planned=len(tasks), timing_is_diagnostic=True)), flush=True)
    return 0 if len(ledger)==len(tasks) and all(r['result']['status']=='completed' for r in ledger) else 2


if __name__ == '__main__':
    raise SystemExit(main())
