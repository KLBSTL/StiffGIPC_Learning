"""One bounded refinement: earliest maximum-work Boxes frame in its completed prefix.

The primary M0 results and their source runner archive stay immutable. This
extension also checks the corrected from-zero/checkpoint metadata end to end.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
import run_innovation_m0 as m


def main():
    parent=ROOT/'runs/innovation_m0_20261014'
    packet=m.read(parent/'seal.json')
    require=m.require
    require(all(m.read(parent/(s+'_ledger.json'))['complete'] for s in ('smoke','prefix','matched','reference')),
        'Primary M0 matrix must be complete')
    frames=m.read(parent/'prefix_boxes_movement/output/stats.json')['frames']
    counts=[sum('pcg' in n for n in f['newton']) for f in frames]
    target=max(range(1,len(counts)+1),key=lambda i:counts[i-1])
    require(target<=20 and counts[target-1]>=6,'No bounded active Boxes state')
    session=ROOT/'runs/innovation_m0_boxes_hard_20261014'
    require(not session.exists(),'Fresh refinement session required')
    selection=dict(rule='Earliest frame with maximum N in completed unchanged B1-MOV first20 prefix',
        parent_stats=m.record(parent/'prefix_boxes_movement/output/stats.json'),physical_frame=target,
        historical_N=counts[target-1],chosen_before_refinement_runs=True)
    prefix=[dict(name='prefix_boxes_hard_movement',scene='Boxes',binary='active',
        config=m.configuration('Boxes',target,'movement_only'),checkpoint_save_frame=target)]
    matched=[]
    for repeat in (1,2):
        modes=['movement_only','legacy','gated','compensated','strict']
        if repeat==2:modes.reverse()
        for mode in modes:
            matched.append(dict(name=f'boxes_hard_{mode}_r{repeat}',scene='Boxes',binary='active',
                config=m.configuration('Boxes',1,'movement_only' if mode=='strict' else mode,
                    rho=1e-6 if mode=='strict' else 1e-4,shadow=True),rule_label=mode,repeat=repeat,
                target_frame=target,checkpoint_from='prefix_boxes_hard_movement/checkpoint'))
    declared=dict(prefix=prefix,matched=matched)
    m.tasks=lambda stage:declared[stage]
    session.mkdir(parents=True)
    m.write_new(session/'selection.json',selection)
    m.write_new(session/'seal.json',packet|dict(planned=declared,
        parent_seal=m.record(parent/'seal.json'),extension_controller=m.record(Path(__file__)),
        current_runner=m.record(Path(m.windows_runner.__file__))))
    m.write_new(session/'smoke_ledger.json',dict(complete=True,
        inherited_same_binary_smoke=m.record(parent/'smoke_ledger.json'),
        scope='Same verified native programs/inputs; origin metadata only changed. Integration verified below.'))
    for stage in ('prefix','matched'):m.run_stage(session,stage)
    for t in prefix+matched:
        request=m.read(session/t['name']/'requested.json')
        actual='checkpoint_from' not in t
        require(request['from_zero']==actual,'Requested origin still mislabeled')
        require(bool(request['checkpoint_load'])==(not actual),'Checkpoint metadata missing')
    m.write_new(session/'origin_metadata_gate.json',dict(passed=True,checked_runs=len(prefix+matched),
        solver_changed=False,parent_measurements_rewritten=False))
    print(json.dumps(dict(refinement_complete=True,physical_frame=target,runs=11,origin_metadata_gate=True)))


if __name__=='__main__':main()
