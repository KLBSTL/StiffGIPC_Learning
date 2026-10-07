"""Close the finite run and record a CPU-only metadata revision, no GPU calls."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('tools/local','tools/bench','tools/diagnostic')]
from config import read
from local_identity import record,verify,tool_inventory,verify_seal
from linux_runner import require,write_new,verify_files
from quality_analysis import validate_base
import report_rounds as rounds
SESSION=rounds.SESSION;REPORT=rounds.REPORT;BUILD=rounds.BUILD

def main():
    old_path=BUILD/'report_seal_capability_v2.json';old=read(old_path)
    for identity in old['programs'].values():
        verify(identity['sources']+identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    changes={}
    for name,path in (('quality_analysis','tools/diagnostic/quality_analysis.py'),
                      ('profiler_windows','tools/local/profiler_windows.py'),
                      ('report_rounds','tools/local/report_rounds.py')):
        snapshot=REPORT/'frozen_tools'/f'{name}.capability_v2.py.txt'
        previous=record(snapshot)
        if name=='profiler_windows':
            requests=[read(SESSION/folder/'requested.json') for folder in
                ('profile_sphere_f49_node','profile_fixed_f65_graph')]
            frozen=[next(r for r in q['new_tools'] if Path(r['path']).resolve()==(ROOT/path).resolve()) for q in requests]
            require(all(previous['sha256']==r['sha256'] for r in frozen),'Frozen capture tool differs')
        elif name!='report_rounds':
            frozen=next(r for r in old['tools'] if Path(r['path']).resolve()==(ROOT/path).resolve())
            require(previous['sha256']==frozen['sha256'],'Frozen pre-fix tool hash differs')
        else:
            require(previous['sha256']==read(REPORT/'PLAN_BEFORE_FINAL_METADATA_FIX.json')['controller']['sha256'],
                    'Frozen pre-fix controller differs')
        changes[name]=dict(previous_snapshot=previous,current=record(ROOT/path))
    new=dict(old,tools=tool_inventory(ROOT),post_run_cpu_revision=True,
        native_programs_unchanged=True,physical_configs_unchanged=True,maximum_profile_attempts_per_program=2)
    write_new(rounds.SEAL,new);verify_seal(ROOT,rounds.SEAL)
    plan=read(SESSION/'plan.json');plan.update(controller=record(ROOT/'tools/local/report_rounds.py'),
        seal=record(rounds.SEAL),finite_gpu_round_closed=True)
    (SESSION/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    indices=[]
    for out in sorted(SESSION.iterdir()):
        if not out.is_dir() or not (out/'result.json').exists():continue
        result=read(out/'result.json');req=read(out/'requested.json')
        verify_files(out,read(out/'evidence.json')['files'])
        require(result['recorded_frames']==100 and result.get('finite') is True,'Partial/nonfinite trajectory')
        kind=('observer' if out.name.startswith('observation_') else 'cost' if out.name.startswith('cost_')
              else 'nsight' if out.name.startswith('profile_') else 'performance_diagnostic')
        if kind=='observer':require(validate_base(out)['passed'],'Observer capability/config mismatch')
        indices.append(dict(name=out.name,kind=kind,raw_status=result['status'],recorded_frames=result['recorded_frames'],
            solver_seconds=result.get('solver_seconds'),wall_seconds=result.get('wall_seconds'),
            config_sha256=req['config_sha256'],exe_sha256=req['exe_sha256'],
            scene=req['expanded_config']['scene'],rho=req['expanded_config']['pcg_rho_tol'],
            initial_position=record(out/'trace/state_0000.bin')['sha256'],
            actual_velocity_export=(out/'trace/velocity_0100.bin').is_file(),
            evidence=record(out/'evidence.json'),quality_certified=False,performance_certified=False))
    require(len(indices)==78,'Declared finite run count differs')
    exe=old['programs']['active']['exe']['sha256'];profile_receipts=[]
    for p in (ROOT/'runs/local_profile_attempts').glob('*/attempt*.json'):
        receipt=read(p);request=Path(receipt['output'])/'requested.json'
        program=receipt.get('exe_sha256')
        if program is None and request.is_file():program=read(request).get('exe_sha256')
        if program==exe:profile_receipts.append(record(p))
    require(len(profile_receipts)==2,'Capture attempts not exactly two for this program')
    revision=dict(schema='report.post_run.cpu_revision.v1',old_seal=record(old_path),new_seal=record(rounds.SEAL),
        changes=changes,native_programs_unchanged=True,physical_configs_unchanged=True,
        cpu_only=True,extra_gpu_runs=0,closed_profile_receipts=profile_receipts,
        reason='Correct actual-velocity capability metadata; preserve two-capture budget across seal revisions. CPU monitor fixture updated to a valid synthetic hash. Old GPU evidence and status remain untouched.')
    write_new(REPORT/'POST_RUN_CPU_REVISION.json',revision)
    write_new(REPORT/'RUN_INDEX.json',dict(rows=indices,completed_native_100_frame_trajectories=78,
        formal_quality_certification=False,formal_performance_certification=False))
    with (REPORT/'RUN_INDEX.csv').open('x',newline='',encoding='utf-8-sig') as f:
        fields=[k for k in indices[0] if k!='evidence'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in indices)
    write_new(REPORT/'FINAL_IDENTITIES.json',dict(programs=new['programs'],
        baseline_observer=read(ROOT/'build/report_base_observer_20261007_v2/identity.json'),
        tools=new['tools'],local_report_tools=[record(p) for p in sorted((ROOT/'tools/local').glob('report_*.py'))],
        requested_configs=plan['stages'],native_programs_unchanged=True,
        scope='All native runs used the v2 seal. The final v3 changes CPU metadata only; hashes of raw evidence are retained. No profile budget reset and no extra GPU launches.'))
    print(json.dumps(dict(native_100_frame_trajectories=len(indices),profile_attempts=len(profile_receipts),
        native_programs_unchanged=True,extra_gpu_runs=0)))

if __name__=='__main__':main()
