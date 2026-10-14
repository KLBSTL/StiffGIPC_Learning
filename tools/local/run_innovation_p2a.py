"""Finite P2-A ablation; identical-binary checkpoints, owned serial GPU jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'/p) for p in ('local','bench')]
import run_innovation_m0 as m
import windows_runner as runner
from local_identity import active_identity,record,verify
from config import expand,environment,read
from linux_runner import require,write_new

SESSION=ROOT/'runs/innovation_p2a_20261014'
SCENES={'Boxes':('boxes1920_cloth_l',17),'Mixed':('bunny_cloth_bunny_l',15),
        'Bunny':('cloth_fixed_bunny_l',14)}
ARMS={'control':(False,False),'scene':(True,False),'abd':(False,True),'both':(True,True)}

def config(scene,steps,arm='control',**overrides):
    lazy,merge=ARMS[arm]
    return expand(dict(scene=SCENES[scene][0],steps=steps,dt=.04,timeout_seconds=120,
        backend='ipc',preset='combined',mas='legacy',pcg_graph_chunk=1,pcg_rho_tol=1e-4,
        ipc_newton_tol=.01,ipc_cumulative_tol=.01,ipc_min_updates=6,ipc_termination='legacy',
        trace_velocity=True,fem_direct_pullback=True,mas_pivot_only_sync=True,
        discrete_bvh_refit=True,discrete_bvh_rebuild_interval=8,
        p2_lazy_scene=lazy,p2_merge_abd_energy=merge)|overrides)

def tasks(stage):
    rows=[]
    def add(name,scene,steps,arm='control',**extra):
        overrides=extra.pop('overrides',{})
        rows.append(dict(name=name,scene=scene,arm=arm,binary='active',
            config=config(scene,steps,arm,**overrides),**extra))
    if stage=='smoke':
        add('smoke_control','Bunny',2)
        add('smoke_both','Bunny',2,'both',overrides=dict(p2_audit=True))
        # A FEM-only scene tests empty ABD output in both existing energy paths.
        for batch in (True,False):
            c=config('Bunny',2,'both',p2_audit=True)|dict(scene='cloth_hang_l',batch=batch)
            rows.append(dict(name='smoke_empty_'+str(int(batch)),scene='Hang',arm='both',
                binary='active',config=expand(c)))
    elif stage=='prefix':
        for scene,(_,frame) in SCENES.items():
            add('prefix_'+scene.lower(),scene,frame,save_checkpoint=frame,
                overrides=dict(ipc_termination='movement_only'))
    elif stage in ('cost','audit','matched'):
        repeats=3 if stage=='matched' else 1
        for repeat in range(1,repeats+1):
            scenes=list(SCENES);arms=list(ARMS)
            if repeat%2==0:scenes.reverse();arms.reverse()
            if stage=='audit':arms=['control','both']
            for scene in scenes:
                for arm in arms:
                    overrides={}
                    if stage=='cost':overrides=dict(diagnostics=['cost'],cost_frames=str(SCENES[scene][1]),cost_events=False)
                    if stage=='audit':overrides=dict(p2_audit=True,discrete_bvh_validate=True,diagnostics=['substeps'])
                    add(f'{stage}_{scene.lower()}_{arm}_r{repeat}',scene,1,arm,repeat=repeat,
                        checkpoint_from='prefix_'+scene.lower()+'/checkpoint',
                        target_frame=SCENES[scene][1],overrides=overrides)
    elif stage=='whole':
        for repeat in (1,2,3):
            scenes=list(SCENES);arms=['control','both']
            if repeat==2:scenes.reverse();arms.reverse()
            for scene in scenes:
                for arm in arms:add(f'whole_{scene.lower()}_{arm}_r{repeat}',scene,SCENES[scene][1],arm,repeat=repeat)
    else:raise ValueError(stage)
    return rows

def seal(session):
    session=Path(session).resolve()
    require(not session.exists(),'Fresh session required; never overwrite measurements')
    build=ROOT/'build/innovation_p2a_20261014'
    identity=active_identity(ROOT,build,build/'Release/gipc.exe',build/'build.log')
    session.mkdir(parents=True)
    write_new(session/'seal.json',dict(schema='innovation_p2a.v1',identity=identity,
        controller=record(Path(__file__)),runner=record(Path(runner.__file__)),power_guid=m.power_guid(),
        plans={s:tasks(s) for s in ('smoke','prefix','cost','audit','matched','whole')},
        selection='M0 earliest high-work Boxes17, Mixed15, Bunny14; freshly captured with this binary',
        hypotheses=dict(line_search_reduction=.15,whole_reduction=.05),
        repeats=3,formal_performance_certification=False,
        restrictions=['No cross-binary checkpoint substitution','No timeout/memory/power failed result accepted',
                      'No profiler or component audit in paired timing','No tolerance/material/CCD change']))
    print(json.dumps(dict(sealed=True,binary_sha256=identity['exe']['sha256'])))

def run_stage(session,stage):
    session=Path(session).resolve()
    packet=read(session/'seal.json');identity=packet['identity']
    verify([packet['controller'],packet['runner']]);require(packet['plans'][stage]==tasks(stage),'Task plan changed')
    path=session/(stage+'_ledger.json');require(not path.exists(),'No implicit repeated stage')
    predecessors={'prefix':'smoke','cost':'prefix','audit':'cost','matched':'audit','whole':'matched'}
    if stage in predecessors:require(read(session/(predecessors[stage]+'_ledger.json'))['complete'],'Predecessor gate failed')
    original_env,original_query=runner.environment,runner.gpu_query
    expected=packet['power_guid'];rows=[]
    try:
        with runner.gpu_lock(ROOT):
            for task in tasks(stage):
                samples=[]
                require(m.power_guid()==expected,'Power mode changed before launch')
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid))
                    require(guid==expected,'Power mode changed during launch');return value
                def env(c,out):
                    value=environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if 'save_checkpoint' in task:
                        value.update(GIPC_CHECKPOINT_FRAME=str(task['save_checkpoint']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
                    if 'checkpoint_from' in task:
                        checkpoint=session/task['checkpoint_from']
                        require(read(checkpoint/'checkpoint.json')['binary_id']==identity['exe']['sha256'],'Cross-binary restore rejected')
                        value['GIPC_CHECKPOINT_LOAD']=str(checkpoint)
                    return value
                runner.environment,runner.gpu_query=env,query
                result=runner.execute(session,task,identity,0);folder=session/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=m.power_guid()))
                row=dict(task=task,result=result)
                try:
                    require(result['status']=='completed' and result.get('cleanup_owned_job_empty'),'Run or ownership cleanup failed')
                    require(m.power_guid()==expected,'Power mode changed after run')
                    row['observation']=m.observation(folder,task)
                    if 'checkpoint_from' in task:
                        row['restore']=m.validate_restore(folder,session/task['checkpoint_from'],identity,task['target_frame'])
                    if stage=='audit' or task['config']['p2_audit']:
                        records=[r for f in read(folder/'output/stats.json')['frames'] for r in f.get('p2_abd_energy_audit',[])]
                        require(bool(records) if task['config']['p2_merge_abd_energy'] else True,'No ABD audit observed')
                        require(all(r['bitwise_equal'] for r in records),'ABD same-state bits differ')
                        row['abd_audit_count']=len(records)
                    if stage=='cost':
                        costs=[json.loads(s) for s in (folder/'cost.jsonl').read_text().splitlines()]
                        require(costs and any(r['stage']=='ipc.line_search.entry_wait' for r in costs),'Missing selected physical-frame costs')
                        row['cost_records']=len(costs)
                    row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row)
                write_new(folder/'p2_validation.json',row)
                if not row['passed']:break
    finally:runner.environment,runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(path,dict(stage=stage,complete=complete,runs=rows))
    require(complete,'Finite P2-A stage stopped; preserve failed/unfinished data')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal','smoke','prefix','cost','audit','matched','whole'))
    parser.add_argument('--session',type=Path,default=SESSION)
    args=parser.parse_args()
    if args.stage=='seal':seal(args.session)
    else:run_stage(args.session,args.stage)

if __name__=='__main__':main()
