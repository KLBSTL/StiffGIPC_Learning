"""Finite P3 F/W study; isolated same-input replays and short paired scenes."""
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
from config import expand,environment,read
from linux_runner import require,write_new
from local_identity import record,verify,verify_active_sources

SESSION=ROOT/'runs/innovation_p3_20261015'
SCENES={'Boxes':('boxes1920_cloth_l',17),'Mixed':('bunny_cloth_bunny_l',15),
        'Bunny':('cloth_fixed_bunny_l',14)}

def config(scene,steps,mode='fused',**overrides):
    return expand(m.configuration(scene,steps)|dict(p3_ee_mode=mode,p3_ee_queue_capacity=1048576)|overrides)

def tasks(stage):
    result=[]
    def add(name,scene,steps,mode='fused',**extra):
        overrides=extra.pop('overrides',{})
        result.append(dict(name=name,scene=scene,binary='active',mode=mode,config=config(scene,steps,mode,**overrides),**extra))
    if stage=='smoke':
        for mode in ('fused','two_stage'):add('smoke_'+mode,'Bunny',2,mode)
    elif stage=='prefix':
        for scene,(_,frame) in SCENES.items():
            add('prefix_'+scene.lower(),scene,frame,save_checkpoint=frame,overrides=dict(ipc_termination='movement_only'))
    elif stage in ('query','matched','fallback'):
        repeats=3 if stage=='matched' else 1
        for repeat in range(1,repeats+1):
            scenes=list(SCENES);modes=['fused','two_stage']
            if repeat==2:scenes.reverse();modes.reverse()
            if stage=='fallback':modes=['two_stage']
            for scene in scenes:
                for mode in modes:
                    frame=SCENES[scene][1];overrides={}
                    if stage=='query':overrides=dict(diagnostics=['p3_ee','fixed'],p3_ee_study_frames=str(frame),fixed_frames=str(frame),fixed_directions='1')
                    if stage=='fallback':overrides=dict(p3_ee_queue_capacity=1)
                    add(f'{stage}_{scene.lower()}_{mode}_r{repeat}',scene,1,mode,repeat=repeat,
                        checkpoint_from='prefix_'+scene.lower()+'/checkpoint',target_frame=frame,overrides=overrides)
    elif stage=='whole':
        for repeat in (1,2,3):
            scenes=list(SCENES);modes=['fused','two_stage']
            if repeat==2:scenes.reverse();modes.reverse()
            for scene in scenes:
                for mode in modes:add(f'whole_{scene.lower()}_{mode}_r{repeat}',scene,SCENES[scene][1],mode,repeat=repeat)
    else:raise ValueError(stage)
    return result

def seal(session):
    session=session.resolve();require(not session.exists(),'Fresh P3 session required')
    build=ROOT/'build/innovation_p3_20261015'
    identity=read(ROOT/'build/innovation_p3_native_snapshot_20261015/snapshot_identity.json')
    require(identity['exe']==record(build/'Release/gipc.exe'),'Wrong frozen executable')
    verify(identity['build_evidence']+identity['objects']+[identity['exe']]+identity['dlls'])
    verify_active_sources(identity)
    session.mkdir(parents=True)
    write_new(session/'seal.json',dict(schema='innovation_p3.v1',identity=identity,controller=record(Path(__file__)),
        runner=record(Path(runner.__file__)),power_guid=m.power_guid(),base_commit='07534faba7d377315ac8f2023e63eab73fef14db',
        planned={s:tasks(s) for s in ('smoke','prefix','query','matched','fallback','whole')},
        fixed_queue_bytes=1048576*8+16,scope='EE only; original F/VF/FullCCD/classifier unchanged, P2 off',
        performance_certified=False,physical_quality_certified=False,
        stop_rule='No full query benefit -> no adaptive selector; 15% stage and 5% full solver room before expansion'))
    print(json.dumps(dict(sealed=True,binary_sha256=identity['exe']['sha256'])))

def run_stage(session,stage):
    session=session.resolve();packet=read(session/'seal.json');identity=packet['identity']
    require(packet['planned'][stage]==tasks(stage),'Finite plan changed');verify([packet['controller'],packet['runner']])
    require(not (session/(stage+'_ledger.json')).exists(),'No implicit retries')
    previous={'prefix':'smoke','query':'prefix','matched':'query','fallback':'matched','whole':'fallback'}
    if stage in previous:require(read(session/(previous[stage]+'_ledger.json'))['complete'],'Predecessor failed')
    original_env,original_query=runner.environment,runner.gpu_query;rows=[];expected=packet['power_guid']
    try:
        with runner.gpu_lock(ROOT):
            for task in tasks(stage):
                require(m.power_guid()==expected,'Power changed before run');samples=[]
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid));require(guid==expected,'Power changed during run')
                    return value
                def env(c,out):
                    value=environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if 'save_checkpoint' in task:
                        value.update(GIPC_CHECKPOINT_FRAME=str(task['save_checkpoint']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
                    if 'checkpoint_from' in task:
                        cp=session/task['checkpoint_from'];require(read(cp/'checkpoint.json')['binary_id']==identity['exe']['sha256'],'Cross-binary restore rejected')
                        value['GIPC_CHECKPOINT_LOAD']=str(cp)
                    return value
                runner.environment,runner.gpu_query=env,query
                result=runner.execute(session,task,identity,0);folder=session/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=m.power_guid()))
                row=dict(task=task,result=result)
                try:
                    require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Native run/resource/ownership failure')
                    require(m.power_guid()==expected,'Power changed after run');row['observation']=m.observation(folder,task)
                    frames=read(folder/'output/stats.json')['frames']
                    require(all(f['p3_ee']['mode']==task['mode'] for f in frames),'Native mode mismatch')
                    if 'checkpoint_from' in task:
                        row['restore']=m.validate_restore(folder,session/task['checkpoint_from'],identity,task['target_frame'])
                    if stage=='query':
                        observations=[json.loads(s) for s in (folder/'p3_ee.jsonl').read_text().splitlines()]
                        require(len(observations)==1 and observations[0]['passed'],'Missing/failed same-query study')
                        study=observations[0]
                        require(study['frame']==task['target_frame'] and study['binary_id']==identity['exe']['sha256'],'Wrong query identity')
                        require(all(p[a]['counts_equal'] and p[a]['typed_multiset_equal'] for p in study['pairs'] for a in ('F','W')),'Query mismatch')
                        row['query_pairs']=len(study['pairs'])
                        snapshots=list((folder/'fixed').glob('*_study.json'));require(len(snapshots)==1,'Missing unique first-direction matrix')
                        matrix=read(snapshots[0]);require(matrix['system_unchanged'] and matrix['primary_restored_bitwise'],'Fixed diagnostic mutated primary')
                    if stage=='fallback':require(sum(f['p3_ee']['complete_fused_fallbacks'] for f in frames)>0,'Tiny queue did not exercise complete fallback')
                    row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row);write_new(folder/'p3_validation.json',row)
                if not row['passed']:break
    finally:runner.environment,runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(session/(stage+'_ledger.json'),dict(stage=stage,complete=complete,runs=rows))
    require(complete,'P3 finite stage stopped; retain all partial/failed outputs')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal','smoke','prefix','query','matched','fallback','whole'))
    parser.add_argument('--session',type=Path,default=SESSION)
    a=parser.parse_args();seal(a.session) if a.stage=='seal' else run_stage(a.session,a.stage)
