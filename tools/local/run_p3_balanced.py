"""P3 balanced query diagnostics and one 100-frame F/W pair per scene."""
from pathlib import Path
import argparse
import json
import time
import run_innovation_p3 as p
from local_identity import freeze_active_identity

SESSION=p.ROOT/'runs/innovation_p3_balanced_20261015'
BUILD=p.ROOT/'build/innovation_p3_balanced_v3_20261015'
STAGES=('smoke','prefix','query','profile','whole')


def tasks(stage):
    rows=[]
    def add(name,scene,steps,mode='fused',**extra):
        overrides=extra.pop('overrides',{})
        rows.append(dict(name=name,scene=scene,mode=mode,binary='active',
            config=p.config(scene,steps,mode,timeout_seconds=600,**overrides),**extra))
    if stage=='smoke':
        for mode in ('fused','two_stage'):add('smoke_'+mode,'Bunny',2,mode)
    elif stage=='prefix':
        for scene,(_,frame) in p.SCENES.items():
            add('prefix_'+scene.lower(),scene,frame,save_checkpoint=frame,
                overrides=dict(ipc_termination='movement_only'))
    elif stage=='query':
        for scene,(_,frame) in p.SCENES.items():
            add('query_'+scene.lower(),scene,1,checkpoint_from='prefix_'+scene.lower()+'/checkpoint',
                target_frame=frame,overrides=dict(diagnostics=['p3_ee'],p3_ee_study_frames=str(frame)))
    elif stage=='profile':
        for index,(scene,(_,frame)) in enumerate(p.SCENES.items()):
            for mode in (('fused','two_stage') if index%2==0 else ('two_stage','fused')):
                add(f'profile_{scene.lower()}_{mode}',scene,3,mode,
                    checkpoint_from='prefix_'+scene.lower()+'/checkpoint',target_frame=frame,
                    overrides=dict(diagnostics=['cost'],cost_frames=f'{frame}-{frame+2}',cost_events=True))
    elif stage=='whole':
        for index,scene in enumerate(p.SCENES):
            for mode in (('fused','two_stage') if index%2==0 else ('two_stage','fused')):
                add(f'whole_{scene.lower()}_{mode}',scene,100,mode)
    else:raise ValueError(stage)
    return rows


def seal(session):
    p.require(not session.exists(),'Fresh session required')
    identity=freeze_active_identity(p.ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log',
        p.ROOT/'build/innovation_p3_balanced_snapshot_20261015')
    session.mkdir(parents=True)
    dependencies=[Path(__file__),Path(p.__file__),Path(p.m.__file__),Path(p.runner.__file__),
        p.ROOT/'tools/bench/config.py',p.ROOT/'tools/bench/validate_run.py',
        p.ROOT/'tools/bench/p3_balanced_analyze.py']
    p.write_new(session/'seal.json',dict(schema='innovation_p3_balanced.v1',identity=identity,
        dependencies=[p.record(f) for f in dependencies],power_guid=p.m.power_guid(),
        planned={s:tasks(s) for s in STAGES},formal_runs=6,formal_frames=600,
        comparison='Current accelerated F/B1 versus fixed W; NOT frozen original StiffGIPC',
        limitations=['One pair per scene; no confidence interval or stable speed certification',
            'WDDM clocks and desktop load uncontrolled','Position differences are not true-solution errors'],
        performance_certified=False,physical_quality_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)


def run_stage(session,stage):
    packet=p.read(session/'seal.json');identity=packet['identity'];expected=packet['power_guid']
    p.verify(packet['dependencies']);p.verify_active_sources(identity)
    p.require(packet['planned'][stage]==tasks(stage),'Sealed plan changed')
    p.require(not (session/(stage+'_ledger.json')).exists(),'No overwriting or implicit retry')
    index=STAGES.index(stage)
    if index:p.require(p.read(session/(STAGES[index-1]+'_ledger.json'))['complete'],'Predecessor incomplete')
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[]
    try:
        with p.runner.gpu_lock(p.ROOT):
            for task in tasks(stage):
                p.require(p.m.power_guid()==expected,'Power changed before run');samples=[]
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=p.m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid))
                    p.require(guid==expected,'Power changed during run');return value
                def env(c,out):
                    value=p.environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if 'save_checkpoint' in task:
                        value.update(GIPC_CHECKPOINT_FRAME=str(task['save_checkpoint']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
                    if 'checkpoint_from' in task:
                        cp=session/task['checkpoint_from']
                        p.require(p.read(cp/'checkpoint.json')['binary_id']==identity['exe']['sha256'],'Cross-binary restore')
                        value['GIPC_CHECKPOINT_LOAD']=str(cp)
                    return value
                p.runner.environment,p.runner.gpu_query=env,query
                result=p.runner.execute(session,task,identity,0);folder=session/task['name']
                p.write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=p.m.power_guid()))
                row=dict(task=task,result=result)
                try:
                    p.require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Native/resource failure')
                    p.require(p.m.power_guid()==expected,'Power changed after run')
                    row['observation']=p.m.observation(folder,task)
                    p.require(p.read(folder/'config_validation.json')['passed'],'Resolved config mismatch')
                    frames=p.read(folder/'output/stats.json')['frames']
                    p.require(len(frames)==task['config']['steps'],'Missing frames')
                    p.require(all(f['p3_ee']['mode']==task['mode'] and f['p3_ee']['w_queries']==
                        (f['p3_ee']['dispatch_queries'] if task['mode']=='two_stage' else 0) for f in frames),'Dispatch mismatch')
                    if 'checkpoint_from' in task:
                        row['restore']=p.m.validate_restore(folder,session/task['checkpoint_from'],identity,task['target_frame'])
                    if stage=='query':
                        study=json.loads((folder/'p3_ee.jsonl').read_text())
                        p.require(study['passed'] and study['binary_id']==identity['exe']['sha256'],'Wrong/failed query')
                        from p3_balanced_analyze import batch_summary
                        row['balanced_timing']=batch_summary(study['batched_timing'])
                    if stage=='profile':
                        costs=[json.loads(x) for x in (folder/'cost.jsonl').read_text().splitlines()]
                        ee=[r for r in costs if r['stage']=='collision.discrete_ee' and r['sample_kind']=='production']
                        p.require(len(ee)==sum(f['p3_ee']['dispatch_queries'] for f in frames),'Not all EE queries profiled')
                        p.require(all(r['gpu_interval_ms'] is not None for r in ee),'Missing EE GPU intervals')
                    row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row);p.write_new(folder/'balanced_validation.json',row)
                print(json.dumps(dict(stage=stage,run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    p.write_new(session/(stage+'_ledger.json'),dict(stage=stage,complete=complete,runs=rows))
    p.require(complete,'Stage incomplete; preserve failed/partial data')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal',*STAGES,'all','analyze'))
    parser.add_argument('--session',type=Path,default=SESSION);a=parser.parse_args();session=a.session.resolve()
    if a.stage in ('seal','all'):seal(session)
    for stage in STAGES:
        if a.stage in (stage,'all'):run_stage(session,stage)
    if a.stage in ('all','analyze'):
        from p3_balanced_analyze import analyze
        analyze(session,p.ROOT/'reports/innovation_p3_balanced_20261015/analysis')


if __name__=='__main__':main()
