"""Finite review regression: four native rejections, five successful short runs."""
from pathlib import Path
import json
import time
import run_innovation_p3 as p
from local_identity import freeze_active_identity

SESSION=p.ROOT/'runs/innovation_p3_review_20261015'


def tasks():
    rows=[]
    def add(name,steps,mode='two_stage',**extra):
        overrides=extra.pop('overrides',{})
        rows.append(dict(name=name,scene='Bunny',mode=mode,binary='active',
            config=p.config('Bunny',steps,mode,**overrides),**extra))
    error='P3 requires IPC with all P2 increments disabled'
    add('reject_backend_without_resolved',1,expect_error=error,env_override=dict(GIPC_CONTACT_BACKEND='toi_al'))
    add('reject_p2_without_resolved',1,expect_error=error,env_override=dict(GIPC_P2_LAZY_SCENE='1'))
    add('reject_pool_without_resolved',1,expect_error='P3 requires the original DCD query, contact pool disabled',env_override=dict(GIPC_CONTACT_POOL='1'))
    add('reject_study_backend_without_resolved',1,'fused',expect_error=error,
        env_override=dict(GIPC_CONTACT_BACKEND='toi_al'),overrides=dict(diagnostics=['p3_ee'],p3_ee_study_frames='1'))
    add('smoke_fused',2,'fused');add('smoke_two_stage',2)
    add('prefix_bunny',14,'fused',save_checkpoint=14,overrides=dict(ipc_termination='movement_only'))
    add('query_bunny_two_stage',1,checkpoint_from='prefix_bunny/checkpoint',target_frame=14,
        overrides=dict(diagnostics=['p3_ee'],p3_ee_study_frames='14'))
    add('fallback_bunny_two_stage',1,checkpoint_from='prefix_bunny/checkpoint',target_frame=14,
        overrides=dict(p3_ee_queue_capacity=1))
    return rows


def main():
    session=SESSION.resolve();p.require(not session.exists(),'Fresh review session required')
    build=p.ROOT/'build/innovation_p3_review_v2_20261015'
    identity=freeze_active_identity(p.ROOT,build,build/'Release/gipc.exe',build/'build.log',
        p.ROOT/'build/innovation_p3_review_native_snapshot_20261015')
    session.mkdir();expected=p.m.power_guid()
    p.write_new(session/'seal.json',dict(identity=identity,controller=p.record(Path(__file__)),
        dependencies=[p.record(Path(p.__file__)),p.record(Path(p.m.__file__)),p.record(p.ROOT/'tools/bench/config.py'),
            p.record(p.ROOT/'tools/bench/validate_run.py'),p.record(Path(p.runner.__file__))],
        tasks=tasks(),expected_successful_frames=20,expected_native_rejections=4,power_guid=expected,
        timing_comparison=False,performance_certified=False))
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[]
    try:
        with p.runner.gpu_lock(p.ROOT):
            for task in tasks():
                p.require(p.m.power_guid()==expected,'Power changed before review run');samples=[]
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=p.m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid))
                    p.require(guid==expected,'Power changed during review run');return value
                def env(c,out):
                    value=p.environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if 'save_checkpoint' in task:value.update(GIPC_CHECKPOINT_FRAME=str(task['save_checkpoint']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
                    if 'checkpoint_from' in task:
                        cp=session/task['checkpoint_from'];p.require(p.read(cp/'checkpoint.json')['binary_id']==identity['exe']['sha256'],'Wrong checkpoint binary')
                        value['GIPC_CHECKPOINT_LOAD']=str(cp)
                    if 'expect_error' in task:
                        value.pop('GIPC_RESOLVED_CONFIG',None);value.update(task['env_override'])
                    return value
                p.runner.environment,p.runner.gpu_query=env,query
                result=p.runner.execute(session,task,identity,0);folder=session/task['name']
                p.write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=p.m.power_guid()))
                row=dict(task=task,result=result)
                try:
                    p.require(result['cleanup_owned_job_empty'] and p.m.power_guid()==expected,'Cleanup/power failed')
                    if 'expect_error' in task:
                        requested=p.read(folder/'requested.json')
                        p.require('GIPC_RESOLVED_CONFIG' not in requested['environment'],'Negative test kept resolved logging')
                        log=(folder/'run.log').read_text(errors='replace')
                        p.require(result['status']=='failed' and result['exit_code']!=0 and result['recorded_frames']==0
                            and task['expect_error'] in log,'Native isolation did not reject at initialization')
                        row['expected_native_rejection']=True
                    else:
                        p.require(result['status']=='completed','Positive native run failed');row['observation']=p.m.observation(folder,task)
                        if 'checkpoint_from' in task:row['restore']=p.m.validate_restore(folder,session/task['checkpoint_from'],identity,task['target_frame'])
                        frames=p.read(folder/'output/stats.json')['frames']
                        p.require(all(f['p3_ee']['w_queries']==(f['p3_ee']['dispatch_queries'] if task['mode']=='two_stage' else 0) for f in frames),'Dispatch accounting differs')
                        if task['name'].startswith('query_'):
                            study=json.loads((folder/'p3_ee.jsonl').read_text())
                            p.require(study['passed'] and study['binary_id']==identity['exe']['sha256'] and study['frame']==14
                                and study['queue_allocation_scope']=='header_and_pair_buffer' and len(study['pairs'])==7,'Review query failed')
                            p.require(all(pair[arm]['counts_equal'] and pair[arm]['typed_multiset_equal'] for pair in study['pairs'] for arm in ('F','W')),'Review tuple mismatch')
                        if task['name'].startswith('fallback_'):p.require(sum(f['p3_ee']['complete_fused_fallbacks'] for f in frames)>0,'Fallback not exercised')
                    row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row);p.write_new(folder/'review_validation.json',row)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks()) and all(r['passed'] for r in rows)
    p.write_new(session/'ledger.json',dict(complete=complete,runs=rows))
    p.require(complete,'Review stopped; preserve partial output')
    print(json.dumps(dict(passed=True,successful_runs=5,successful_frames=20,native_rejections=4)))


if __name__=='__main__':main()
