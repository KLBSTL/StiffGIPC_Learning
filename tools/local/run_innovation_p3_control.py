"""Three predeclared F/F captures after cross-process trial geometry differed."""
import json
from pathlib import Path
import time
import run_innovation_p3 as p


def main():
    session=p.SESSION.resolve();packet=p.read(session/'seal.json');identity=packet['identity']
    p.require(p.read(session/'query_ledger.json')['complete'],'Original query stage incomplete')
    p.verify([packet['controller'],packet['runner']]);p.verify_active_sources(identity)
    tasks=[]
    for t in p.tasks('query'):
        if t['mode']=='fused':tasks.append(t|dict(name=t['name'].replace('_r1','_control_r2')))
    plan=session/'control_query_plan.json'
    p.require(not plan.exists(),'No implicit control retries')
    p.write_new(plan,dict(tasks=tasks,controller=p.record(Path(__file__)),
        reason='First queried trial positions differed between restored processes; calibrate with one extra F per scene',
        original_seal=p.record(session/'seal.json'),performance_runs=False))
    original_env,original_query=p.runner.environment,p.runner.gpu_query
    expected=packet['power_guid'];rows=[]
    try:
        with p.runner.gpu_lock(p.ROOT):
            for task in tasks:
                p.require(p.m.power_guid()==expected,'Power changed before control');samples=[]
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=p.m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid))
                    p.require(guid==expected,'Power changed during control');return value
                def env(c,out):
                    value=p.environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',
                        GIPC_BINARY_ID=identity['exe']['sha256'],GIPC_CHECKPOINT_LOAD=str(session/task['checkpoint_from']))
                    return value
                p.runner.environment,p.runner.gpu_query=env,query
                result=p.runner.execute(session,task,identity,0);folder=session/task['name']
                p.write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=p.m.power_guid()))
                row=dict(task=task,result=result)
                try:
                    p.require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Control run failed')
                    p.require(p.m.power_guid()==expected,'Power changed after control')
                    row['restore']=p.m.validate_restore(folder,session/task['checkpoint_from'],identity,task['target_frame'])
                    row['observation']=p.m.observation(folder,task)
                    q=json.loads((folder/'p3_ee.jsonl').read_text())
                    p.require(q['passed'] and q['frame']==task['target_frame'] and q['binary_id']==identity['exe']['sha256'],'Control query failed')
                    studies=list((folder/'fixed').glob('*_study.json'));p.require(len(studies)==1,'Missing control matrix')
                    matrix=p.read(studies[0]);p.require(matrix['system_unchanged'] and matrix['primary_restored_bitwise'],'Control diagnostic mutated system')
                    row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row);p.write_new(folder/'p3_validation.json',row)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==3 and all(r['passed'] for r in rows)
    p.write_new(session/'control_query_ledger.json',dict(complete=complete,runs=rows))
    p.require(complete,'F/F capture failed; preserve evidence')


if __name__=='__main__':main()
