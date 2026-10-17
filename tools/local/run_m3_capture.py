"""Six finite native captures, reusing the sealed B1 binary and its checkpoints."""
from pathlib import Path
import json
import time
import run_innovation_p3 as p

SESSION=p.ROOT/'runs/innovation_m3_20261015'
PARENT=p.ROOT/'runs/innovation_p3_balanced_20261015'


def tasks():
    result=[]
    for scene in p.SCENES:
        result.append(dict(name='ordinary_'+scene.lower(),scene=scene,binary='active',kind='ordinary',
            config=p.config(scene,1,'fused',timeout_seconds=600,diagnostics=['fixed'],fixed_frames='1',
                fixed_directions='1',fixed_mas_stage_study=True)))
    for scene,(_,frame) in p.SCENES.items():
        result.append(dict(name='adjacent_'+scene.lower(),scene=scene,binary='active',kind='adjacent',target_frame=frame,
            checkpoint_from='prefix_'+scene.lower()+'/checkpoint',
            config=p.config(scene,1,'fused',timeout_seconds=600,diagnostics=['fixed'],fixed_frames=str(frame),fixed_directions='1,2')))
    return result


def main():
    p.require(not SESSION.exists(),'Fresh M3 capture required')
    parent=p.read(PARENT/'seal.json');identity=parent['identity'];p.verify_active_sources(identity)
    p.verify(identity['sources']+[identity['exe']]+identity['dlls'])
    SESSION.mkdir();expected=p.m.power_guid()
    p.write_new(SESSION/'seal.json',dict(schema='innovation_m3_capture.v1',identity=identity,
        parent=p.record(PARENT/'seal.json'),controller=p.record(Path(__file__)),power_guid=expected,tasks=tasks(),
        native_solver_modified=False,performance_certified=False,physical_quality_certified=False))
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[]
    try:
        with p.runner.gpu_lock(p.ROOT):
            for task in tasks():
                samples=[];p.require(p.m.power_guid()==expected,'Power changed before capture')
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=p.m.power_guid()
                    samples.append(dict(epoch_seconds=time.time(),guid=guid));p.require(guid==expected,'Power changed')
                    return value
                def env(c,out):
                    value=p.environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if task['kind']=='ordinary':value['GIPC_MAS_AUDIT']=str(out/'operator')
                    if 'checkpoint_from' in task:value['GIPC_CHECKPOINT_LOAD']=str(PARENT/task['checkpoint_from'])
                    return value
                p.runner.environment,p.runner.gpu_query=env,query
                result=p.runner.execute(SESSION,task,identity,0);folder=SESSION/task['name']
                p.write_new(folder/'power_samples.json',dict(expected=expected,after=p.m.power_guid(),samples=samples))
                row=dict(task=task,result=result)
                try:
                    p.require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Capture failed')
                    row['observation']=p.m.observation(folder,task)
                    p.require(p.read(folder/'config_validation.json')['passed'],'Resolved configuration mismatch')
                    if 'checkpoint_from' in task:row['restore']=p.m.validate_restore(folder,PARENT/task['checkpoint_from'],identity,task['target_frame'])
                    files=sorted((folder/'fixed').glob('*_study.json'))
                    p.require(len(files)==(1 if task['kind']=='ordinary' else 2),'Wrong number of frozen systems')
                    for file in files:
                        d=p.read(file)
                        p.require(d['system_unchanged'] and d['primary_restored_bitwise'] and d['system']['preconditioner_export_complete'],
                            'Incomplete or mutated A/b/MAS')
                        if task['kind']=='ordinary':
                            p.require(d['full_snapshot_unchanged_including_scratch'] and d['z_restored_bitwise']
                                and d['graph_signature_restored'],'Stage diagnostic did not restore all state')
                        else:p.require(all(not r['error'] and not r['limit'] for r in d['runs']),'Native fixed replay failed')
                    row['systems']=[p.record(f) for f in files];row['passed']=True
                except Exception as error:row.update(passed=False,error=str(error))
                rows.append(row);p.write_new(folder/'m3_validation.json',row)
                print(json.dumps(dict(run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==6 and all(r['passed'] for r in rows)
    p.write_new(SESSION/'ledger.json',dict(complete=complete,runs=rows))
    p.require(complete,'Capture stopped; retain partial data')


if __name__=='__main__':main()
