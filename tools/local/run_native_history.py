"""Finite protected native host-history runs; no cross-binary checkpoint restore."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
import run_innovation_p3 as p
from local_identity import active_identity,record,verify
from linux_runner import require,write_new

RAW=ROOT/'runs/innovation_p5_native_history_20261016'
REPORT=ROOT/'reports/innovation_p5_native_history_20261016'
BUILD=ROOT/'build/native_history_20261016'
RESTORE=('full_system_restored','primary_restored_bitwise','workspace_restored','production_graph_restored',
         'operator_signature_restored','production_info_restored','solve_context_restored','config_restored','cost_trace_restored')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def tasks(stage):
    result=[]
    cases=[('smoke','Bunny',1,'1'),('fault','Bunny',1,'1')] if stage=='small' else [
        ('bunny','Bunny',14,'1,2'),('mixed','Mixed',15,'1,2')]
    for name,scene,frame,directions in cases:
        result.append(dict(name=name,scene=scene,expected_fault=name=='fault',target_frame=frame,
            binary='active',config=p.config(scene,frame,'fused',timeout_seconds=240,diagnostics=['fixed'],
                fixed_frames=str(frame),fixed_directions=directions,fixed_graph_chunk_study=False)))
    return result


def seal():
    require(not RAW.exists(),'Fresh native history batch required')
    identity=active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log')
    RAW.mkdir();write_new(RAW/'seal.json',dict(identity=identity,runner=record(Path(__file__)),
        protocol=record(REPORT/'PLAN.md'),power_guid=p.m.power_guid(),tasks={s:tasks(s) for s in ('small','representative')},
        native_execution='host',production_threshold_changed=False,performance_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)


def run(stage):
    packet=read(RAW/'seal.json');identity=packet['identity'];verify([packet['runner'],packet['protocol']])
    require(packet['tasks'][stage]==tasks(stage),'Task protocol changed')
    require(not (RAW/(stage+'_ledger.json')).exists(),'No implicit retries')
    if stage=='representative':require(read(RAW/'small_ledger.json')['complete'],'Small contracts required')
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[];expected=packet['power_guid']
    try:
        with p.runner.gpu_lock(ROOT):
            for task in tasks(stage):
                samples=[];require(p.m.power_guid()==expected,'Power changed')
                def query(gpu,timeout=10):
                    sample=original_query(gpu,timeout=timeout);guid=p.m.power_guid();samples.append(dict(epoch=time.time(),guid=guid))
                    require(guid==expected,'Power changed during run');return sample
                def environment(c,out):
                    value=p.environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'],
                        GIPC_FIXED_NATIVE_HISTORY_STUDY='1',GIPC_NATIVE_HISTORY_FAIL_AFTER_CAPTURE=str(int(task['expected_fault'])))
                    return value
                p.runner.environment,p.runner.gpu_query=environment,query
                result=p.runner.execute(RAW,task,identity,0);folder=RAW/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,after=p.m.power_guid(),samples=samples))
                row=dict(task=task,result=result,passed=False)
                try:
                    require(result['cleanup_owned_job_empty'],'Owned Job not cleaned')
                    files=sorted((folder/'fixed').glob('*_native_history_study.json'))
                    require(len(files)==(1 if stage=='small' else 2),'Wrong number of native history systems')
                    for file in files:
                        study=read(file);require(all(study.get(k) for k in RESTORE) and not study['restoration_error'],
                            'Protected state was not restored')
                        if task['expected_fault']:
                            require(not study['passed'] and study.get('error')=='Native history injected failure after captured solve'
                                and result.get('exit_code') not in (0,None),'Wrong injected failure')
                        else:
                            require(study['passed'] and len(study['runs'])==6,'Native study incomplete')
                            for pair in (1,2,3):
                                arms=[r for r in study['runs'] if r['pair']==pair]
                                require(len(arms)==2 and {r['capture'] for r in arms}=={False,True},'Missing pair')
                    if not task['expected_fault']:
                        require(result['status']=='completed' and read(folder/'config_validation.json')['passed'],
                            'Native run/config failed')
                    row.update(passed=True,studies=[record(f) for f in files])
                except Exception as error:row['error']=str(error)
                write_new(folder/'history_validation.json',row);rows.append(row)
                print(json.dumps(dict(run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(RAW/(stage+'_ledger.json'),dict(complete=complete,runs=rows));require(complete,'Native history stage stopped; retain outputs')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['seal','small','representative']);args=parser.parse_args()
    seal() if args.action=='seal' else run(args.action)
