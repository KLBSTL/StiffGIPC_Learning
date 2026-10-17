"""Source n1 -> target n2, protected native PCG; finite serial local studies."""
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

RAW=ROOT/'runs/innovation_p5_native_balanced_20261016'
REPORT=ROOT/'reports/innovation_p5_native_balanced_20261016'
BUILD=ROOT/'build/native_balanced_v2_20261016'
RESTORE=('full_system_restored','primary_restored_bitwise','workspace_restored','production_graph_restored',
         'operator_signature_restored','production_info_restored','solve_context_restored','config_restored','cost_trace_restored')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def tasks(stage):
    cases={
        'small':[('smoke','Bunny',1,'none')],
        'faults':[('coarse_rejection','Bunny',14,'coarse_reject'),('after_apply_failure','Bunny',14,'after_apply')],
        'representative':[('bunny','Bunny',14,'none'),('mixed','Mixed',15,'none')],
    }[stage]
    return [dict(name=name,scene=scene,target_frame=frame,fault=fault,binary='active',
        config=p.config(scene,frame,'fused',timeout_seconds=240,diagnostics=['fixed'],
                        fixed_frames=str(frame),fixed_directions='1,2',fixed_graph_chunk_study=False))
        for name,scene,frame,fault in cases]

def seal():
    require(not RAW.exists(),'Fresh native balanced batch required')
    require(read(ROOT/'build/native_balanced_verification_full/verification.json')['passed'],'Full CTest required')
    identity=active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log')
    RAW.mkdir();write_new(RAW/'seal.json',dict(identity=identity,runner=record(Path(__file__)),
        protocol=record(REPORT/'PLAN.md'),power_guid=p.m.power_guid(),tasks={s:tasks(s) for s in ('small','faults','representative')},
        native_execution='host',production_threshold_changed=False,performance_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)

def run(stage):
    packet=read(RAW/'seal.json');identity=packet['identity'];verify([packet['runner'],packet['protocol']])
    require(packet['tasks'][stage]==tasks(stage),'Task protocol changed')
    require(not (RAW/(stage+'_ledger.json')).exists(),'No implicit retry')
    if stage!='small':require(read(RAW/'small_ledger.json')['complete'],'Small contracts required')
    if stage=='representative':require(read(RAW/'faults_ledger.json')['complete'],'Fallback/exception tests required')
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[];expected=packet['power_guid']
    try:
        with p.runner.gpu_lock(ROOT):
            required_free=2048 if stage=='small' else 4608
            require(original_query(0)['memory.free']>=required_free,
                    'Native balanced preflight memory not ready; no stage launched')
            for task in tasks(stage):
                samples=[];require(p.m.power_guid()==expected,'Power changed')
                def query(gpu,timeout=10):
                    sample=original_query(gpu,timeout=timeout);guid=p.m.power_guid();samples.append(dict(epoch=time.time(),guid=guid))
                    require(guid==expected,'Power changed during run');return sample
                def environment(c,out):
                    values=p.environment(c,out)
                    flags=dict(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'],
                        GIPC_FIXED_NATIVE_HISTORY_STUDY='0',GIPC_NATIVE_HISTORY_FAIL_AFTER_CAPTURE='0',
                        GIPC_FIXED_NATIVE_BALANCED_STUDY='1',GIPC_NATIVE_BALANCED_FAULT=task['fault'])
                    values.update(flags);write_new(out/'native_balanced_environment.json',flags);return values
                p.runner.environment,p.runner.gpu_query=environment,query
                result=p.runner.execute(RAW,task,identity,0);folder=RAW/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,after=p.m.power_guid(),samples=samples))
                row=dict(task=task,result=result,passed=False)
                try:
                    require(result['cleanup_owned_job_empty'],'Owned Job not cleaned')
                    files=sorted((folder/'fixed').glob('*_native_balanced_study.json'))
                    require(len(files)==2,'Missing n1/n2 native balanced studies')
                    source,target=map(read,files)
                    for study in (source,target):
                        require(all(study.get(k) for k in RESTORE) and not study['restoration_error'],'Protected state not restored')
                    require(source['passed'] and len(source['runs'])==2,'Source capture failed')
                    if task['fault']=='after_apply':
                        require(not target['passed'] and target.get('error')=='Native balanced injected failure after apply'
                            and result.get('exit_code') not in (0,None),'Wrong exception test')
                    else:
                        require(target['passed'] and len(target['runs'])==6 and result['status']=='completed'
                                and read(folder/'config_validation.json')['passed'],'Target run/config incomplete')
                        for pair in (1,2,3):
                            arms=[r for r in target['runs'] if r['pair']==pair]
                            require(len(arms)==2 and {r['requested_balanced'] for r in arms}=={False,True},'Missing AB/BA pair')
                        if task['fault']=='coarse_reject' or stage=='small':
                            require(all(not r['used_balanced'] and r['fallback']==r['requested_balanced'] for r in target['runs']),
                                    'Expected explicit MAS fallback')
                        # Numerical candidate fallbacks remain visible scientific failures,
                        # rather than be recast as an implementation-contract failure here.
                    row.update(passed=True,studies=[record(f) for f in files])
                except Exception as error:row['error']=str(error)
                write_new(folder/'balanced_validation.json',row);rows.append(row)
                print(json.dumps(dict(run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(RAW/(stage+'_ledger.json'),dict(complete=complete,runs=rows));require(complete,'Native balanced stage stopped; retain outputs')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('seal','small','faults','representative'));args=parser.parse_args()
    seal() if args.action=='seal' else run(args.action)
