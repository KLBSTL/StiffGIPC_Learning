"""Finite M/reference/fused native replay; production and 100-frame runs stay off."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
import run_native_balanced as previous
from local_identity import active_identity,record,verify
from linux_runner import require,write_new
p=previous.p
RAW=ROOT/'runs/innovation_p5_fused_apply_20261016'
REPORT=ROOT/'reports/innovation_p5_fused_apply_20261016'
BUILD=ROOT/'build/native_fused_apply_20261016'
RESTORE=previous.RESTORE

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def tasks(stage):
    rows=previous.tasks(stage)
    for row in rows:
        row.update(quality_study=True,apply_study=True)
        if row['fault']=='after_apply':row.update(name='after_fused_failure',fault='after_fused_apply')
    if stage=='small':
        legacy=previous.tasks('small')[0];legacy.update(name='legacy_smoke',quality_study=False,apply_study=False)
        rows.insert(0,legacy)
    return rows

def seal():
    require(not RAW.exists(),'Fresh fused batch required')
    full=ROOT/'build/fused_apply_verification_full/verification.json'
    retry=ROOT/'build/fused_apply_verification_full_retry1/verification.json'
    if retry.exists():
        failed=read(full);require(not failed['passed'] and failed['cleanup_owned_job_empty']
            and failed['error']=='PermissionError: [WinError 5] Member image','Unexpected retry reason')
        full=retry
    evidence=read(full);require(evidence['passed'] and evidence['expected']==29 and evidence['cleanup_owned_job_empty'],'Full CTest required')
    verify([evidence['source']])
    identity=active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log')
    RAW.mkdir();write_new(RAW/'seal.json',dict(identity=identity,runner=record(Path(__file__)),
        protocol=record(REPORT/'PLAN.md'),full_verification=record(full),power_guid=p.m.power_guid(),
        tasks={s:tasks(s) for s in ('small','faults','representative')},performance_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)

def run(stage):
    packet=read(RAW/'seal.json');identity=packet['identity'];verify([packet['runner'],packet['protocol']])
    require(packet['tasks'][stage]==tasks(stage) and not (RAW/(stage+'_ledger.json')).exists(),'Task changed/already run')
    if stage!='small':require(read(RAW/'small_ledger.json')['complete'],'Small contracts required')
    if stage=='representative':require(read(RAW/'faults_ledger.json')['complete'],'Fault recovery required')
    original_env,original_query=p.runner.environment,p.runner.gpu_query;rows=[];expected=packet['power_guid']
    try:
        with p.runner.gpu_lock(ROOT):
            require(original_query(0)['memory.free']>=(2048 if stage=='small' else 4608),'Memory not ready; no stage launched')
            for task in tasks(stage):
                samples=[];require(p.m.power_guid()==expected,'Power changed')
                def query(gpu,timeout=10):
                    value=original_query(gpu,timeout=timeout);guid=p.m.power_guid();samples.append(dict(epoch=time.time(),guid=guid))
                    require(guid==expected,'Power changed during run');return value
                def environment(c,out):
                    values=p.environment(c,out)
                    flags=dict(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'],
                        GIPC_FIXED_NATIVE_HISTORY_STUDY='0',GIPC_NATIVE_HISTORY_FAIL_AFTER_CAPTURE='0',
                        GIPC_FIXED_NATIVE_BALANCED_STUDY='1',GIPC_NATIVE_BALANCED_FAULT=task['fault'],
                        GIPC_NATIVE_BALANCED_QUALITY_STUDY=str(int(task['quality_study'])),
                        GIPC_NATIVE_BALANCED_APPLY_STUDY=str(int(task['apply_study'])))
                    values.update(flags);write_new(out/'fused_environment.json',flags);return values
                p.runner.environment,p.runner.gpu_query=environment,query
                result=p.runner.execute(RAW,task,identity,0);folder=RAW/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,after=p.m.power_guid(),samples=samples))
                row=dict(task=task,result=result,passed=False)
                try:
                    require(result['cleanup_owned_job_empty'],'Owned Job not cleaned')
                    files=sorted((folder/'fixed').glob('*_native_balanced_study.json'));require(len(files)==2,'Missing n1/n2')
                    source,target=map(read,files)
                    require(all(all(s.get(k) for k in RESTORE) and not s['restoration_error'] for s in (source,target)),
                        'Protected state not restored')
                    require(source['passed'] and len(source['runs'])==2,'Source capture failed')
                    if task['fault']=='after_fused_apply':
                        require(not target['passed'] and target.get('error')=='Native balanced injected failure after fused apply'
                            and result.get('exit_code') not in (0,None) and len(target['runs'])==2,'Wrong injected exception')
                        require(target['failed_target_candidate']['fused_apply'],'Failure did not follow fused apply')
                    else:
                        require(target['passed'] and len(target['runs'])==(9 if task['apply_study'] else 6)
                            and result['status']=='completed' and read(folder/'config_validation.json')['passed'],
                            'Target/config failed')
                        require(target['fused_apply_study']==task['apply_study'],'Wrong apply selector')
                        for pair in (1,2,3):
                            arms=[r for r in target['runs'] if r['pair']==pair]
                            wanted={(False,False),(True,False),(True,True)} if task['apply_study'] else {(False,False),(True,False)}
                            require(len(arms)==len(wanted) and {(r['requested_balanced'],r['requested_fused']) for r in arms}==wanted,
                                'Missing matched arm')
                        if target['coarse_ready'] and task['apply_study']:
                            require(target['coarse_restored_bitwise'] and target['apply_batch_inputs_unchanged']
                                and len(target['apply_batches'])==12,'Warm/batch calibration failed')
                        if task['fault']=='coarse_reject' or stage=='small':
                            require(all(not r['used_balanced'] and not r['used_fused'] and r['fallback']==r['requested_balanced']
                                for r in target['runs']),'Missing explicit MAS fallback')
                    row.update(passed=True,studies=[record(f) for f in files])
                except Exception as error:row['error']=str(error)
                write_new(folder/'fused_validation.json',row);rows.append(row)
                print(json.dumps(dict(run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(RAW/(stage+'_ledger.json'),dict(complete=complete,runs=rows));require(complete,'Fused stage stopped; preserve outputs')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('seal','small','faults','representative'));args=parser.parse_args()
    seal() if args.action=='seal' else run(args.action)
