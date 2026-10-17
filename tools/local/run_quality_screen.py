"""Four matched native arms; no production rollout or 100-frame launch."""
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
RAW=ROOT/'runs/innovation_p5_quality_screen_20261016'
REPORT=ROOT/'reports/innovation_p5_quality_screen_20261016'
BUILD=ROOT/'build/native_quality_screen_20261016'
RESTORE=previous.RESTORE

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def tasks(stage):
    rows=previous.tasks(stage)
    for row in rows:row['quality_study']=True
    if stage=='small':
        legacy=previous.tasks('small')[0];legacy.update(name='legacy_smoke',quality_study=False);rows.insert(0,legacy)
    return rows

def seal():
    require(not RAW.exists(),'Fresh quality batch required')
    require(read(ROOT/'build/quality_screen_verification_full/verification.json')['passed'],'Full CTest required')
    identity=active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log')
    RAW.mkdir();write_new(RAW/'seal.json',dict(identity=identity,runner=record(Path(__file__)),
        protocol=record(REPORT/'PLAN.md'),power_guid=p.m.power_guid(),
        tasks={s:tasks(s) for s in ('small','faults','representative')},performance_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)

def run(stage):
    packet=read(RAW/'seal.json');identity=packet['identity'];verify([packet['runner'],packet['protocol']])
    require(packet['tasks'][stage]==tasks(stage) and not (RAW/(stage+'_ledger.json')).exists(),'Task changed/already run')
    if stage!='small':require(read(RAW/'small_ledger.json')['complete'],'Small contracts required')
    if stage=='representative':require(read(RAW/'faults_ledger.json')['complete'],'Fallback/exception tests required')
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
                        GIPC_NATIVE_BALANCED_QUALITY_STUDY=str(int(task['quality_study'])))
                    values.update(flags);write_new(out/'quality_environment.json',flags);return values
                p.runner.environment,p.runner.gpu_query=environment,query
                result=p.runner.execute(RAW,task,identity,0);folder=RAW/task['name']
                write_new(folder/'power_samples.json',dict(expected=expected,after=p.m.power_guid(),samples=samples))
                row=dict(task=task,result=result,passed=False)
                try:
                    require(result['cleanup_owned_job_empty'],'Owned Job not cleaned')
                    files=sorted((folder/'fixed').glob('*_native_balanced_study.json'));require(len(files)==2,'Missing n1/n2')
                    source,target=map(read,files)
                    require(all(all(s.get(k) for k in RESTORE) and not s['restoration_error'] for s in (source,target)),
                            'State not restored')
                    require(source['passed'] and len(source['runs'])==2,'Source capture failed')
                    if task['fault']=='after_apply':
                        require(not target['passed'] and target.get('error')=='Native balanced injected failure after apply'
                                and result.get('exit_code') not in (0,None),'Wrong injected exception')
                    else:
                        count=12 if task['quality_study'] else 6
                        require(target['passed'] and len(target['runs'])==count and result['status']=='completed'
                                and read(folder/'config_validation.json')['passed'],'Target/config failed')
                        require(target['quality_frequency_study']==task['quality_study'],'Wrong quality selector')
                        for pair in (1,2,3):
                            arms=[r for r in target['runs'] if r['pair']==pair]
                            wanted={(a,s) for a in (False,True) for s in ((False,True) if task['quality_study'] else (False,))}
                            require(len(arms)==len(wanted) and {(r['requested_balanced'],r['screened']) for r in arms}==wanted,
                                    'Missing matched arm')
                        if target['coarse_ready'] and task['quality_study']:
                            require(target['coarse_restored_bitwise'] and len(target['coarse_warm_calibration'])==3,'Warm coarse not restored')
                        if task['fault']=='coarse_reject' or stage=='small':
                            require(all(not r['used_balanced'] and r['fallback']==r['requested_balanced'] for r in target['runs']),
                                    'Missing expected MAS fallback')
                    row.update(passed=True,studies=[record(f) for f in files])
                except Exception as error:row['error']=str(error)
                write_new(folder/'quality_validation.json',row);rows.append(row)
                print(json.dumps(dict(run=task['name'],passed=row['passed'],error=row.get('error'))),flush=True)
                if not row['passed']:break
    finally:p.runner.environment,p.runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all(r['passed'] for r in rows)
    write_new(RAW/(stage+'_ledger.json'),dict(complete=complete,runs=rows));require(complete,'Quality stage stopped; keep outputs')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('seal','small','faults','representative'));args=parser.parse_args()
    seal() if args.action=='seal' else run(args.action)
