"""Re-run existing tiny/apply contracts only; no build, timing, or new algorithm."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_owned_job import OwnedJob
from windows_runner import gpu_lock,gpu_query,gpu_processes,driver_model
from run_innovation_m0 import power_guid
from local_identity import record,verify,tree
from linux_runner import require,write_new
from prepare_tiny_prefetch import CUDA,read
from run_tiny_prefetch import validate_identity
from tiny_prefetch_review import RAW,REPORT,OLD_RAW,BUILD,strict_metadata
import analyze_tiny_prefetch as numeric

def deadline_ok(elapsed):
    require(0<elapsed<=240,'Absolute native deadline exceeded')

def main(action):
    require(read(RAW/'history.json')['passed'],'Strict history review required')
    packet,identity,manifest=validate_identity();expected=packet['power_guid']
    folder=RAW/('fresh_'+action);require(not folder.exists(),'No implicit retry');folder.mkdir()
    code=[record(Path(__file__)),record(ROOT/'tools/bench/tiny_prefetch_review.py')]
    write_new(folder/'before.json',dict(source=code,history=record(RAW/'history.json'),protocol=record(REPORT/'PLAN.md'),
        manifest=packet['manifest'],exe=identity['exe'],parent_runtime=record(OLD_RAW/'runtime_seal.json')))
    start=time.monotonic();job=None;result=dict(passed=False,action=action,samples=[],performance_measured=False)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Require 3GiB free')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Reserve insufficient')
            model=driver_model(0);foreign=gpu_processes(0,None,model);require(not foreign['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')};env['CUDA_VISIBLE_DEVICES']='0'
            env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            command=[identity['exe']['path'],action,str(OLD_RAW/'input_manifest.json'),str(folder)]
            result.update(command=command,exe=identity['exe'],gpu_before=before,process_before=foreign,
                power_guid=expected,memory_budget_mib=budget,deadline_seconds=240)
            def sample():
                gpu=gpu_query(0);guid=power_guid();processes=gpu_processes(0,job.pid if job else None,model,ownership=job.owns_pid if job else None)
                row=dict(epoch=time.time(),gpu=gpu,power_guid=guid,processes=processes);result['samples'].append(row)
                require(guid==expected and gpu['uuid']==before['uuid'],'Power/GPU changed')
                require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                require(gpu['memory.free']>=768 and before['memory.free']-gpu['memory.free']<=budget,'Memory budget')
                return row
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():
                    job.observe();sample();deadline_ok(time.monotonic()-start);time.sleep(.25)
            # Check the last-poll completion interval explicitly, even after native exit.
            deadline_ok(time.monotonic()-start);require(job.poll()==0,'Existing native contracts failed')
            result['post_exit_resource_check']=sample()
            validate_identity();verify(code)
            data=read(folder/'data.json');exact_uuid=before['uuid'].removeprefix('GPU-').replace('-','').lower()
            require(data['device']['uuid_hex'].lower()==exact_uuid,'Native GPU differs')
            result['metadata']=strict_metadata(manifest,data,folder,action)
            result['passed']=True
    except Exception as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        if job is not None:
            try:
                if not job.finished():job.terminate()
                result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            finally:job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    require(result['passed'],'Contract rerun stopped; preserve outputs')
    # Full independent numeric audit is outside the owned native deadline.
    data=read(folder/'data.json')
    outputs=numeric.tiny_audit(manifest,data) if action=='tiny' else numeric.full_audit(manifest,data)
    require(len(outputs)==(420 if action=='tiny' else 36),'Wrong fresh output count')
    numeric.csv_new(REPORT/'tables'/('fresh_'+action+'_CPU.csv'),outputs)
    write_new(folder/'audit.json',dict(passed=True,outputs=len(outputs),maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),
        tiny_bitwise_equal=True if action=='tiny' else None,source=code,inputs=tree(folder)))
    print(dict(action=action,passed=True,outputs=len(outputs),max_CPU_relative=max(r['CPU_relative'] for r in outputs)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tiny','contracts']);main(p.parse_args().action)
