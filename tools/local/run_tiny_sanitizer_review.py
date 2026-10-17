"""NVIDIA dynamic checks on the unchanged 70-case tiny executable path."""
import argparse
import os
from pathlib import Path
import re
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
from tiny_prefetch_review import RAW,REPORT,OLD_RAW,strict_metadata
from run_tiny_review import deadline_ok
import analyze_tiny_prefetch as numeric

def main(tool):
    require(read(RAW/'fresh_tiny/audit.json')['passed'] and read(RAW/'fresh_contracts/audit.json')['passed'],'Fresh contracts required')
    packet,identity,manifest=validate_identity();expected=packet['power_guid'];folder=RAW/('sanitizer_'+tool)
    require(not folder.exists(),'No implicit retry');folder.mkdir();job=None;start=time.monotonic()
    sanitizer=CUDA/'compute-sanitizer/compute-sanitizer.exe';sources=[record(Path(__file__)),record(ROOT/'tools/bench/tiny_prefetch_review.py'),record(ROOT/'tools/local/run_tiny_review.py')]
    result=dict(passed=False,tool=tool,samples=[],sources=sources,sanitizer=record(sanitizer),exe=identity['exe'],performance_measured=False)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Require 3GiB free')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Reserve insufficient')
            model=driver_model(0);require(not gpu_processes(0,None,model)['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')};env['CUDA_VISIBLE_DEVICES']='0'
            env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            command=[str(sanitizer),'--tool',tool,'--error-exitcode','99','--target-processes','all','--log-file',str(folder/'sanitizer.log'),
                identity['exe']['path'],'tiny',str(OLD_RAW/'input_manifest.json'),str(folder)]
            result.update(command=command,gpu_before=before,memory_budget_mib=budget,power_guid=expected)
            def sample():
                gpu=gpu_query(0);guid=power_guid();processes=gpu_processes(0,job.pid,model,ownership=job.owns_pid)
                result['samples'].append(dict(epoch=time.time(),gpu=gpu,power_guid=guid,processes=processes))
                require(guid==expected and gpu['uuid']==before['uuid'],'Power/GPU changed')
                require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                require(gpu['memory.free']>=768 and before['memory.free']-gpu['memory.free']<=budget,'Memory budget')
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():
                    job.observe();sample();deadline_ok(time.monotonic()-start);time.sleep(.25)
            deadline_ok(time.monotonic()-start);require(job.poll()==0,'Sanitized native path failed')
            sample();verify(sources+[result['sanitizer']]);validate_identity()
            log=(folder/'sanitizer.log').read_text(encoding='utf-8',errors='replace')
            pattern=r'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)' if tool=='racecheck' else r'ERROR SUMMARY: 0 errors'
            require(re.search(pattern,log) is not None,'Missing zero-error sanitizer summary')
            data=read(folder/'data.json');require(data['device']['uuid_hex'].lower()==before['uuid'].removeprefix('GPU-').replace('-','').lower(),'Native device differs')
            result['metadata']=strict_metadata(manifest,data,folder,'tiny');result['passed']=True
    except Exception as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        if job is not None:
            try:
                if not job.finished():job.terminate()
                result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            finally:job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    require(result['passed'],'Sanitizer review stopped; preserve evidence')
    outputs=numeric.tiny_audit(manifest,read(folder/'data.json'));require(len(outputs)==420,'Wrong sanitized output count')
    numeric.csv_new(REPORT/'tables'/('sanitizer_'+tool+'_CPU.csv'),outputs)
    write_new(folder/'audit.json',dict(passed=True,outputs=420,tiny_bitwise_equal=True,maximum_CPU_relative=max(r['CPU_relative'] for r in outputs),inputs=tree(folder)))
    print(dict(tool=tool,passed=True,outputs=420,wall_seconds=result['wall_seconds']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tool',choices=['memcheck','racecheck','synccheck']);main(p.parse_args().tool)
