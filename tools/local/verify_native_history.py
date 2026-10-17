"""One serial, bounded CTest Job for the new native build."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_runner import gpu_lock,gpu_query
from windows_owned_job import OwnedJob
from run_innovation_m0 import power_guid
from local_identity import record,verify
from linux_runner import require,write_new


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retry-outside-sandbox',action='store_true');args=parser.parse_args()
    build=ROOT/'build/native_history_20261016'
    folder=ROOT/'build'/('native_history_verification_v2' if args.retry_outside_sandbox else 'native_history_verification')
    require((build/'windows_manifest.json').exists() and not folder.exists(),'Completed build / fresh verification required')
    folder.mkdir();ctest=Path('D:/computer/cmake/bin/ctest.exe');job=None;samples=[];power=power_guid();start=time.monotonic()
    result=dict(passed=False,source=record(Path(__file__)),power_guid=power,performance_certified=False)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=2048,'Small fixtures need 2GiB free')
            command=[str(ctest),'--test-dir',str(build),'-C','Release','--output-on-failure','-j','1']
            result.update(command=command,gpu_before=before)
            env=os.environ.copy();cuda=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
            env['PATH']=os.pathsep.join([str(build/'Release'),str(cuda/'bin/x64'),str(cuda/'bin'),env.get('PATH','')])
            env['GIPC_FIXED_NATIVE_HISTORY_STUDY']='0';env['GIPC_NATIVE_HISTORY_FAIL_AFTER_CAPTURE']='0'
            job=OwnedJob()
            with (folder/'ctest.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0);samples.append(sample)
                    require(power_guid()==power,'Power changed')
                    require(time.monotonic()-start<180,'CTest deadline exceeded')
                    require(before['memory.free']-sample['memory.free']<1024 and sample['memory.free']>=768,'CTest memory guard')
                    time.sleep(.25)
            require(job.poll()==0,'CTest failed')
            text=(folder/'ctest.log').read_text(encoding='utf-8-sig',errors='replace')
            require('100% tests passed' in text and 'out of 27' in text,'Expected 27 CTests missing')
            result['passed']=True
    except Exception as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll());job.close()
        result.update(wall_seconds=time.monotonic()-start,samples=samples)
        write_new(folder/'verification.json',result)
    print(json.dumps({k:result.get(k) for k in ('passed','exit_code','wall_seconds','error')}),flush=True)
    require(result['passed'],'Native verification failed')


if __name__=='__main__':main()
