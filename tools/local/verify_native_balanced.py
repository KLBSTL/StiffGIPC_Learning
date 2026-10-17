"""Bounded, owned CTest: two new contracts first, then the full 29-test suite."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_runner import gpu_lock,gpu_query
from windows_owned_job import OwnedJob
from run_innovation_m0 import power_guid
from local_identity import record
from linux_runner import require,write_new

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('contracts','full'));args=parser.parse_args()
    build=ROOT/'build/native_balanced_v2_20261016';folder=ROOT/'build'/('native_balanced_verification_'+args.phase)
    require(not folder.exists(),'Fresh verification required');folder.mkdir()
    expected=2 if args.phase=='contracts' else 29
    command=['D:/computer/cmake/bin/ctest.exe','--test-dir',str(build),'-C','Release','--output-on-failure','-j','1']
    if args.phase=='contracts':command+=['-R','^native_balanced_']
    job=None;samples=[];power=power_guid();start=time.monotonic()
    result=dict(passed=False,command=command,source=record(Path(__file__)),power_guid=power,expected=expected)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=2048,'Small contracts need 2GiB free')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            cuda=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
            env['PATH']=os.pathsep.join([str(build/'Release'),str(cuda/'bin/x64'),str(cuda/'bin'),env.get('PATH','')])
            env.update(GIPC_FIXED_NATIVE_HISTORY_STUDY='0',GIPC_FIXED_NATIVE_BALANCED_STUDY='0',GIPC_NATIVE_BALANCED_FAULT='none')
            job=OwnedJob()
            with (folder/'ctest.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0);samples.append(sample)
                    require(power_guid()==power,'Power changed')
                    require(time.monotonic()-start<180,'CTest deadline')
                    require(before['memory.free']-sample['memory.free']<1024 and sample['memory.free']>=768,'CTest memory budget')
                    time.sleep(.25)
            require(job.poll()==0,'CTest failed')
            log=(folder/'ctest.log').read_text(encoding='utf-8-sig',errors='replace')
            require(f'0 tests failed out of {expected}\n' in log and '100% tests passed' in log,'Unexpected CTest count')
            result['passed']=True
    except Exception as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll());job.close()
        result.update(wall_seconds=time.monotonic()-start,samples=samples)
        write_new(folder/'verification.json',result)
    print(json.dumps({k:result.get(k) for k in ('passed','exit_code','wall_seconds','error','expected')}),flush=True)
    require(result['passed'],'Native balanced verification failed')

if __name__=='__main__':main()
