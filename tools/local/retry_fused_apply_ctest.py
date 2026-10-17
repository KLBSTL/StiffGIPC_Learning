"""One explicit full-suite retry; retain the WinError5 interruption unchanged."""
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
from local_identity import record,verify
from linux_runner import require,write_new

def main():
    previous=ROOT/'build/fused_apply_verification_full/verification.json'
    failed=json.loads(previous.read_text(encoding='utf-8-sig'))
    require(not failed['passed'] and failed['error']=='PermissionError: [WinError 5] Member image'
        and failed['cleanup_owned_job_empty'],'Only the retained monitor interruption permits this retry')
    verify([failed['source']])
    build=ROOT/'build/native_fused_apply_20261016';folder=ROOT/'build/fused_apply_verification_full_retry1'
    require(not folder.exists(),'Single fresh retry required');folder.mkdir()
    command=['D:/computer/cmake/bin/ctest.exe','--test-dir',str(build),'-C','Release','--output-on-failure','-j','1']
    require(command==failed['command'],'Retry command differs')
    job=None;samples=[];power=power_guid();start=time.monotonic()
    result=dict(passed=False,command=command,source=record(Path(__file__)),power_guid=power,expected=29,
        previous_interruption=record(previous),monitor_source=record(ROOT/'tools/local/windows_owned_job.py'))
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=2048,'CTest needs 2GiB free')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            cuda=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
            env['PATH']=os.pathsep.join([str(build/'Release'),str(cuda/'bin/x64'),str(cuda/'bin'),env.get('PATH','')])
            env.update(GIPC_FIXED_NATIVE_HISTORY_STUDY='0',GIPC_FIXED_NATIVE_BALANCED_STUDY='0',
                GIPC_NATIVE_BALANCED_FAULT='none',GIPC_NATIVE_BALANCED_QUALITY_STUDY='0',GIPC_NATIVE_BALANCED_APPLY_STUDY='0')
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
            require('0 tests failed out of 29\n' in log and '100% tests passed' in log,'Unexpected test count')
            result['passed']=True
    except Exception as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            job.close()
        result.update(wall_seconds=time.monotonic()-start,samples=samples);write_new(folder/'verification.json',result)
    print(json.dumps({k:result.get(k) for k in ('passed','expected','exit_code','error')}),flush=True)
    require(result['passed'],'Full CTest retry failed')
if __name__=='__main__':main()
