"""One bounded Job-owned GPU probe, with immutable input and build checks."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_runner import gpu_lock,gpu_query,gpu_processes,driver_model,remaining
from windows_owned_job import OwnedJob
from local_identity import record,verify
from linux_runner import require,write_new

SESSION=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def power():
    result=subprocess.run(['powercfg','/getactivescheme'],capture_output=True,check=True,timeout=10)
    match=re.search(rb'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',result.stdout)
    require(match is not None,'Power scheme unavailable');return match[0].decode().lower()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--input',choices=['inputs','fixture_positive_inputs','fixture_negative_inputs'],required=True)
    parser.add_argument('--build',default='cgstats_gpu_v2_20261015');parser.add_argument('--expected-statistics-rejection',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();require(re.fullmatch(r'[a-zA-Z0-9_-]+',args.name) is not None and
        re.fullmatch(r'[a-zA-Z0-9_-]+',args.build) is not None,'Simple names required')
    folder=SESSION/args.input;out=SESSION/args.name;require(not out.exists(),'Fresh GPU run required');out.mkdir()
    output=out/'data';output.mkdir();identity=read(ROOT/'build'/args.build/'identity.json')
    manifest=read(folder/'manifest.json');dependencies=identity['sources']+identity['evidence']+[identity['exe']]+manifest['records']+[record(folder/'manifest.json')]
    verify(dependencies)
    expected=power();samples=[];job=None;started=time.monotonic();status='not_launched'
    result=dict(passed=False,performance_certified=False,expected_statistics_rejection=args.expected_statistics_rejection,
        runner=record(Path(__file__)),input=record(folder/'manifest.json'),identity=identity,
        power_guid=expected,timeout_seconds=180,memory_budget_mib=1024)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);mode=driver_model(0);processes=gpu_processes(0,None,mode)
            require(before['memory.free']>=2048 and not processes['blocking_foreign_pids'],'GPU resource preflight failed')
            result.update(gpu_before=before,gpu_driver_model=mode,prelaunch_processes=processes,desktop_load_uncontrolled=mode=='WDDM')
            env=os.environ.copy();cuda=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
            env['PATH']=os.pathsep.join([str(cuda/'bin/x64'),str(cuda/'bin'),env.get('PATH','')])
            env['CUDA_VISIBLE_DEVICES']=before['uuid'];command=[identity['exe']['path'],str(folder),str(output)]
            if args.memcheck:
                require(args.input=='fixture_positive_inputs','Memcheck scope is the small positive fixture')
                sanitizer=cuda/'compute-sanitizer/compute-sanitizer.exe'
                require(sanitizer.is_file(),'Compute Sanitizer missing')
                command=[str(sanitizer),'--tool','memcheck','--error-exitcode','99','--target-processes','all']+command
                result['sanitizer']=record(sanitizer)
            result['command']=command;job=OwnedJob()
            with (out/'run.log').open('xb') as log:
                job.launch(command,out,env,log);status='running'
                while not job.finished():
                    job.observe();sample=gpu_query(0);sample['power_guid']=power();samples.append(sample)
                    require(sample['power_guid']==expected,'Power changed')
                    if time.monotonic()-started>180:status='timeout'
                    elif sample['memory.free']<768 or before['memory.free']-sample['memory.free']>1024:status='memory_budget'
                    if status!='running':job.terminate();break
                    time.sleep(.25)
            # Close the last-poll race: an exit between monitor samples must
            # still satisfy the absolute deadline before accepting completion.
            remaining(started+180,180)
            exit_code=job.poll();require(status=='running','GPU probe guard: '+status)
            result.update(exit_code=exit_code,owned_processes=job.observe(),cleanup_owned_job_empty=job.finished())
            if args.expected_statistics_rejection:
                require(exit_code!=0 and (output/'failure.json').exists(),'Invalid-statistics fixture was accepted')
                failure=read(output/'failure.json');require('Invalid CG energy statistics' in failure['error'],'Unexpected failure')
                status='expected_statistics_rejection'
            else:
                require(exit_code==0 and read(output/'result.json')['passed'],'GPU probe failed');status='completed'
            verify(dependencies);require(power()==expected,'Power changed after GPU probe');result['passed']=True
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error);status='failed'
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(owned_processes=job.observe(),cleanup_owned_job_empty=job.finished(),exit_code=job.poll());job.close()
        result.update(status=status,wall_seconds=time.monotonic()-started,samples=samples)
        write_new(out/'launch.json',result)
    print(json.dumps({k:result.get(k) for k in ('passed','status','exit_code','wall_seconds','error')}),flush=True)
    require(result['passed'],'Probe failed; preserve output')


if __name__=='__main__':main()
