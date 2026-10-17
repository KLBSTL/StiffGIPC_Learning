"""Owned, serial GPU stages. Timing is gated by independent CPU contracts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_owned_job import OwnedJob
from windows_runner import gpu_lock,gpu_query,gpu_processes,driver_model
from run_innovation_m0 import power_guid
from local_identity import record,verify,text
from linux_runner import require,write_new
from prepare_tiny_prefetch import RAW,REPORT,BUILD,CUDA,FROZEN,read

def validate_identity():
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json')
    verify([packet[k] for k in ['identity','manifest','protocol','amendment','inspector_amendment','runner']]+packet['compiled']+packet['helpers'])
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+m['original_inputs']+m['generated_inputs'])
    verify(read(RAW/'production_before.json'))
    return packet,identity,m

def seal():
    require(not (RAW/'runtime_seal.json').exists(),'Already sealed');identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+m['original_inputs']+m['generated_inputs']+[m['source'],m['protocol']])
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};xml=ET.parse(BUILD/'tiny_prefetch.vcxproj');units=[]
    for tag in ['CudaCompile','ClCompile']:
        for e in xml.findall('.//m:'+tag,ns):
            if e.attrib.get('Include'):
                p=Path(e.attrib['Include']);units.append((p if p.is_absolute() else BUILD/p).resolve())
    require(set(units)=={ROOT/'tools/bench/tiny_prefetch/driver.cu',FROZEN/'gipc/statistics.cpp'},'Compile units differ')
    objects=list((BUILD/'tiny_prefetch.dir/Release').glob('*.obj'));tracks=[p for p in BUILD.glob('tiny_prefetch.dir/Release/**/*.tlog') if p.is_file()]
    links=[p for p in tracks if p.name.lower().startswith('link.read')];require(len(objects)>=2 and links,'Missing build tracking')
    linked='\n'.join(text(p).replace('\\','/').casefold() for p in links)
    require(all(str(p.resolve()).replace('\\','/').casefold() in linked for p in objects),'Missing link input')
    helpers=[ROOT/'tools/local'/n for n in ['windows_owned_job.py','windows_runner.py','run_innovation_m0.py','local_identity.py','local_plan.py','prepare_tiny_prefetch.py']]
    helpers += [ROOT/'tools/bench'/n for n in ['linux_runner.py','config.py','analyze_tiny_prefetch.py','m3_numeric.py']]
    write_new(RAW/'runtime_seal.json',dict(identity=record(BUILD/'identity.json'),manifest=record(RAW/'input_manifest.json'),
        protocol=m['protocol'],amendment=record(REPORT/'PLAN_AMENDMENT_001.md'),runner=record(Path(__file__)),compiled=[record(p) for p in objects+tracks],helpers=[record(p) for p in helpers],
        inspector_amendment=record(REPORT/'PLAN_AMENDMENT_002.md'),
        production_before=record(RAW/'production_before.json'),power_guid=power_guid(),performance_certified=False))
    print(dict(sealed=True,binary=identity['exe']['sha256']),flush=True)

def run(action):
    packet,identity,m=validate_identity()
    if action!='tiny':require(read(RAW/'tiny/audit.json')['passed'],'Tiny independent contracts required')
    if action in ['tiny_measure','measure']:require(read(RAW/'contracts/audit.json')['passed'],'Full action contracts required')
    folder=RAW/action;require(not folder.exists(),'No implicit retry');folder.mkdir();job=None;start=time.monotonic();expected=packet['power_guid']
    require(expected=='52521609-efc9-4268-b9ba-67dea73f18b2','Unexpected planned power mode')
    result=dict(passed=False,action=action,power_guid=expected,samples=[],binary=identity['exe'],performance_certified=False)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Require 3GiB free')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Reserve insufficient')
            model=driver_model(0);foreign=gpu_processes(0,None,model);require(not foreign['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed');env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            env['CUDA_VISIBLE_DEVICES']='0';env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            command=[identity['exe']['path'],action,str(RAW/'input_manifest.json'),str(folder)]
            result.update(command=command,gpu_before=before,process_before=foreign,memory_budget_mib=budget,timeout_seconds=240)
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0);guid=power_guid();processes=gpu_processes(0,job.pid,model,ownership=job.owns_pid)
                    result['samples'].append(dict(epoch=time.time(),power_guid=guid,gpu=sample,processes=processes))
                    require(guid==expected and sample['uuid']==before['uuid'],'Power/GPU changed')
                    require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                    require(sample['memory.free']>=768 and before['memory.free']-sample['memory.free']<=budget,'Memory budget')
                    require(time.monotonic()-start<=240,'Deadline');time.sleep(.25)
            require(job.poll()==0,'Native probe failed');data=read(folder/'data.json')
            require(data['passed'] and data['action']==action and data['device']['uuid_hex'].lower()==before['uuid'].removeprefix('GPU-').replace('-','').lower(),'Runtime result/device differs')
            require(len(data['cases'])==dict(tiny=70,contracts=6,tiny_measure=2,measure=2)[action],'Wrong cases')
            require(all(c['guards_passed'] and (action.startswith('tiny') or c['restored']) for c in data['cases']),'Missing guards/restoration')
            require(len(data['samples'])==dict(tiny=0,contracts=0,tiny_measure=192,measure=384)[action],'Wrong windows')
            validate_identity();result['passed']=True
    except Exception as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle);job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ['passed','action','exit_code','error','wall_seconds']}),flush=True)
    require(result['passed'],'Stage stopped; preserve outputs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','tiny','contracts','tiny_measure','measure']);a=p.parse_args()
    seal() if a.action=='seal' else run(a.action)
