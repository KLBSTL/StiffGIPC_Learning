"""Owned, serial execution of native full-apply fixed-system contracts and timings."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from windows_owned_job import OwnedJob
from windows_runner import gpu_lock,gpu_query,gpu_processes,driver_model
from run_innovation_m0 import power_guid
from local_identity import record,verify,text
from linux_runner import require,write_new
from prepare_full_apply_graph import RAW,REPORT,BUILD,CUDA,FROZEN,read

def seal():
    require(not (RAW/'runtime_seal.json').exists(),'Already sealed')
    identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']])
    verify(manifest['original_inputs']+manifest['generated_inputs']+[manifest['source'],manifest['protocol']])
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};xml=ET.parse(BUILD/'full_apply_graph.vcxproj')
    units=[]
    for tag in ['CudaCompile','ClCompile']:
        for e in xml.findall('.//m:'+tag,ns):
            if not e.attrib.get('Include'):continue
            p=Path(e.attrib['Include']);units.append((p if p.is_absolute() else BUILD/p).resolve())
    require(set(units)=={ROOT/'tools/bench/full_apply_graph/driver.cu',FROZEN/'gipc/statistics.cpp'},'Compile units differ')
    objects=list((BUILD/'full_apply_graph.dir/Release').glob('*.obj'))
    tracks=[p for p in BUILD.glob('full_apply_graph.dir/Release/**/*.tlog') if p.is_file()]
    links=[p for p in tracks if p.name.lower().startswith('link.read')]
    require(len(objects)>=2 and links,'Actual objects/link tracking missing')
    linked='\n'.join(text(p).replace('\\','/').casefold() for p in links)
    require(all(str(p.resolve()).replace('\\','/').casefold() in linked for p in objects),'Object missing from link evidence')
    helpers=[ROOT/'tools/local'/name for name in ['windows_owned_job.py','windows_runner.py','run_innovation_m0.py',
        'local_identity.py','local_plan.py','prepare_full_apply_graph.py']]
    helpers += [ROOT/'tools/bench'/name for name in ['linux_runner.py','config.py']]
    write_new(RAW/'runtime_seal.json',dict(identity=record(BUILD/'identity.json'),manifest=record(RAW/'input_manifest.json'),
        protocol=record(REPORT/'PLAN.md'),runner=record(Path(__file__)),compiled=[record(p) for p in objects+tracks],
        helpers=[record(p) for p in helpers],power_guid=power_guid(),performance_certified=False))
    print(json.dumps(dict(sealed=True,binary=identity['exe']['sha256'])),flush=True)

def run(action):
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify([packet[k] for k in ['identity','manifest','protocol','runner']]+packet['compiled']+packet['helpers'])
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']])
    verify(manifest['original_inputs']+manifest['generated_inputs'])
    if action=='measure':require(read(RAW/'contracts/result.json')['passed'],'Contracts required')
    folder=RAW/action;require(not folder.exists(),'No implicit retry');folder.mkdir()
    expected=packet['power_guid'];start=time.monotonic();job=None
    result=dict(passed=False,action=action,power_guid=expected,binary=identity['exe'],samples=[],performance_certified=False)
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Require 3GiB free for guarded replay')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Reserve insufficient')
            model=driver_model(0);foreign=gpu_processes(0,None,model);require(not foreign['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            env['CUDA_VISIBLE_DEVICES']='0';env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            cmd=[identity['exe']['path'],action,str(RAW/'input_manifest.json'),str(folder)]
            result.update(command=cmd,gpu_before=before,process_before=foreign,memory_budget_mib=budget,timeout_seconds=240,
                environment_overrides=dict(CUDA_VISIBLE_DEVICES='0',CUDA_BIN=str(CUDA/'bin/x64'),GIPC_environment_removed=True))
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(cmd,ROOT,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0);guid=power_guid();processes=gpu_processes(0,job.pid,model,ownership=job.owns_pid)
                    result['samples'].append(dict(epoch=time.time(),power_guid=guid,gpu=sample,processes=processes))
                    require(guid==expected and sample['uuid']==before['uuid'],'Power/GPU changed')
                    require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                    require(sample['memory.free']>=768 and before['memory.free']-sample['memory.free']<=budget,'Memory budget')
                    require(time.monotonic()-start<=240,'Replay deadline')
                    time.sleep(.25)
            require(job.poll()==0,'Native replay failed')
            data=read(folder/('contracts.json' if action=='contracts' else 'measurements.json'))
            uuid=before['uuid'].removeprefix('GPU-').replace('-','').lower()
            require(data['passed'] and data['device']['uuid_hex'].lower()==uuid,'Runtime device/result differs')
            require(len(data['cases'])==(6 if action=='contracts' else 4),'Case count differs')
            require(all(len(c['warm'])==9 and len(c['graphs'])==6 and c['restoration_passed']
                and c['post_graph_exception_restored'] for c in data['cases']),'Incomplete contracts')
            require(len(data['samples'])==(0 if action=='contracts' else 648),'Measurement count differs')
            verify(manifest['original_inputs']+manifest['generated_inputs']);result['passed']=True
    except Exception as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ['passed','action','exit_code','error','wall_seconds']}),flush=True)
    require(result['passed'],'Native replay stopped; preserve outputs')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','contracts','measure']);a=p.parse_args()
    seal() if a.action=='seal' else run(a.action)
