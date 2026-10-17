"""Serial owned fixed-system PCG stages, with unchanged resource safeguards."""
import argparse
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cache_pcg import RAW,REPORT,BUILD,CUDA,FROZEN,read
from cache_pcg_experiment import verify_identity
from local_identity import record,verify,text
from linux_runner import require,write_new
from windows_owned_job import OwnedJob
from windows_runner import gpu_lock,gpu_query,gpu_processes,driver_model
from run_innovation_m0 import power_guid
from run_tiny_review import deadline_ok

def seal():
    require(not (RAW/'runtime_seal.json').exists(),'Already sealed')
    identity=read(BUILD/'identity.json');m=read(RAW/'input_manifest.json');require(read(RAW/'compiler/audit.json')['passed'],'No-spill gate')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']]+m['original_inputs']+m['generated_inputs']+m['pcg_inputs'])
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};xml=ET.parse(BUILD/'cache_pcg.vcxproj');units=[]
    for tag in ['CudaCompile','ClCompile']:
        for e in xml.findall('.//m:'+tag,ns):
            if e.attrib.get('Include'):
                p=Path(e.attrib['Include']);units.append((p if p.is_absolute() else BUILD/p).resolve())
    require(set(units)=={BUILD/'project/driver.cu',FROZEN/'gipc/statistics.cpp'},'Unexpected actual compile units')
    objects=list((BUILD/'cache_pcg.dir/Release').glob('*.obj'));tracks=[p for p in BUILD.glob('cache_pcg.dir/Release/**/*.tlog') if p.is_file()]
    links=[p for p in tracks if p.name.lower().startswith('link.read')];require(len(objects)>=2 and links,'Missing object/link evidence')
    linked='\n'.join(text(p).replace('\\','/').casefold() for p in links)
    require(all(str(p.resolve()).replace('\\','/').casefold() in linked for p in objects),'Objects not linked')
    require(power_guid()=='52521609-efc9-4268-b9ba-67dea73f18b2','Power differs from planned mode')
    closure={Path(v.__file__).resolve() for v in sys.modules.values() if getattr(v,'__file__',None) and str(v.__file__).endswith('.py') and Path(v.__file__).resolve().is_relative_to(ROOT)}
    files=[record(p) for p in sorted(closure)]+[record(p) for p in objects+tracks]
    files+=[record(BUILD/'identity.json'),record(RAW/'input_manifest.json'),record(REPORT/'PLAN.md'),record(RAW/'compiler/audit.json')]
    write_new(RAW/'runtime_seal.json',dict(files=files,power_guid=power_guid()));print(dict(sealed=True,exe=identity['exe']['sha256']),flush=True)

def run(stage):
    packet,identity,manifest=verify_identity();action='contracts' if stage=='memcheck' else stage
    if stage!='contracts':require(read(RAW/'contracts/audit.json')['passed'],'Independent contracts required')
    if stage=='measure':require(read(RAW/'memcheck/audit.json')['passed'],'Full recurrence memcheck required')
    folder=RAW/stage;require(not folder.exists(),'No implicit retry');folder.mkdir();job=None;start=time.monotonic();expected=packet['power_guid']
    result=dict(passed=False,stage=stage,action=action,samples=[],instrumented=stage=='memcheck')
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Require 3GiB free')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Reserve insufficient')
            model=driver_model(0);foreign=gpu_processes(0,None,model);require(not foreign['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed');env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            env['CUDA_VISIBLE_DEVICES']='0';env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            command=[identity['exe']['path'],action,str(RAW/'input_manifest.json'),str(folder)]
            if stage=='memcheck':
                tool=CUDA/'compute-sanitizer/compute-sanitizer.exe';result['sanitizer']=record(tool)
                command=[str(tool),'--tool','memcheck','--error-exitcode','99','--target-processes','all','--log-file',str(folder/'sanitizer.log')]+command
            result.update(command=command,exe=identity['exe'],gpu_before=before,process_before=foreign,memory_budget_mib=budget,power_guid=expected,timeout_seconds=240)
            def sample():
                gpu=gpu_query(0);guid=power_guid();processes=gpu_processes(0,job.pid,model,ownership=job.owns_pid)
                row=dict(epoch=time.time(),gpu=gpu,power_guid=guid,processes=processes);result['samples'].append(row)
                require(guid==expected and gpu['uuid']==before['uuid'],'Power/GPU changed')
                require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                require(gpu['memory.free']>=768 and before['memory.free']-gpu['memory.free']<=budget,'Memory budget');return row
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(command,ROOT,env,log)
                while not job.finished():job.observe();sample();deadline_ok(time.monotonic()-start);time.sleep(.25)
            deadline_ok(time.monotonic()-start);require(job.poll()==0,'Native/sanitized stage failed');result['post_exit_resource_check']=sample()
            if stage=='memcheck':
                log=(folder/'sanitizer.log').read_text(encoding='utf-8',errors='replace')
                require(re.search(r'ERROR SUMMARY: 0 errors',log) is not None,'No zero-error summary');verify([result['sanitizer']])
            data=read(folder/'data.json');require(data['passed'] and data['action']==action,'Wrong action')
            require(data['device']['uuid_hex'].lower()==before['uuid'].removeprefix('GPU-').replace('-','').lower(),'Native UUID differs')
            verify_identity();result['passed']=True
    except Exception as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        if job is not None:
            try:
                if not job.finished():job.terminate()
                result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            finally:job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    print({k:result.get(k) for k in ['passed','stage','error','wall_seconds']},flush=True);require(result['passed'],'Stage stopped; preserve evidence')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','contracts','memcheck','measure']);args=p.parse_args();seal() if args.stage=='seal' else run(args.stage)
