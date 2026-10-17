"""Serial owned GPU contracts/measurements for the immutable stage-only probe."""
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
from local_identity import record,verify,tree,text
from linux_runner import require,write_new
from prepare_projection_probe import RAW,REPORT,BUILD,CUDA,read

def seal():
    require(not (RAW/'runtime_seal.json').exists(),'Runtime already sealed')
    identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe'],identity['parent']])
    verify(manifest['generated_inputs']+manifest['original_inputs']+[manifest['source'],manifest['protocol']])
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'};project=ET.parse(BUILD/'projection_probe.vcxproj')
    units=[e.attrib['Include'] for e in project.findall('.//m:CudaCompile',ns) if e.attrib.get('Include')]
    require(len(units)==1,'Unexpected CUDA source count')
    unit=Path(units[0]);unit=(unit if unit.is_absolute() else BUILD/unit).resolve()
    require(unit==ROOT/'tools/bench/projection_probe/probe.cu','Unexpected CUDA source')
    objects=list((BUILD/'projection_probe.dir/Release').glob('*.obj'));require(len(objects)>=1,'Compiled object missing')
    tracks=[p for p in BUILD.glob('projection_probe.dir/Release/**/*.tlog') if p.is_file()];require(tracks,'Tracking evidence missing')
    links=[p for p in tracks if p.name.lower().startswith('link.read')];require(links,'Link inputs missing')
    linked='\n'.join(text(p).replace('\\','/').casefold() for p in links)
    require(all(str(p.resolve()).replace('\\','/').casefold() in linked for p in objects),'Object absent from linker inputs')
    evidence=[record(p) for p in objects+tracks]
    write_new(RAW/'runtime_seal.json',dict(identity=record(BUILD/'identity.json'),manifest=record(RAW/'input_manifest.json'),
        protocol=record(REPORT/'PLAN.md'),runner=record(Path(__file__)),compiled=evidence,power_guid=power_guid(),
        parent=identity['parent'],performance_certified=False))
    print(json.dumps(dict(runtime_sealed=True,binary=identity['exe']['sha256'])),flush=True)

def run(action):
    packet=read(RAW/'runtime_seal.json');identity=read(BUILD/'identity.json');manifest=read(RAW/'input_manifest.json')
    verify([packet[k] for k in ('identity','manifest','protocol','runner','parent')]+packet['compiled'])
    verify(identity['sources']+identity['runtime']+identity['build_evidence']+[identity['exe']])
    verify(manifest['generated_inputs']+manifest['original_inputs'])
    if action=='measure':require(read(RAW/'contracts/result.json')['passed'],'Contracts required')
    folder=RAW/action;require(not folder.exists(),'No implicit retry');folder.mkdir()
    expected=packet['power_guid'];result=dict(passed=False,action=action,power_guid=expected,source=record(Path(__file__)),
        binary=identity['exe'],performance_certified=False,samples=[]);job=None;start=time.monotonic()
    try:
        with gpu_lock(ROOT):
            before=gpu_query(0);require(before['memory.free']>=3072,'Probe requires 3GiB free')
            budget=min(.75*before['memory.free'],before['memory.free']-1536);require(budget>=1024,'Memory reserve insufficient')
            model=driver_model(0);foreign=gpu_processes(0,None,model);require(not foreign['blocking_foreign_pids'],'Foreign compute present')
            require(power_guid()==expected,'Power changed')
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
            env['CUDA_VISIBLE_DEVICES']='0';env['PATH']=os.pathsep.join([str(CUDA/'bin/x64'),str(CUDA/'bin'),env.get('PATH','')])
            cmd=[identity['exe']['path'],'contracts',str(folder)] if action=='contracts' else [identity['exe']['path'],'run',str(RAW/'input_manifest.json'),str(folder)]
            result.update(command=cmd,gpu_before=before,process_before=foreign,memory_budget_mib=budget,
                environment_overrides=dict(CUDA_VISIBLE_DEVICES='0',CUDA_BIN=str(CUDA/'bin/x64')),timeout_seconds=240)
            job=OwnedJob()
            with (folder/'process.log').open('xb') as log:
                job.launch(cmd,ROOT,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0);guid=power_guid()
                    processes=gpu_processes(0,job.pid,model,ownership=job.owns_pid)
                    result['samples'].append(dict(epoch=time.time(),power_guid=guid,gpu=sample,processes=processes))
                    require(guid==expected and sample['uuid']==before['uuid'],'Power/GPU changed')
                    require(not processes['blocking_foreign_pids'],'Foreign compute appeared')
                    require(sample['memory.free']>=768 and before['memory.free']-sample['memory.free']<=budget,'Memory budget')
                    require(time.monotonic()-start<=240,'Probe timeout')
                    time.sleep(.25)
            require(job.poll()==0,'Probe executable failed')
            output=read(folder/('contracts.json' if action=='contracts' else 'measurements.json'))
            require(output['passed'],'Probe did not pass')
            uuid=before['uuid'].removeprefix('GPU-').replace('-','').lower()
            require(output['device']['uuid_hex'].lower()==uuid,'Runtime device differs')
            require(len(output['warm'])==(80 if action=='contracts' else 40),'Wrong warm/contract count')
            if action=='measure':require(len(output['samples'])==720 and output['read_only_inputs_unchanged'],'Incomplete measurements')
            verify(manifest['generated_inputs']+manifest['original_inputs'])
            result['passed']=True
    except Exception as error:result['error']=type(error).__name__+': '+str(error)
    finally:
        if job is not None:
            if not job.finished():job.terminate()
            result.update(cleanup_owned_job_empty=job.finished(),exit_code=job.poll(),members=job.member_evidence(),lifecycle=job.lifecycle)
            job.close()
        result['wall_seconds']=time.monotonic()-start;write_new(folder/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('passed','action','exit_code','error','wall_seconds')}),flush=True)
    require(result['passed'],'Projection probe stage failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('seal','contracts','measure'));args=p.parse_args()
    seal() if args.action=='seal' else run(args.action)
