"""Profile the existing MAS precision fixture; no simulation or native edits."""
from pathlib import Path
import argparse
import os
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for folder in ('bench','local'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from config import read
from linux_runner import require,write_new,gpu_query
from local_identity import record,verify,verify_active_sources
from windows_runner import gpu_lock,driver_model,gpu_processes,remaining,final_status
from windows_owned_job import OwnedJob


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--session',required=True)
    parser.add_argument('--native-only',action='store_true')
    args=parser.parse_args();require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session;require(not out.exists(),'Fresh output required');out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    identity=read(args.identity);verify_active_sources(identity)
    evidence=identity['sources']+identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls']
    verify(evidence)
    prefix=ROOT/'runs/deep_toi_native_allowed_20261009/linear113/events/e1'
    meta=read(Path(str(prefix)+'_meta.json'));owner=next(p for p in meta['local_preconditioners'] if p['kind']=='MAS_full_owned_buffers')
    input_folder=out/'input';input_folder.mkdir();private=input_folder/'system'
    source_records=[record(Path(str(prefix)+'_meta.json'))]
    shutil.copyfile(source_records[0]['path'],str(private)+'_meta.json')
    names=('d_goingNext','d_prefixOriginal','d_fineConnectMask','d_partId_map_real','d_real_map_partId',
        'd_coarseTable','d_precondMatMas','d_inverseMatMas')
    for name in names:
        source=Path(str(prefix)+'_mas_'+name+'.bin');source_records.append(record(source))
        shutil.copyfile(source,str(private)+'_mas_'+name+'.bin')
    source_rhs=Path(str(prefix)+'_rhs.bin');source_records.append(record(source_rhs))
    rhs=np.frombuffer(source_rhs.read_bytes(),dtype='<f8').copy();require(len(rhs)==meta['dofs'],'RHS ABI mismatch')
    rng=np.random.default_rng(20261009);probes=[rhs,np.zeros_like(rhs),np.ones_like(rhs)]
    probes.extend(rng.standard_normal(len(rhs))*s for s in (1,1e-8,1e8))
    probes.extend([np.sin(np.arange(len(rhs))),np.where(np.arange(len(rhs))%2,1.,-1.),
        rng.standard_normal(len(rhs))*np.power(10.,rng.uniform(-12,12,len(rhs))),rhs*1e-8])
    for index,values in enumerate(probes):values.astype('<f8').tofile(str(private)+f'_v{index}.bin')
    profiler=Path(r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe')
    require(profiler.is_file(),'Profiler unavailable')
    command=[str(profiler),'profile','--trace=cuda','--sample=none','--cpuctxsw=none',
        '--cuda-graph-trace=node','--export=none','--force-overwrite=false','--output='+str(out/'trace'),identity['exe']['path']]
    if args.native_only:command=[identity['exe']['path']]
    env={k:v for k,v in os.environ.items() if not k.upper().startswith('GIPC_')}
    env.update(GIPC_MAS_REPLAY_FIXTURE=str(private),GIPC_MAS_FIXTURE_OUTPUT=str(out/'fixture'))
    (out/'tmp').mkdir();env['TMP']=env['TEMP']=str(out/'tmp')
    timeout=120;job=None;samples=[]
    result=dict(status='launcher_failed',controller=record(__file__),native_identity=record(args.identity),
        profiler=record(profiler),command=command,input_source_records=source_records,
        scope='Saved TOI e1 factors/maps, real RHS plus nine labeled synthetic probes. Existing native MAS fixture; optional CUDA kernel tracing. Fixed-operator diagnosis only; no whole-scene performance/physics certificate.',
        native_performance_certified=False,physical_quality_certified=False)
    try:
        with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
            mode=driver_model(0);before=gpu_query(0);processes=gpu_processes(0,None,mode)
            require(not processes['blocking_foreign_pids'],'Foreign compute gate failed')
            free=before['memory.free'];budget=min(.75*free,free-1536)
            require(budget>=1024 and shutil.disk_usage(out).free>=4*1024**3,'Resource reserve insufficient')
            env['CUDA_VISIBLE_DEVICES']=before['uuid']
            result.update(gpu_before=before,memory_budget_mib=budget,process_before=processes)
            start=time.monotonic();deadline=start+timeout;status='running'
            with (out/'run.log').open('xb') as log:
                job=OwnedJob();job.launch(command,out,env,log)
                while not job.finished():
                    job.observe();sample=gpu_query(0,timeout=remaining(deadline,timeout))
                    processes=gpu_processes(0,None,mode,timeout=remaining(deadline,timeout),ownership=job.owns_pid)
                    require(sample['uuid']==before['uuid'],'GPU identity changed')
                    samples.append(sample|{'processes':processes})
                    status=('foreign_gpu_load' if processes['blocking_foreign_pids'] else
                        'disk_reserve' if shutil.disk_usage(out).free<1024**3 else
                        'memory_budget' if sample['memory.free']<768 or free-sample['memory.free']>budget else 'running')
                    if status!='running':job.terminate();break
                    if job.poll() not in (None,0):status='failed';job.terminate();break
                    time.sleep(max(0,min(.5,deadline-time.monotonic())))
                elapsed=time.monotonic()-start
                result.update(status=final_status(status,job.poll(),elapsed,timeout),exit_code=job.poll(),wall_seconds=elapsed)
            verify(evidence+source_records)
            if result['status']=='completed':
                fixture=read(out/'fixture/fixture.json')
                require(fixture['full64_graph_passed'] and fixture['off_signature_restored']
                    and fixture['growth_changes_graph_signature'],'Native precision fixture contract failed')
                result.update(passed=True,fixture=record(out/'fixture/fixture.json'))
                if not args.native_only:
                    require((out/'trace.nsys-rep').is_file(),'CUDA trace report missing')
                    result['trace']=record(out/'trace.nsys-rep')
    except Exception as error:
        result.update(status='launcher_or_validation_failed',error=type(error).__name__+': '+str(error),passed=False)
    finally:
        if job is not None:
            try:
                if not job.finished():job.terminate()
                result.update(owned_processes=job.observe(),cleanup_owned_job_empty=job.finished())
            finally:job.close()
        result['gpu_samples']=samples;write_new(out/'result.json',result)
    print({k:v for k,v in result.items() if k in ('status','passed','wall_seconds','exit_code','error')},flush=True)
    return 0 if result.get('passed') else 2


if __name__=='__main__':raise SystemExit(main())
