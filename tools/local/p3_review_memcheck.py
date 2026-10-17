"""One owned, resource-bounded Compute Sanitizer run of the full P3 fixture."""
from pathlib import Path
import os
import time
import run_innovation_p3 as p
from windows_owned_job import OwnedJob


def main():
    session=p.ROOT/'runs/innovation_p3_review_20261015';seal=p.read(session/'seal.json');identity=seal['identity']
    folder=session/'memcheck_monitor_fixed';p.require(not folder.exists(),'Fresh memcheck output required');folder.mkdir()
    sanitizer=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/compute-sanitizer/compute-sanitizer.exe')
    p.verify(identity['sources']+identity['objects']+identity['build_evidence']+[identity['exe']]+identity['dlls'])
    p.verify_active_sources(identity)
    env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
    env['GIPC_P3_EE_FIXTURE']=str(folder/'fixture.json');expected=seal['power_guid']
    command=[str(sanitizer),'--error-exitcode','99','--tool','memcheck','--leak-check','no',
        '--log-file',str(folder/'sanitizer.log'),identity['exe']['path']]
    result=dict(command=command,sanitizer=p.record(sanitizer),binary=identity['exe'],controller=p.record(Path(__file__)),samples=[])
    job=None;start=time.monotonic()
    with p.runner.gpu_lock(p.ROOT):
        try:
            before=p.runner.gpu_query(0);budget=min(.75*before['memory.free'],before['memory.free']-1536)
            p.require(budget>=1024 and p.m.power_guid()==expected,'Resource/power preflight failed')
            result.update(before=before,memory_budget_mib=budget)
            with (folder/'native.log').open('xb') as log:
                job=OwnedJob();job.launch(command,folder,env,log)
                while not job.finished():
                    sample=p.runner.gpu_query(0);result['samples'].append(sample)
                    p.require(time.monotonic()-start<60,'Memcheck timeout')
                    p.require(sample['memory.free']>=768 and before['memory.free']-sample['memory.free']<=budget,'Memcheck memory reserve exceeded')
                    p.require(p.m.power_guid()==expected,'Power changed during memcheck');time.sleep(.25)
                result.update(exit_code=job.poll(),owned_processes=job.observe())
            p.require(result['exit_code']==0,'Compute Sanitizer returned an error')
            p.require('ERROR SUMMARY: 0 errors' in (folder/'sanitizer.log').read_text(errors='replace'),'Missing clean memcheck summary')
            fixture=p.read(folder/'fixture.json');p.require(fixture['passed'],'Native fixture did not complete')
            result.update(passed=True,fixture_cases=len(fixture['cases']))
        except Exception as error:result.update(passed=False,error=str(error))
        finally:
            if job:
                if not job.finished():job.terminate()
                result['cleanup_owned_job_empty']=job.finished();job.close()
            result['wall_seconds']=time.monotonic()-start;result['power_after']=p.m.power_guid()
            p.write_new(folder/'result.json',result)
    p.require(result['passed'] and result['cleanup_owned_job_empty'] and result['power_after']==expected,'Review memcheck failed; retain evidence')
    print('Compute Sanitizer memcheck passed: '+str(result['fixture_cases'])+' fixture cases, zero errors')


if __name__=='__main__':main()
