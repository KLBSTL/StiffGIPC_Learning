"""One finite clean build, GPU matrix, offline analysis and export on AutoDL."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CONTROL=Path('/root/autodl-tmp/codex_recording105_20261013').resolve()
ROOT=CONTROL/'payload'
PYTHON='/root/miniconda3/bin/python'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save(state):
    path=CONTROL/'controller_state.json'
    temp=CONTROL/'controller_state.tmp'
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(path)


def main():
    assert ROOT.is_dir() and not (ROOT/'build').exists()
    state=dict(pid=os.getpid(),status='running',stage='configure',steps=[],root=str(ROOT),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        expected_trajectories=36,frames=105,repeats=3,table_frame=100,
        stages=dict(smoke=12,performance=36,checkpoint=63,cost=9))
    env=os.environ.copy()
    env['PATH']='/usr/local/cuda-12.8/bin:/root/miniconda3/bin:'+env['PATH']
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    build=ROOT/'build/observer'
    build.mkdir(parents=True)
    controller='tools/remote/run_recording105_autodl.py'
    commands=[
        ('configure',['cmake','-S','.', '-B','build/observer','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                      '-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc'],build/'configure.log'),
        ('build',['cmake','--build','build/observer','--target','gipc','--parallel','2'],build/'build.log'),
        ('seal',[PYTHON,'-B',controller,'seal'],CONTROL/'seal.log'),
        ('gpu_suite',['xvfb-run','-a',PYTHON,'-B','-u',controller,'run'],CONTROL/'gpu_suite.log'),
        ('analysis',[PYTHON,'-B',controller,'analyze'],CONTROL/'analysis.log'),
        ('render',['xvfb-run','-a',PYTHON,'-B',controller,'render'],CONTROL/'render.log'),
    ]
    try:
        for stage,command,log in commands:
            state.update(stage=stage,command=command)
            save(state)
            begin=time.monotonic()
            with log.open('xb') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state['child_pid']=process.pid
                save(state)
                code=process.wait()
            state['steps'].append(dict(stage=stage,exit_code=code,wall_seconds=time.monotonic()-begin,log=str(log)))
            save(state)
            if code:
                raise RuntimeError(f'{stage} failed with exit code {code}')
        state['stage']='archive'
        save(state)
        archive=CONTROL/'autodl_recording105_20261013_results.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=2) as tar:
            for name in ('StiffGIPC','MeshProcess','Assets','tests','tools','build','runs','reports'):
                tar.add(ROOT/name,arcname=name)
            for name in ('CMakeLists.txt','FLATTEN_SOURCE_MAP.json','EXPERIMENT_PLAN.json',
                         'TRANSFER_MANIFEST.json','LOCAL_SOURCE_COMPARISON.json','ANALYSIS_ADAPTATION.json'):
                tar.add(ROOT/name,arcname=name)
        receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
        (CONTROL/(archive.name+'.json')).write_text(json.dumps(receipt,indent=2))
        state.update(status='completed',stage='completed',archive=receipt,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save(state)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}',
                     failed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save(state)
        raise


if __name__=='__main__':
    main()
