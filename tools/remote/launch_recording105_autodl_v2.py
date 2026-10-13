"""Resume only the GPU/data stages using the already clean-built native binary."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

CONTROL=Path('/root/autodl-tmp/codex_recording105_20261013')
ROOT=CONTROL/'payload'
PYTHON='/root/miniconda3/bin/python'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save(state):
    path=CONTROL/'controller_state_v2.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(path)


def main():
    assert not (ROOT/'runs/autodl_recording105_20261013_v2').exists()
    identity=json.loads((ROOT/'build/program_identity.json').read_text())
    original=json.loads((CONTROL/'controller_state.json').read_text())
    assert original['status']=='failed' and original['stage']=='gpu_suite'
    assert all(s['exit_code']==0 for s in original['steps'] if s['stage'] in ('configure','build','seal'))
    assert sha(ROOT/'build/observer/gipc')==identity['exe']['sha256']
    state=dict(pid=os.getpid(),status='running',stage='gpu_suite',steps=[],
        build_from='Original clean build; binary hash unchanged',native_exe_sha256=identity['exe']['sha256'],
        first_attempt='Zero frames; required Linux extra_environment metadata absent',
        controller_overlay_sha256=sha(ROOT/'tools/remote/run_recording105_autodl_v2.py'))
    env=os.environ.copy()
    env['PATH']='/usr/local/cuda-12.8/bin:/root/miniconda3/bin:'+env['PATH']
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    script='tools/remote/run_recording105_autodl_v2.py'
    commands=[('gpu_suite',['xvfb-run','-a',PYTHON,'-B','-u',script,'run']),
              ('analysis',[PYTHON,'-B',script,'analyze']),
              ('render',['xvfb-run','-a',PYTHON,'-B',script,'render'])]
    try:
        for stage,command in commands:
            state.update(stage=stage,command=command)
            save(state)
            begin=time.monotonic()
            log=CONTROL/(stage+'_v2.log')
            with log.open('xb') as f:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                state['child_pid']=process.pid
                save(state)
                code=process.wait()
            state['steps'].append(dict(stage=stage,exit_code=code,wall_seconds=time.monotonic()-begin,log=str(log)))
            save(state)
            if code:
                raise RuntimeError(f'{stage} failed with exit code {code}')
        state.update(stage='archive')
        save(state)
        archive=CONTROL/'autodl_recording105_20261013_v2_results.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=2) as tar:
            for name in ('StiffGIPC','MeshProcess','Assets','tests','tools','build','runs','reports'):
                tar.add(ROOT/name,arcname=name)
            for name in ('CMakeLists.txt','FLATTEN_SOURCE_MAP.json','EXPERIMENT_PLAN.json',
                         'TRANSFER_MANIFEST.json','LOCAL_SOURCE_COMPARISON.json','ANALYSIS_ADAPTATION.json'):
                tar.add(ROOT/name,arcname=name)
        receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
        (CONTROL/(archive.name+'.json')).write_text(json.dumps(receipt,indent=2))
        state.update(status='completed',stage='completed',archive=receipt)
        save(state)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}')
        save(state)
        raise


if __name__=='__main__':
    main()
