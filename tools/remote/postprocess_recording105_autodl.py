"""Render measured geometry via software EGL, then collect finite review/full archives."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
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


def save(value):
    target=CONTROL/'postprocess_state.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(target)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def render():
    wrapper=load('render_adapter',ROOT/'tools/remote/run_recording105_autodl_v2.py')
    base=wrapper.base
    base.install_compatibility()
    import moderngl
    original=moderngl.create_standalone_context
    moderngl.create_standalone_context=lambda **kw:original(backend='egl',**kw)
    output=base.OUTPUT
    folder=output/'visuals'
    if folder.exists():
        assert not any(folder.iterdir()),'Preserve any existing visual output'
        folder.rmdir()
    report=base.load_module('egl_report',ROOT/'tools/remote/report_recording105_autodl.py')
    report.visuals(base.SESSION,output)
    path=folder/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest.update(context_backend='EGL; software Mesa llvmpipe; system libstdc++ preload for conda compatibility',
        adapter_sha256=sha(Path(__file__)),numerical_simulation_rerun=False)
    path.write_text(json.dumps(manifest,indent=2))


def collect(review):
    name='autodl_recording105_20261013_review.tar.gz' if review else 'autodl_recording105_20261013_full.tar.gz'
    target=CONTROL/name
    assert not target.exists()
    count=0
    with tarfile.open(target,'w:gz',compresslevel=2) as tar:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            rel=path.relative_to(ROOT)
            if review and 'trace' in rel.parts and path.name.startswith(('state_','velocity_')) and path.suffix=='.bin':
                frame=int(path.stem.rsplit('_',1)[1])
                if frame not in (0,25,50,75,100,105):
                    continue
            tar.add(path,arcname=rel.as_posix(),recursive=False)
            count+=1
    receipt=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=count,
                 review_package=review,omitted= 'Non-keyframe raw positions/velocities only; all numeric logs retained' if review else 'None')
    (CONTROL/(name+'.json')).write_text(json.dumps(receipt,indent=2))
    return receipt


def main():
    status=dict(pid=os.getpid(),status='running',stage='render')
    save(status)
    env=os.environ.copy()
    env.update(LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',LIBGL_ALWAYS_SOFTWARE='1',
               PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    try:
        with (CONTROL/'render_egl.log').open('xb') as f:
            subprocess.run([PYTHON,'-B',str(Path(__file__)),'render'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        status['stage']='verify'
        save(status)
        with (CONTROL/'verify_cloud.log').open('xb') as f:
            subprocess.run([PYTHON,'-B',str(ROOT/'tools/remote/verify_recording105_autodl.py'),'--root',str(ROOT)],
                           cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        status['stage']='review_archive'
        save(status)
        status['review']=collect(True)
        status['stage']='full_archive'
        save(status)
        status['full']=collect(False)
        status.update(status='completed',stage='completed')
        save(status)
    except BaseException as e:
        status.update(status='failed',error=f'{type(e).__name__}: {e}')
        save(status)
        raise


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='render':
        render()
    else:
        main()
