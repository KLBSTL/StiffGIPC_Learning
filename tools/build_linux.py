"""Clean standalone Linux builds with immutable per-attempt command logs."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def checked(command,log,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:
        f.write('ARGV '+json.dumps(command)+'\n');f.flush()
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-50:]))
        raise SystemExit(result.returncode)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--compiler',default='/usr/local/cuda-12.8/bin/nvcc')
    ap.add_argument('--kind',choices=['active','base'],required=True)
    a=ap.parse_args();assert os.name=='posix' and 1<=a.jobs<=8
    compiler=Path(a.compiler).resolve();assert compiler.is_file()
    env=os.environ.copy();env['PATH']=str(compiler.parent)+':'+env['PATH']
    build=ROOT/'build'/a.kind
    assert not build.exists(),'Use a new build directory; never reuse objects from another identity'
    source=ROOT if a.kind=='active' else ROOT/'baseline'
    logs=ROOT/'build_logs'/a.kind
    checked(['cmake','-S',str(source),'-B',str(build),'-G','Ninja',
             '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89',
             '-DCMAKE_CUDA_COMPILER='+str(compiler)],logs/'configure.log',env)
    checked(['cmake','--build',str(build),'--parallel',str(a.jobs)],logs/'build.log',env)
    print(json.dumps({'build':'completed','kind':a.kind,'executable':str(build/'gipc')}))

if __name__=='__main__':main()
