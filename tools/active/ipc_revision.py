"""IPC revision delivery helpers. Existing evidence is never overwritten."""
import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from config import ROOT, read, sha

TAG='ipc_revision_20261005'

def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

def freeze():
    target=ROOT/'manifests/ipc_reference_20261005.json'
    if target.exists():
        saved=read(target)
        for p,r in saved['binaries'].items(): assert sha(ROOT/p)==r['sha256']
        return saved
    m=read(ROOT/'builds/active/manifest.json'); saved=copy.deepcopy(m)
    folder=ROOT/'builds/ipc-reference/Release';folder.mkdir(parents=True,exist_ok=False)
    saved['binaries']={}
    for p,r in m['binaries'].items():
        src=ROOT/p; assert sha(src)==r['sha256']
        dst=folder/src.name;shutil.copy2(src,dst)
        saved['binaries'][dst.relative_to(ROOT).as_posix()]={'sha256':sha(dst),'bytes':dst.stat().st_size}
    saved['reference_scope']='Pre-revision IPC host/Graph, legacy MAS and legacy termination; AL is historical only'
    saved['original_manifest_sha256']=sha(ROOT/'builds/active/manifest.json')
    write_new(target,saved)
    print(json.dumps({'reference':str(target),'binaries':saved['binaries']}))
    return saved

def init_sources():
    freeze()
    for logical in ['core/GIPC.cu']:
        dst=ROOT/'sources/stiff_active/StiffGIPC'/logical
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'sources/stiff_perf_v50/StiffGIPC'/logical,dst)
    # One observer-only baseline translation unit; no frozen source is edited.
    dst=ROOT/'sources/stiff_base_observed/StiffGIPC/app/gl_main.cu'
    if not dst.exists():
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'sources/stiff_base/StiffGIPC/app/gl_main.cu',dst)

def observed_build():
    cmake=Path('D:/computer/cmake/bin/cmake.exe')
    folder=ROOT/'builds/base-observed';folder.mkdir(parents=True,exist_ok=True)
    commands=[[str(cmake),'-S',str(ROOT/'sources/stiff_base_observed'),'-B',str(folder),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
        '-DVCPKG_TARGET_TRIPLET=x64-windows','-DCMAKE_CUDA_ARCHITECTURES=86'],
        [str(cmake),'--build',str(folder),'--config','Release','--parallel','2']]
    for i,cmd in enumerate(commands):
        with (folder/f'build_{i}.log').open('w') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError((folder/f'build_{i}.log').read_text(errors='replace')[-9000:])
    m=copy.deepcopy(read(ROOT/'manifests/perf_v34.json'))
    src=ROOT/'sources/stiff_base_observed/StiffGIPC/app/gl_main.cu'
    m['files'].append({'path':src.relative_to(ROOT).as_posix(),'sha256':sha(src)})
    m['binaries']={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
                    for p in sorted((folder/'Release').glob('*')) if p.suffix in ('.exe','.dll')}
    m['observation_only']='Actual velocity export outside solver timer; inherited frozen base algorithms'
    write_new(ROOT/'manifests/base_observed_20261005.json',m)
    print(json.dumps({'observed_build':'completed','binaries':m['binaries']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','init','observed-build']);a=p.parse_args()
    {'freeze':freeze,'init':init_sources,'observed-build':observed_build}[a.action]()
