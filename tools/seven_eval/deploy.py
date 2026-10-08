"""CPU-only cache preparation and two clean builds for the 2026-10-08 suite."""
import json
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('/root/stiffgipc_full_20261006_5e2f9bb')
sys.path[:0] = [str(ROOT/'tools/bench'),str(ROOT/'tools/full_eval')]
from linux_runner import sha, inventory, write_new, require
import build_identity

def logged(argv, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(argv)+'\n'); stream.flush()
        result = subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0, 'Command failed; see '+str(path))

def configure_time_bound(raw,before_ns):
    match=re.fullmatch(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{9}) ([+-]\d{4})',raw.strip())
    require(match is not None,'Filesystem configure-log birth time unavailable')
    stamp=dt.datetime.strptime(match[1]+'.'+match[2][:6]+' '+match[3],'%Y-%m-%d %H:%M:%S.%f %z')
    # One microsecond margin keeps the floating-point epoch conservative; this
    # is a real filesystem event, never a fabricated supervisor launch time.
    bound=stamp.timestamp()-1e-6
    require(int(bound*1e9)>before_ns,'Configure log birth does not strictly follow the original source snapshot')
    return {'time_unix':bound,'actual_supervisor_start_available':False,
            'timestamp_source':'Filesystem birth time of first configure log; conservative lower bound',
            'raw_filesystem_birth_time':raw.strip(),'precision_margin_seconds':1e-6}

def recover_identity():
    report=ROOT/'reports/autodl_seven_20261008'
    require(not (report/'BUILD_IDENTITY.json').exists(),'Identity already exists')
    require(not (ROOT/'build_start.json').exists(),'Existing start receipt must not be replaced')
    require(not (ROOT/'runs/autodl_seven_20261008').exists(),'Recovery only before any simulation')
    before=ROOT/'build_pre_sources.json';log=ROOT/'build_logs/active/configure.log'
    raw=subprocess.run(['stat','-c','%w',str(log)],capture_output=True,text=True,check=True).stdout
    marker=configure_time_bound(raw,before.stat().st_mtime_ns)
    marker.update(recovery_created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_snapshot_sha256=sha(before),configure_log_sha256=sha(log),
                  reason='Original build launcher omitted its supervisor receipt. Both clean builds already completed; no compile or GPU retry.',
                  metadata_key_note='time_unix supplies the legacy auditor with a conservative observed configuration-file birth bound, not actual supervisor start.')
    write_new(ROOT/'build_start.json',marker)
    packet=build_identity.verify(ROOT,'build_pre_sources.json')
    packet['temporal_freshness'].update(actual_supervisor_start_available=False,
        timestamp_source=marker['timestamp_source'],recovery_adapter_sha256=sha(__file__),
        scope='Original fresh-directory source snapshot precedes observed configure-log filesystem birth and all emitted object/log mtimes. Actual supervisor launch UTC unavailable; compile UTC remains a conservative bound, not an exact timestamp.')
    write_new(report/'BUILD_IDENTITY.json',packet)
    print(json.dumps({'stage':'identity_recovered','counts':{k:v['counts'] for k,v in packet['builds'].items()},
                      'actual_supervisor_start_available':False,'gpu_run':False}),flush=True)

def main():
    require(os.name=='posix','Linux only')
    if sys.argv[1:]==['--recover-identity']:
        recover_identity();return
    require(not (ROOT/'build').exists(),'New root and absent build directories required')
    report = ROOT/'reports/autodl_seven_20261008'
    report.mkdir(parents=True,exist_ok=True)
    added=[]
    for source in sorted((OLD/'Assets/sorted_mesh').glob('*')):
        if not source.is_file(): continue
        for prefix in ('Assets','baseline/Assets'):
            target=ROOT/prefix/'sorted_mesh'/source.name
            if target.exists():
                require(sha(target)==sha(source),'Existing cache differs: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                added.append(target)
    # The additional 23k cloth cache is prepared without loading CUDA or running
    # a simulation. Old METIS/GKlib libraries are explicitly hashed below.
    stem='square_cloth_23k';cache=ROOT/'Assets/sorted_mesh'
    needed=[cache/(stem+'_sorted.16'+suffix) for suffix in ('.obj','.part')]
    require(not any(p.exists() for p in needed),'Unexpected partial/new 23k cache')
    prep=ROOT/'runs/cache_preparation';prep.mkdir(parents=True)
    entry=prep/'main.cpp'
    entry.write_text('#include <vector>\n#include "metis_sort.h"\nint main(int argc,char** argv){ if(argc!=2)return 2; return metis_sort(argv[1],2).size()==2?0:3; }\n')
    src=ROOT/'MeshProcess/metis_partition/src'
    libs=[OLD/'build/active/MeshProcess/External/METIS/libmetis/libmetis.a',
          OLD/'build/active/MeshProcess/External/GKlib/libGKlib.a']
    sources=[src/n for n in ('mesh.cpp','metis_sort.cpp','node_edge_model.cpp')]
    compiler=shutil.which('g++');require(compiler is not None,'g++ unavailable')
    # A compiler definition is one argv token, including the C++ string quotes.
    argv=[compiler,'-O2','-std=c++17','-I'+str(src),'-I/usr/include/eigen3',
          '-I'+str(ROOT/'MeshProcess/External/METIS/include'),
          '-DOUTPUT_DIR="'+str(cache)+'/"']
    argv+=['-DASSETS_DIR="'+str(ROOT/'Assets')+'/"',str(entry),*map(str,sources),*map(str,libs),'-lm','-o',str(prep/'prepare')]
    logged(argv,prep/'compile.log')
    logged([str(prep/'prepare'),str(ROOT/'Assets/benchmark_meshes/triMesh'/ (stem+'.obj'))],prep/'prepare.log')
    require(all(p.is_file() for p in needed),'Cache preparation did not produce both files')
    for source in needed:
        target=ROOT/'baseline/Assets/sorted_mesh'/source.name
        require(not target.exists(),'Baseline 23k cache unexpectedly exists')
        shutil.copyfile(source,target);added += [source,target]
    write_new(report/'CACHE_PREPARATION.json',{
        'gpu_run':False,'added_files':inventory(ROOT,added),'command':argv,
        'library_inputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in libs],
        'prepare_exe_sha256':sha(prep/'prepare'),'compiler_sha256':sha(compiler)})
    write_new(ROOT/'build_pre_sources.json',build_identity.snapshot(ROOT))
    write_new(ROOT/'build_start.json',{'time_unix':time.time(),'pid':os.getpid(),
                                     'actual_supervisor_start_available':True})
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    jobs=4 if available>=12*1024**2 else 2
    for kind in ('active','base'):
        print(json.dumps({'stage':'build_started','kind':kind,'jobs':jobs}),flush=True)
        logged([sys.executable,'tools/build_linux.py','--kind',kind,'--jobs',str(jobs)],report/(kind+'_launcher.log'))
        print(json.dumps({'stage':'build_completed','kind':kind}),flush=True)
    packet=build_identity.verify(ROOT,'build_pre_sources.json')
    write_new(report/'BUILD_IDENTITY.json',packet)
    print(json.dumps({'stage':'identity_verified','counts':{k:v['counts'] for k,v in packet['builds'].items()},'gpu_run':False}),flush=True)

if __name__=='__main__': main()
