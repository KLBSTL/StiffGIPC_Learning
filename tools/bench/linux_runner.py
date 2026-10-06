"""Portable finite Linux runner for the flattened repository. Never builds or cleans."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from config import ROOT, read, digest, expand, environment
from plans import STAGES, tasks, protocol
from validate_run import validate

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def child(root, relative):
    root=Path(root).resolve(); rel=Path(relative)
    require(not rel.is_absolute() and rel.parts and all(p not in ('..','.') for p in rel.parts), 'Unsafe relative path')
    path=root/rel
    for p in (path,*path.parents):
        if p==root: break
        require(not p.is_symlink(), 'Symlink path rejected: '+str(p))
    require(path.resolve().is_relative_to(root), 'Path outside root')
    return path

def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)

def inventory(root, paths):
    rows=[]
    for path in sorted(set(paths)):
        require(path.is_file() and not path.is_symlink(), 'Missing or linked identity file: '+str(path))
        before=path.stat(); value=sha(path); after=path.stat()
        require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino), 'File changed while hashing')
        rows.append({'path':path.relative_to(root).as_posix(),'bytes':after.st_size,'sha256':value})
    return rows

def source_inventory(root):
    paths=[]
    for name in ('StiffGIPC','MeshProcess','Assets','tools/bench','tests'):
        base=child(root,name); require(base.is_dir(), 'Missing source root '+name)
        for p in base.rglob('*'):
            if '__pycache__' in p.parts or p.suffix=='.pyc': continue
            require(not p.is_symlink(), 'Linked source entry rejected')
            if p.is_file(): paths.append(p)
    paths += [child(root,'CMakeLists.txt'),child(root,'FLATTEN_SOURCE_MAP.json')]
    return inventory(root,paths)

def verify_files(root, rows):
    for row in rows:
        p=child(root,row['path'])
        require(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], 'Identity changed: '+row['path'])

def seal_build(root, build_dir, exe, build_log, output):
    build_dir=child(root,build_dir); exe=child(root,exe); log=child(root,build_log)
    require(exe.is_relative_to(build_dir) and exe.is_file(), 'Executable outside selected build directory')
    commands=read(build_dir/'compile_commands.json')
    native={p.resolve() for p in (root/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
    actual=set()
    for row in commands:
        p=Path(row['file']); p=p if p.is_absolute() else Path(row['directory'])/p
        text=row.get('command',' '.join(row.get('arguments',[])))
        if 'gipc.dir' in text:
            require(p.resolve() in native, 'gipc compiles outside flattened native tree: '+str(p))
            actual.add(p.resolve())
    require(native==actual and native, 'Compile command/native source coverage mismatch')
    evidence=[build_dir/'compile_commands.json',build_dir/'CMakeCache.txt',log]
    if (build_dir/'CMakeFiles/gipc.dir/link.txt').is_file(): evidence.append(build_dir/'CMakeFiles/gipc.dir/link.txt')
    sources=source_inventory(root)
    manifest={'schema':'flat_build_identity.v1','source_digest':digest(sources),'sources':sources,
              'exe_path':exe.relative_to(root).as_posix(),'exe_sha256':sha(exe),
              'build_evidence':inventory(root,evidence),'compile_source_count':len(actual),
              'plan':protocol(),'plan_sha256':digest(protocol()),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'scope':'Post-build source/executable/commands identity; producer must perform clean build. This tool does not compile or certify compiler correctness.'}
    write_new(child(root,output),manifest)
    return manifest

def verify_manifest(root,path):
    m=read(path)
    require(m['schema']=='flat_build_identity.v1' and m['plan']==protocol() and m['plan_sha256']==digest(protocol()), 'Build/plan contract changed')
    require(digest(source_inventory(root))==m['source_digest'] and digest(m['sources'])==m['source_digest'], 'Source or runner changed since build seal')
    verify_files(root,m['sources']);verify_files(root,m['build_evidence'])
    require(sha(child(root,m['exe_path']))==m['exe_sha256'],'Executable changed since seal')
    return m

def gpu_query(gpu, timeout=10):
    fields=['uuid','name','driver_version','utilization.gpu','memory.free','temperature.gpu','power.draw','clocks.sm','compute_cap']
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=timeout)
    rows=list(csv.reader(p.stdout.splitlines(),skipinitialspace=True));require(len(rows)==1,'One physical GPU required')
    result=dict(zip(fields,rows[0]));require(len(result)==len(fields),'Incomplete nvidia-smi fields')
    for key in ('utilization.gpu','memory.free','temperature.gpu'):result[key]=float(result[key])
    return result

def compute_pids(gpu, timeout=10):
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=timeout)
    return [int(x) for x in p.stdout.splitlines() if x.strip()]

@contextmanager
def gpu_lock(root):
    import fcntl
    lock=child(root,'runs/.gpu.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally: os.close(fd)

def stop_owned(process, immediate=False):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGKILL if immediate else signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)

def execute(root,session,t,manifest,gpu):
    c=expand(t['config']);out=child(session,t['name']);require(not out.exists(),'Run output already exists')
    out.mkdir();(out/'output').mkdir();process=None
    heavy=bool(c['diagnostics']) or any(c[k] for k in ('contact_pool_validate','discrete_bvh_validate','bounded_ccd_validate','bvh_eligibility_validate'))
    result={'status':'launcher_failed','recorded_frames':0,'physical_quality_certified':False,
            'performance_certified':False,'heavy_diagnostics':heavy,
            'timing_scope':('Guard validation adds private reference queries/copies and energy audit; not a performance arm.' if heavy else
                            'Whole solver CUDA event; no Nsight/cost probes. State and actual velocity export remains enabled.')}
    try:
        before=[gpu_query(gpu)];time.sleep(.25);before.append(gpu_query(gpu))
        require(not compute_pids(gpu) and all(x['utilization.gpu']<=5 for x in before),'GPU load gate failed; no foreign process stopped')
        free=before[-1]['memory.free'];budget=min(.75*free,free-1536)
        require(budget>=1024 and shutil.disk_usage(session).free>=4*1024**3,'Insufficient GPU memory/disk reserve')
        verify_files(root,manifest['sources'])
        exe=child(root,manifest['exe_path']);require(sha(exe)==manifest['exe_sha256'],'Executable changed')
        env=environment(c,out);env['CUDA_VISIBLE_DEVICES']=before[-1]['uuid']
        req={'requested_config':t['config'],'expanded_config':c,'config_sha256':digest(c),'binary':'active',
             'source_digest':manifest['source_digest'],'exe_sha256':manifest['exe_sha256'],
             'runner_sha256':sha(__file__),'command':[str(exe)],'gpu_index':gpu,'gpu_before':before,
             'gpu_memory_budget_mib':budget,'environment':{k:v for k,v in env.items() if k.startswith('GIPC_')},'from_zero':True}
        write_new(out/'requested.json',req);write_new(out/'build_manifest.json',manifest)
        start=time.monotonic();deadline=start+c['timeout_seconds'];status='running';samples=[]
        with (out/'run.log').open('xb') as log:
            process=subprocess.Popen([str(exe)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_new(out/'process.json',{'pid':process.pid,'exe':str(exe)})
            while process.poll() is None:
                try:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:status='timeout';stop_owned(process,True);break
                    sample=gpu_query(gpu,timeout=min(10,remaining))
                    remaining=deadline-time.monotonic()
                    if remaining<=0:status='timeout';stop_owned(process,True);break
                    foreign=[p for p in compute_pids(gpu,timeout=min(10,remaining)) if p!=process.pid]
                except subprocess.TimeoutExpired:
                    status='timeout' if time.monotonic()>=deadline else 'monitor_timeout'
                    stop_owned(process,True);break
                samples.append(sample|{'foreign_compute_pids':foreign})
                status=('timeout' if time.monotonic()-start>=c['timeout_seconds'] else
                        'disk_reserve' if shutil.disk_usage(session).free<1024**3 else
                        'memory_budget' if sample['memory.free']<768 or free-sample['memory.free']>budget else
                        'foreign_gpu_load' if foreign else 'running')
                if status!='running':stop_owned(process,status=='timeout');break
                time.sleep(max(0,min(.5,deadline-time.monotonic())))
            process.wait(timeout=10)
        result.update(status=('completed' if process.returncode==0 else 'failed') if status=='running' else status,
                      exit_code=process.returncode,wall_seconds=time.monotonic()-start,gpu_samples=samples)
        frames=out/'trace/frames.csv'
        if frames.exists():
            with frames.open() as stream: rows=list(csv.DictReader(stream))
            result.update(recorded_frames=len(rows),solver_seconds=sum(float(r['solver_ms']) for r in rows)/1000)
        final=out/'final.bin'
        if final.is_file():
            data=final.read_bytes();result['finite']=bool(data) and len(data)%8==0 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<d',data))
        if result['status']=='completed' and (result['recorded_frames']!=c['steps'] or not result.get('finite')):result['status']='incomplete_or_nonfinite'
        require(sha(exe)==manifest['exe_sha256'],'Executable changed during run')
        verify_files(root,manifest['sources'])
    except Exception as e:
        result.update(status='launcher_or_monitor_failed',error=type(e).__name__+': '+str(e))
    finally:
        stop_owned(process);write_new(out/'result.json',result)
    if result['status']=='completed':
        try: check=validate(out)
        except Exception as e: check={'passed':False,'error':str(e)}
        write_new(out/'config_validation.json',check)
        if not check['passed']:
            result['status']='configuration_failed'
            (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_new(out/'evidence.json',{'files':inventory(out,[p for p in out.rglob('*') if p.is_file()])})
    print(json.dumps({'run':t['name'],'status':result['status'],'frames':result['recorded_frames'],'solver_seconds':result.get('solver_seconds')}),flush=True)
    return result

def verify_stage(session,stage,manifest_sha):
    ledger_path=child(session,stage+'_batch.json');analysis_path=child(session,stage+'_analysis.json')
    ledger=read(ledger_path);receipt=read(child(session,stage+'_receipt.json'))
    require(ledger['tasks']==tasks(stage) and ledger['manifest_sha256']==manifest_sha,'Prior ledger contract differs')
    require(receipt['batch_sha256']==sha(ledger_path) and receipt['analysis_sha256']==sha(analysis_path),'Prior analysis/ledger changed')
    require(receipt['manifest_sha256']==manifest_sha and receipt['plan_sha256']==digest(protocol()),'Prior receipt identity differs')
    require(len(ledger['runs'])==len(tasks(stage)),'Prior stage incomplete')
    for row,t in zip(ledger['runs'],tasks(stage)):
        require(row['name']==t['name'] and row['result']['status']=='completed','Prior run failed/reordered')
        out=child(session,t['name']);ev=out/'evidence.json'
        require(sha(ev)==row['evidence_sha256'],'Prior evidence changed');verify_files(out,read(ev)['files'])
        require(read(out/'result.json')==row['result'],'Prior result differs')
    a=read(analysis_path)
    require(a['stage']==stage and a['allow_next_round'] is True,'Prior analyzed gate did not pass')
    return receipt

def run_stage(root,manifest_path,session_name,stage,gpu):
    require(sys.platform.startswith('linux'),'GPU runner requires Linux')
    require(os.environ.get('DISPLAY'),'GLUT needs DISPLAY; launch through /usr/bin/xvfb-run -a')
    require(re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',session_name),'Invalid session name')
    from pool_analysis import analyze,aggregate  # Preflight CPU dependencies before any GPU work.
    manifest=verify_manifest(root,manifest_path);manifest_sha=sha(manifest_path)
    session=child(root,'runs/'+session_name)
    if stage=='guards':require(not session.exists(),'New guard session must not exist');session.mkdir(parents=True)
    else:require(session.is_dir(),'Missing prior session')
    for previous in STAGES[:STAGES.index(stage)]:verify_stage(session,previous,manifest_sha)
    require(not any((session/(stage+s)).exists() for s in ('_batch.json','_analysis.json','_receipt.json','_started.json')),'Stage already attempted; never rerun a failed arm')
    require(all(not (session/t['name']).exists() for t in tasks(stage)),'A planned run already exists')
    write_new(session/(stage+'_started.json'),{'stage':stage,'manifest_sha256':manifest_sha,'plan_sha256':digest(protocol()),'pid':os.getpid()})
    rows=[]
    with gpu_lock(root):
        for t in tasks(stage):
            result=execute(root,session,t,manifest,gpu)
            rows.append({'name':t['name'],'result':result,'evidence_sha256':sha(session/t['name']/'evidence.json')})
            if result['status']!='completed':break
        # CPU analysis completes while the stage holds the GPU lease; no later stage is spawned.
        batch={'stage':stage,'tasks':tasks(stage),'runs':rows,'manifest_sha256':manifest_sha,'plan_sha256':digest(protocol()),'skipped':[t['name'] for t in tasks(stage)[len(rows):]]}
        write_new(session/(stage+'_batch.json'),batch)
        try:
            verify_manifest(root,manifest_path)
            report=analyze(session,stage,manifest)
            verify_manifest(root,manifest_path)
        except Exception as e:report={'stage':stage,'allow_next_round':False,'analysis_error':type(e).__name__+': '+str(e),'performance_certified':False,'quality_certified':False}
        report['allow_next_round'] &= len(rows)==len(tasks(stage)) and all(r['result']['status']=='completed' for r in rows)
        write_new(session/(stage+'_analysis.json'),report)
        write_new(session/(stage+'_receipt.json'),{'stage':stage,'batch_sha256':sha(session/(stage+'_batch.json')),
             'analysis_sha256':sha(session/(stage+'_analysis.json')),'manifest_sha256':manifest_sha,'plan_sha256':digest(protocol())})
        if stage=='r3' and report['allow_next_round']:write_new(session/'screen_summary.json',aggregate(session))
    print(json.dumps({'stage':stage,'allow_next_round':report['allow_next_round'],'analysis':str(session/(stage+'_analysis.json'))}),flush=True)
    return 0 if report['allow_next_round'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    seal=sub.add_parser('seal-build');seal.add_argument('--build-dir',required=True);seal.add_argument('--exe',required=True)
    seal.add_argument('--build-log',required=True);seal.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--manifest',required=True);r.add_argument('--session',required=True)
    r.add_argument('--stage',choices=STAGES,required=True);r.add_argument('--gpu',type=int,default=0)
    sub.add_parser('plan');a=p.parse_args();root=a.root.resolve()
    if a.command=='plan':print(json.dumps(protocol(),indent=2));return 0
    if a.command=='seal-build':seal_build(root,a.build_dir,a.exe,a.build_log,a.output);return 0
    require(a.gpu>=0,'GPU index must be nonnegative')
    return run_stage(root,child(root,a.manifest),a.session,a.stage,a.gpu)

if __name__=='__main__':raise SystemExit(main())
