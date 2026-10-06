"""Independent fixed-cloth diagnosis. Frozen bench code and guard evidence are read only."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
from quality_plan import ROOT, tasks, plan
from config import read, expand, digest
from linux_runner import (require, child, sha, inventory, verify_files, write_new,
                          verify_manifest, gpu_query, compute_pids, gpu_lock, stop_owned,
                          execute as execute_active)
from quality_analysis import validate_base, analyze_run, analyze

def baseline_environment(c,out):
    env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
    env.update(GIPC_SCENE=c['scene'],GIPC_STEPS=str(c['steps']),GIPC_DT=str(c['dt']),
               GIPC_NEWTON_TOL=str(c['ipc_newton_tol']),GIPC_PCG_TOL=str(c['pcg_rho_tol']),
               GIPC_TRACE_DIR=str(out/'trace'),GIPC_TRACE_STRIDE='1',GIPC_TRACE_PHYSICS='0',
               GIPC_DEFER_STATS='1',GIPC_DUMP_STATE=str(out/'final.bin'),GIPC_OUTPUT_PATH=str(out/'output'))
    return env

def execute_base(root,session,t,manifest,gpu):
    c=expand(t['config']);out=child(session,t['name']);require(not out.exists(),'Run output already exists')
    out.mkdir();(out/'output').mkdir();process=None
    heavy=bool(c['diagnostics']) or any(c[k] for k in ('contact_pool_validate','discrete_bvh_validate','bounded_ccd_validate','bvh_eligibility_validate'))
    result={'status':'launcher_failed','recorded_frames':0,'physical_quality_certified':False,
            'performance_certified':False,'heavy_diagnostics':heavy,
            'timing_scope':('Guard validation adds private reference queries/copies and energy audit; not a performance arm.' if heavy else
                            'Diagnostic runtime observation; baseline exports positions only, with no resolved-config or actual velocity capability.')}
    try:
        before=[gpu_query(gpu)];time.sleep(.25);before.append(gpu_query(gpu))
        require(not compute_pids(gpu) and all(x['utilization.gpu']<=5 for x in before),'GPU load gate failed; no foreign process stopped')
        free=before[-1]['memory.free'];budget=min(.75*free,free-1536)
        require(budget>=1024 and shutil.disk_usage(session).free>=4*1024**3,'Insufficient GPU memory/disk reserve')
        verify_files(root,manifest['sources'])
        exe=child(root,manifest['exe_path']);require(sha(exe)==manifest['exe_sha256'],'Executable changed')
        env=baseline_environment(c,out);env['CUDA_VISIBLE_DEVICES']=before[-1]['uuid']
        req={'requested_config':t['config'],'expanded_config':c,'config_sha256':digest(c),'binary':'base',
             'source_digest':manifest['source_digest'],'exe_sha256':manifest['exe_sha256'],
             'runner_sha256':sha(__file__),'command':[str(exe)],'gpu_index':gpu,'gpu_before':before,
             'gpu_memory_budget_mib':budget,'environment':{k:v for k,v in env.items() if k.startswith('GIPC_')},'from_zero':True,'capabilities':{'resolved_config':False,'actual_velocity':False}}
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
        try: check=validate_base(out)
        except Exception as e: check={'passed':False,'error':str(e)}
        write_new(out/'config_validation.json',check)
        if not check['passed']:
            result['status']='configuration_failed'
            (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_new(out/'evidence.json',{'files':inventory(out,[p for p in out.rglob('*') if p.is_file()])})
    print(json.dumps({'run':t['name'],'status':result['status'],'frames':result['recorded_frames'],'solver_seconds':result.get('solver_seconds')}),flush=True)
    return result


def diagnostic_files(root):
    return inventory(root,[p for p in (root/'tools/diagnostic').rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'])

def compiler_identity(cache):
    values={line.split('=',1)[0].split(':',1)[0]:line.split('=',1)[1]
            for line in cache.read_text().splitlines() if '=' in line and not line.startswith(('#','//'))}
    rows=[];missing=[]
    for key in ('CMAKE_CXX_COMPILER','CMAKE_CUDA_COMPILER'):
        value=values.get(key);p=Path(value) if value else None
        if p is None or not p.is_absolute() or not p.is_file():missing.append(key);continue
        target=p.resolve();rows.append({'cache_key':key,'path':str(p),'resolved_path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
    return {'files':rows,'missing':missing,'complete':not missing,
            'scope':'Compiler binary identity from CMakeCache. System shared libraries are not exhaustively frozen.'}

def base_seal(root,build_log):
    base=child(root,'baseline');build=child(root,'build/base');exe=child(root,'build/base/gipc')
    require(exe.is_file(),'Missing baseline executable')
    cache=build/'CMakeCache.txt';text=cache.read_text()
    home=next((line.split('=',1)[1] for line in text.splitlines() if line.startswith('CMAKE_HOME_DIRECTORY:')),None)
    require(home and Path(home).resolve()==base,'Baseline build root differs')
    native={p.resolve() for p in (base/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
    actual=set()
    for row in read(build/'compile_commands.json'):
        command=row.get('command',' '.join(row.get('arguments',[])))
        if 'gipc.dir' not in command:continue
        p=Path(row['file']);p=p if p.is_absolute() else Path(row['directory'])/p
        require(p.resolve() in native,'Baseline compiles a non-baseline translation unit')
        require(str(base/'Assets') in command,'Baseline asset compile definition not identified')
        actual.add(p.resolve())
    require(actual==native and native,'Baseline compile source coverage mismatch')
    stopping=(base/'StiffGIPC/core/GIPC.cu').read_text()
    require(re.search(r'Kmin\s*=\s*6\s*;',stopping) and re.search(r'beta\s*<=\s*Newton_solver_threshold',stopping),
            'Baseline source stopping semantics not identified')
    paths=[]
    for name in ('StiffGIPC','Assets','MeshProcess'):
        folder=base/name;require(folder.is_dir(),'Missing baseline source/input root '+name)
        for p in folder.rglob('*'):
            require(not p.is_symlink(),'Linked baseline source/input rejected')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':paths.append(p)
    paths.append(base/'CMakeLists.txt')
    asset_rows=inventory(base/'Assets',[p for p in (base/'Assets').rglob('*') if p.is_file()])
    active_assets=child(root,'Assets')
    require(active_assets.is_dir(),'Missing active Assets root')
    for p in active_assets.rglob('*'):require(not p.is_symlink(),'Linked active asset rejected')
    require(asset_rows==inventory(active_assets,[p for p in active_assets.rglob('*') if p.is_file()]),
            'Baseline and active Assets differ')
    evidence=[cache,build/'compile_commands.json',child(root,build_log)]
    missing=[];link=build/'CMakeFiles/gipc.dir/link.txt'
    if link.is_file():evidence.append(link)
    else:missing.append('CMakeFiles/gipc.dir/link.txt (generator may not emit a separate link file)')
    sources=inventory(root,paths)
    return {'schema':'baseline_build_identity.v1','sources':sources,'source_digest':digest(sources),
            'exe_path':'build/base/gipc','exe_sha256':sha(exe),'compile_source_count':len(actual),
            'build_evidence':inventory(root,evidence),'missing_build_evidence':missing,
            'compiler_identity':compiler_identity(cache),
            'assets_equivalence':{'passed':True,'files':len(asset_rows),'relative_inventory_sha256':digest(asset_rows)},
            'stopping_source_evidence':{'path':'baseline/StiffGIPC/core/GIPC.cu','sha256':sha(base/'StiffGIPC/core/GIPC.cu'),
                  'min_updates':6,'cumulative_tolerance':'Newton_solver_threshold','runtime_resolved_available':False},
            'capabilities':{'resolved_config':False,'actual_velocity':False},
            'scope':'Independent baseline source/input/compile/executable identity; existing clean build supplied by owner.'}

def original_guard_evidence(root,path,active):
    guard=read(path);session=path.parent
    require(guard['stage']=='guards' and guard['allow_next_round'] is False,'Expected unchanged failed guard')
    require(guard['source_digest']==active['source_digest'] and guard['exe_sha256']==active['exe_sha256'],'Guard/active identity mismatch')
    require(guard['protocol_sha256']==sha(root/'tools/bench/quality_protocol.json'),'Original quality protocol changed')
    rows=guard['runs'];require([r['name'] for r in rows]==['guards_hang_on','guards_fixed_on','guards_mixed_off','guards_mixed_on'],'Unexpected original guard sequence')
    for row in rows:
        require(row['configuration']['passed'] and row['pool']['passed'],'Original guard has a configuration/typed/energy failure')
        if row['name']=='guards_fixed_on':
            require(row['failures']==['ValueError: Frozen cloth material bounds failed'],'Fixed guard failed beyond the diagnosed material bound')
            require(row['material']['p99_stretch']>row['material_bounds']['p99_stretch'],'Missing original p99 excess')
        else:require(row['hard_checks_passed'] and not row['failures'],'Unexpected original hard failure')
    receipt_path=session/'guards_receipt.json';batch_path=session/'guards_batch.json'
    receipt=read(receipt_path);batch=read(batch_path)
    require(receipt['analysis_sha256']==sha(path) and receipt['batch_sha256']==sha(batch_path),'Old guard receipt mismatch')
    require(len(batch['runs'])==4 and not batch['skipped'],'Original guard incomplete')
    for entry,row in zip(batch['runs'],rows):
        require(entry['name']==row['name'] and entry['result']['status']=='completed','Original run reordered/incomplete')
        run=session/entry['name'];require(sha(run/'evidence.json')==entry['evidence_sha256'],'Old run inventory changed')
        verify_files(run,read(run/'evidence.json')['files'])
    return {'path':path.relative_to(root).as_posix(),'sha256':sha(path),'receipt_path':receipt_path.relative_to(root).as_posix(),
            'receipt_sha256':sha(receipt_path),'batch_path':batch_path.relative_to(root).as_posix(),'batch_sha256':sha(batch_path),
            'remains_failed':True,'fixed_original_material':rows[1]['material'],'fixed_original_bounds':rows[1]['material_bounds']}

def seal(root,active_path,guard_path,base_log,output):
    active=verify_manifest(root,active_path)
    prior=original_guard_evidence(root,guard_path,active)
    require(read(child(root,prior['receipt_path']))['manifest_sha256']==sha(active_path),'Original active manifest SHA differs')
    base=base_seal(root,base_log)
    packet={'schema':'fixed_quality_diagnosis_seal.v1','plan':plan(),'plan_sha256':digest(plan()),
            'active_manifest_path':active_path.relative_to(root).as_posix(),'active_manifest_sha256':sha(active_path),
            'active_manifest':active,'base_manifest':base,'original_guard':prior,
            'diagnostic_tools':diagnostic_files(root),'quality_protocol_sha256':sha(root/'tools/bench/quality_protocol.json'),
            'active_compiler_identity':compiler_identity(root/Path(active['exe_path']).parent/'CMakeCache.txt')}
    write_new(output,packet);return packet

def verify_seal(root,path):
    packet=read(path);require(packet['schema']=='fixed_quality_diagnosis_seal.v1' and packet['plan']==plan() and packet['plan_sha256']==digest(plan()),'Diagnosis plan changed')
    active_path=child(root,packet['active_manifest_path'])
    require(sha(active_path)==packet['active_manifest_sha256'],'Original active manifest changed')
    require(verify_manifest(root,active_path)==packet['active_manifest'],'Active identity changed')
    require(diagnostic_files(root)==packet['diagnostic_tools'],'Diagnostic tools changed')
    base=packet['base_manifest'];verify_files(root,base['sources']);verify_files(root,base['build_evidence'])
    require(sha(child(root,base['exe_path']))==base['exe_sha256'],'Baseline executable changed')
    require(sha(root/'tools/bench/quality_protocol.json')==packet['quality_protocol_sha256'],'Quality protocol changed')
    for comp in (base['compiler_identity'],packet['active_compiler_identity']):
        for row in comp['files']:
            p=Path(row['path']);require(str(p.resolve())==row['resolved_path'] and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Compiler changed')
    prior=packet['original_guard']
    for field,hashfield in (('path','sha256'),('receipt_path','receipt_sha256'),('batch_path','batch_sha256')):
        require(sha(child(root,prior[field]))==prior[hashfield],'Original guard evidence changed')
    return packet

def run(root,seal_path,session_name,gpu):
    require(sys.platform.startswith('linux') and os.environ.get('DISPLAY'),'Linux + DISPLAY required; use xvfb-run')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',session_name) and gpu>=0,'Invalid session/GPU')
    packet=verify_seal(root,seal_path);session=child(root,'runs/'+session_name)
    require(not session.exists(),'Diagnostic session must be new; no retries or overwrite')
    session.mkdir(parents=True);write_new(session/'plan.json',plan())
    write_new(session/'started.json',{'seal_sha256':sha(seal_path),'plan_sha256':digest(plan()),'pid':os.getpid()})
    rows=[]
    with gpu_lock(root):
        for task in tasks():
            verify_seal(root,seal_path)
            identity=packet['base_manifest'] if task['binary']=='base' else packet['active_manifest']
            result=(execute_base if task['binary']=='base' else execute_active)(root,session,task,identity,gpu)
            check=analyze_run(session,task,packet)
            check_path=session/(task['name']+'_check.json');write_new(check_path,check)
            rows.append({'name':task['name'],'result':result,'evidence_sha256':sha(session/task['name']/'evidence.json'),
                         'check_sha256':sha(check_path),'hard_checks_passed':check['hard_checks_passed']})
            if result['status']!='completed' or not check['hard_checks_passed']:break
        batch={'plan':plan(),'seal_sha256':sha(seal_path),'runs':rows,'skipped':[t['name'] for t in tasks()[len(rows):]]}
        write_new(session/'batch.json',batch)
        try:
            verify_seal(root,seal_path);report=analyze(session,packet,batch);verify_seal(root,seal_path)
        except Exception as e:
            report={'diagnosis_complete':False,'error':type(e).__name__+': '+str(e),'original_guard_remains_failed':True,
                    'quality_certified':False,'performance_certified':False,'allow_performance_screen':False,'automatic_long_run':False}
        write_new(session/'analysis.json',report)
        write_new(session/'receipt.json',{'seal_sha256':sha(seal_path),'batch_sha256':sha(session/'batch.json'),'analysis_sha256':sha(session/'analysis.json')})
    print({'diagnosis_complete':report['diagnosis_complete'],'runs':len(rows),'analysis':str(session/'analysis.json'),'allow_performance_screen':False},flush=True)
    return 0 if report['diagnosis_complete'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('seal');s.add_argument('--active-manifest',required=True);s.add_argument('--prior-guard',required=True)
    s.add_argument('--base-build-log',required=True);s.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--seal',required=True);r.add_argument('--session',required=True);r.add_argument('--gpu',type=int,default=0)
    sub.add_parser('plan');a=p.parse_args();root=a.root.resolve()
    if a.action=='plan':
        print(json.dumps(plan(),indent=2));return 0
    if a.action=='seal':seal(root,child(root,a.active_manifest),child(root,a.prior_guard),a.base_build_log,child(root,a.output));return 0
    return run(root,child(root,a.seal),a.session,a.gpu)

if __name__=='__main__':raise SystemExit(main())
