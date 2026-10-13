"""Finite Linux/AutoDL runner; no installation, upload, retries or foreign cleanup."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from protocol import ROOT,plan,env_for,input_inventory,read,sha,write,require
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from linux_runner import gpu_query,compute_pids,gpu_lock,stop_owned
from validate_run import validate
from quality_analysis import validate_base
from pool_metrics import export_evidence


def records(paths,root=ROOT):
    return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(set(paths),key=lambda p:p.relative_to(root).as_posix())]


def source_files():
    paths=[]
    for name in ('StiffGIPC','MeshProcess','tests','baseline/StiffGIPC','baseline/MeshProcess','baseline/tests',
                 'tools/bench','tools/diagnostic','tools/paper_suite'):
        folder=ROOT/name
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if any(x in p.parts for x in ('.git','__pycache__','build')):continue
            require(not p.is_symlink(),'Linked source rejected: '+str(p))
            if p.is_file() and p.suffix not in ('.pyc','.exe','.dll','.obj'):paths.append(p)
    paths.extend([ROOT/'CMakeLists.txt',ROOT/'baseline/CMakeLists.txt'])
    for row in input_inventory():
        for f in row['files']:
            paths.extend([ROOT/'Assets'/f['path'],ROOT/'baseline/Assets'/f['path']])
    for kind in ('Assets','baseline/Assets'):
        paths.extend(p for p in (ROOT/kind/'scene').glob('*') if p.is_file())
    return sorted(set(paths),key=lambda p:p.relative_to(ROOT).as_posix())


def verify_rows(rows):
    for r in rows:
        p=(ROOT/r['path']).resolve()
        require(p.is_relative_to(ROOT) and p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],
            'Changed/missing identity: '+r['path'])


def seal(active,base,output):
    require(sys.platform.startswith('linux'),'Seal Linux builds on AutoDL, not Windows binaries')
    sources=records(source_files());programs={}
    for kind,build,source in (('active',active,ROOT),('base',base,ROOT/'baseline')):
        build=build.resolve();exe=build/'gipc'
        require(build.is_relative_to(ROOT) and exe.is_file(),'Build/executable outside payload')
        commands=read(build/'compile_commands.json')
        actual={}
        for r in commands:
            text=r.get('command',' '.join(r.get('arguments',[])))
            if 'gipc.dir' not in text:continue
            f=Path(r['file']);f=(f if f.is_absolute() else Path(r['directory'])/f).resolve()
            require(f not in actual,'Duplicate compilation unit');actual[f]=text
        expected={p.resolve() for p in (source/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
        require(set(actual)==expected,'Compilation coverage mismatch for '+kind)
        evidence=[build/'compile_commands.json',build/'CMakeCache.txt',build/'configure.log',build/'build.log']
        link=build/'CMakeFiles/gipc.dir/link.txt'
        if link.exists():evidence.append(link)
        objects=[p for p in (build/'CMakeFiles/gipc.dir').rglob('*.o')]
        require(len(objects)>=len(expected),'Missing compiled objects')
        programs[kind]=dict(exe=records([exe])[0],build_evidence=records(evidence+objects),translation_units=len(expected))
    result=dict(schema='paper_suite_linux_build.v1',plan=plan(),sources=sources,programs=programs,
        system=dict(platform=sys.platform,nvcc=subprocess.check_output(['nvcc','--version'],text=True).strip()),
        clean_build_required=True,scope='Post-clean-build exact source/unit/object/executable identity; dynamic system libraries not exhaustively sealed')
    write(output,result);print(json.dumps(dict(sealed=True,programs={k:v['translation_units'] for k,v in programs.items()})))


def verify(manifest):
    require(manifest['schema']=='paper_suite_linux_build.v1' and manifest['plan']==plan(),'Manifest/plan differs')
    require(records(source_files())==manifest['sources'],'Source/input/runner inventory differs')
    verify_rows(manifest['sources'])
    for p in manifest['programs'].values():verify_rows(p['build_evidence']+[p['exe']])


def checkpoint_evidence(cp):
    meta=read(cp/'checkpoint.json');require(meta['schema']=='gipc-frame-boundary-2','Unsupported checkpoint')
    rows=[dict(path='checkpoint.json',bytes=(cp/'checkpoint.json').stat().st_size,sha256=sha(cp/'checkpoint.json'))]
    for name,r in meta['buffers'].items():
        require(name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name),'Invalid member name')
        p=cp/(name+'.bin');require(p.stat().st_size==r['bytes'],'Checkpoint size mismatch')
        rows.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    return dict(meta=meta,files=rows)


def finish_run(out,t,result):
    frames=[]
    if (out/'trace/frames.csv').exists():
        with (out/'trace/frames.csv').open() as f:frames=list(csv.DictReader(f))
    result['recorded_frames']=len(frames)
    if frames:result['solver_seconds']=sum(float(r['solver_ms']) for r in frames)/1000
    if result['status']=='completed':
        try:
            require(len(frames)==t['config']['steps'],'Partial run')
            require(export_evidence(out,len(frames),'state')['passed'],'Nonfinite/incomplete position export')
            if t['binary']!='base':require(export_evidence(out,len(frames),'velocity')['passed'],'Missing actual velocities')
            stats=read(out/'output/stats.json')['frames']
            require(len(stats)==len(frames),'Frame statistics missing')
            require(not any(f.get('newton_exit')=='iteration_limit' for f in stats),'Nonlinear iteration cap')
            p=[n['pcg'] for f in stats for n in f['newton'] if 'pcg' in n]
            require(p and not any(x.get('iteration_limit') or x.get('breakdown') for x in p),'PCG failure')
            if 'checkpoint_from' in t:
                restore=read(out/'output/checkpoint_restore.json')
                require(restore['device_bytes_verified'] and restore['context_verified']
                    and restore['completed_physical_frames']==t['target_frame']-1,'Restore contract failed')
                require(all(x.get('frame')==t['target_frame'] for x in p),'Wrong absolute PCG frame')
            result['finite']=True
        except Exception as e:result.update(status='incomplete_or_invalid',error=str(e))
    write(out/'result.json',result)
    if result['status']=='completed':
        try:
            check=validate(out) if t['binary']=='active' else validate_base(out)
            if t['binary']=='active':
                require(read(out/'resolved_config.json')['ipc_pcg_forcing']['mode']=='off','Unexpected PCG forcing')
            require(check['passed'],'Configuration/execution contract failed')
        except Exception as e:
            check=dict(passed=False,error=str(e));result.update(status='configuration_failed',error=str(e))
            # Original result is retained separately; final outcome is explicit.
            write(out/'final_status.json',result)
        write(out/'config_validation.json',check)
    if result['status']=='completed' and 'checkpoint_from' in t:
        cp=Path(result['checkpoint_source']);meta=read(cp/'checkpoint.json')
        for name,trace_name in (('vertices','state_0000.bin'),('velocities','velocity_0000.bin')):
            require((cp/(name+'.bin')).read_bytes()==(out/'trace'/trace_name).read_bytes(),'Restored state is not bitwise initial input')
    return result


def wait_idle_gpu(gpu,receipt,timeout_seconds=10):
    deadline=time.monotonic()+timeout_seconds;low=[];identity=None
    report=dict(timeout_seconds=timeout_seconds,utilization_max=5,required_consecutive_samples=2,
        foreign_processes_never_stopped=True,samples=[],passed=False)
    try:
        while True:
            remaining=deadline-time.monotonic();require(remaining>0,'GPU did not become idle before bounded preflight deadline')
            sample=gpu_query(gpu,timeout=min(10,remaining))
            remaining=deadline-time.monotonic();require(remaining>0,'GPU preflight query exceeded deadline')
            pids=compute_pids(gpu,timeout=min(10,remaining))
            report['samples'].append(dict(gpu=sample,foreign_compute_pids=pids))
            require(not pids,'GPU has foreign compute processes; they are untouched')
            identity=identity or sample['uuid'];require(sample['uuid']==identity,'GPU identity changed during preflight')
            low=low+[sample] if sample['utilization.gpu']<=5 else []
            if len(low)==2:
                report['passed']=True;return low
            time.sleep(min(.25,max(0,deadline-time.monotonic())))
    finally:write(receipt,report)


def execute(session,t,manifest,gpu):
    out=session/t['name'];out.mkdir();(out/'output').mkdir();p=None
    result=dict(status='launcher_failed',recorded_frames=0,performance_certified=False,physical_quality_certified=False,
        timing_scope='Synchronized host elapsed solver_ms; exports outside timer; checkpoint restores rebuild derived caches')
    started=None
    try:
        before=wait_idle_gpu(gpu,out/'gpu_preflight.json')
        require(before[0]['uuid']==before[1]['uuid'],'GPU identity changed')
        free=before[-1]['memory.free'];budget=min(.75*free,free-1536)
        require(budget>=1024 and shutil.disk_usage(session).free>=4*1024**3,'Memory/disk reserve insufficient')
        program=manifest['programs'][t['binary']];exe=ROOT/program['exe']['path'];verify_rows([program['exe']])
        env=env_for(t,out,program['exe']['sha256'],session);env['CUDA_VISIBLE_DEVICES']=before[-1]['uuid']
        cp=None
        if 'checkpoint_from' in t:
            cp=session/t['checkpoint_from'];result['checkpoint_source']=str(cp)
        requested=dict(task=t,expanded_config=t['config'],binary=t['binary'],capabilities={'actual_velocity':False},
            command=[str(exe)],environment={k:v for k,v in env.items() if k.startswith('GIPC_')},gpu_before=before,
            exe_sha256=program['exe']['sha256'],runner_sha256=sha(__file__),
            checkpoint=checkpoint_evidence(cp) if cp else None,
            resource_policy=dict(timeout_seconds=t['config']['timeout_seconds'],memory_budget_mib=budget,
                memory_free_min_mib=768,disk_before_bytes=4*1024**3,disk_during_bytes=1024**3,kill_only_owned_process_group=True))
        write(out/'requested.json',requested)
        samples=[];started=time.monotonic();deadline=started+t['config']['timeout_seconds'];status='running'
        with (out/'run.log').open('xb') as log:
            p=subprocess.Popen([str(exe)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                if time.monotonic()>=deadline:status='timeout';stop_owned(p);break
                remaining=max(.01,min(10,deadline-time.monotonic()))
                sample=gpu_query(gpu,timeout=remaining);foreign=[x for x in compute_pids(gpu,timeout=remaining) if x!=p.pid]
                samples.append(sample|{'foreign_compute_pids':foreign})
                require(sample['uuid']==before[-1]['uuid'],'GPU identity changed')
                status=('foreign_gpu_load' if foreign else 'disk_reserve' if shutil.disk_usage(session).free<1024**3 else
                    'memory_budget' if sample['memory.free']<768 or free-sample['memory.free']>budget else 'running')
                if status!='running':stop_owned(p);break
                time.sleep(min(.5,max(0,deadline-time.monotonic())))
        elapsed=time.monotonic()-started
        if status=='running':status='completed' if p.returncode==0 and elapsed<=t['config']['timeout_seconds'] else 'failed_or_timeout'
        result.update(status=status,exit_code=p.poll(),wall_seconds=elapsed,gpu_samples=samples)
    except Exception as e:result.update(status='launcher_or_monitor_failed',error=type(e).__name__+': '+str(e))
    finally:
        stop_owned(p)
        result['owned_process_finished']=p is None or p.poll() is not None
    try:return finish_run(out,t,result)
    except Exception as e:
        result.update(status='analysis_failed',error=type(e).__name__+': '+str(e));write(out/'analysis_failure.json',result);return result


def run(manifest_path,session,stage,gpu):
    require(sys.platform.startswith('linux'),'Use local_smoke.py for bounded Windows checks')
    require(os.environ.get('DISPLAY'),'Run under xvfb-run -a; GL window creation is still needed')
    manifest=read(manifest_path);verify(manifest)
    session=session.resolve();require(session.is_relative_to(ROOT/'runs'),'Session must be under payload/runs')
    session.mkdir(parents=True,exist_ok=True)
    require(not (session/(stage+'_ledger.json')).exists() and not (session/(stage+'_journal.jsonl')).exists(),'Stage already attempted')
    if stage!='smoke':
        smoke=read(session/'smoke_ledger.json');require(smoke['complete'],'Smoke did not pass for all seven scenes')
    if stage=='matched':require(read(session/'seed_ledger.json')['complete'],'Checkpoint seed failed')
    if stage=='performance':
        gate=read(session/'checkpoint_input_gate.json');require(gate['passed'],'Run analyze after matched; exact checkpoint input gate must pass before long timing')
    tasks=plan()['stages'][stage];rows=[];skipped=[];failed=set()
    with gpu_lock(ROOT):
        with (session/(stage+'_journal.jsonl')).open('x',encoding='utf8') as journal:
            for t in tasks:
                if t['scene_key'] in failed:skipped.append(t);continue
                require(not (session/t['name']).exists(),'Prior output exists')
                result=execute(session,t,manifest,gpu);row=dict(task=t,result=result);rows.append(row)
                journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                print(json.dumps(dict(name=t['name'],status=result['status'],frames=result['recorded_frames'])),flush=True)
                if result['status']!='completed':failed.add(t['scene_key'])
    verify(manifest)
    packet=dict(stage=stage,manifest_sha256=sha(manifest_path),plan=plan(),runs=rows,skipped=skipped,
        complete=not skipped and len(rows)==len(tasks) and all(r['result']['status']=='completed' for r in rows))
    write(session/(stage+'_ledger.json'),packet);return 0 if packet['complete'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    s=sub.add_parser('seal');s.add_argument('--active',type=Path,required=True);s.add_argument('--base',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--manifest',type=Path,required=True);r.add_argument('--session',type=Path,required=True)
    r.add_argument('--stage',choices=tuple(plan()['stages']),required=True);r.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.action=='seal':seal(args.active,args.base,args.output)
    else:raise SystemExit(run(args.manifest,args.session,args.stage,args.gpu))
