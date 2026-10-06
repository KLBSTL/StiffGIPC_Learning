"""One bounded diagnostic round on Windows. No build, cleanup, remote or auto-next operation."""
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
import time
from contextlib import contextmanager
from local_plan import ROOT,ROUNDS,tasks,plan,predecessor
from config import read,expand,digest,environment
from linux_runner import require,child,sha,write_new,inventory,gpu_query,verify_files
from fixed_quality import baseline_environment
from local_identity import create_seal,verify_seal,verify
from local_analysis import analyze_one,analyze_round
from quality_analysis import validate_base
from validate_run import validate

@contextmanager
def gpu_lock(root):
    import msvcrt
    path=child(root,'runs/.local_gpu.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)

def stop_owned(proc):
    # Native executable is launched directly, with no profiler/launcher descendants.
    # Popen keeps the exact Windows process handle; never terminate a foreign PID.
    if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=10)

def remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise subprocess.TimeoutExpired('owned native budget',120)
    return min(10,value)

def process_rows(stdout,own_pid,driver_model):
    rows=[];foreign=[]
    for values in csv.reader(stdout.splitlines(),skipinitialspace=True):
        if not values:continue
        require(len(values)==3,'Incomplete GPU process query')
        pid=int(values[0]);name=values[1].strip();raw=values[2].strip()
        try:used=float(raw)
        except ValueError:used=None
        require(used is None or (math.isfinite(used) and used>=0),'Invalid process GPU memory')
        known_solver=name.replace('\\','/').split('/')[-1].lower()=='gipc.exe'
        blocking=pid!=own_pid and (driver_model!='WDDM' or known_solver or (used is not None and used>0))
        if blocking:foreign.append(pid)
        rows.append({'pid':pid,'process_name':name,'used_gpu_memory_mib':used,'raw_memory':raw,
                     'owned':pid==own_pid,'blocking_foreign_compute':blocking,
                     'classification':'owned' if pid==own_pid else 'foreign_compute_evidence' if blocking else 'unknown_WDDM_desktop_or_compute'})
    return {'rows':rows,'blocking_foreign_pids':foreign,'unknown_load':any(r['classification']=='unknown_WDDM_desktop_or_compute' for r in rows)}

def gpu_processes(gpu,own_pid,driver_model,timeout=10):
    result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name,used_gpu_memory',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=timeout)
    return process_rows(result.stdout,own_pid,driver_model)

def driver_model(gpu):
    result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=driver_model.current','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=10)
    value=result.stdout.strip().upper();require(value in ('WDDM','TCC'),'Unrecognized Windows GPU driver model')
    return value

def execute(session,t,identity,gpu):
    c=expand(t['config']);out=child(session,t['name']);require(not out.exists(),'Run output exists')
    (out/'output').mkdir(parents=True);proc=None
    result={'status':'launcher_failed','recorded_frames':0,'heavy_diagnostics':False,
            'performance_certified':False,'physical_quality_certified':False,'timing_is_diagnostic':True,
            'timing_scope':'Shared Windows desktop diagnostic. Baseline lacks velocity/resolved/breakdown telemetry; no performance certification.'}
    try:
        mode=driver_model(gpu);before=[gpu_query(gpu)];time.sleep(.25);before.append(gpu_query(gpu));process_before=gpu_processes(gpu,None,mode)
        require(not process_before['blocking_foreign_pids'] and (mode=='WDDM' or all(g['utilization.gpu']<=5 for g in before)),
                'Confirmed foreign compute/load gate failed; no process stopped')
        result.update(gpu_driver_model=mode,desktop_load_uncontrolled=mode=='WDDM',prelaunch_processes=process_before)
        free=before[-1]['memory.free'];budget=min(.75*free,free-1536)
        require(budget>=1024 and shutil.disk_usage(session).free>=4*1024**3,'Memory/disk reserve insufficient')
        verify(identity['sources']+[identity['exe']]+identity['dlls'])
        exe=Path(identity['exe']['path']);env=baseline_environment(c,out) if t['binary']=='base' else environment(c,out)
        env['CUDA_VISIBLE_DEVICES']=before[-1]['uuid']
        req={'requested_config':t['config'],'expanded_config':c,'config_sha256':digest(c),'binary':t['binary'],
             'source_digest':identity['source_digest'],'exe_sha256':identity['exe']['sha256'],
             'runner_sha256':sha(__file__),'command':[str(exe)],'gpu_index':gpu,'gpu_before':before,'memory_budget_mib':budget,
             'environment':{k:v for k,v in env.items() if k.startswith('GIPC_')},'from_zero':True,'capabilities':identity['capabilities'],
             'gpu_driver_model':mode,'gpu_processes_before':process_before,'desktop_load_uncontrolled':mode=='WDDM'}
        write_new(out/'requested.json',req);write_new(out/'build_manifest.json',identity)
        start=time.monotonic();deadline=start+120;status='running';samples=[]
        with (out/'run.log').open('xb') as log:
            startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
            proc=subprocess.Popen([str(exe)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
            write_new(out/'process.json',{'pid':proc.pid,'exe':str(exe)})
            while proc.poll() is None:
                try:
                    sample=gpu_query(gpu,timeout=remaining(deadline))
                    processes=gpu_processes(gpu,proc.pid,mode,timeout=remaining(deadline));foreign=processes['blocking_foreign_pids']
                    remaining(deadline)
                except subprocess.TimeoutExpired:
                    status='timeout' if time.monotonic()>=deadline else 'monitor_timeout';stop_owned(proc);break
                samples.append(sample|{'gpu_processes':processes,'foreign_compute_pids':foreign})
                status=('foreign_gpu_load' if foreign else 'disk_reserve' if shutil.disk_usage(session).free<1024**3 else
                        'memory_budget' if sample['memory.free']<768 or free-sample['memory.free']>budget else 'running')
                if status!='running':stop_owned(proc);break
                time.sleep(max(0,min(.5,deadline-time.monotonic())))
            proc.wait(timeout=10)
        result.update(status=('completed' if proc.returncode==0 else 'failed') if status=='running' else status,
                      exit_code=proc.returncode,wall_seconds=time.monotonic()-start,gpu_samples=samples)
        if (out/'trace/frames.csv').exists():
            with (out/'trace/frames.csv').open() as stream:frames=list(csv.DictReader(stream))
            result.update(recorded_frames=len(frames),solver_seconds=sum(float(f['solver_ms']) for f in frames)/1000)
        if (out/'final.bin').is_file():
            raw=(out/'final.bin').read_bytes();result['finite']=bool(raw) and len(raw)%8==0 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<d',raw))
        if result['status']=='completed' and (result['recorded_frames']!=c['steps'] or not result.get('finite')):result['status']='incomplete_or_nonfinite'
        verify(identity['sources']+[identity['exe']]+identity['dlls'])
    except Exception as e:result.update(status='launcher_or_monitor_failed',error=type(e).__name__+': '+str(e))
    finally:stop_owned(proc);write_new(out/'result.json',result)
    if result['status']=='completed':
        try:check=validate(out) if t['binary']=='active' else validate_base(out)
        except Exception as e:check={'passed':False,'error':type(e).__name__+': '+str(e)}
        write_new(out/'config_validation.json',check)
        if not check['passed']:
            result['status']='configuration_failed';(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_new(out/'evidence.json',{'files':inventory(out,[p for p in out.rglob('*') if p.is_file()])})
    print(json.dumps({'run':t['name'],'status':result['status'],'frames':result['recorded_frames']}),flush=True)
    return result

def check_previous(session,stage,seal_sha,reviewed_sha):
    previous=predecessor(stage)
    if previous is None:
        require(reviewed_sha is None,'First round has no previous analysis');return []
    path=session/previous/'analysis.json';require(reviewed_sha and sha(path)==reviewed_sha,'Supply the actual previous analysis SHA after review')
    reports=[]
    while previous:
        folder=session/previous;receipt=read(folder/'receipt.json');analysis=read(folder/'analysis.json');batch=read(folder/'batch.json')
        require(receipt['seal_sha256']==seal_sha and receipt['analysis_sha256']==sha(folder/'analysis.json') and receipt['batch_sha256']==sha(folder/'batch.json'),'Predecessor receipt differs')
        require(analysis['stage']==previous and analysis['diagnosis_complete'],'Previous diagnostic round incomplete/hard-failed')
        require(batch['stage']==previous and batch['plan']==plan(),'Predecessor plan differs')
        require([r['name'] for r in batch['runs']]==[t['name'] for t in tasks(previous)] and not batch['skipped'],'Predecessor ledger differs')
        for r in batch['runs']:
            out=folder/r['name'];require(sha(out/'evidence.json')==r['evidence_sha256'],'Previous evidence inventory changed')
            verify_files(out,read(out/'evidence.json')['files'])
        reports.append(analysis);previous=predecessor(previous)
    return reports

def run(root,seal_path,session_name,stage,reviewed_sha,gpu):
    require(os.name=='nt','Windows runner only');require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',session_name) and gpu>=0,'Invalid session/GPU')
    seal=verify_seal(root,seal_path);session=child(root,'runs/'+session_name);ssha=sha(seal_path)
    previous=check_previous(session,stage,ssha,reviewed_sha);folder=child(session,stage)
    require(not folder.exists(),'Round output exists; no overwrite/retry');folder.mkdir(parents=True)
    write_new(folder/'started.json',{'seal_sha256':ssha,'stage':stage,'previous_analysis_reviewed_sha256':reviewed_sha,'pid':os.getpid()})
    rows=[];failure=None
    with gpu_lock(root):
        try:
            for t in tasks(stage):
                verify_seal(root,seal_path);res=execute(folder,t,seal['programs'][t['binary']],gpu)
                check=analyze_one(folder,t,seal);cp=folder/(t['name']+'_check.json');write_new(cp,check)
                rows.append({'name':t['name'],'result':res,'evidence_sha256':sha(folder/t['name']/'evidence.json'),'check_sha256':sha(cp)})
                if not check['hard_checks_passed']:break
            verify_seal(root,seal_path)
        except Exception as e:failure=type(e).__name__+': '+str(e)
        batch={'stage':stage,'plan':plan(),'runs':rows,'skipped':[t['name'] for t in tasks(stage)[len(rows):]],'failure':failure}
        write_new(folder/'batch.json',batch)
        try:
            report=analyze_round(folder,stage,seal,batch);verify_seal(root,seal_path)
            if failure:report.update(diagnosis_complete=False,failure=failure)
            allrows=[r for prior in previous for r in prior['runs']]+report['runs']
            report['cumulative_observed_ranges']={v:{k:{'min':min(r['material'][k] for r in allrows if r['variant']==v and r['hard_checks_passed']),
                'max':max(r['material'][k] for r in allrows if r['variant']==v and r['hard_checks_passed'])}
                for k in ('max_stretch','p99_stretch','fixed_drift_m')} for v in ('off','base','on') if any(r['variant']==v and r['hard_checks_passed'] for r in allrows)}
        except Exception as e:
            report={'diagnosis_complete':False,'stage':stage,'error':type(e).__name__+': '+str(e),'allow_performance_screen':False,
                    'old_quality_gate_unchanged':True,'quality_certified':False,'performance_certified':False,'automatic_next_round':False}
        write_new(folder/'analysis.json',report)
        write_new(folder/'receipt.json',{'seal_sha256':ssha,'batch_sha256':sha(folder/'batch.json'),'analysis_sha256':sha(folder/'analysis.json')})
    print(json.dumps({'stage':stage,'diagnosis_complete':report['diagnosis_complete'],'analysis':str(folder/'analysis.json'),
                      'analysis_sha256':sha(folder/'analysis.json'),'automatic_next_round':False}),flush=True)
    return 0 if report['diagnosis_complete'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('seal');s.add_argument('--build-dir',required=True);s.add_argument('--exe',required=True);s.add_argument('--build-log',required=True)
    s.add_argument('--base-root',type=Path,required=True);s.add_argument('--base-manifest',default='manifests/perf_v34.json');s.add_argument('--base-exe',default='builds/local-base/Release/gipc.exe')
    s.add_argument('--prior-guard',required=True);s.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--seal',required=True);r.add_argument('--session',required=True);r.add_argument('--stage',choices=ROUNDS,required=True)
    r.add_argument('--reviewed-previous-sha');r.add_argument('--gpu',type=int,default=0);sub.add_parser('plan');a=p.parse_args();root=a.root.resolve()
    if a.action=='plan':print(json.dumps(plan(),indent=2));return 0
    if a.action=='seal':
        base=a.base_root.resolve();create_seal(root,child(root,a.build_dir),child(root,a.exe),child(root,a.build_log),base,
            child(base,a.base_manifest),child(base,a.base_exe),child(root,a.prior_guard),child(root,a.output));return 0
    return run(root,child(root,a.seal),a.session,a.stage,a.reviewed_previous_sha,a.gpu)

if __name__=='__main__':raise SystemExit(main())
