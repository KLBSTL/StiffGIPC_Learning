"""Detached finite queue authorized by the user; no simulation retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as suite

def must_stop(result,failures):
    return result.get('status') in {'configuration_failed','launcher_or_monitor_failed',
        'launcher_failed','monitor_timeout','disk_reserve','memory_budget','foreign_gpu_load'} or any(
        'identity differs' in str(f).lower() or 'configuration evidence' in str(f).lower() for f in failures)

def main():
    p=argparse.ArgumentParser();p.add_argument('--build-pid',type=int,required=True);a=p.parse_args()
    import fcntl
    directory=suite.ROOT/suite.REPORT
    lock=suite.ROOT/'runs/.batch.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as owned:
        fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=directory/'STATUS.json'
        suite.require(not status_path.exists(),'Batch already launched; no automatic restart')
        started=time.monotonic();deadline=started+6000
        status={'pid':os.getpid(),'build_pid':a.build_pid,'planned_runs':21,
                'recorded_runs':0,'automatic_retry':False,'total_timeout_seconds':6000}
        def update(stage,**fields):
            status.update(stage=stage,elapsed_seconds=time.monotonic()-started,**fields)
            temp=status_path.with_suffix('.tmp')
            temp.write_text(json.dumps(status,indent=2,allow_nan=False))
            temp.replace(status_path)
            print(json.dumps(status),flush=True)
        try:
            update('waiting_for_build')
            ready=directory/'BUILD_IDENTITY.json';wait_deadline=min(deadline,started+1800)
            while not ready.is_file():
                suite.require(time.monotonic()<wait_deadline,'Build wait exceeded 1800 seconds')
                suite.require(Path(f'/proc/{a.build_pid}/cmdline').exists(),'Build owner exited before identity was produced')
                time.sleep(5)
            update('sealing')
            suite.seal()
            for index in range(1,22):
                suite.require(time.monotonic()<deadline,'Batch total budget expired')
                update('running',current_index=index,current_task=suite.tasks()[index-1]['name'])
                command=[sys.executable,str(Path(__file__).with_name('run.py')),'run','--index',str(index)]
                completed=subprocess.run(command,cwd=suite.ROOT,timeout=min(360,deadline-time.monotonic()))
                suite.require(completed.returncode==0,'Controller/analysis failed; queue stopped without retry')
                row=suite.read(directory/'runs'/f'{index:02d}.json')
                update('analyzed',recorded_runs=index,last_result=row['result']['status'],
                       last_hard_checks_passed=row['analysis']['hard_checks_passed'],last_failures=row['analysis']['failures'])
                suite.require(not must_stop(row['result'],row['analysis']['failures']),
                              'Resource/configuration failure; remaining queue suspended without retry')
            update('finalizing')
            result=suite.report()
            update('completed',report=result['result'],
                   completed_native_runs=sum(r['result']['status']=='completed' for r in suite.records()),
                   hard_checks_passed=sum(r['analysis']['hard_checks_passed'] for r in suite.records()))
        except Exception as e:
            update('stopped',error=type(e).__name__+': '+str(e))
            raise

if __name__=='__main__':main()
