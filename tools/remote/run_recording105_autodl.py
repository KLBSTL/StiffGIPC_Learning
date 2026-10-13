"""Finite serial AutoDL reproduction of the sealed local 105-step experiment."""
import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('tools/local','tools/paper_suite','tools/bench','tools/diagnostic')]
from config import read,expand
from linux_runner import require,write_new,sha,gpu_lock
import run_suite as linux
from analyze import observation

SESSION=ROOT/'runs/autodl_recording105_20261013'
OUTPUT=ROOT/'reports/autodl_recording105_20261013'
SELECTED=('Bunny','Mixed','Boxes')
ARMS=('original','pure','ours','strict')
LABELS={'original':'Movement only','pure':'Cumulative TOI .01',
        'ours':'Compensated + residual gate','strict':'Movement only, rho=1e-6 reference'}


def record(path):
    path=Path(path).resolve()
    before=path.stat()
    value=sha(path)
    after=path.stat()
    require(before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'File changed while hashing')
    return dict(path=str(path),bytes=after.st_size,sha256=value)


def verify_records(values):
    for value in values:
        require(record(value['path'])==value,'Source/build identity changed: '+value['path'])


def plan():
    return read(ROOT/'EXPERIMENT_PLAN.json')


# Reuse the original observer's recording validation function byte-for-byte.
original=ROOT/'tools/local/run_recording105_suite.py'
node=next(n for n in ast.parse(original.read_text(encoding='utf-8')).body
          if isinstance(n,ast.FunctionDef) and n.name=='recording_evidence')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(original),'exec'),globals())


def install_compatibility():
    module=types.ModuleType('run_recording105_suite')
    module.SELECTED, module.ARMS, module.LABELS=SELECTED,ARMS,LABELS
    module.plan, module.recording_evidence=plan,recording_evidence
    sys.modules['run_recording105_suite']=module
    from protocol import SCENES,KEYFRAMES,CHECKPOINTS
    combined=types.ModuleType('run_combination_suite')
    combined.SCENES,combined.KEYFRAMES,combined.CHECKPOINTS=SCENES,KEYFRAMES,CHECKPOINTS
    sys.modules['run_combination_suite']=combined


def manifest():
    return read(ROOT/'build/program_identity.json')


def seal():
    require(sys.platform.startswith('linux'),'Linux binary required')
    build=ROOT/'build/observer'
    exe=build/'gipc'
    native={p.resolve() for p in (ROOT/'StiffGIPC').rglob('*') if p.suffix in ('.cpp','.cu')}
    commands=read(build/'compile_commands.json')
    actual={Path(r['file']).resolve() for r in commands if 'gipc.dir' in r.get('command','')}
    require(native==actual and native,'Exact native compilation coverage failed')
    objects=list((build/'CMakeFiles/gipc.dir').rglob('*.o'))
    require(len(objects)>=len(native),'Missing compiled translation units')
    payload=read(ROOT/'TRANSFER_MANIFEST.json')
    sources=[]
    for item in payload['files']:
        path=ROOT/item['path']
        require(path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Transferred source differs')
        sources.append(record(path))
    evidence=[build/'CMakeCache.txt',build/'compile_commands.json',build/'configure.log',build/'build.log']+objects
    identity=dict(sources=sources,build_evidence=[record(p) for p in evidence],
                  exe=record(exe),dlls=[],objects=[],translation_units=len(native))
    for path in ('/usr/local/cuda-12.8/bin/nvcc','/usr/bin/g++'):
        identity.setdefault('compilers',[]).append(record(path))
    write_new(ROOT/'build/program_identity.json',identity)
    print(json.dumps(dict(sealed=True,translation_units=len(native),exe_sha256=identity['exe']['sha256'])),flush=True)


def verify_identity(identity):
    verify_records(identity['sources']+identity['build_evidence']+[identity['exe']]+identity['compilers'])


def run():
    require(os.environ.get('DISPLAY'),'Run under xvfb-run')
    identity=manifest()
    verify_identity(identity)
    SESSION.mkdir(parents=True,exist_ok=False)
    hardware=subprocess.check_output(['nvidia-smi','-q'],text=True)
    (SESSION/'hardware_before.txt').write_text(hardware)
    packet=dict(plan=plan(),programs={'active':identity},controller=record(__file__),
                platform=dict(os=sys.platform,gpu_query=linux.gpu_query(0),
                    nvcc=subprocess.check_output(['/usr/local/cuda-12.8/bin/nvcc','--version'],text=True)),
                local_source_comparison=read(ROOT/'LOCAL_SOURCE_COMPARISON.json'))
    write_new(SESSION/'identities.json',packet)
    original_env=linux.env_for
    original_gpu=linux.gpu_query
    def environment(t,out,exe_hash,checkpoint_root=None):
        value=original_env(t,out,exe_hash,checkpoint_root)
        value['GIPC_RECORD_TIMING']='1' if t['recording_enabled'] else '0'
        return value
    def gpu_query(gpu,timeout=10):
        begin=time.monotonic()
        result=original_gpu(gpu,timeout)
        result.update(sample_epoch_seconds=time.time(),query_host_ms=1000*(time.monotonic()-begin),
            timestamp_scope='Host time after existing nvidia-smi query; not per-frame GPU activity')
        return result
    linux.env_for=environment
    linux.gpu_query=gpu_query
    with gpu_lock(ROOT):
        for stage,tasks in plan()['stages'].items():
            results=[]
            write_new(SESSION/(stage+'_tasks.json'),dict(tasks=tasks))
            for index,task in enumerate(tasks,1):
                require(expand(task['config'])==task['config'],'Expanded configuration changed')
                result=linux.execute(SESSION,task,{'programs':{'active':{'exe':dict(path='build/observer/gipc',
                    sha256=identity['exe']['sha256'],bytes=identity['exe']['bytes'])}}},0)
                row=dict(task=task,result=result)
                if result['status']=='completed':
                    row['observation']=observation(SESSION/task['name'],task,result)
                    row['recording']=recording_evidence(SESSION/task['name'],task)
                    if 'checkpoint_save_frame' in task:
                        row['saved_checkpoint']=linux.checkpoint_evidence(SESSION/task['name']/'checkpoint')
                results.append(row)
                with (SESSION/(stage+'_journal.jsonl')).open('a') as f:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps(dict(stage=stage,done=index,total=len(tasks),run=task['name'],
                    status=result['status'],frames=result['recorded_frames'],solver_seconds=result.get('solver_seconds'))),flush=True)
                if result['status']!='completed':
                    break
            ledger=dict(stage=stage,runs=results,skipped=tasks[len(results):],
                        complete=len(results)==len(tasks) and all('recording' in r for r in results))
            write_new(SESSION/(stage+'_ledger.json'),ledger)
            require(ledger['complete'],'Stage incomplete: '+stage)
            verify_identity(identity)
    (SESSION/'hardware_after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    print(json.dumps(dict(complete=True,trajectories=36,steps=3780,checkpoint_runs=63,cost_runs=9)),flush=True)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def analyze_results():
    install_compatibility()
    module=load_module('autodl_analysis',ROOT/'tools/remote/analyze_recording105_autodl.py')
    module.analyze(SESSION,OUTPUT)


def render_results():
    install_compatibility()
    module=load_module('autodl_report',ROOT/'tools/remote/report_recording105_autodl.py')
    summary=read(OUTPUT/'summary.json')
    module.tables(OUTPUT,summary)
    module.plots(OUTPUT)
    module.visuals(SESSION,OUTPUT)
    print(json.dumps(dict(complete=True,tables=9,plots=3,visuals=12,obj=162)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','run','analyze','render'])
    action=parser.parse_args().action
    {'seal':seal,'run':run,'analyze':analyze_results,'render':render_results}[action]()
