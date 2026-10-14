"""Finite local M0 evidence runs. No solver edits, remote use, overwrite or auto-retry."""
from __future__ import annotations
import argparse
import ctypes
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'/name) for name in ('local','bench','diagnostic','paper_suite')]
from config import expand,environment,digest,read
from linux_runner import require,write_new
from local_identity import active_identity,record,verify,compiler_records
import windows_runner

SCENES={'Boxes':('boxes1920_cloth_l',20),'Mixed':('bunny_cloth_bunny_l',15),
        'Bunny':('cloth_fixed_bunny_l',14)}
PHASES=('assembly','pcg','ccd','line_search','state_update')


def power_guid():
    api=ctypes.WinDLL('PowrProf',use_last_error=True)
    api.PowerGetActiveScheme.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
    api.PowerGetActiveScheme.restype=ctypes.c_ulong
    pointer=ctypes.c_void_p()
    code=api.PowerGetActiveScheme(None,ctypes.byref(pointer))
    require(code==0 and pointer.value,'Cannot query active power scheme')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    try:return str(uuid.UUID(bytes_le=ctypes.string_at(pointer,16)))
    finally:kernel.LocalFree(pointer)


def configuration(scene,steps,mode='legacy',rho=1e-4,outer=.01,shadow=False):
    return expand(dict(scene=SCENES[scene][0],steps=steps,dt=.04,timeout_seconds=120,
        backend='ipc',preset='combined',mas='legacy',pcg_graph_chunk=1,pcg_rho_tol=rho,
        ipc_newton_tol=outer,ipc_cumulative_tol=.01,ipc_min_updates=6,
        ipc_termination=mode,ipc_residual_rel_tol=.03,ipc_residual_shadow=shadow,
        trace_velocity=True,fem_direct_pullback=True,mas_pivot_only_sync=True,
        discrete_bvh_refit=True,discrete_bvh_rebuild_interval=8,diagnostics=[],profile='none'))


def tasks(stage):
    result=[]
    def add(name,scene,config,binary='active',**extra):
        if binary=='base':
            config=expand(config|dict(execution='host',refit=False,batch=False,reuse=False,
                fem_direct_pullback=False,mas_pivot_only_sync=False,discrete_bvh_refit=False,trace_velocity=False))
        result.append(dict(name=name,scene=scene,config=config,binary=binary,**extra))
    if stage=='smoke':
        for binary in ('base','active'):
            add('smoke_'+binary,'Bunny',configuration('Bunny',2),binary)
    elif stage=='prefix':
        for scene,(_,steps) in SCENES.items():
            add('prefix_'+scene.lower()+'_base',scene,configuration(scene,steps),'base')
            add('prefix_'+scene.lower()+'_legacy',scene,configuration(scene,steps))
            add('prefix_'+scene.lower()+'_movement',scene,configuration(scene,steps,'movement_only'),
                checkpoint_save_frame=steps)
    elif stage=='matched':
        for repeat in (1,2):
            scenes=list(SCENES) if repeat==1 else list(reversed(SCENES))
            modes=['movement_only','legacy','gated','compensated','strict']
            if repeat==2:modes.reverse()
            for scene in scenes:
                for mode in modes:
                    rule='movement_only' if mode=='strict' else mode
                    add(f'matched_{scene.lower()}_{mode}_r{repeat}',scene,
                        configuration(scene,1,rule,rho=1e-6 if mode=='strict' else 1e-4,shadow=True),
                        checkpoint_from='prefix_'+scene.lower()+'_movement/checkpoint',
                        target_frame=SCENES[scene][1],rule_label=mode,repeat=repeat)
    elif stage=='reference':
        # Starting from the identical pristine scene allows outer-tolerance
        # changes without bypassing the checkpoint's native context guard.
        for repeat in (1,2):
            for label,rho,outer in [('nominal',1e-4,.01),('inner',1e-6,.01),
                    ('inner_outer',1e-6,.001),('tighter',1e-8,.0001)]:
                add(f'reference_bunny_{label}_r{repeat}','Bunny',
                    configuration('Bunny',6,'movement_only',rho,outer),reference_label=label,repeat=repeat)
    else:raise ValueError('Unknown stage')
    return result


def seal(session):
    require(not session.exists(),'Fresh session required')
    build=ROOT/'build/innovation_m0_b1_20261014_v2'
    active=active_identity(ROOT,build,build/'Release/gipc.exe',build/'build.log')
    base_build=ROOT/'build/innovation_m0_b0_20261014'
    base=read(base_build/'identity.json')
    base.update(sources=base['source_inventory'],source_digest=digest(base['source_inventory']),
        capabilities=dict(resolved_config=False,actual_velocity=False),compilers=compiler_records(base_build))
    verify(base['sources']+[base['exe']]+base['dlls']+base['build_evidence']+base['objects'])
    # Only selected scene inputs and their derived partitions matter. Check
    # exact equality rather than waiving arbitrary baseline-only asset files.
    inputs=[]
    for scene,(key,_) in SCENES.items():
        a=ROOT/'Assets/benchmark_scenes'/(key+'.json')
        b=ROOT/'baseline/Assets/benchmark_scenes'/(key+'.json')
        require(a.read_bytes()==b.read_bytes(),'B0/B1 scene JSON differs')
        paths=[a.relative_to(ROOT/'Assets')]
        spec=read(a)
        for obj in spec['objects']:
            mesh=Path(obj['stiff_mesh'])
            paths.append(mesh)
            suffix='.msh' if obj['dimension']==3 else '.obj'
            stem=mesh.stem+'_sorted.16'
            if obj['body_type']=='FEM':
                paths.extend([Path('sorted_mesh')/(stem+suffix),Path('sorted_mesh')/(stem+'.part')])
        for rel in sorted(set(paths)):
            aa,bb=ROOT/'Assets'/rel,ROOT/'baseline/Assets'/rel
            require(aa.read_bytes()==bb.read_bytes(),'B0/B1 mesh/partition bytes differ: '+str(rel))
            inputs.extend([record(aa),record(bb)])
    session.mkdir(parents=True)
    write_new(session/'seal.json',dict(schema='innovation_m0_local.v1',programs=dict(active=active,base=base),
        inputs=inputs,controller=record(Path(__file__)),power_guid=power_guid(),
        planned={stage:tasks(stage) for stage in ('smoke','prefix','matched','reference')},
        limitations=['Windows WDDM desktop','Two diagnostic repeats, not performance statistics',
                    'Frozen B0 has no actual-velocity/resolved-config/checkpoint capability'],
        performance_certified=False,physical_quality_certified=False))
    print(json.dumps(dict(sealed=True,session=str(session),active_sha256=active['exe']['sha256'],base_sha256=base['exe']['sha256'])))


def validate_restore(folder,checkpoint,identity,target):
    meta=read(checkpoint/'checkpoint.json')
    report=read(folder/'output/checkpoint_restore.json')
    require(meta['binary_id']==identity['exe']['sha256'] and meta['next_physical_frame']==target,
        'Wrong checkpoint binary/frame')
    require(report['device_bytes_verified'] and report['context_verified'] and
        report['restored_buffers']==len(meta['buffers']),'Incomplete native restore')
    for member,trace in [('vertices','state_0000.bin'),('velocities','velocity_0000.bin')]:
        require((checkpoint/(member+'.bin')).read_bytes()==(folder/'trace'/trace).read_bytes(),
            'Initial restored position/actual velocity differs')
    context=read(folder/'trace/frame_context.json')
    require(context['completed_physical_frames_at_start']==target-1,'Wrong restored physical frame')
    return dict(passed=True,checkpoint_metadata=record(checkpoint/'checkpoint.json'),report=report)


def observation(folder,t):
    frames=read(folder/'output/stats.json')['frames']
    pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    require(not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg),'PCG hard failure')
    exits=Counter(f.get('newton_exit') for f in frames)
    require(not any(x in exits for x in ('iteration_limit',None)),'Missing/failed Newton exit')
    if t['binary']=='active':
        require(all('phase_ms' in n for f in frames for n in f['newton']),'Missing native Newton phase')
    with (folder/'trace/frames.csv').open(newline='') as stream:timings=list(csv.DictReader(stream))
    return dict(N=len(pcg),P=sum(p['iterations'] for p in pcg),frames=len(frames),exits=dict(exits),
        solver_ms=sum(float(r['solver_ms']) for r in timings),
        phase_ms={phase:sum(n.get('phase_ms',{}).get(phase,0) for f in frames for n in f['newton'])
                  if t['binary']=='active' else None for phase in PHASES},
        true_linear_residual_available=False,accepted_alpha=[n['alpha'] for f in frames for n in f['newton'] if 'alpha' in n],
        scalar_observation_scope='rho ratios are preconditioned native diagnostics, not explicit true linear residuals')


def run_stage(session,stage):
    packet=read(session/'seal.json')
    require(packet['controller']==record(Path(__file__)),'Controller changed since seal')
    require(packet['planned'][stage]==tasks(stage),'Declared tasks changed')
    path=session/(stage+'_ledger.json')
    require(not path.exists(),'Stage exists; no implicit retry')
    if stage!='smoke':require(read(session/'smoke_ledger.json')['complete'],'Smoke gate required')
    if stage=='matched':require(read(session/'prefix_ledger.json')['complete'],'Prefix gate required')
    expected=packet['power_guid'];rows=[]
    original_env,original_query=windows_runner.environment,windows_runner.gpu_query
    try:
        with windows_runner.gpu_lock(ROOT):
            for t in tasks(stage):
                identity=packet['programs'][t['binary']]
                verify(packet['inputs']+[packet['controller']])
                require(power_guid()==expected,'Power scheme changed before run')
                samples=[]
                def query(gpu,timeout=10):
                    sample=original_query(gpu,timeout=timeout)
                    guid=power_guid();samples.append(dict(epoch_seconds=time.time(),guid=guid))
                    require(guid==expected,'Power scheme changed during run')
                    return sample
                def env(c,out):
                    value=environment(c,out)
                    value.update(GIPC_RECORD_TIMING='1',GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
                    if 'checkpoint_save_frame' in t:
                        value.update(GIPC_CHECKPOINT_FRAME=str(t['checkpoint_save_frame']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
                    if 'checkpoint_from' in t:
                        meta=read(session/t['checkpoint_from']/'checkpoint.json')
                        require(meta['binary_id']==identity['exe']['sha256'],'Cross-binary checkpoint rejected')
                        value['GIPC_CHECKPOINT_LOAD']=str(session/t['checkpoint_from'])
                    return value
                windows_runner.environment,windows_runner.gpu_query=env,query
                result=windows_runner.execute(session,t,identity,0)
                folder=session/t['name']
                write_new(folder/'power_samples.json',dict(expected=expected,samples=samples,after=power_guid()))
                row=dict(task=t,result=result)
                try:
                    require(result['status']=='completed' and result.get('cleanup_owned_job_empty'),'Native run/cleanup failed')
                    require(power_guid()==expected,'Power changed after run')
                    row['observation']=observation(folder,t)
                    if 'checkpoint_from' in t:
                        row['restore']=validate_restore(folder,session/t['checkpoint_from'],identity,t['target_frame'])
                    if 'checkpoint_save_frame' in t:
                        meta=read(folder/'checkpoint/checkpoint.json')
                        require(meta['next_physical_frame']==t['checkpoint_save_frame'],'Saved off-by-one checkpoint')
                        row['checkpoint_metadata']=record(folder/'checkpoint/checkpoint.json')
                except Exception as error:row['error']=type(error).__name__+': '+str(error)
                rows.append(row)
                with (session/(stage+'_journal.jsonl')).open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                small=row.get('observation',{})
                print(json.dumps(dict(stage=stage,run=t['name'],status=result['status'],error=row.get('error'),
                    N=small.get('N'),P=small.get('P'),solver_ms=small.get('solver_ms')),ensure_ascii=False),flush=True)
                if 'error' in row:break
    finally:
        windows_runner.environment,windows_runner.gpu_query=original_env,original_query
    complete=len(rows)==len(tasks(stage)) and all('error' not in r for r in rows)
    write_new(path,dict(stage=stage,runs=rows,skipped=tasks(stage)[len(rows):],complete=complete,
        performance_certified=False,physical_quality_certified=False))
    require(complete,'Stage incomplete; evidence retained and further stages stopped')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal','smoke','prefix','matched','reference','all'))
    parser.add_argument('--session',type=Path,default=ROOT/'runs/innovation_m0_20261014')
    args=parser.parse_args()
    session=args.session.resolve()
    require(session.is_relative_to(ROOT/'runs') and session!=ROOT/'runs','Session outside private runs directory')
    if args.stage=='seal':seal(session)
    elif args.stage=='all':
        for stage in ('smoke','prefix','matched','reference'):run_stage(session,stage)
    else:run_stage(session,args.stage)


if __name__=='__main__':main()
