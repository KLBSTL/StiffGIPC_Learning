"""Serial local seven-scene comparison of retained and current execution combinations."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from config import expand,environment,read
from linux_runner import require,write_new
from local_identity import record,verify
from compare_graph_history import identities,power
from protocol import SCENES,KEYFRAMES,CHECKPOINTS,input_inventory
from run_suite import checkpoint_evidence
from analyze import observation,checkpoint_gate
import windows_runner

ARMS=('stiff','old_host','old','host','ours')
LABELS={'stiff':'StiffGIPC','old_host':'Frozen combination (Host)',
        'old':'Frozen 2.084x combination (Graph)','host':'Current combination (Host)',
        'ours':'Current combination (Graph)','original':'Movement only (Graph)',
        'pure':'Cumulative TOI .01 (Graph)','gate':'Compensated residual gate .01 (Graph)',
        'strict':'Movement only, rho=1e-6 (Graph)'}


def task(scene,arm,repeat,steps=100,stage='performance'):
    combined=arm!='stiff'
    cfg=dict(scene=SCENES[scene],steps=steps,dt=.01,timeout_seconds=600,
        preset='combined' if combined else 'base',
        execution='host' if arm in ('stiff','old_host','host') else 'conditional_graph',
        ipc_newton_tol=.01,pcg_rho_tol=1e-4,ipc_min_updates=6,
        ipc_termination='legacy',ipc_cumulative_tol=.01,
        fem_direct_pullback=combined,mas_pivot_only_sync=combined,
        discrete_bvh_refit=combined,trace_velocity=arm!='stiff')
    # The main performance arm "ours" is the execution combination requested
    # by the user. Only matched one-step "ours" is a residual-gate diagnostic.
    if stage in ('seed','matched'):
        if arm in ('original','strict'):cfg['ipc_termination']='movement_only'
        if arm=='strict':cfg['pcg_rho_tol']=1e-6
        if arm=='ours':cfg.update(ipc_termination='compensated',ipc_residual_rel_tol=.03)
    t=dict(name=f'{stage}_{scene.lower()}_{arm}_r{repeat}',scene_key=scene,arm=arm,
        repeat=repeat,binary='base' if arm=='stiff' else 'active',
        identity_key='base' if arm=='stiff' else 'historical' if arm in ('old','old_host') else 'active',
        config=expand(cfg))
    if stage=='seed':t['checkpoint_save_frame']=CHECKPOINTS[scene]
    if stage=='matched':t.update(target_frame=CHECKPOINTS[scene],checkpoint_from=f'seed_{scene.lower()}_original_r1/checkpoint')
    return t


def plan():
    performance=[]
    for repeat in (1,2,3):
        arms=ARMS[repeat-1:]+ARMS[:repeat-1]
        scenes=list(SCENES) if repeat%2 else list(reversed(SCENES))
        for scene in scenes:performance.extend(task(scene,arm,repeat) for arm in arms)
    matched=[]
    for scene in CHECKPOINTS:
        for repeat in (1,2,3):
            arms=('original','pure','ours','strict')
            arms=arms[repeat-1:]+arms[:repeat-1]
            matched.extend(task(scene,arm,repeat,1,'matched') for arm in arms)
    return dict(schema='local_combination_suite.v1',scenes=SCENES,arms=ARMS,labels=LABELS,
        frames=100,repeats=3,keyframes=KEYFRAMES,checkpoints=CHECKPOINTS,
        stages=dict(smoke=[task(s,a,1,2,'smoke') for s in SCENES for a in ARMS],
            performance=performance,seed=[task(s,'original',1,CHECKPOINTS[s],'seed') for s in CHECKPOINTS],matched=matched),
        definitions=dict(dt='Uniform .01, including Hang and Stack; rho=1e-4 throughout main matrix',
            main='Retained exact 2.084x binary and current binary, same complete execution optimizations and legacy cumulative threshold .01',
            graph='Within each binary, complete combination with host versus Graph; execution is the only differing expanded field',
            checkpoint='Current binary only; shared cold checkpoint: movement_only, legacy cumulative .01, compensated .01 plus residual .03',
            reference='Movement-only rho=1e-6, numerical reference rather than exact solution',
            stopping='Movement threshold .01; min_updates=6 applies only to additional exit, movement can exit sooner',
            timing='Synchronized native solver time, including graph capture; initialization/trace export excluded; profile=none'),
        scope='Local shared WDDM desktop; no power changes, no native edits; three paired observations',
        performance_certified=False,physical_quality_certified=False)


def checkpoint_env(c,out,t,identity,session):
    value=environment(c,out)
    value.update(GIPC_IPC_PCG_FORCING='off',GIPC_BINARY_ID=identity['exe']['sha256'])
    if 'checkpoint_save_frame' in t:
        value.update(GIPC_CHECKPOINT_FRAME=str(t['checkpoint_save_frame']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
    if 'checkpoint_from' in t:
        cp=session/t['checkpoint_from'];meta=read(cp/'checkpoint.json')
        require(meta['binary_id']==identity['exe']['sha256'] and meta['next_physical_frame']==t['target_frame'],'Wrong checkpoint identity/frame')
        value['GIPC_CHECKPOINT_LOAD']=str(cp)
    return value


def verify_restore(session,t):
    out=session/t['name'];cp=session/t['checkpoint_from']
    restore=read(out/'output/checkpoint_restore.json');meta=read(cp/'checkpoint.json')
    require(restore['device_bytes_verified'] and restore['context_verified'],'Native checkpoint restore failed')
    require(restore['restored_buffers']==len(meta['buffers']),'Missing restored buffers')
    for name,trace in [('vertices','state_0000.bin'),('velocities','velocity_0000.bin')]:
        require((cp/(name+'.bin')).read_bytes()==(out/'trace'/trace).read_bytes(),'Initial checkpoint bytes differ')
    stats=read(out/'output/stats.json')['frames']
    require(all(n['pcg']['frame']==t['target_frame'] for f in stats for n in f['newton'] if 'pcg' in n),'Wrong physical frame context')


def run(session_name,stage):
    session=ROOT/'runs'/session_name;session.mkdir(parents=True,exist_ok=True)
    p=plan();ids=identities()
    packet=dict(plan=p,programs=ids,controller=record(__file__),power=power(),inputs=input_inventory())
    if (session/'identities.json').exists():
        require(read(session/'identities.json')==json.loads(json.dumps(packet)),'Program, controller, plan, inputs or power changed')
    else:
        write_new(session/'identities.json',packet);shutil.copyfile(__file__,session/'controller_at_run.py')
    if stage!='smoke':require(read(session/'smoke_ledger.json')['complete'],'Smoke gate required')
    stages=('seed','matched') if stage=='checkpoint' else (stage,)
    original=windows_runner.environment
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for part in stages:
            require(not (session/(part+'_ledger.json')).exists(),'Stage exists; no overwrite or implicit retry')
            rows=[];tasks=p['stages'][part];write_new(session/(part+'_tasks.json'),dict(tasks=tasks))
            for t in tasks:
                identity=ids[t['identity_key']]
                if 'checkpoint_from' in t:checkpoint_evidence(session/t['checkpoint_from'])
                windows_runner.environment=lambda c,out,t=t,identity=identity:checkpoint_env(c,out,t,identity,session)
                try:result=windows_runner.execute(session,t,identity,0)
                finally:windows_runner.environment=original
                row=dict(task=t,result=result);rows.append(row)
                if result['status']=='completed':
                    try:
                        if part=='matched':verify_restore(session,t)
                        row['observation']=observation(session/t['name'],t,result)
                        if identity['capabilities'].get('actual_velocity'):
                            from pool_metrics import export_evidence
                            require(export_evidence(session/t['name'],t['config']['steps'],'velocity')['passed'],'Incomplete actual velocity exports')
                    except Exception as error:row['analysis_error']=f'{type(error).__name__}: {error}'
                with (session/(part+'_journal.jsonl')).open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                if 'observation' in row:
                    m=row['observation'];print(json.dumps(dict(stage=part,run=t['name'],seconds=m['TimeTot'],N=m['N_tot'],P=m['P_tot'])),flush=True)
                else:break
            ledger=dict(stage=part,runs=rows,skipped=tasks[len(rows):],
                complete=len(rows)==len(tasks) and all('observation' in r for r in rows),power_after=power())
            write_new(session/(part+'_ledger.json'),ledger)
            require(ledger['power_after']==packet['power'],'Power changed during stage')
            if not ledger['complete']:return 2
            if part=='matched':write_new(session/'checkpoint_input_gate.json',checkpoint_gate(session,ledger))
    for identity in ids.values():verify(identity['sources']+identity['build_evidence']+[identity['exe']]+identity['dlls']+identity.get('objects',[]))
    print(json.dumps(dict(stage=stage,complete=True)),flush=True)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','smoke','performance','checkpoint'))
    parser.add_argument('--session',default='local_combination_seven_20261011')
    args=parser.parse_args()
    if args.stage=='plan':print(json.dumps(plan(),ensure_ascii=False,indent=2))
    else:raise SystemExit(run(args.session,args.stage))
