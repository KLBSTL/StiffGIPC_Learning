"""Local, serial historical Graph comparison with retained build identities."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/paper_suite'), str(ROOT/'tools/local'),
               str(ROOT/'tools/bench'), str(ROOT/'tools/diagnostic')]
from config import expand, read, digest, environment
from linux_runner import require, write_new
from local_identity import record, verify
from local_smoke import local_program_identity
import windows_runner
from toi_pcg_audit import summary
from quality_analysis import input_comparison
from pool_metrics import state_comparison

SCENES = ('cloth_hang_l', 'cloth_fixed_bunny_l')
ARMS = ('frozen_stiff', 'current_host', 'current_graph', 'historical_best',
        'current_best', 'current_ours')


def power():
    return subprocess.check_output(['powercfg', '/getactivescheme']).decode(
        'utf-8', errors='replace').strip()


def identities():
    active = local_program_identity(ROOT/'build/paper_suite_20261010')
    historical = read(ROOT/'build/audit_corrected_frozen/snapshot_identity.json')
    verify(historical['sources'] + historical['build_evidence'] + historical['objects']
           + [historical['exe']] + historical['dlls'])
    # The native snapshot remains frozen. Four baseline runtime input caches
    # were restored later; the current clean-build identity audits these exact
    # additions against their retained baseline sources.
    historical = dict(historical, runtime_input_additions=active['runtime_input_additions'])
    base = read(ROOT/'reports/report_execution_20261007/FROZEN_REFERENCE.json')['programs']['base']
    verify(base['sources'] + base['build_evidence'] + [base['exe']] + base['dlls'])
    return {'active': active, 'historical': historical, 'base': base}


def task(scene, arm, repeat, frames, stage):
    combined = arm in ('historical_best', 'current_best', 'current_ours')
    cfg = dict(scene=scene, steps=frames, dt=.01, timeout_seconds=240,
               ipc_newton_tol=.01, pcg_rho_tol=1e-4, ipc_min_updates=6,
               ipc_termination='legacy', ipc_cumulative_tol=.01,
               preset='combined' if combined else 'graph' if arm=='current_graph' else 'base',
               fem_direct_pullback=combined, mas_pivot_only_sync=combined,
               discrete_bvh_refit=combined, trace_velocity=arm!='frozen_stiff')
    if arm=='current_ours':
        cfg.update(ipc_termination='compensated', ipc_cumulative_tol=.001,
                   ipc_residual_rel_tol=.03)
    return dict(name=f'{stage}_{scene}_{arm}_r{repeat}', scene=scene, arm=arm,
                repeat=repeat, binary='base' if arm=='frozen_stiff' else 'active',
                identity_key='base' if arm=='frozen_stiff' else 'historical' if arm=='historical_best' else 'active',
                config=expand(cfg))


def env(config, out):
    value = environment(config, out)
    value['GIPC_IPC_PCG_FORCING'] = 'off'
    return value


def run(session_name, stage):
    session = ROOT/'runs'/session_name
    session.mkdir(parents=True, exist_ok=True)
    require(not (session/(stage+'_ledger.json')).exists(), 'Stage exists; no overwrite or implicit retry')
    ids = identities()
    packet = dict(programs=ids, controller=record(__file__), power=power(),
                  scenes=SCENES, arms=ARMS, frames=100, repeats=3,
                  early_v4_exact_recovery=False, scope='Local shared WDDM desktop measurements')
    path = session/'identities.json'
    if path.exists():
        require(read(path)==json.loads(json.dumps(packet)), 'Program/controller/power identity changed')
    else:
        write_new(path, packet)
        shutil.copyfile(__file__, session/'controller_at_run.py')
    if stage=='performance':
        previous = read(session/'smoke_ledger.json')
        require(previous['complete'], 'Complete smoke checks required')
    tasks=[]
    for repeat in range(1, 2 if stage=='smoke' else 4):
        order = ARMS[repeat-1:] + ARMS[:repeat-1]
        scenes = SCENES if repeat%2 else tuple(reversed(SCENES))
        for scene in scenes:
            tasks.extend(task(scene, arm, repeat, 2 if stage=='smoke' else 100, stage) for arm in order)
    write_new(session/(stage+'_tasks.json'), dict(tasks=tasks))
    rows=[]
    original = windows_runner.environment
    try:
        windows_runner.environment = env
        with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
            for t in tasks:
                result=windows_runner.execute(session, t, ids[t['identity_key']], 0)
                row=dict(task=t, result=result)
                rows.append(row)
                if result['status']!='completed':
                    break
                try:
                    row['summary']=summary(session/t['name'],result)
                except Exception as error:
                    row['analysis_error']=f'{type(error).__name__}: {error}'
                    break
                with (session/(stage+'_journal.jsonl')).open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps(dict(name=t['name'], **row['summary'])),flush=True)
    finally:
        windows_runner.environment = original
    final=dict(stage=stage, runs=rows, skipped=tasks[len(rows):],
               complete=len(rows)==len(tasks) and all('summary' in row for row in rows),
               power_after=power())
    write_new(session/(stage+'_ledger.json'), final)
    verify([packet['controller']])
    require(final['power_after']==packet['power'], 'Power mode changed during stage')
    print(json.dumps(dict(stage=stage, complete=final['complete'], runs=len(rows))),flush=True)
    return 0 if final['complete'] else 2


def analyze(session_name):
    session=ROOT/'runs'/session_name
    ledger=read(session/'performance_ledger.json')
    require(ledger['complete'], 'Only complete formal matrix may be summarized')
    rows=ledger['runs']; pairs=[]; grouped=[]
    for scene in SCENES:
        for repeat in range(1,4):
            by={r['task']['arm']:r for r in rows if r['task']['scene']==scene and r['task']['repeat']==repeat}
            for reference,candidate in [('frozen_stiff',a) for a in ARMS[1:]] + [
                ('current_host','current_graph'),('historical_best','current_best'),
                ('current_graph','current_best'),('current_best','current_ours')]:
                a,b=by[reference],by[candidate]
                initial=input_comparison(session/a['task']['name'],session/b['task']['name'])
                require(initial['passed'], 'Different initial inputs')
                pairs.append(dict(scene=scene,repeat=repeat,reference=reference,candidate=candidate,
                    speedup=a['summary']['solver_seconds']/b['summary']['solver_seconds'],
                    input=initial,trajectory=state_comparison(session/a['task']['name'],session/b['task']['name'])))
        for arm in ARMS:
            runs=[r for r in rows if r['task']['scene']==scene and r['task']['arm']==arm]
            values={k:[r['summary'][k] for r in runs] for k in
                    ('solver_seconds','directions','pcg','pcg_per_direction','max_stretch','p99_stretch','graph_captures','graph_capture_ms')}
            s=next((p for p in pairs if p['scene']==scene and p['candidate']==arm and p['reference']=='frozen_stiff'),None)
            grouped.append(dict(scene=scene,arm=arm,medians={k:statistics.median(v) for k,v in values.items()},raw=values,
                speedup_median=None if s is None else statistics.median(p['speedup'] for p in pairs if p['scene']==scene and p['candidate']==arm and p['reference']=='frozen_stiff')))
    write_new(session/'analysis.json',dict(rows=grouped,pairs=pairs,physical_quality_certified=False,
        early_v4_exact_recovery=False,scope='Three paired full 100-frame local observations; shared WDDM desktop'))
    print(json.dumps(grouped,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('smoke','performance','analyze'))
    parser.add_argument('--session',default='local_graph_history_20261011')
    args=parser.parse_args()
    raise SystemExit(analyze(args.session) if args.stage=='analyze' else run(args.session,args.stage))
