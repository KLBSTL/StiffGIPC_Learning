"""Focused read-only analysis and bounded native diagnostics; no solver edits."""
from pathlib import Path
import argparse
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in ('bench', 'diagnostic', 'local', 'full_eval'):
    sys.path.insert(0, str(ROOT/'tools'/folder))
from config import read, expand
from linux_runner import require, write_new
from local_identity import record, verify, verify_active_sources
import windows_runner
from toi_pcg_audit import summary
from quality_analysis import input_comparison
from pool_metrics import state_comparison


def scan(out):
    origin = ROOT/'runs/audit_v4_toi_120'
    data = {}
    evidence = []
    for arm in ('base', 'toi'):
        path = origin/f'bunny_cloth_bunny_l_{arm}_r1/output/stats.json'
        evidence.append(record(path))
        frames = read(path)['frames']
        rows = []
        for index, frame in enumerate(frames, 1):
            directions = [n for n in frame['newton'] if 'pcg' in n]
            outer = frame.get('toi', [])
            rows.append(dict(frame=index, directions=len(directions),
                pcg=sum(n['pcg']['iterations'] for n in directions),
                pcg_max=max((n['pcg']['iterations'] for n in directions), default=0),
                outer=len(outer), inner_max=max((s['inner_iterations'] for s in outer), default=0),
                mu=[s['mu'] for s in outer],
                selected_safe=sum(s.get('initial_guess_selection', {}).get('selected')=='safe' for s in outer),
                restart_floor=sum(s.get('restart_minimum_inner_iterations', 0)==6 for s in outer),
                restart_blocked=sum(n.get('restart_full_step_blocked', False) for n in directions),
                blocked_pcg=sum(n['pcg']['iterations'] for n in directions if n.get('restart_full_step_blocked', False)),
                half_steps=sum(n.get('line_search_r', 1)<1 for n in directions),
                contacts=max((s['active_self']+s['active_ground'] for s in outer), default=0)))
        data[arm] = rows
    paired = []
    for a, b in zip(data['base'], data['toi']):
        paired.append(dict(b, base_directions=a['directions'], base_pcg=a['pcg'],
            pcg_ratio=b['pcg']/a['pcg'] if a['pcg'] else None))
    all_mu = [v for r in data['toi'] for v in r['mu']]
    report = dict(evidence=evidence, frames=paired,
        totals={arm:{key:sum(r[key] for r in rows) for key in
            ('directions','pcg','outer','selected_safe','restart_floor','restart_blocked','blocked_pcg','half_steps')}
            for arm,rows in data.items()},
        mu_min=min(all_mu),mu_max=max(all_mu),
        top_frames=sorted(paired,key=lambda r:r['pcg'],reverse=True)[:10],
        cautions=['Blocked-step count is descriptive, not a counterfactual speedup.',
                  'The paired trajectories are not certified physically equivalent.'])
    write_new(out/'scan.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('frames','evidence')}, indent=2))


def capture(out, identity_path):
    identity = read(identity_path)
    verify_active_sources(identity)
    verify(identity['build_evidence']+identity.get('objects', [])+[identity['exe']]+identity['dlls'])
    controller = record(__file__)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    tasks = []
    for name, last, first, second, directions in (
        ('stall55',55,25,65,'1,50,90'),
        ('linear113',113,2,3,'2,3,8,9')):
        cfg = expand(dict(preset='toi', scene='bunny_cloth_bunny_l', steps=last,
            timeout_seconds=180, mas='legacy', fem_direct_pullback=True,
            mas_pivot_only_sync=True, diagnostics=['fixed','outer_probe'],
            cost_frames=str(last), fixed_frames=str(last), fixed_directions=directions))
        extra = dict(GIPC_PROFILE='1',GIPC_DIRECTION_EVENT_FROM=str(last),
            GIPC_DIRECTION_EVENT_FIRST=str(first),GIPC_DIRECTION_EVENT_SECOND=str(second),
            GIPC_DIRECTION_EVENT_VELOCITY='0',GIPC_DIRECTION_EVENT_DIR=str(out/name/'events'))
        tasks.append(dict(name=name,binary='active',config=cfg,diagnostic_environment=extra))
    write_new(out/'identity.json', dict(program=identity, controller=controller,
        tasks=tasks, scope='Existing v4 observer and fixed-system diagnostics; no native source changes.'))
    original_environment = windows_runner.environment
    ledger = []
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            # Create event output after execute has prepared its fresh directory.
            def environment(config, path, extra=task['diagnostic_environment']):
                (path/'events').mkdir()
                return dict(original_environment(config,path), **extra)
            windows_runner.environment = environment
            try:
                result = windows_runner.execute(out,task,identity,0)
            finally:
                windows_runner.environment = original_environment
            row = dict(task=task,result=result)
            ledger.append(row)
            if result['status']!='completed':
                break
            try:
                row['summary']=summary(out/task['name'],result)
            except Exception as error:
                row['analysis_error']=str(error)
                break
            print(json.dumps(dict(name=task['name'],summary=row['summary'])),flush=True)
    verify([controller])
    write_new(out/'ledger.json',dict(runs=ledger,skipped=tasks[len(ledger):]))
    return 0 if len(ledger)==len(tasks) and all('summary' in r for r in ledger) else 2


def controls(out,identity_path,remaining=False,diagonal=False,profile=False):
    """Existing native options; explicitly separate algorithm controls."""
    active=read(identity_path)
    verify_active_sources(active)
    base=read(ROOT.parent/'StiffGIPC_Learning/runs/report56_stage1_20261008/REFERENCE_IDENTITY.json')['programs']['base']
    for program in (active,base):
        verify(program['sources']+program['build_evidence']+program.get('objects',[])+[program['exe']]+program['dlls'])
    programs=dict(active=active,base=base);controller=record(__file__)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    tasks=[]
    definitions=(
        ('base','base',1,False),('toi_default','active',1,True),
        ('toi_mu01','active',.1,True),('toi_warm_only','active',1,False),
        ('toi_default_repeat','active',1,True))
    reference=out/'base'
    if remaining:
        definitions=definitions[3:]
        reference=ROOT/'runs/deep_toi_controls_20261009/base'
        require(read(reference/'result.json')['status']=='completed','Completed reference required')
    if diagonal:
        definitions=(('toi_diagonal_profile' if profile else 'toi_diagonal','active',1,True),)
        reference=ROOT/'runs/deep_toi_controls_20261009/base'
        require(read(reference/'result.json')['status']=='completed','Completed reference required')
    for name,arm,scale,choose in definitions:
        cfg=expand(dict(preset='host' if arm=='base' else 'toi',scene='bunny_cloth_bunny_l',
            steps=120,timeout_seconds=180,mas='legacy',
            fem_direct_pullback=arm!='base',mas_pivot_only_sync=arm!='base',
            choose_start=choose,restart_guard=choose))
        extra={'GIPC_TOI_MU_SCALE':str(scale)}
        if diagonal:extra['GIPC_PCG_PRECONDITIONER']='diag'
        if profile:extra['GIPC_PROFILE']='1'
        tasks.append(dict(name=name,binary=arm,config=cfg,
            diagnostic_environment=extra,
            algorithm_changed=scale!=1 or (arm!='base' and not choose),expected_mu_scale=scale))
    write_new(out/'identity.json',dict(programs=programs,controller=controller,tasks=tasks,
        reference=str(reference),
        scope='Bounded existing-option causal controls, no native source changes; no profile or fixed-study overhead.'))
    original=windows_runner.environment;ledger=[]
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            def environment(config,path,extra=task['diagnostic_environment']):
                return dict(original(config,path),**extra)
            windows_runner.environment=environment
            try:result=windows_runner.execute(out,task,programs[task['binary']],0)
            finally:windows_runner.environment=original
            row=dict(task=task,result=result);ledger.append(row)
            if result['status']!='completed':break
            try:
                row['summary']=summary(out/task['name'],result)
                if task['binary']=='active':
                    resolved=read(out/task['name']/'resolved_config.json')
                    require(resolved['toi']['mu_scale']==task['expected_mu_scale'],'Native penalty scale mismatch')
                    if diagonal:require(resolved['effective_preconditioner']=='diag','Native diagonal mode mismatch')
                if task['name']!='base':
                    row['initial_input']=input_comparison(reference,out/task['name'])
                    require(row['initial_input']['passed'],'Initial input mismatch')
                    row['trajectory']=state_comparison(reference,out/task['name'])
            except Exception as error:
                row['analysis_error']=str(error);break
            print(json.dumps(dict(name=task['name'],summary=row['summary'])),flush=True)
    verify([controller]);write_new(out/'ledger.json',dict(runs=ledger,skipped=tasks[len(ledger):]))
    return 0 if len(ledger)==len(tasks) and all('summary' in r and 'analysis_error' not in r for r in ledger) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('scan','capture','controls'))
    p.add_argument('--session',required=True)
    p.add_argument('--identity',type=Path)
    p.add_argument('--remaining-controls',action='store_true')
    p.add_argument('--diagonal-only',action='store_true')
    p.add_argument('--native-profile',action='store_true')
    args=p.parse_args()
    require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session
    require(not out.exists(),'Fresh output required')
    out.mkdir(parents=True)
    if args.action=='scan':
        scan(out)
        return 0
    require(args.identity is not None,'Frozen identity required')
    return capture(out,args.identity) if args.action=='capture' else controls(out,args.identity,args.remaining_controls,args.diagonal_only,args.native_profile)


if __name__=='__main__':
    raise SystemExit(main())
