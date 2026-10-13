"""Finite IPC forcing probes using the existing bounded Windows runner."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in ('bench', 'diagnostic', 'local', 'full_eval'):
    sys.path.insert(0, str(ROOT/'tools'/folder))
from config import read, expand
from linux_runner import require, write_new
from local_identity import verify, verify_active_sources, record
import windows_runner
from toi_pcg_audit import summary
from quality_analysis import input_comparison
from pool_metrics import state_comparison


def cfg(scene, steps, **extra):
    return expand(dict(preset='combined', scene=scene, steps=steps,
        timeout_seconds=180, mas='legacy', fem_direct_pullback=True,
        mas_pivot_only_sync=True, discrete_bvh_refit=True, **extra))


def forcing_stats(path, known_metadata_defect_probe=False):
    requested=read(path/'requested.json')['environment'].get('GIPC_IPC_PCG_FORCING','off')
    resolved=read(path/'resolved_config.json')['ipc_pcg_forcing']
    declaration_valid=isinstance(resolved,dict)
    if not declaration_valid:
        require(known_metadata_defect_probe and read(path/'requested.json')['exe_sha256']==
            '0f4139500ead3bc47cfd5587d54414094797f614076c88bed1e6a8a0d56931a3',
            'Malformed feature declaration blocks normal validation')
        # Continue only physical-recovery diagnosis of the already recorded v4
        # defect. Missing constant keys are not reconstructed or declared valid.
        resolved={item[0]:item[1] for item in resolved if len(item)==2}
    require(resolved['mode']==requested and resolved['previous_rho_timing_retained'] and
        resolved['legacy_outer_stopping_retained'] and resolved['physical_error_certified'] is False,
        'Forcing feature declaration disagrees with requested scope')
    if declaration_valid:
        require(resolved['eta_min']==.01 and resolved['eta_max']==.1 and
            resolved['near_reference_ratio']==.03,'Forcing constant fields differ or are missing')
    rows = [n for f in read(path/'output/stats.json')['frames'] for n in f['newton'] if 'pcg' in n]
    decisions = [n['ipc_pcg_forcing'] for n in rows if 'ipc_pcg_forcing' in n]
    require((not decisions) if requested=='off' else len(decisions)==len(rows),
        'Forcing not observed on every production solve')
    retries = sum(n.get('baseline_retry_count', 0) for n in decisions)
    attempts = [a for n in rows for a in n['pcg'].get('forcing_attempts', [])]
    for n in rows:
        if 'ipc_pcg_forcing' in n:
            d=n['ipc_pcg_forcing']
            require(d['mode']==requested and .01<=d['eta_accepted']<=.1 and
                abs(n['pcg']['rho_tolerance']-d['eta_accepted']**2)<1e-15,
                'Requested/accepted eta and actual solver tolerance disagree')
        if 'forcing_attempts' in n['pcg']:
            require(n['pcg']['iterations']==sum(a['pcg'].get('iterations',0) for a in n['pcg']['forcing_attempts']),
                'Retry PCG work was omitted')
        if 'forcing_rejected_trial' in n:
            require(n['energy_evaluations']>n['forcing_rejected_trial']['energy_evaluations'],
                'Rejected trial energy work was omitted')
    return dict(feature_declaration_valid=declaration_valid,directions=len(rows), baseline_retries=retries,
        retry_fraction=retries/len(rows) if rows else 0,
        attempts=len(attempts), failed_attempts=sum(a['failed'] for a in attempts),
        eta_accepted_counts={str(eta):sum(n.get('eta_accepted')==eta for n in decisions)
            for eta in sorted({n['eta_accepted'] for n in decisions})},
        objective_epochs=max((n['objective_epoch'] for n in decisions),default=0),
        rollback_checks=[n['forcing_rollback_bitwise_verified'] for n in rows if 'forcing_rollback_bitwise_verified' in n],
        attempt_reasons=sorted({a['reason'] for a in attempts}),
        true_checks_passed=all(n['pcg'].get('forcing_direction_check', {'passed': True})['passed'] for n in rows))


def bounds_check(scene, observed):
    bounds = read(ROOT/'reports/AUDIT_REFERENCE_OBSERVATION.json')['scenes'][scene]['bounds']
    failures = {key:dict(observed=observed[key], bound=value) for key,value in bounds.items()
        if value is not None and (observed[key] < value if key=='fem_min_J' else observed[key] > value)}
    return dict(passed=not failures, failures=failures, frozen_bounds=bounds,
        quality_certified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('guards', 'steps', 'chunk_restore', 'capture', 'screen', 'pairs'))
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--scenes', nargs='+', default=['cloth_fixed_bunny_l'])
    p.add_argument('--arms', nargs='+', choices=('off','monitor','fixed','adaptive'), default=['off','monitor','fixed','adaptive'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--known-metadata-defect-probe',action='store_true',
        help='v4 guards only: diagnose physical recovery while preserving the known failed metadata contract')
    args = p.parse_args()
    require(args.session.replace('_','').isalnum(), 'Invalid session')
    require(1<=args.repeats<=7, 'Repeat budget')
    require(not args.known_metadata_defect_probe or args.phase=='guards',
        'Known metadata defect cannot be used for timing/capture/acceptance')
    out = ROOT/'runs'/args.session
    require(not out.exists(), 'Fresh session required')
    out.mkdir(parents=True)
    ident = read(args.identity)
    verify_active_sources(ident)
    verify(ident['build_evidence']+ident.get('objects', [])+[ident['exe']]+ident['dlls'])
    controller = record(__file__)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    tasks = []
    if args.phase=='guards':
        for scene in args.scenes:
            for kind in ('direction','small','line_search'):
                tasks.append(dict(name=scene+'_'+kind, scene=scene, arm='fixed', binary='active',
                    config=cfg(scene, 2), extra_environment=dict(GIPC_IPC_PCG_FORCING='fixed',
                        GIPC_IPC_PCG_ETA='.03', GIPC_IPC_PCG_FAILURE_FIXTURE=kind,
                        GIPC_MAS_SNAPSHOT='1'), fixture=kind))
    elif args.phase=='chunk_restore':
        scene='bunny_cloth_bunny_l'
        tasks.append(dict(name='chunk_restore',scene=scene,arm='off',binary='active',
            config=cfg(scene,2,diagnostics=['fixed'],fixed_frames='1',fixed_directions='1',
                fixed_graph_chunk_study=True),extra_environment=dict(GIPC_IPC_PCG_FORCING='off')))
    elif args.phase=='steps':
        for scene in args.scenes:
            for name,mode,eta in (('baseline','monitor',None),('eta03','fixed','.03'),('eta1','fixed','.1'),
                ('adaptive','adaptive',None)):
                tasks.append(dict(name=scene+'_'+name,scene=scene,arm=name,binary='active',
                    config=cfg(scene,2),extra_environment=dict(GIPC_IPC_PCG_FORCING=mode,
                        **({'GIPC_IPC_PCG_ETA':eta} if eta else {}))))
    elif args.phase=='capture':
        for scene,frame in (('cloth_fixed_bunny_l',57),('bunny_cloth_bunny_l',34)):
            tasks.append(dict(name=scene+'_capture', scene=scene, arm='off', binary='active',
                config=cfg(scene,frame,diagnostics=['fixed'],fixed_frames=str(frame),fixed_directions='1,2,3'),
                extra_environment=dict(GIPC_IPC_PCG_FORCING='off',GIPC_FIXED_PCG_ETA_STUDY='1')))
    else:
        for repeat in range(args.repeats):
            order=args.arms if repeat%2==0 else list(reversed(args.arms))
            for scene in args.scenes:
                for arm in order:
                    tasks.append(dict(name=f'{scene}_{arm}_r{repeat+1}',scene=scene,arm=arm,repeat=repeat+1,
                        binary='active',config=cfg(scene,120),extra_environment=dict(GIPC_IPC_PCG_FORCING=arm,
                            **({'GIPC_IPC_PCG_ETA':'.03'} if arm=='fixed' else {}))))
    write_new(out/'identity.json', dict(program=ident,controller=controller,tasks=tasks,
        performance_certified=False,quality_certified=False,power_mode='unchanged quiet'))
    ledger=[]
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            original=windows_runner.environment
            windows_runner.environment=lambda c,path:dict(original(c,path), **task['extra_environment'])
            try:
                result=windows_runner.execute(out,task,ident,0)
            finally:
                windows_runner.environment=original
            entry=dict(task=task,result=result)
            ledger.append(entry)
            if result['status']!='completed':
                break
            try:
                entry['summary']=summary(out/task['name'],result)
                entry['forcing']=forcing_stats(out/task['name'],args.known_metadata_defect_probe)
                if args.phase in ('screen','pairs'):
                    entry['quality_bounds']=bounds_check(task['scene'],entry['summary'])
                if args.phase=='guards':
                    f=entry['forcing']
                    reason={'direction':'direction_failure_baseline_retry','small':'small_direction_baseline_confirmation',
                        'line_search':'line_search_baseline_retry'}[task['fixture']]
                    require(reason in f['attempt_reasons'] and f['true_checks_passed'], 'Recovery not exercised')
                    if task['fixture']=='line_search':
                        require(f['rollback_checks']==[True], 'Exact rollback not verified once')
                    entry['fixture_passed']=True
                if args.phase=='chunk_restore':
                    from graph_chunk_rounds import analyze, numerical_comparison
                    path=out/task['name']
                    study=read(path/'fixed/f1_n1_graph_chunk_study.json')
                    entry['chunk_restore']=analyze(study)
                    entry['chunk_numerical']=numerical_comparison(study,path)
                    require(entry['chunk_numerical']['within_observed_K1_range_plus_roundoff'],
                        'Chunk diagnostic numerical restoration guard failed')
                if args.phase=='capture':
                    studies=[]
                    for file in sorted((out/task['name']/'fixed').glob('*_study.json')):
                        s=read(file)
                        require(s['eta_study'] and s['system_unchanged'] and s['primary_restored_bitwise'],
                            'Same-system precision study changed operator or primary solution')
                        require(all(not r['error'] and not r['limit'] for r in s['runs']),
                            'Fixed precision solve failed or hit cap')
                        studies.append(dict(file=record(file),runs=s['runs'],system_unchanged=True,
                            primary_restored_bitwise=True))
                    require(len(studies)==3,'Missing fixed Newton systems')
                    entry['fixed_studies']=studies
            except Exception as error:
                entry['analysis_error']=str(error)
                break
            print(json.dumps(dict(name=task['name'],summary=entry['summary'],forcing=entry['forcing'],
                quality_bounds=entry.get('quality_bounds'))),flush=True)
    comparisons=[]
    if args.phase=='steps':
        for scene in args.scenes:
            rows=[r for r in ledger if r['task']['scene']==scene and 'summary' in r]
            if len(rows)!=4:continue
            baseline=out/rows[0]['task']['name']
            initial_residual=None
            for row in rows:
                path=out/row['task']['name']
                initial=input_comparison(baseline,path)
                require(initial['passed'],'Step initial input mismatch')
                newton=read(path/'output/stats.json')['frames'][0]['newton']
                first=newton[0]; second=newton[1]
                r0=first['ipc_pcg_forcing']['physical_residual']
                if initial_residual is None:initial_residual=r0
                comparisons.append(dict(scene=scene,arm=row['task']['arm'],initial_input=initial,
                    initial_gradient_monitor_bitwise_match=r0==initial_residual,
                    initial_gradient_monitor_relative_difference=abs(r0-initial_residual)/max(abs(initial_residual),1e-300),
                    first_accepted_step=dict(eta=first['ipc_pcg_forcing']['eta_accepted'],
                        pcg=first['pcg']['iterations'],physical_residual_before=r0,
                        physical_residual_after=second['ipc_pcg_forcing']['physical_residual'],
                        alpha=first['alpha'],energy_evaluations=first['energy_evaluations'],
                        energy_backtracks=first['energy_backtracks'],
                        intersection_backtracks=first['intersection_backtracks'],
                        direction_check=first['pcg']['forcing_direction_check']),
                    scope='Same native initial state first accepted step; no late-Newton checkpoint replay or speed claim'))
    for scene in args.scenes:
        for repeat in range(1,args.repeats+1):
            arms={r['task']['arm']:r for r in ledger if r['task']['scene']==scene and
                r['task'].get('repeat')==repeat and 'summary' in r}
            if 'off' not in arms:continue
            for arm,b in arms.items():
                if arm=='off':continue
                a=arms['off']; ap,bp=(out/r['task']['name'] for r in (a,b))
                initial=input_comparison(ap,bp)
                require(initial['passed'], 'Initial input mismatch')
                pcga=a['summary']['phase_ms']['pcg']; pcgb=b['summary']['phase_ms']['pcg']
                comparisons.append(dict(scene=scene,repeat=repeat,arm=arm,
                    speedup=a['summary']['solver_seconds']/b['summary']['solver_seconds'],
                    pcg_phase_reduction=1-pcgb/pcga,initial_input=initial,
                    trajectory=state_comparison(ap,bp),retry_fraction=b['forcing']['retry_fraction'],
                    quality_bounds=b['quality_bounds']))
    write_new(out/'ledger.json',dict(runs=ledger,skipped=tasks[len(ledger):],comparisons=comparisons))
    verify([controller,record(out/'controller_at_run.py')])
    print(json.dumps(dict(comparisons=comparisons,complete=len(ledger)==len(tasks)),indent=2),flush=True)
    return 0 if len(ledger)==len(tasks) and all('summary' in r and 'analysis_error' not in r for r in ledger) else 2


if __name__=='__main__':
    raise SystemExit(main())
