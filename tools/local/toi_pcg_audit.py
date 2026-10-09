"""Serial, protected comparisons for this isolated TOI/PCG audit.

Uses the existing owned Windows Job and resource monitor. Historical results
are never overwritten. Complete runs only; failures remain in the ledger.
"""
from pathlib import Path
import argparse
import csv
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for name in ('bench', 'diagnostic', 'local', 'full_eval'):
    sys.path.insert(0, str(ROOT / 'tools' / name))
from config import read, expand, digest
from linux_runner import require, write_new
from local_identity import record, verify, active_identity, verify_active_sources
from windows_runner import execute, gpu_lock
from pool_metrics import metrics, export_evidence, state_comparison, pool_evidence
from quality_analysis import input_comparison


def summary(out, result):
    require(result['status'] == 'completed', 'Native run failed: ' + result['status'])
    m = metrics(out)
    frames = read(out / 'output/stats.json')['frames']
    pcgs = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    require(not m['pcg_failures'] and m['finite'], 'PCG/state check failed')
    require(export_evidence(out, len(frames), 'state')['passed'], 'Incomplete state export')
    if read(out / 'requested.json')['binary'] == 'active':
        require(export_evidence(out, len(frames), 'velocity')['passed'], 'Incomplete actual velocity export')
    require(not any(f.get('newton_exit') == 'iteration_limit' or f.get('toi_exit') == 'iteration_limit' for f in frames), 'Nonlinear iteration cap')
    row = {k: m[k] for k in ('directions', 'pcg', 'max_stretch', 'p99_stretch', 'fixed_drift_m', 'phase_ms')}
    row.update(solver_seconds=result['solver_seconds'], frames=len(frames),
        pcg_per_direction=m['pcg']/m['directions'] if m['directions'] else 0,
        fem_min_J=min((r['fem_min_J'] for r in m['frames'] if r['fem_min_J'] is not None), default=None),
        fem_nonpositive=max((r['fem_nonpositive'] for r in m['frames']), default=0),
        graph_captures=sum(not p.get('graph_cache_hit', False) for p in pcgs if p.get('execution') == 'conditional_graph'),
        graph_capture_ms=sum(p.get('graph_capture_instantiate_host_ms', 0) for p in pcgs),
        toi_outer=sum(len(f.get('toi', [])) for f in frames),
        hard_checks_passed=True, quality_certified=False, performance_certified=False)
    if read(out / 'requested.json')['expanded_config']['contact_pool']:
        evidence = pool_evidence(frames, True, False)
        require(evidence['passed'], 'Contact pool runtime contract failed')
        row['contact_pool'] = {k: evidence[k] for k in
            ('passed', 'complete_contract', 'totals', 'pool_bytes_peak') if k in evidence}
    return row


def run(args):
    session = ROOT / 'runs' / args.session
    require(not session.exists(), 'Fresh session required; no overwrite or implicit retry')
    session.mkdir(parents=True)
    controller = record(__file__)
    with (session/'controller_at_run.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    saved_controller = record(session/'controller_at_run.py')
    require((controller['bytes'], controller['sha256']) ==
            (saved_controller['bytes'], saved_controller['sha256']), 'Controller changed while retaining it')
    build = ROOT / 'build' / args.build
    if args.frozen_identity:
        active = read(args.frozen_identity)
        verify_active_sources(active)
        verify(active['build_evidence'] + active.get('objects', []) + [active['exe']] + active['dlls'])
    elif args.active_identity:
        active = read(args.active_identity)
        origin = Path(active['source_root']).resolve()
        transported = []
        for item in active['sources']:
            path = Path(item['path']).resolve()
            relative = path.relative_to(origin)
            # The archived executable still reads its originally baked Assets.
            # Native source is bound to this clean, byte-identical checkout.
            target = path if relative.parts[0] == 'Assets' else ROOT/relative
            actual = record(target)
            require(actual['bytes']==item['bytes'] and actual['sha256']==item['sha256'],
                    'Archived native bytes differ: '+str(relative))
            transported.append(actual)
        verify(active['build_evidence'] + active.get('objects', []) + [active['exe']] + active['dlls'])
        active = dict(active, sources=transported, original_source_root=str(origin),
            source_root=str(ROOT), source_digest=digest(transported),
            archived_identity=record(args.active_identity),
            scope='Archived compiled executable; all saved native/build/input bytes reverified. Native source bound to identical isolated checkout; baked Assets remain original.')
    else:
        active = active_identity(ROOT, build, build/'Release/gipc.exe', build/'build.log')
    old = read(ROOT.parent/'StiffGIPC_Learning/runs/report56_stage1_20261008/REFERENCE_IDENTITY.json')
    base = old['programs']['base']
    verify(base['sources'] + base['build_evidence'] + [base['exe']] + base['dlls'])
    identity = {'active': active, 'base': base}
    tasks = []
    for repeat in range(args.repeats):
        order = args.arms[repeat % len(args.arms):] + args.arms[:repeat % len(args.arms)]
        for scene in args.scenes:
            for arm in order:
                cfg = expand({'scene': scene, 'steps': args.steps, 'timeout_seconds': args.timeout,
                    'preset': {'base':'host', 'host':'host', 'graph':'graph', 'all':'combined', 'direct':'combined',
                        'pivot':'combined', 'best':'combined', 'pool':'combined', 'toi':'toi'}[arm],
                    'discrete_bvh_refit': arm in ('all','direct','pivot','best','pool'),
                    'fem_direct_pullback': (args.direct_pullback and arm != 'base') or arm in ('direct','best','pool'),
                    'mas_pivot_only_sync': (args.pivot_sync and arm != 'base') or arm in ('pivot','best','pool'),
                    'contact_pool': arm == 'pool',
                    **({'mas':args.toi_mas,'mas_restrict':args.mas_restrict} if arm=='toi' else {}),
                    **({'diagnostics':['cost'], 'cost_frames':'24-26,33-35'} if args.cost and arm!='base' else {}),
                    **({'diagnostics':['fixed'], 'fixed_frames':str(args.fixed_frame),
                        'fixed_directions':'1', 'fixed_mas_stage_study':args.mas_stage,
                        'fixed_restrict_study':args.restrict_study}
                        if args.fixed_frame and arm!='base' else {})})
                tasks.append(dict(name=f'{scene}_{arm}_r{repeat+1}', scene=scene, arm=arm,
                    repeat=repeat+1, binary='base' if arm=='base' else 'active', config=cfg))
    write_new(session/'identity.json', dict(programs=identity, controller=controller,
        retained_controller=saved_controller, tasks=tasks))
    ledger = []
    with gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for task in tasks:
            result = execute(session, task, identity[task['binary']], 0)
            entry = dict(task, result=result)
            ledger.append(entry)
            if result['status'] != 'completed':
                break
            try:
                entry['summary'] = summary(session/task['name'], result)
            except Exception as error:
                entry['analysis_error'] = str(error)
                break
            print(json.dumps(dict(name=task['name'], **entry['summary'])), flush=True)
    write_new(session/'ledger.json', dict(runs=ledger, skipped=tasks[len(ledger):]))
    verify([controller, saved_controller])
    return 0 if len(ledger) == len(tasks) and all('summary' in r for r in ledger) else 2


def analyze(args):
    session = ROOT/'runs'/args.session
    ledger = read(session/'ledger.json')
    comparisons = []
    for scene in sorted({r['scene'] for r in ledger['runs']}):
        for repeat in sorted({r['repeat'] for r in ledger['runs']}):
            arms = {r['arm']:r for r in ledger['runs'] if r['scene']==scene and r['repeat']==repeat and 'summary' in r}
            for reference in ('base', 'host', 'graph', 'all', 'direct', 'best'):
                for candidate in ('graph', 'all', 'direct', 'pivot', 'best', 'pool', 'toi'):
                    if reference==candidate or reference not in arms or candidate not in arms:
                        continue
                    a, b = arms[reference], arms[candidate]
                    matching = input_comparison(session/a['name'], session/b['name'])
                    require(matching['passed'], 'Initial scene/state mismatch')
                    comparisons.append(dict(scene=scene, repeat=repeat, reference=reference,
                        candidate=candidate, speedup=a['summary']['solver_seconds']/b['summary']['solver_seconds'],
                        initial_input=matching, trajectory=state_comparison(session/a['name'],session/b['name']),
                        quality_certified=False, performance_certified=False))
    write_new(session/'comparisons.json', dict(comparisons=comparisons, skipped=ledger['skipped']))
    print(json.dumps([dict(scene=r['scene'], repeat=r['repeat'], reference=r['reference'],
        candidate=r['candidate'], speedup=r['speedup']) for r in comparisons], indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('run', 'analyze'))
    p.add_argument('--session', required=True)
    p.add_argument('--build', default='audit_clean_active')
    p.add_argument('--active-identity', type=Path)
    p.add_argument('--frozen-identity', type=Path)
    p.add_argument('--direct-pullback', action='store_true')
    p.add_argument('--pivot-sync', action='store_true')
    p.add_argument('--scenes', nargs='+', default=['bunny_cloth_bunny_l'])
    p.add_argument('--arms', nargs='+', choices=('base','host','graph','all','direct','pivot','best','pool','toi'), default=['base','host','graph','all','toi'])
    p.add_argument('--steps', type=int, default=35)
    p.add_argument('--repeats', type=int, default=1, choices=range(1,8))
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--toi-mas', choices=('legacy','cholesky'), default='cholesky')
    p.add_argument('--mas-restrict', choices=('serial','warp'), default='serial')
    p.add_argument('--restrict-study', action='store_true')
    p.add_argument('--cost', action='store_true')
    p.add_argument('--fixed-frame', type=int, default=0)
    p.add_argument('--mas-stage', action='store_true')
    args = p.parse_args()
    require(args.session.replace('_','').isalnum(), 'Invalid session path')
    require(not args.fixed_frame or 1<=args.fixed_frame<=args.steps, 'Fixed frame outside budget')
    require(not args.mas_stage or args.fixed_frame==args.steps, 'MAS stage requires final frame')
    require(not args.fixed_frame or not args.cost, 'Separate fixed and cost probes')
    require(not args.restrict_study or args.fixed_frame, 'Restriction study requires a fixed frame')
    return run(args) if args.action=='run' else analyze(args)


if __name__ == '__main__':
    raise SystemExit(main())
