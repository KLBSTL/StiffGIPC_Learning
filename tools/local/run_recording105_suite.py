"""Serial 105-frame trajectories and cold frame100 rule/recording diagnostics."""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('tools/local', 'tools/paper_suite', 'tools/bench', 'tools/diagnostic')]
from run_combination_suite import task as old_task, checkpoint_env, verify_restore
from compare_graph_history import power
from local_identity import record, verify, active_identity
from config import read, expand
from linux_runner import require, write_new
from protocol import SCENES, input_inventory
from analyze import observation, physical_inputs_equal, position_error
from run_suite import checkpoint_evidence
from pool_metrics import export_evidence
import windows_runner

COPY = ROOT / 'runs/recording_build_20261011/source'
BUILD = COPY / 'build/recording105'
SELECTED = tuple(s for s in SCENES if s != 'Stack')
ARMS = ('original', 'pure', 'ours', 'strict')
LABELS = {'original': 'Movement only', 'pure': 'Cumulative TOI .01',
          'ours': 'Compensated + residual gate', 'strict': 'Movement only, rho=1e-6 reference'}


def task(scene, arm, repeat, stage, recording=True):
    steps = 105 if stage == 'performance' else 2 if stage == 'smoke' else 1
    t = old_task(scene, arm, repeat, steps, 'recording_' + stage)
    cfg = t['config'] | dict(dt=.04, ipc_termination='legacy', pcg_rho_tol=1e-4)
    if arm in ('original', 'strict'):
        cfg['ipc_termination'] = 'movement_only'
    if arm == 'strict':
        cfg['pcg_rho_tol'] = 1e-6
    if arm == 'ours':
        cfg.update(ipc_termination='compensated', ipc_residual_rel_tol=.03)
    if stage == 'cost':
        cfg.update(diagnostics=['cost'], cost_frames='100', cost_events=True)
    t.update(name=f'{stage}_{scene.lower()}_{arm}_r{repeat}', config=expand(cfg),
             recording_enabled=recording, identity_key='active')
    if stage == 'performance' and arm == 'original' and repeat == 1:
        t['checkpoint_save_frame'] = 100
    if stage in ('matched', 'overhead', 'cost'):
        t.update(target_frame=100, checkpoint_from=f'performance_{scene.lower()}_original_r1/checkpoint')
    return t


def plan():
    performance, checkpoint = [], []
    for repeat in (1, 2, 3):
        scenes = SELECTED if repeat % 2 else tuple(reversed(SELECTED))
        order = ARMS[repeat - 1:] + ARMS[:repeat - 1]
        for scene in scenes:
            performance.extend(task(scene, arm, repeat, 'performance') for arm in order)
            for arm in order:
                modes = (True, False) if repeat % 2 else (False, True)
                for enabled in ((True,) if arm == 'strict' else modes):
                    checkpoint.append(task(scene, arm, repeat, 'matched' if enabled else 'overhead', enabled))
    return dict(schema='recording105.v1', scenes={s: SCENES[s] for s in SELECTED},
        frames=105, table_physical_frame=100, repeats=3, dt=.04, labels=LABELS,
        stages=dict(smoke=[task(s, a, 1, 'smoke') for s in SELECTED for a in ARMS],
            performance=performance, checkpoint=checkpoint,
            cost=[task(s, a, 1, 'cost') for s in SELECTED for a in ARMS[:3]]),
        definitions=dict(candidates='Three outer rules; current source, same complete execution combination',
            reference='Movement-only rho=1e-6; stricter numerical reference, not exact solution',
            fastest_alias='Frozen historical 2.084x binary remains untouched; current observer copy is a separate program',
            checkpoint='Shared physical state after step99, from movement-only repeat1; cold reconstruction',
            overhead='Same observer binary and cold input; instrumentation/batched logging on versus old logging',
            timing='Synchronized host solver_ms, existing CUDA phase intervals, separate export/checkpoint/stats host times',
            certification='Shared WDDM desktop; instrumented observations and three repeats; no performance/physical certification'),
        excluded_scenes=['Stack'])


def identities():
    return dict(active=active_identity(COPY, BUILD, BUILD / 'Release/gipc.exe', BUILD / 'build.log'))


def env(config, out, t, identity, session):
    value = checkpoint_env(config, out, t, identity, session)
    value['GIPC_RECORD_TIMING'] = '1' if t['recording_enabled'] else '0'
    return value


def recording_evidence(folder, t):
    if not t['recording_enabled']:
        require(not (folder / 'trace/recording_summary.json').exists(), 'Recording unexpectedly enabled')
        return dict(enabled=False)
    summary = read(folder / 'trace/recording_summary.json')
    require(summary['enabled'] and summary['stats_mode'] == 'one_final_write', 'Wrong recording mode')
    with (folder / 'trace/recording_costs.csv').open() as f:
        costs = list(csv.DictReader(f))
    with (folder / 'trace/frames.csv').open() as f:
        timings = list(csv.DictReader(f))
    native = read(folder / 'output/stats.json')['frames']
    require(len(costs) == len(native) == len(timings) == t['config']['steps'], 'Missing recording rows')
    offset = 99 if 'checkpoint_from' in t else 0
    for index, (c, f, timing) in enumerate(zip(costs, native, timings), 1):
        require(int(c['local_frame']) == index and int(c['physical_frame']) == offset + index, 'Wrong recording frame')
        require(float(c['solver_ms']) == float(timing['solver_ms']), 'Different solver timer')
        require(all(math.isfinite(float(v)) and float(v) >= 0 for k, v in c.items() if k.endswith('_ms')), 'Invalid recording time')
        require(all('phase_ms' in n and 'phase_in_legacy_frame_sum' in n for n in f['newton']), 'Missing Newton phases')
        for phase, value in f['recording_phase_ms'].items():
            measured = sum(n['phase_ms'][phase] for n in f['newton'])
            require(abs(measured - value) <= 1e-5 * max(1, value), 'Newton/frame phase sum mismatch')
    return dict(enabled=True, summary=summary, files=[record(folder / ('trace/' + n)) for n in
        ('recording_costs.csv', 'recording_summary.json')])


def run(session, stage):
    session.mkdir(parents=True, exist_ok=True)
    p, ids = plan(), identities()
    packet = dict(plan=p, programs=ids, controller=record(__file__), power=power(), inputs=input_inventory())
    path = session / 'identities.json'
    if path.exists():
        require(read(path) == json.loads(json.dumps(packet)), 'Program/input/power/controller changed')
    else:
        write_new(path, packet)
        shutil.copyfile(__file__, session / 'controller_at_run.py')
    if stage != 'smoke':
        require(read(session / 'smoke_ledger.json')['complete'], 'Smoke gate required')
    if stage in ('checkpoint', 'cost'):
        require(read(session / 'performance_ledger.json')['complete'], 'Complete full trajectories required')
    require(not (session / (stage + '_ledger.json')).exists(), 'Stage already exists; no implicit retry')
    tasks, rows = p['stages'][stage], []
    write_new(session / (stage + '_tasks.json'), dict(tasks=tasks))
    original = windows_runner.environment
    original_gpu_query = windows_runner.gpu_query
    def gpu_query_with_time(gpu, timeout=10):
        begin = time.monotonic()
        sample = original_gpu_query(gpu, timeout=timeout)
        sample.update(sample_epoch_seconds=time.time(), query_host_ms=1000 * (time.monotonic() - begin),
                      timestamp_scope='Host time after existing nvidia-smi query; not a per-frame GPU measurement')
        return sample
    windows_runner.gpu_query = gpu_query_with_time
    with windows_runner.gpu_lock(ROOT.parent / 'StiffGIPC_Learning'):
        for index, t in enumerate(tasks, 1):
            identity = ids['active']
            if 'checkpoint_from' in t:
                checkpoint_evidence(session / t['checkpoint_from'])
            windows_runner.environment = lambda c, out, t=t: env(c, out, t, identity, session)
            try:
                result = windows_runner.execute(session, t, identity, 0)
            finally:
                windows_runner.environment = original
            row = dict(task=t, result=result)
            rows.append(row)
            if result['status'] == 'completed':
                try:
                    if 'checkpoint_from' in t:
                        verify_restore(session, t)
                    require(export_evidence(session / t['name'], t['config']['steps'], 'velocity')['passed'], 'Invalid velocities')
                    row['observation'] = observation(session / t['name'], t, result)
                    row['recording'] = recording_evidence(session / t['name'], t)
                    if 'checkpoint_save_frame' in t:
                        row['saved_checkpoint'] = checkpoint_evidence(session / t['name'] / 'checkpoint')
                except Exception as error:
                    row['analysis_error'] = f'{type(error).__name__}: {error}'
            with (session / (stage + '_journal.jsonl')).open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if 'observation' not in row or 'recording' not in row:
                print(json.dumps(dict(run=t['name'], error=row.get('analysis_error'), result=result['status'])), flush=True)
                break
            m = row['observation']
            print(json.dumps(dict(stage=stage, done=index, total=len(tasks), run=t['name'],
                seconds=m['TimeTot'], N=m['N_tot'], P=m['P_tot']), ensure_ascii=False), flush=True)
        ledger = dict(stage=stage, runs=rows, skipped=tasks[len(rows):],
            complete=len(rows) == len(tasks) and all('recording' in r for r in rows), power_after=power())
        write_new(session / (stage + '_ledger.json'), ledger)
    windows_runner.gpu_query = original_gpu_query
    require(ledger['power_after'] == packet['power'], 'Power changed')
    for identity in ids.values():
        verify(identity['sources'] + identity['build_evidence'] + [identity['exe']] + identity['dlls'] + identity.get('objects', []))
    verify([packet['controller']])
    return 0 if ledger['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'smoke', 'performance', 'checkpoint', 'cost', 'all'))
    parser.add_argument('--session', type=Path, default=ROOT / 'runs/local_recording105_dt0p04_20261011')
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
    elif args.stage == 'all':
        for stage in ('smoke', 'performance', 'checkpoint', 'cost'):
            code = run(args.session.resolve(), stage)
            if code:
                raise SystemExit(code)
    else:
        raise SystemExit(run(args.session.resolve(), args.stage))
