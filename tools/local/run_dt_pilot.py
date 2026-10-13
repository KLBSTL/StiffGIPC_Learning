"""One paired 100-frame observation per dt, using unmodified frozen executables."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('tools/local', 'tools/paper_suite', 'tools/bench', 'tools/diagnostic')]
from run_combination_suite import task as old_task, checkpoint_env
from compare_graph_history import identities, power
from local_identity import record, verify
from protocol import input_inventory
from config import expand, read
from linux_runner import require, write_new
from analyze import observation, physical_inputs_equal, position_error
import windows_runner

DTS = (.01, .04, .004)
LABELS = {'stiff': 'Base', 'old': '最快版'}


def plan():
    stages = {}
    for stage, steps in (('smoke', 2), ('performance', 100)):
        tasks = []
        for index, dt in enumerate(DTS):
            for arm in (('stiff', 'old') if index % 2 == 0 else ('old', 'stiff')):
                t = old_task('Hang', arm, 1, steps, 'dt_' + stage)
                t['config'] = expand(t['config'] | {'dt': dt})
                t.update(name=f'{stage}_hang_dt{str(dt).replace(".", "p")}_{arm}_r1', dt=dt)
                tasks.append(t)
        stages[stage] = tasks
    return dict(schema='cloth_dt_pilot.v1', scene='cloth_hang_l', dts=DTS, frames=100,
        repeats=1, labels=LABELS, stages=stages, recording='Existing native logging unchanged',
        definitions=dict(rho=1e-4, movement=.01, cumulative=.01, minimum_updates=6,
            timing='Synchronized solver_ms including graph capture; initialization/export excluded',
            physical_times_seconds=[100 * dt for dt in DTS],
            scope='Speedup comparisons within each dt; different dt endpoints have different physical times'),
        performance_certified=False, physical_quality_certified=False)


def run(session, stage):
    session.mkdir(parents=True, exist_ok=True)
    p = plan()
    ids = identities()
    packet = dict(plan=p, programs={k: ids[k] for k in ('base', 'historical')},
        controller=record(__file__), power=power(), inputs=input_inventory())
    path = session / 'identities.json'
    if path.exists():
        require(read(path) == json.loads(json.dumps(packet)), 'Program/input/power/controller changed')
    else:
        write_new(path, packet)
        shutil.copyfile(__file__, session / 'controller_at_run.py')
    if stage == 'performance':
        require(read(session / 'smoke_ledger.json')['complete'], 'Smoke gate required')
    require(not (session / (stage + '_ledger.json')).exists(), 'Existing stage; no overwrite/retry')
    tasks = p['stages'][stage]
    write_new(session / (stage + '_tasks.json'), dict(tasks=tasks))
    rows = []
    original = windows_runner.environment
    with windows_runner.gpu_lock(ROOT.parent / 'StiffGIPC_Learning'):
        for t in tasks:
            identity = packet['programs'][t['identity_key']]
            windows_runner.environment = lambda c, out, t=t, identity=identity: checkpoint_env(c, out, t, identity, session)
            try:
                result = windows_runner.execute(session, t, identity, 0)
            finally:
                windows_runner.environment = original
            row = dict(task=t, result=result)
            rows.append(row)
            if result['status'] == 'completed':
                try:
                    row['observation'] = observation(session / t['name'], t, result)
                except Exception as error:
                    row['analysis_error'] = f'{type(error).__name__}: {error}'
            with (session / (stage + '_journal.jsonl')).open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if 'observation' not in row:
                break
            m = row['observation']
            print(json.dumps(dict(stage=stage, dt=t['dt'], method=LABELS[t['arm']],
                seconds=m['TimeTot'], N=m['N_tot'], P=m['P_tot']), ensure_ascii=False), flush=True)
        ledger = dict(stage=stage, runs=rows, skipped=tasks[len(rows):],
            complete=len(rows) == len(tasks) and all('observation' in r for r in rows), power_after=power())
        write_new(session / (stage + '_ledger.json'), ledger)
    require(ledger['power_after'] == packet['power'], 'Power changed during tests')
    for identity in packet['programs'].values():
        verify(identity['sources'] + identity['build_evidence'] + [identity['exe']] + identity['dlls'] + identity.get('objects', []))
    verify([packet['controller']])
    return 0 if ledger['complete'] else 2


def analyze(session, output):
    ledger = read(session / 'performance_ledger.json')
    require(ledger['complete'], 'All six formal runs required')
    output.mkdir(parents=True, exist_ok=False)
    values = []
    for dt in DTS:
        runs = {r['task']['arm']: r for r in ledger['runs'] if r['task']['dt'] == dt}
        base = session / runs['stiff']['task']['name']
        fast = session / runs['old']['task']['name']
        equal = physical_inputs_equal(base, fast)
        require(equal['passed'], 'Paired physical input mismatch')
        obs = {arm: observation(session / r['task']['name'], r['task'], read(session / r['task']['name'] / 'result.json'))
               for arm, r in runs.items()}
        worst = {'rms_mm': 0., 'max_mm': 0.}
        endpoint = None
        for frame in range(101):
            error = position_error(fast / f'trace/state_{frame:04d}.bin', base / f'trace/state_{frame:04d}.bin')
            worst = {k: max(worst[k], error[k]) for k in worst}
            if frame == 100:
                endpoint = error
        a, b = obs['stiff'], obs['old']
        values.append(dict(dt=dt, physical_seconds=100 * dt,
            base_seconds=a['TimeTot'], fastest_seconds=b['TimeTot'], speedup=a['TimeTot'] / b['TimeTot'],
            base_N=a['N_tot'], fastest_N=b['N_tot'], base_P=a['P_tot'], fastest_P=b['P_tot'],
            fastest_ms_per_step=b['ms_per_step'], fastest_solver_seconds_per_simulated_second=b['TimeTot'] / (100 * dt),
            trajectory_frame100_rms_mm=endpoint['rms_mm'], trajectory_frame100_max_mm=endpoint['max_mm'],
            trajectory_worst_rms_mm=worst['rms_mm'], trajectory_worst_max_mm=worst['max_mm'],
            physical_input=equal, base=a, fastest=b,
            base_wall_seconds=runs['stiff']['result']['wall_seconds'], fastest_wall_seconds=runs['old']['result']['wall_seconds']))
    write_new(output / 'summary.json', dict(complete=True, rows=values, labels=LABELS, repeats=1,
        timing='Existing native solver timing; logs unchanged; shared WDDM desktop; no stability certification',
        error_scope='Same-dt Base/fastest trajectory difference, not error against exact solution',
        identities=record(session / 'identities.json')))
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in values]
    with (output / 'speedups.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(flat, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'smoke', 'performance', 'analyze'))
    parser.add_argument('--session', type=Path, default=ROOT / 'runs/local_hang_dt_pilot_20261011')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_hang_dt_pilot_20261011')
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
    elif args.stage == 'analyze':
        analyze(args.session.resolve(), args.output.resolve())
    else:
        raise SystemExit(run(args.session.resolve(), args.stage))
