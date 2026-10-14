"""Read-only M0 distributions; validate flattened rows against native and frame data.

No simulation is launched. Missing observations remain missing. Dataset identities
and Windows/AutoDL results stay separate; every output directory must be fresh.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ('assembly', 'pcg', 'ccd', 'line_search', 'state_update')
OBSERVED = ('energy_evaluations', 'energy_backtracks', 'intersection_backtracks',
            'initial_energy_reused', 'alpha', 'beta', 'active_pairs',
            'kappa_before', 'kappa_after', 'alpha_geometric',
            'acceptance_margin', 'capacity_growth')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), 'Input changed while hashing')
    return dict(path=str(path), bytes=after.st_size, sha256=value)


def number(value, name, integer=False):
    require(not isinstance(value, bool) and value is not None and value != '',
            'Missing or boolean numeric observation: ' + name)
    result = float(value)
    require(math.isfinite(result), 'Nonfinite observation: ' + name)
    if integer:
        require(result.is_integer() and result >= 0, 'Invalid count: ' + name)
        return int(result)
    return result


def distribution(values):
    if not values:
        return dict(count=0, min=None, p50=None, p90=None, p99=None, max=None)
    values = sorted(values)
    def quantile(q):
        index = (len(values) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(count=len(values), min=values[0], p50=quantile(.5),
                p90=quantile(.9), p99=quantile(.99), max=values[-1])


def csv_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                'Missing or duplicate CSV columns')
        for row in reader:
            require(None not in row, 'Row has extra CSV columns')
            yield row


def parse_native(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    native = json.loads(text, parse_constant=invalid)
    require(isinstance(native, dict), 'Native row must be an object')
    return native


def analyze(newton, frames, label):
    inputs = [record(newton), record(frames)]
    frame_map = {}
    exits = defaultdict(Counter)
    for row in csv_rows(frames):
        if not row['run'].startswith('performance_'):
            continue
        physical = number(row['physical_frame'], 'physical_frame', True)
        require(physical == number(row['local_frame'], 'local_frame', True),
                'From-zero performance frame mapping differs')
        require(number(row['dt'], 'dt') == .04, 'Mixed time steps in dataset')
        key = (row['run'], physical)
        require(key not in frame_map, 'Duplicate frame key')
        frame_map[key] = row
        if physical <= 100:
            exits[(row['scene'], row['method'])][row['exit_reason']] += 1
    require(frame_map, 'No from-zero trajectory frames')
    groups = defaultdict(lambda: dict(records=0, solves=0, trials=0, missing=Counter(),
        phase_ms=Counter(), P=[], alpha=[], pairs=[], energy=Counter(), intersection=Counter(),
        first_accept=0, epoch_changes=0, iteration_limits=0, terminal_records=0))
    checks = defaultdict(lambda: dict(records=0, P=0, N=0, phases=Counter()))
    event_rows = defaultdict(list)
    previous = {}
    ignored = 0
    for row in csv_rows(newton):
        if not row['run'].startswith('performance_'):
            ignored += 1
            continue
        f = number(row['physical_frame'], 'physical_frame', True)
        key = (row['run'], f)
        require(key in frame_map, 'Newton row lacks its frame')
        frame = frame_map[key]
        require(all(row[k] == frame[k] for k in ('scene', 'method', 'repeat')),
                'Newton/frame identity differs')
        require(f == number(row['local_frame'], 'local_frame', True), 'Wrong local frame')
        index = number(row['record_index_0'], 'record_index', True)
        require(index == checks[key]['records'], 'Duplicate/unordered/noncontiguous Newton record')
        native = parse_native(row['native_json'])
        checks[key]['records'] += 1
        if f <= 100:
            group = groups[(row['scene'], row['method'])]
            group['records'] += 1
        else:
            group = None
        for phase in PHASES:
            value = number(row[phase + '_ms'], phase)
            require(value >= 0, 'Negative phase time')
            actual = native.get('phase_ms', {}).get(phase)
            require(actual is not None and math.isclose(value, number(actual, phase),
                    rel_tol=1e-12, abs_tol=1e-9), 'Flattened phase differs from native')
            checks[key]['phases'][phase] += value
            if group is not None:
                group['phase_ms'][phase] += value
        pcg = native.get('pcg')
        if pcg is not None:
            count = number(row['P'], 'P', True)
            require(isinstance(pcg, dict), 'Invalid PCG object')
            require(count == number(pcg['iterations'], 'iterations', True), 'PCG count mismatch')
            require(number(pcg['frame'], 'pcg.frame', True) == f, 'PCG physical frame mismatch')
            checks[key]['N'] += 1
            checks[key]['P'] += count
            if group is not None:
                group['solves'] += 1
                group['P'].append(count)
                group['iteration_limits'] += int(pcg.get('iteration_limit') is True)
        else:
            # A terminal assembly has no PCG observation; empty P is valid.
            # Zero work follows from absence of a solve, not an imputed count.
            require(row['P'] in ('', '0', '0.0'), 'Iterations without a PCG solve')
            count = 0
            if group is not None:
                group['terminal_records'] += 1
        # Scalar CSV defaults are not evidence of an observed native field.
        if group is not None:
            for field in OBSERVED:
                if field not in native:
                    group['missing'][field] += 1
        if 'alpha' not in native:
            continue
        require(pcg is not None, 'Accepted update without recorded direction solve')
        alpha = number(native['alpha'], 'alpha')
        require(0 < alpha <= 1, 'Accepted alpha outside (0,1]')
        require(alpha == number(row['alpha'], 'csv.alpha'), 'Flattened alpha mismatch')
        require(number(native['beta'], 'beta') == number(row['beta'], 'csv.beta'), 'Beta mismatch')
        pairs = number(native['active_pairs'], 'active_pairs', True)
        eb = number(native['energy_backtracks'], 'energy_backtracks', True)
        ib = number(native['intersection_backtracks'], 'intersection_backtracks', True)
        evaluations = number(native['energy_evaluations'], 'energy_evaluations', True)
        require(evaluations >= 1, 'Trial has no energy evaluation')
        before = number(native['kappa_before'], 'kappa_before')
        after = number(native['kappa_after'], 'kappa_after')
        if group is not None:
            group['trials'] += 1
            group['alpha'].append(alpha)
            group['pairs'].append(pairs)
            group['energy'][eb] += 1
            group['intersection'][ib] += 1
            group['first_accept'] += int(eb == ib == 0)
            group['epoch_changes'] += int(before != after)
        if row['scene'] in ('Boxes', 'Mixed', 'Bunny') and row['method'] in ('original', 'graph_mov') and row['repeat'] == '1':
            observation = dict(run=row['run'], scene=row['scene'], method=row['method'],
                physical_frame=f, record_index_0=index, P=count, alpha_accepted=alpha,
                active_pairs=pairs, energy_backtracks=eb, intersection_backtracks=ib,
                kappa_before=before, kappa_after=after,
                contact_birth=pairs > 0 and previous.get(row['run'], 0) == 0,
                contact_release=pairs == 0 and previous.get(row['run'], 0) > 0)
            event_rows[row['scene']].append(observation)
            previous[row['run']] = pairs
    for key, frame in frame_map.items():
        values = checks[key]
        require(values['records'] == number(frame['newton_records'], 'frame.records', True),
                'Newton/frame record coverage mismatch')
        require(values['N'] == number(frame['N'], 'frame.N', True) and
                values['P'] == number(frame['P'], 'frame.P', True), 'Newton/frame N or P mismatch')
        for phase in PHASES:
            require(math.isclose(values['phases'][phase], number(frame[phase + '_ms'], phase),
                    rel_tol=1e-8, abs_tol=1e-5), 'Newton/frame phase sum mismatch')
    runs = sorted({run for run, _ in frame_map})
    require(all({f for run, f in frame_map if run == name} == set(range(1, 106)) for name in runs),
            'Performance trajectory must contain exactly frames 1..105')
    summary = []
    for (scene, method), g in sorted(groups.items()):
        summary.append(dict(scene=scene, method=method, records=g['records'], solves=g['solves'],
            accepted_trials=g['trials'], terminal_records=g['terminal_records'],
            first_trial_accepted_fraction=g['first_accept'] / g['trials'] if g['trials'] else None,
            energy_backtracks=dict(sorted(g['energy'].items())),
            intersection_backtracks=dict(sorted(g['intersection'].items())),
            accepted_alpha=distribution(g['alpha']), P_per_solve=distribution(g['P']),
            active_pairs=distribution(g['pairs']), parent_phase_seconds={k:v/1000 for k,v in g['phase_ms'].items()},
            kappa_changes=g['epoch_changes'], pcg_iteration_limits=g['iteration_limits'],
            missing_native_fields={k:g['missing'][k] for k in OBSERVED},
            frame_exits=dict(exits[scene, method])))
    events = {}
    for scene, rows in sorted(event_rows.items()):
        # Fixed before any new candidate measurements. Up to eight distinct states.
        predicates = {
            'ordinary': lambda r: r['active_pairs'] == 0 and r['alpha_accepted'] == 1,
            'first_contact': lambda r: r['contact_birth'],
            'contact_release': lambda r: r['contact_release'],
            'energy_rejection': lambda r: r['energy_backtracks'] > 0,
            'intersection_backtrack': lambda r: r['intersection_backtracks'] > 0,
        }
        selected = {}
        for event, test in predicates.items():
            selected[event] = next((r for r in rows if test(r)), None)
        selected['dense_contact'] = max(rows, key=lambda r:r['active_pairs']) if rows else None
        selected['small_accepted_alpha'] = min(rows, key=lambda r:r['alpha_accepted']) if rows else None
        selected['pcg_tail'] = max(rows, key=lambda r:r['P']) if rows else None
        events[scene] = dict(selected=selected, capacity_growth=None,
            missing_reason='Capacity growth and pre-line-search geometric alpha are not recorded; accepted alpha is not a CCD-only limit.',
            available_natural_state_count=len({(r['physical_frame'],r['record_index_0']) for r in selected.values() if r}),
            scope='Historical identity-specific event selection, not exact checkpoints of the new binary.')
    require([record(newton), record(frames)] == inputs, 'Input changed during analysis')
    return dict(dataset=label, input_identity=inputs, first100_summary=summary,
        selected_states=events, validated_frames=len(frame_map), validated_runs=len(runs),
        ignored_nontrajectory_newton_rows=ignored, row_frame_crosschecks_passed=True,
        metric_scope='Frames1..100; first trial accepted means no energy or intersection backtrack. Initial alpha/CCD reductions unavailable.',
        recording_cost_deducted=False, performance_certified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-report', type=Path, default=ROOT.parent/'stiff_speed_audit_20261009/reports/five_arm_retest_20261014')
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output directory required')
    datasets = [analyze(ROOT/'history/autodl_recording105_20261013/data/per_newton.csv.gz',
        ROOT/'history/autodl_recording105_20261013/data/per_frame.csv', 'AutoDL_a4_snapshot'),
        analyze(args.local_report/'per_newton.csv', args.local_report/'per_frame.csv', 'Windows_five_arm_20261014')]
    args.output.mkdir(parents=True)
    (args.output/'offline_summary.json').write_text(json.dumps(dict(schema='m0_offline.v1',
        controller=record(Path(__file__)), datasets=datasets), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for dataset in datasets:
        print(json.dumps(dict(dataset=dataset['dataset'], frames=dataset['validated_frames'],
            runs=dataset['validated_runs'], cells=len(dataset['first100_summary']), passed=True)))


if __name__ == '__main__':
    main()
