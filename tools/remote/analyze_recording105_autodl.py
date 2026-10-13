"""Offline frame/Newton/error accounting for the six-scene recording experiment."""
import argparse
import csv
import importlib.util
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('tools/local', 'tools/paper_suite', 'tools/bench', 'tools/diagnostic')]
from run_recording105_suite import SELECTED, ARMS, LABELS, plan, recording_evidence
from analyze import observation, physical_inputs_equal, position_error
from config import read
from linux_runner import require, write_new, sha
from local_identity import verify, record
from run_suite import checkpoint_evidence
from render import geometry, positions

PHASES = ('assembly', 'pcg', 'ccd', 'line_search', 'state_update')


def csv_file(output, name, rows):
    require(rows, 'Empty table: ' + name)
    with (output / (name + '.csv')).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_run(session, run):
    task = run['task']
    folder = session / task['name']
    result = read(folder / 'result.json')
    require(result['status'] == 'completed' and result['recorded_frames'] == task['config']['steps'], 'Incomplete native run')
    value = observation(folder, task, result)
    require(value == run['observation'], 'Raw observation changed')
    require(recording_evidence(folder, task) == run['recording'], 'Recording evidence changed')
    with (folder / 'trace/frames.csv').open() as f:
        timings = list(csv.DictReader(f))
    native = read(folder / 'output/stats.json')['frames']
    costs = []
    if task['recording_enabled']:
        with (folder / 'trace/recording_costs.csv').open() as f:
            costs = list(csv.DictReader(f))
    return task, folder, result, value, native, timings, costs


def frame_rows(session, runs):
    frames_out, newton_out, hardware_out, run_rows = [], [], [], []
    for run in runs:
        task, folder, result, obs, native, timings, costs = read_run(session, run)
        offset = 99 if 'checkpoint_from' in task else 0
        for local_frame, (f, timing) in enumerate(zip(native, timings), 1):
            pcg = [n['pcg'] for n in f['newton'] if 'pcg' in n]
            phase = f.get('recording_phase_ms', f['phase_ms'])
            stop = f.get('ipc_stopping', {})
            row = dict(run=task['name'], scene=task['scene_key'], method=task['arm'], repeat=task['repeat'],
                recording_enabled=task['recording_enabled'], measurement=task['name'].split('_')[0],
                local_frame=local_frame, physical_frame=offset + local_frame,
                dt=task['config']['dt'], rho_tolerance=task['config']['pcg_rho_tol'],
                solver_ms=float(timing['solver_ms']), **{k + '_ms': phase[k] for k in PHASES},
                regular_assembly_ms=f['phase_ms']['assembly'], terminal_assembly_ms=f.get('ipc_exit_assembly_ms', 0),
                unclassified_solver_envelope_ms=float(timing['solver_ms']) - sum(phase.values()),
                N=len(pcg), P=sum(p['iterations'] for p in pcg), newton_records=len(f['newton']),
                accepted_updates=stop.get('accepted_updates'), exit_reason=f.get('newton_exit'),
                final_beta=stop.get('final_beta'), gate_budget=stop.get('budget'), gate_audits=stop.get('budget_audits'),
                energy_evaluations=sum(n.get('energy_evaluations', 0) for n in f['newton']),
                energy_backtracks=sum(n.get('energy_backtracks', 0) for n in f['newton']),
                intersection_backtracks=sum(n.get('intersection_backtracks', 0) for n in f['newton']),
                candidate_pairs=int(timing['candidate_pairs']), ground_candidates=int(timing['ground_candidates']),
                graph_cache_hits=sum(bool(p.get('graph_cache_hit')) for p in pcg),
                graph_capture_host_ms=sum(p.get('graph_capture_instantiate_host_ms', 0) for p in pcg),
                graph_body_executions=sum(p.get('graph_body_executions', 0) for p in pcg),
                residual_observations=sum(len(n.get('ipc_residual_observations', [])) for n in f['newton']),
                residual_observe_host_ms=sum(o['observe_host_ms'] for n in f['newton'] for o in n.get('ipc_residual_observations', [])),
                trace_export_host_ms=float(costs[local_frame - 1]['trace_export_host_ms']) if costs else None,
                checkpoint_host_ms=float(costs[local_frame - 1]['checkpoint_host_ms']) if costs else None,
                stats_write_host_ms=float(costs[local_frame - 1]['stats_write_host_ms']) if costs else None,
                native_contact_json=json.dumps(f.get('contact_geometry', {}), separators=(',', ':')))
            row['P_per_direction'] = row['P'] / row['N'] if row['N'] else None
            row['pcg_stage_us_per_iteration'] = 1000 * phase['pcg'] / row['P'] if row['P'] else None
            require(all(p.get('frame', row['physical_frame']) == row['physical_frame'] for p in pcg), 'Wrong physical PCG frame')
            frames_out.append(row)
            for index, n in enumerate(f['newton']):
                p = n.get('pcg', {})
                rho_initial, rho_stop = p.get('rho_initial'), p.get('rho_stop')
                newton_out.append(dict(run=task['name'], scene=task['scene_key'], method=task['arm'], repeat=task['repeat'],
                    local_frame=local_frame, physical_frame=row['physical_frame'], record_index_0=index,
                    P=p.get('iterations'), rho_initial=rho_initial, rho_stop=rho_stop,
                    rho_ratio=rho_stop / rho_initial if rho_initial and rho_stop is not None else None,
                    residual_observations=len(n.get('ipc_residual_observations', [])),
                    residual_observe_host_ms=sum(o['observe_host_ms'] for o in n.get('ipc_residual_observations', [])),
                    **{k + '_ms': n.get('phase_ms', {}).get(k) for k in PHASES},
                    alpha=n.get('alpha'), beta=n.get('beta'), active_pairs=n.get('active_pairs'),
                    kappa_before=n.get('kappa_before'), kappa_after=n.get('kappa_after'),
                    native_json=json.dumps(n, ensure_ascii=False, separators=(',', ':'))))
        for index, sample in enumerate(result.get('gpu_samples', [])):
            hardware_out.append(dict(run=task['name'], sample_index=index, epoch_seconds=sample.get('sample_epoch_seconds'),
                query_host_ms=sample.get('query_host_ms'), utilization_gpu=sample['utilization.gpu'],
                memory_free_mib=sample['memory.free'], temperature_c=sample['temperature.gpu'],
                power_draw_w_raw=sample.get('power.draw'), sm_clock_mhz_raw=sample.get('clocks.sm')))
        summary = run['recording'].get('summary', {})
        run_rows.append(dict(run=task['name'], scene=task['scene_key'], method=task['arm'], repeat=task['repeat'],
            recording_enabled=task['recording_enabled'], steps=task['config']['steps'],
            solver_seconds=obs['TimeTot'], process_wall_seconds=result['wall_seconds'],
            non_solver_process_envelope_seconds=result['wall_seconds'] - obs['TimeTot'],
            initialization_ms=summary.get('initialization_host_ms'),
            checkpoint_restore_ms=summary.get('checkpoint_restore_host_ms'),
            trace_export_ms=summary.get('trace_export_host_ms'),
            checkpoint_hook_ms=summary.get('checkpoint_save_host_ms'),
            checkpoint_saved_before_physical_frame=task.get('checkpoint_save_frame'),
            stats_write_ms=summary.get('stats_write_host_ms')))
    return frames_out, newton_out, hardware_out, run_rows


def masks(folder):
    raw = np.fromfile(folder / 'trace/topology.bin', dtype='<u4')
    nv, nf, nt = map(int, raw[:3])
    faces = raw[3:3 + 3 * nf].reshape(-1, 3)
    tets = raw[3 + 3 * nf:].reshape(-1, 4)
    boundary = np.fromfile(folder / 'trace/boundary_types.bin', dtype='<i4')
    abd = read(folder / 'trace/metadata.json')['abd_point_num']
    volume = np.zeros(nv, bool)
    volume[tets.ravel()] = True
    in_abd = np.arange(nv) < abd
    return dict(all=np.ones(nv, bool), cloth_dynamic=~volume & ~in_abd & (boundary == 0),
                fem=volume & ~in_abd, abd=in_abd), faces, boundary


def trajectory_input_gate(session, runs):
    checks = []
    for run in runs:
        task = run['task']
        anchor = session / f"performance_{task['scene_key'].lower()}_original_r1"
        folder = session / task['name']
        equal = physical_inputs_equal(anchor, folder)
        require(equal['passed'], 'Trajectory initial physical inputs differ: ' + task['name'])
        checks.append(dict(run=task['name'], anchor=str(anchor), **equal))
    return dict(passed=True, exact_initial_runs=len(checks), runs=checks,
                scope='Same physical initial positions, actual velocities, topology, masses, boundaries and frozen scene; solver rule/tolerances deliberately differ')


def errors(session):
    output, quality, reference_dispersion = [], [], []
    for scene in SELECTED:
        folders = {(a, r): session / f'performance_{scene.lower()}_{a}_r{r}' for a in ARMS for r in (1, 2, 3)}
        partition, faces, boundary = masks(folders['original', 1])
        x0 = np.fromfile(folders['original', 1] / 'trace/state_0000.bin', dtype='<f8').reshape(-1, 3)
        dynamic_faces = faces[partition['cloth_dynamic'][faces].any(axis=1)]
        edges = np.unique(np.sort(np.concatenate([dynamic_faces[:, [0, 1]], dynamic_faces[:, [1, 2]], dynamic_faces[:, [2, 0]]]), axis=1), axis=0)
        initial_lengths = np.linalg.norm(x0[edges[:, 0]] - x0[edges[:, 1]], axis=1) if len(edges) else np.array([])
        for frame in range(106):
            x = {k: np.fromfile(p / f'trace/state_{frame:04d}.bin', dtype='<f8').reshape(-1, 3) for k, p in folders.items()}
            v = {k: np.fromfile(p / f'trace/velocity_{frame:04d}.bin', dtype='<f8').reshape(-1, 3) for k, p in folders.items()}
            require(all(a.shape == x0.shape and np.isfinite(a).all() for a in list(x.values()) + list(v.values())), 'Invalid exported positions/velocities')
            for left, right in itertools.combinations((1, 2, 3), 2):
                distances = np.linalg.norm(x['strict', left] - x['strict', right], axis=1)
                speeds = np.linalg.norm(v['strict', left] - v['strict', right], axis=1)
                for name, mask in partition.items():
                    if mask.any():
                        reference_dispersion.append(dict(scene=scene, frame=frame, partition=name,
                            reference_left=left, reference_right=right,
                            position_rms_mm=1000 * float(np.sqrt(np.mean(distances[mask] ** 2))),
                            position_max_mm=1000 * float(distances[mask].max()),
                            velocity_rms_m_per_s=float(np.sqrt(np.mean(speeds[mask] ** 2))),
                            velocity_max_m_per_s=float(speeds[mask].max())))
            for arm in ARMS:
                for repeat in (1, 2, 3):
                    a = x[arm, repeat]
                    drift = np.linalg.norm(a[boundary != 0] - x0[boundary != 0], axis=1)
                    stretch = np.linalg.norm(a[edges[:, 0]] - a[edges[:, 1]], axis=1) / np.maximum(initial_lengths, 1e-30) if len(edges) else np.array([])
                    dynamic = (partition['cloth_dynamic'] | partition['fem'] | partition['abd']) & (boundary == 0)
                    quality.append(dict(scene=scene, method=arm, repeat=repeat, frame=frame,
                        dynamic_cloth_max_stretch=float(stretch.max()) if len(stretch) else None,
                        dynamic_cloth_p99_stretch=float(np.quantile(stretch, .99)) if len(stretch) else None,
                        nonzero_boundary_max_displacement_mm=1000 * float(drift.max()) if len(drift) else None,
                        minimum_dynamic_vertex_ground_gap_m=float((a[dynamic, 1] + 1).min()) if dynamic.any() else None,
                        finite=True))
                    if arm == 'strict':
                        continue
                    for ref in (1, 2, 3):
                        distances = np.linalg.norm(a - x['strict', ref], axis=1)
                        speed_diff = np.linalg.norm(v[arm, repeat] - v['strict', ref], axis=1)
                        for name, mask in partition.items():
                            if not mask.any():
                                continue
                            output.append(dict(scene=scene, method=arm, repeat=repeat, reference_repeat=ref, frame=frame,
                                partition=name, position_rms_mm=1000 * float(np.sqrt(np.mean(distances[mask] ** 2))),
                                position_max_mm=1000 * float(distances[mask].max()),
                                velocity_rms_m_per_s=float(np.sqrt(np.mean(speed_diff[mask] ** 2))), velocity_max_m_per_s=float(speed_diff[mask].max())))
    return output, quality, reference_dispersion


def checkpoint_rows(session, ledger):
    require(ledger['complete'] and len(ledger['runs']) == 63, 'Incomplete cold comparison')
    rows, overhead, gate, hot_cold, reference_dispersion = [], [], [], [], []
    by = {r['task']['name']: r for r in ledger['runs']}
    for scene in SELECTED:
        cp = session / f'performance_{scene.lower()}_original_r1/checkpoint'
        meta = read(cp / 'checkpoint.json')
        checkpoint_evidence(cp)
        require(meta['next_physical_frame'] == 100 and meta['completed_physical_frames'] == 99, 'Wrong checkpoint boundary')
        for left, right in itertools.combinations((1, 2, 3), 2):
            reference_dispersion.append(dict(scene=scene, reference_left=left, reference_right=right,
                **position_error(session / f'matched_{scene.lower()}_strict_r{left}/trace/state_0001.bin',
                                 session / f'matched_{scene.lower()}_strict_r{right}/trace/state_0001.bin')))
        for run in ledger['runs']:
            t = run['task']
            if t['scene_key'] != scene:
                continue
            folder = session / t['name']
            restored = read(folder / 'output/checkpoint_restore.json')
            request = read(folder / 'requested.json')
            checks = dict(vertices=(cp / 'vertices.bin').read_bytes() == (folder / 'trace/state_0000.bin').read_bytes(),
                velocities=(cp / 'velocities.bin').read_bytes() == (folder / 'trace/velocity_0000.bin').read_bytes(),
                device_bytes=restored['device_bytes_verified'], context=restored['context_verified'],
                binary=restored['binary_id'] == meta['binary_id'] == request['exe_sha256'],
                physical_frame=restored['completed_physical_frames'] == meta['completed_physical_frames'] == 99,
                all_buffers=restored['restored_buffers'] == len(meta['buffers']))
            require(all(checks.values()), 'Cold checkpoint input gate failed')
            gate.append(dict(run=t['name'], checks=checks, restore=record(folder / 'output/checkpoint_restore.json')))
        for arm in ARMS[:3]:
            candidates = [by[f'matched_{scene.lower()}_{arm}_r{r}'] for r in (1, 2, 3)]
            differences = [position_error(session / c['task']['name'] / 'trace/state_0001.bin',
                session / f'matched_{scene.lower()}_strict_r{ref}/trace/state_0001.bin')
                for c, ref in itertools.product(candidates, (1, 2, 3))]
            observations = [c['observation'] for c in candidates]
            rows.append(dict(state=f'{scene}, step 100', scene=scene, method=arm,
                N=statistics.median(c['N_tot'] for c in observations), P=statistics.median(c['P_tot'] for c in observations),
                t=statistics.median(c['TimeTot'] for c in observations),
                e_RMS_mm=max(e['rms_mm'] for e in differences), e_max_mm=max(e['max_mm'] for e in differences)))
            for repeat, on in enumerate(candidates, 1):
                off = by[f'overhead_{scene.lower()}_{arm}_r{repeat}']
                a, b = on['observation'], off['observation']
                equal = physical_inputs_equal(session / on['task']['name'], session / off['task']['name'])
                require(equal['passed'], 'Overhead initial input differs')
                diff = position_error(session / on['task']['name'] / 'trace/state_0001.bin', session / off['task']['name'] / 'trace/state_0001.bin')
                selected_counters = []
                for candidate in (on, off):
                    frame = read(session / candidate['task']['name'] / 'output/stats.json')['frames'][0]
                    pcg = [n['pcg'] for n in frame['newton'] if 'pcg' in n]
                    selected_counters.append(dict(N=candidate['observation']['N_tot'], P=candidate['observation']['P_tot'],
                        newton_records=len(frame['newton']),
                        **{k: sum(n.get(k, 0) for n in frame['newton']) for k in ('energy_evaluations', 'energy_backtracks', 'intersection_backtracks')},
                        graph_cache_hits=sum(bool(p.get('graph_cache_hit')) for p in pcg),
                        graph_body_executions=sum(p.get('graph_body_executions', 0) for p in pcg)))
                overhead.append(dict(scene=scene, method=arm, repeat=repeat, on_solver_ms=1000 * a['TimeTot'],
                    off_solver_ms=1000 * b['TimeTot'], solver_ratio_on_off=a['TimeTot'] / b['TimeTot'],
                    on_N=a['N_tot'], off_N=b['N_tot'], on_P=a['P_tot'], off_P=b['P_tot'],
                    same_N_P=a['N_tot'] == b['N_tot'] and a['P_tot'] == b['P_tot'],
                    same_selected_counters=selected_counters[0] == selected_counters[1],
                    on_work_counters=json.dumps(selected_counters[0], separators=(',', ':')),
                    off_work_counters=json.dumps(selected_counters[1], separators=(',', ':')),
                    output_rms_mm=diff['rms_mm'], output_max_mm=diff['max_mm'],
                    on_process_wall_seconds=on['result']['wall_seconds'], off_process_wall_seconds=off['result']['wall_seconds']))
        for repeat in (1, 2, 3):
            cold = session / f'matched_{scene.lower()}_original_r{repeat}'
            hot = session / f'performance_{scene.lower()}_original_r1'
            hot_cold.append(dict(scene=scene, repeat=repeat,
                position=position_error(cold / 'trace/state_0001.bin', hot / 'trace/state_0100.bin'),
                actual_velocity=position_error(cold / 'trace/velocity_0001.bin', hot / 'trace/velocity_0100.bin'),
                velocity_scope='rms_mm/max_mm keys represent velocity multiplied by 1000; units mm/s'))
    return rows, overhead, dict(passed=True, exact_input_runs=len(gate), runs=gate, hot_cold=hot_cold,
        reference_repeat_dispersion=reference_dispersion,
        scope='All rule rows use cold reconstruction; hot/cold differences retained; not physical-quality certification')


def analyze(session, output):
    packet = read(session / 'identities.json')
    require(packet['plan'] == json.loads(json.dumps(plan())), 'Recorded plan changed')
    perf = read(session / 'performance_ledger.json')
    require(perf['complete'] and len(perf['runs']) == 36, 'Complete full trajectories required')
    input_gate = trajectory_input_gate(session, perf['runs'])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'trajectory_input_gate.json', input_gate)
    frames, newton, gpu, runs = frame_rows(session, perf['runs'])
    require(len(frames) == 3780, 'Missing trajectory frames')
    csv_file(output, 'per_frame', frames)
    csv_file(output, 'frame100_actual', [r for r in frames if r['physical_frame'] == 100])
    csv_file(output, 'per_newton', newton)
    csv_file(output, 'gpu_samples', gpu)
    csv_file(output, 'run_costs', runs)
    differences, quality, dispersion = errors(session)
    csv_file(output, 'trajectory_errors', differences)
    csv_file(output, 'per_frame_quality', quality)
    csv_file(output, 'reference_repeat_dispersion', dispersion)
    trajectory100, phases = [], []
    for scene in SELECTED:
        for arm in ARMS[:3]:
            final = [r for r in frames if r['scene'] == scene and r['method'] == arm and r['physical_frame'] == 100]
            err = [e for e in differences if e['scene'] == scene and e['method'] == arm and e['frame'] == 100 and e['partition'] == 'all']
            require(len(final) == 3 and len(err) == 9, 'Missing three-by-three endpoint comparison')
            trajectory100.append(dict(state=f'{scene}, step 100', scene=scene, method=arm,
                N=statistics.median(r['N'] for r in final), P=statistics.median(r['P'] for r in final),
                t=statistics.median(r['solver_ms'] / 1000 for r in final),
                e_RMS_mm=max(e['position_rms_mm'] for e in err), e_max_mm=max(e['position_max_mm'] for e in err)))
            prefix = [r for r in frames if r['scene'] == scene and r['method'] == arm and r['physical_frame'] <= 100]
            groups = [[r for r in prefix if r['repeat'] == repeat] for repeat in (1, 2, 3)]
            time = statistics.median(sum(r['solver_ms'] for r in group) / 1000 for group in groups)
            phase_row = dict(scene=scene, method=arm, **{k: statistics.median(sum(r[k + '_ms'] for r in group) / 1000 for group in groups) for k in PHASES},
                terminal_assembly=statistics.median(sum(r['terminal_assembly_ms'] for r in group) / 1000 for group in groups),
                TimeTot=time, N=statistics.median(sum(r['N'] for r in group) for group in groups),
                P=statistics.median(sum(r['P'] for r in group) for group in groups),
                P_bar=statistics.median(sum(r['P'] for r in group) / sum(r['N'] for r in group) for group in groups),
                N_bar=statistics.median(sum(r['N'] for r in group) / len(group) for group in groups),
                Misc=statistics.median(sum(r['solver_ms'] - sum(r[k + '_ms'] for k in PHASES[:4]) for r in group) / 1000 for group in groups),
                pcg_stage_us_per_iteration=statistics.median(1000 * sum(r['pcg_ms'] for r in group) / sum(r['P'] for r in group) for group in groups))
            phases.append(phase_row)
    for row in phases:
        baseline = next(r for r in phases if r['scene'] == row['scene'] and r['method'] == 'original')
        row['S_movement_over_rule'] = baseline['TimeTot'] / row['TimeTot']
    checkpoint_ledger = read(session / 'checkpoint_ledger.json')
    matched, overhead, gate = checkpoint_rows(session, checkpoint_ledger)
    cold_frames, cold_newton, cold_gpu, cold_costs = frame_rows(session,
        [r for r in checkpoint_ledger['runs'] if r['task']['recording_enabled']])
    require(len(cold_frames) == 36 and all(r['physical_frame'] == 100 for r in cold_frames), 'Missing cold physical frame100 records')
    csv_file(output, 'checkpoint_per_frame', cold_frames)
    csv_file(output, 'checkpoint_per_newton', cold_newton)
    csv_file(output, 'checkpoint_run_costs', cold_costs)
    if cold_gpu:
        csv_file(output, 'checkpoint_gpu_samples', cold_gpu)
    csv_file(output, 'trajectory_frame100_table', trajectory100)
    csv_file(output, 'checkpoint_frame100_table', matched)
    csv_file(output, 'recording_overhead', overhead)
    overhead_summary = []
    for scene in SELECTED:
        for arm in ARMS[:3]:
            pairs = [r for r in overhead if r['scene'] == scene and r['method'] == arm]
            selected = [r['solver_ratio_on_off'] for r in pairs if r['same_selected_counters']]
            overhead_summary.append(dict(scene=scene, method=arm, pairs=len(pairs),
                same_N_P_pairs=sum(r['same_N_P'] for r in pairs), same_selected_counter_pairs=len(selected),
                median_solver_ratio_all_pairs=statistics.median(r['solver_ratio_on_off'] for r in pairs),
                median_solver_ratio_selected=statistics.median(selected) if selected else None,
                minimum_solver_ratio_selected=min(selected) if selected else None,
                maximum_solver_ratio_selected=max(selected) if selected else None))
    csv_file(output, 'recording_overhead_summary', overhead_summary)
    csv_file(output, 'phase_table_first100', phases)
    write_new(output / 'checkpoint_gate.json', gate)
    costs = read(session / 'cost_ledger.json')
    require(costs['complete'] and len(costs['runs']) == 9, 'Missing selected cost diagnostics')
    detailed = []
    for run in costs['runs']:
        _, folder, _, _, _, _, _ = read_run(session, run)
        with (folder / 'cost.jsonl').open() as f:
            for line in f:
                if line.strip():
                    native = json.loads(line)
                    require(native['schema'] == 'gipc.cost.v1' and native['frame'] == 100
                            and native['diagnostic_only'] and native['gpu_events_enabled']
                            and not native['operator_probe_enabled'], 'Unexpected cost diagnostic scope')
                    require(math.isfinite(native['cpu_submit_ms']) and native['cpu_submit_ms'] >= 0
                            and (native['gpu_interval_ms'] is None or (math.isfinite(native['gpu_interval_ms'])
                                 and native['gpu_interval_ms'] >= 0)), 'Invalid cost interval')
                    detailed.append(dict(run=run['task']['name'], scene=run['task']['scene_key'], method=run['task']['arm'],
                        **{k: native.get(k) for k in ('stage', 'frame', 'outer', 'inner', 'scope_id', 'parent_scope_id',
                           'pcg_iteration', 'index', 'cpu_submit_ms', 'gpu_interval_ms', 'inclusive', 'sample_kind',
                           'contact_model_id', 'linear_system_id', 'measurement_mode')}, native_json=line.strip()))
    csv_file(output, 'cost_diagnostics', detailed)
    for identity in packet['programs'].values():
        verify(identity['sources'] + identity['build_evidence'] + [identity['exe']] + identity['dlls'] + identity.get('objects', []))
    verify([packet['controller']])
    summary = dict(complete=True, selected_scenes=list(SELECTED), excluded_scenes=['Sphere','Table','Stack','Hang'],
        dt=.04, candidate_runs=27, reference_runs=9, frames_per_trajectory=105,
        physical_frame_for_tables=100, trajectory_frames=3780, trajectory100=trajectory100,
        trajectory_initial_input_gate_passed=input_gate['passed'],
        exact_trajectory_initial_runs=input_gate['exact_initial_runs'],
        matched100=matched, phases_first100=phases, checkpoint_input_gate_passed=gate['passed'],
        exact_checkpoint_runs=gate['exact_input_runs'], logger_overhead_pairs=len(overhead),
        logger_matching_counter_pairs=sum(r['same_selected_counters'] for r in overhead),
        cost_diagnostic_runs=len(costs['runs']), identity=record(session / 'identities.json'),
        analysis_controller=record(__file__),
        error_scope='Worst of nine pairings versus movement-only rho=1e-6; not exact solution',
        reference_dispersion_frame100=[r for r in dispersion if r['frame'] == 100 and r['partition'] == 'all'],
        phase_scope='assembly includes terminal assembly; terminal_assembly is now a subset, not added twice',
        performance_certified=False, physical_quality_certified=False,
        output_sha256={p.name: sha(p) for p in output.glob('*.csv')})
    write_new(output / 'summary.json', summary)
    print(json.dumps(dict(complete=True, frames=len(frames), newton_records=len(newton), error_rows=len(differences), matched_rows=len(matched), checkpoint_inputs=gate['exact_input_runs'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=ROOT / 'runs/local_recording105_dt0p04_20261011')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_recording105_dt0p04_20261011')
    args = parser.parse_args()
    analyze(args.session.resolve(), args.output.resolve())
