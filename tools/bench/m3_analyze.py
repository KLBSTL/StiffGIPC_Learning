"""Independently re-open M3 inputs/solutions; export every accepted comparison.

This does not run a native simulation or change any captured data. Output must
be fresh. Timings are diagnostics; conditional GPU budgets are not speedups.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import time

import numpy as np

from m3_numeric import ROOT, FrozenSystem, allow_transfer, need, residual_metrics, rel
from cpu_fixed_reference import decimal_residual
from local_identity import record, verify
from linux_runner import write_new
from m0_analyze import csv_write

REPORT = ROOT / 'reports/innovation_m3_20261015'
RAW = ROOT / 'runs/innovation_m3_20261015'
OUT = REPORT / 'verified'
FRAMES = dict(Bunny=14, Mixed=15, Boxes=17)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def frozen_path(scene, kind, frame, direction):
    return RAW / f'{kind}_{scene.lower()}/fixed' / f'f{frame}_n{direction}'


def validate_capture():
    seal = read(RAW / 'seal.json')
    identity = seal['identity']
    verify(identity['sources'] + [identity['exe']] + identity['dlls'])
    # Validate current native sources against the compiled original source
    # bytes, independently of the frozen copies used above.
    verify(identity['frozen_source']['original_sources'])
    verify([seal['controller'], seal['parent']])
    ledger = read(RAW / 'ledger.json')
    need(ledger['complete'] and len(ledger['runs']) == 6, 'Incomplete capture')
    rows = []
    for entry in ledger['runs']:
        task, result = entry['task'], entry['result']
        folder = RAW / task['name']
        need(entry['passed'] and result['status'] == 'completed' and
             result['exit_code'] == 0 and result['cleanup_owned_job_empty'], 'Native capture failed')
        need(read(folder / 'config_validation.json')['passed'], 'Configuration mismatch')
        power = read(folder / 'power_samples.json')
        need(power['expected'] == power['after'] == seal['power_guid'] and power['samples'] and
             all(s['guid'] == seal['power_guid'] for s in power['samples']), 'Capture power changed')
        if task['kind'] == 'adjacent':
            restore = entry['restore']
            need(restore['passed'] and restore['report']['restored_buffers'] == 40 and
                 restore['report']['device_bytes_verified'] and restore['report']['context_verified'],
                 'Incomplete checkpoint restoration')
            verify([restore['checkpoint_metadata']])
        verify(entry['systems'])
        for saved in entry['systems']:
            d = read(saved['path'])
            need(d['system_unchanged'] and d['primary_restored_bitwise'], 'Native fixed-study mutation')
            if task['kind'] == 'ordinary':
                need(d['full_snapshot_unchanged_including_scratch'] and d['z_restored_bitwise'] and
                     d['graph_signature_restored'], 'Ordinary stage scratch restoration failed')
            else:
                need(all(not r['error'] and not r['limit'] for r in d['runs']), 'Native replay failed')
            rows.append(dict(task=task['name'], frame=d['frame'], direction=d['direction'],
                system_unchanged=True, primary_restored_bitwise=True,
                full_scratch_restored=d['full_snapshot_unchanged_including_scratch'],
                power_guid=seal['power_guid'], study_sha256=saved['sha256']))
    need(len(rows) == 9, 'Expected nine frozen systems')
    return rows, identity['exe']


def main():
    started = time.perf_counter()
    need(not OUT.exists(), 'Fresh verification directory required')
    OUT.mkdir()
    native, executable = validate_capture()
    for name in ('analysis_v2', 'full_cg_control', 'arnoldi_control'):
        plan = read(REPORT / name / 'plan.json')
        verify(plan['dependencies'])
        status = 'ledger.json' if name == 'analysis_v2' else 'summary.json'
        need(read(REPORT / name / status)['complete'], 'Incomplete diagnostic: ' + name)
    # Old v1 sources are retained under new paths, so compare bytes/hash rather
    # than pretending that their former working paths still contain v1.
    old_plan = read(REPORT / 'analysis/plan.json')
    for original in old_plan['dependencies']:
        archived = record(REPORT / 'prior_cpu_v1' / Path(original['path']).name)
        need((archived['bytes'], archived['sha256']) == (original['bytes'], original['sha256']),
             'Prior v1 source archive changed')

    methods, matrix_rows, calibrations, decimal_checks, transfers, spectral = [], [], [], [], [], []
    replays = []
    residual_count = 0
    old_decimal_count = 0
    maximum_metric_difference = 0.

    def validate_solution(system, method, tolerance, scene, label, family):
        nonlocal residual_count, maximum_metric_difference
        need(method.get('converged') is True, 'Unconverged method: ' + str(method))
        verify([method['solution']])
        x = np.load(method['solution']['path'], allow_pickle=False)
        need(x.shape == system.b.shape and np.isfinite(x).all(), 'Invalid saved solution shape/data')
        measured, _ = residual_metrics(system.A, system.b, x, system.apply)
        need(max(measured.values()) <= tolerance, 'Explicit original-A residual target failed')
        delta = max(abs(measured[k] - method[k]) for k in measured)
        need(delta < 1e-12, 'Saved/recomputed residual mismatch')
        maximum_metric_difference = max(maximum_metric_difference, delta)
        residual_count += 1
        return x

    def method_row(scene, label, family, m, d, name):
        n, rank = d['matrix']['active_dofs'], m.get('actual_rank', 0)
        base = d['methods'][0]['iterations']
        unit = d['native_default_iteration_ms']
        iterations = m['iterations']
        saved = (base - iterations) * unit if unit is not None else None
        return dict(scene=scene, label=label, family=family, method=name,
            baseline_iterations=base, iterations=iterations,
            iteration_ratio=base / iterations, requested_rank=m.get('requested_rank', 0), actual_rank=rank,
            relative_l2=m['relative_l2'], common_MAS_relative=m['common_MAS_relative'],
            CPU_solve_seconds=m['seconds'], CPU_setup_seconds=m.get('setup_seconds'),
            CPU_construction_seconds=(m.get('construction') or {}).get('seconds'),
            Z_W_bytes=16*n*rank, logical_matrix_bytes_per_apply=32*n*rank,
            native_assumed_ms_per_iteration=unit, conditional_saved_GPU_ms_before_all_setup=saved,
            conditional_apply_budget_us=(1000*saved/iterations if saved is not None else None),
            scope='Offline iterations; native average iteration cost assumed; no GPU acceleration certified')

    for scene, frame in FRAMES.items():
        previous = None
        arnoldi = read(REPORT / 'arnoldi_control' / scene / 'result.json')
        verify([arnoldi['basis']])
        need(arnoldi['spectral']['status'] == 'complete' and arnoldi['spectral']['qualified'] == 32,
             'Keep and report incomplete Arnoldi result instead of a complete oracle table')
        for kind, physical_frame, direction in (('ordinary',1,1), ('adjacent',frame,1), ('adjacent',frame,2)):
            label = f'{kind}_f{physical_frame}_n{direction}'
            case = REPORT / 'analysis_v2' / scene / label
            d = read(case / 'result.json')
            verify(d['input_records'] + [d['source_packet']])
            system = FrozenSystem(frozen_path(scene, kind, physical_frame, direction))
            before = [record(Path(r['path'])) for r in d['input_records']]
            need(d['context'] == {k: system.meta[k] for k in d['context']}, 'Wrong system context')
            packet = read(d['source_packet']['path'])
            verify([packet['bases']])
            need(packet['source_tolerance'] == .01 and packet['target_solution_used'] is False,
                 'Oracle contaminated practical source')
            matrix_rows.append(dict(scene=scene, label=label, dofs=len(system.b),
                scalar_nnz=system.A.nnz, exact_zero_rows=len(system.details['exact_zero_rows']),
                relative_A_asymmetry=system.details['relative_frobenius_asymmetry'],
                maximum_absolute_A_asymmetry=system.details['max_absolute_asymmetry'],
                minimum_local_inverse_eigenvalue=min(v['minimum_eigenvalue'] for v in system.local_checks),
                maximum_local_symmetrization=max(v['explicit_symmetrization_relative'] for v in system.local_checks),
                A_symmetrized=False, negative_eigenvalues_clipped=False,
                native_spmv_us=(d['native_operator_us'] or {}).get('spmv'),
                native_MAS_us=(d['native_operator_us'] or {}).get('preconditioner'),
                source_Ritz_seconds=packet['source_ritz_info']['seconds'],
                source_POD_seconds=packet['source_pod_info']['seconds']))
            for c in d['native_calibration']:
                calibrations.append(dict(scene=scene,label=label,**c))
            for m in d['methods']:
                x = validate_solution(system,m,.01,scene,label,'sampled')
                methods.append(method_row(scene,label,'sampled',m,d,m['method']))
                if m['method'].startswith('transfer_'):
                    need(previous is not None and m['source'] == previous['packet_record'], 'Wrong transfer source')
                    allow_transfer(previous['context'],d['context'])
                    need(packet['layout'] == previous['packet']['layout'], 'Transfer changed global layout')
                    verify([m['source'],previous['packet']['bases']])
            validate_solution(system,d['reference'],1e-8,scene,label,'high_accuracy_reference')
            old_decimal_count += len(d['independent_scalar_residuals'])
            for c in d['independent_scalar_residuals']:
                need(c['relative_l2'] <= .01, 'Original Decimal validation failed')
            full = read(REPORT / 'full_cg_control' / scene / label / 'result.json')
            verify([full['parent'],full['source_packet']['basis_file']])
            for scope, info in full['basis_diagnostics'].items():
                need(info['replay_solution_bitwise_equal'] and info['maximum_rho_relative_replay_error'] == 0,
                     'Nonidentical full-CG replay')
                replays.append(dict(scene=scene,label=label,scope=scope,steps=info['steps'],
                    raw_modes=info['raw_ritz_modes'],actual_rank=info['independent_rank'],
                    replay_bitwise_equal=True,construction_seconds=info['construction_seconds']))
            for m in full['methods']:
                validate_solution(system,m,.01,scene,label,'full_CG')
                methods.append(method_row(scene,label,'full_CG',m,d,m['method']))
            if kind == 'adjacent':
                for m in arnoldi['methods']:
                    if m['direction'] != direction:
                        continue
                    validate_solution(system,m,.01,scene,label,'Arnoldi')
                    name = f"{m['source_kind']}_{m['space']}_{m['requested_rank']}"
                    methods.append(method_row(scene,label,'Arnoldi',m,d,name))
                if direction == 1:
                    # Recompute the invariant subspace defect after the saved
                    # modes were reorthogonalized, independently of eigs checks.
                    z = np.load(arnoldi['basis']['path'],allow_pickle=False)
                    cz = np.column_stack([system.apply(system.A@v) for v in z.T])
                    h = z.T@cz
                    defect = rel(cz,z@h)
                    need(np.linalg.norm(z.T@z-np.eye(32)) < 1e-10 and defect < 1e-4,
                         'Saved spectral space failed independent orthogonality/invariance')
                    spectral.append(dict(scene=scene,qualified_rank=32,
                        spectral_seconds=arnoldi['spectral']['seconds'],matvec_calls=arnoldi['spectral']['calls'],
                        max_original_eigen_residual=max(v['relative_eigen_residual'] for v in arnoldi['spectral']['eigenpairs']),
                        saved_space_relative_invariance_defect=defect))
                    candidate = min((m for m in arnoldi['methods'] if m['direction']==1), key=lambda m:m['iterations'])
                    selected_name = 'Arnoldi_'+candidate['space']+'_'+str(candidate['requested_rank'])
                    previous = dict(context=d['context'],packet=packet,packet_record=d['source_packet'])
                else:
                    candidate = min((m for m in d['methods'] if m['method'].startswith('transfer_')),
                                    key=lambda m:m['iterations'])
                    selected_name = candidate['method']
                    transfers.append(dict(scene=scene,source_frame=frame,source_inner=0,target_inner=1,
                        source_iterations=previous['packet']['source_solve_iterations'],
                        best_practical_method=selected_name,target_iterations=candidate['iterations'],
                        baseline_iterations=d['methods'][0]['iterations'],
                        source_Ritz_seconds=previous['packet']['source_ritz_info']['seconds'],
                        source_POD_seconds=previous['packet']['source_pod_info']['seconds'],
                        target_solution_used=False))
                x = np.load(candidate['solution']['path'],allow_pickle=False)
                _, independent = decimal_residual(system.A,system.b,x)
                need(independent['relative_l2'] <= .01 and
                     abs(independent['relative_l2']-candidate['relative_l2']) < 1e-9,
                     'New independent Decimal scalar residual failed')
                decimal_checks.append(dict(scene=scene,label=label,method=selected_name,**independent))
            need(before == [record(Path(r['path'])) for r in d['input_records']], 'Input changed during analysis')
            print(json.dumps(dict(verified=scene+'/'+label,solutions=residual_count)),flush=True)

    # Calibrations have heterogeneous keys; preserve every original field.
    keys = sorted({k for r in calibrations for k in r})
    calibrations = [{k:r.get(k) for k in keys} for r in calibrations]
    for name,rows in (('native_capture',native),('matrices',matrix_rows),('all_methods',methods),
                      ('calibration',calibrations),('new_decimal_checks',decimal_checks),
                      ('transfer',transfers),('spectral',spectral),('full_CG_replays',replays)):
        csv_write(OUT/(name+'.csv'),rows)
    need(residual_count == 252 and len(methods) == 243 and len(replays) == 18,
         'Unexpected comparison coverage')
    write_new(OUT/'validation.json',dict(passed=True,systems=9,native_runs=6,
        candidate_solves=len(methods),saved_solution_rechecks=residual_count,
        original_decimal_checks=old_decimal_count,new_decimal_checks=len(decimal_checks),
        full_CG_bitwise_replays=len(replays),maximum_saved_metric_difference=maximum_metric_difference,
        executable=executable,seconds=time.perf_counter()-started,
        native_solver_modified=False,GPU_candidate_implemented=False,performance_certified=False,
        source=record(Path(__file__))))


if __name__ == '__main__':
    main()
