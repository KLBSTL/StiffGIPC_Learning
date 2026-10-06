"""CPU-only A/b and 14-solution reference for a protected graph-chunk study.

The public CLI owns one Windows Job, stops it at 60 seconds or 2048 MiB sampled
aggregate RSS, and never overwrites an output. Default PCG solutions are reported
at their actual accuracy; only the independent reference must reach 1e-8.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

from cpu_fixed_reference import (REFERENCE_RESIDUAL_LIMIT, decimal_residual,
                                 fnv1a64, load_snapshot, record)

TIMEOUT_SECONDS = 60
MEMORY_MIB = 2048
RESTORATION_KEYS = (
    'full_system_restored', 'primary_restored_bitwise', 'workspace_restored',
    'production_graph_restored', 'operator_signature_restored',
    'production_info_restored', 'solve_context_restored', 'config_restored',
    'cost_trace_restored',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tool_identity():
    directory = Path(__file__).resolve().parent
    return {name: record(directory / name) for name in
            ('graph_chunk_cpu_reference.py', 'cpu_fixed_reference.py', 'windows_owned_job.py')}


def load_study(prefix):
    """Fail closed on incomplete schedules, stale paths, or unprotected output."""
    prefix = Path(prefix).resolve()
    path = Path(str(prefix) + '_graph_chunk_study.json')
    study = json.loads(path.read_text(encoding='utf-8'))
    require(study.get('schema') == 'fixed_graph_chunk_v1' and study.get('passed') is True,
            'Native study not completed/passed')
    require(all(study.get(k) is True for k in RESTORATION_KEYS), 'Missing restoration proof')
    require(study.get('restoration_error') == '', 'Native restoration error')
    require(study.get('system') == study.get('system_after'), 'Native full system changed')
    require(study['system'].get('preconditioner_export_complete') is True, 'Incomplete native M snapshot')
    require(study.get('mas') == 'legacy' and study.get('zero_initial_guess') is True,
            'Expected frozen legacy MAS and zero initial guess')
    require(study.get('rho_tolerance') == 1e-4, 'PCG rho was changed')
    meta = json.loads(Path(str(prefix) + '_meta.json').read_text(encoding='utf-8'))
    require(meta == study['system'], 'A/b metadata differs from native study identity')
    require(study.get('warmup_pairs') == 2 and study.get('measured_pairs') == 7,
            'Unexpected replay budget')
    runs = study.get('runs', [])
    require(len(runs) == 18, 'Expected 4 warmup and 14 measured solves')
    expected = [(warm, pair, k) for warm, count in ((True, 2), (False, 7))
                for pair in range(1, count + 1) for k in (1, 4)]
    actual = [(r.get('warmup'), r.get('pair'), r.get('chunk')) for r in runs]
    require(sorted(actual) == sorted(expected), 'Incomplete or duplicated replay arm/pair')
    require(all(r.get('passed') is True and r.get('finite') is True for r in runs),
            'Native solve failed')
    measured = []
    for row in runs:
        require(row.get('cache_contract_passed') is True, 'Cache contract failed')
        require(isinstance(row.get('iterations'), int) and 0 <= row['iterations'] < study['max_iter'],
                'Invalid iteration count')
        require(not row.get('pcg', {}).get('iteration_limit', False)
                and 'breakdown' not in row.get('pcg', {}), 'Native PCG breakdown/limit')
        if row['warmup']:
            continue
        expected_path = Path(str(prefix) + f"_k{row['chunk']}_pair{row['pair']}_x.bin")
        path = Path(row['solution_file'])
        require(not path.is_symlink() and path.resolve() == expected_path.resolve(),
                'Solution is not the exact expected local study file')
        measured.append((row, expected_path))
    return study, measured, record(Path(str(prefix) + '_graph_chunk_study.json'))


def verify_evidence(evidence):
    for entry in evidence.values():
        require(record(entry['path']) == entry, 'CPU input changed during evaluation: ' + entry['path'])


def solve(prefix):
    import numpy as np
    import scipy
    from scipy.sparse.linalg import splu

    started = time.monotonic()
    tools_before = tool_identity()
    prefix = Path(prefix).resolve()
    study, rows, study_identity = load_study(prefix)
    # Reuses the existing loader's exact size/FNV/ABI checks, symmetric block
    # expansion, and exact zero-row/column + zero-RHS elimination policy.
    matrix, rhs, active, zero, inputs, details = load_snapshot(prefix)
    inputs['study'] = study_identity
    output = {
        'schema': 'graph_chunk_cpu_reference.v1', 'status': 'completed', 'passed': False,
        'cpu_only': True, 'prefix': str(prefix), 'tools': tools_before,
        'matrix': details, 'inputs': inputs, 'saved_solutions': [],
        'reference_method': 'FP64 SuperLU; at most two improvements using independent Decimal50 residual',
        'residual_method': 'Decimal precision 50 scalar CSR dot and norms; no GPU or scipy matvec',
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'numpy_longdouble_precision_bits': np.finfo(np.longdouble).nmant + 1,
        'reference_residual_limit': REFERENCE_RESIDUAL_LIMIT,
        'default_pcg_true_residual_requirement': None,
        'default_pcg_rho_tolerance': study['rho_tolerance'],
        'condition_number_certified': False, 'physics_certified': False,
        'performance_certified': False,
        'pass_scope': 'Reference residual <=1e-8, exact input identity and 14 finite independently evaluated solutions; not a K4 numerical-equivalence or physics gate',
    }
    for name in ('requested.json', 'build_manifest.json'):
        path = prefix.parent.parent / name
        if path.exists():
            inputs[name] = record(path)
            if name == 'requested.json':
                request = json.loads(path.read_text(encoding='utf-8'))
                output['native_identity'] = {key: request.get(key) for key in
                                             ('source_digest', 'exe_sha256', 'config_sha256', 'from_zero')}
    output.setdefault('native_identity', {'available': False,
                                         'reason': 'requested.json not present alongside fixed/ directory'})
    factor = splu(matrix[active][:, active].tocsc()) if active.any() else None
    reference = np.zeros(len(rhs))
    if factor is not None:
        reference[active] = factor.solve(rhs[active])
    residual, metrics = decimal_residual(matrix, rhs, reference)
    history = [metrics]
    for _ in range(2):
        if metrics['relative_l2'] <= 1e-12 or factor is None:
            break
        candidate = reference.copy()
        candidate[active] += factor.solve(residual[active])
        next_residual, next_metrics = decimal_residual(matrix, rhs, candidate)
        history.append(next_metrics)
        if next_metrics['relative_l2'] >= metrics['relative_l2']:
            break
        reference, residual, metrics = candidate, next_residual, next_metrics
    output['reference'] = {**metrics, 'refinement_evaluations': history,
                           'residual_passed': metrics['relative_l2'] <= REFERENCE_RESIDUAL_LIMIT,
                           'lu_nnz': int(factor.L.nnz + factor.U.nnz) if factor is not None else 0}
    norm_reference = float(np.linalg.norm(reference[active]))
    require(math.isfinite(norm_reference), 'Nonfinite reference norm')
    for row, path in sorted(rows, key=lambda item: (item[0]['pair'], item[0]['chunk'])):
        raw = path.read_bytes()
        identity = row['solution_identity']
        require(len(raw) == identity['bytes'] == len(rhs) * 8
                and fnv1a64(raw) == identity['fnv1a64'], 'Solution size or hash mismatch')
        candidate = np.frombuffer(raw, dtype='<f8')
        _, actual = decimal_residual(matrix, rhs, candidate)
        error = float(np.linalg.norm((candidate - reference)[active]))
        relative_error = error / norm_reference if norm_reference else error
        max_error = float(np.max(np.abs(candidate - reference))) if len(rhs) else 0.0
        require(math.isfinite(relative_error) and math.isfinite(max_error), 'Nonfinite solution error')
        evidence_key = f"solution_k{row['chunk']}_pair{row['pair']}"
        inputs[evidence_key] = record(path)
        output['saved_solutions'].append({
            'chunk': row['chunk'], 'pair': row['pair'], 'order': row.get('order'),
            'iterations': row['iterations'], 'solution': inputs[evidence_key],
            'cpu_true_residual': actual,
            'native_true_relative_residual': row['true_relative_residual'],
            'relative_error_to_cpu_reference': relative_error,
            'max_abs_error_to_cpu_reference': max_error,
            'exact_zero_row_solution_max_abs': float(np.max(np.abs(candidate[zero]))) if len(zero) else 0.0,
            'residual_at_most_1e_minus_8': actual['relative_l2'] <= REFERENCE_RESIDUAL_LIMIT,
        })
    verify_evidence(inputs)
    require(tool_identity() == tools_before, 'CPU reference tool changed during worker')
    output['passed'] = output['reference']['residual_passed'] and len(output['saved_solutions']) == 14
    output['wall_seconds_diagnostic_only'] = time.monotonic() - started
    return output


def bounded_worker(prefix, seconds=TIMEOUT_SECONDS, memory_mib=MEMORY_MIB):
    import psutil
    from windows_owned_job import OwnedJob
    require(0 < seconds <= TIMEOUT_SECONDS and 0 < memory_mib <= MEMORY_MIB, 'Invalid CPU budget')
    started = time.monotonic()
    deadline = started + seconds
    peak = 0
    tools_before = tool_identity()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    guard = None
    with tempfile.TemporaryDirectory(prefix='graph_chunk_cpu_') as temporary:
        folder = Path(temporary)
        worker_output = folder / 'worker.json'
        with (folder / 'worker.log').open('wb') as log, OwnedJob() as job:
            job.launch([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                        '--prefix', str(Path(prefix).resolve()), '--worker-output', str(worker_output)],
                       folder, env, log)
            while True:
                if time.monotonic() >= deadline:
                    guard = 'timeout'
                    break
                rss = 0
                job.observe()
                for pid in job.pids():
                    if not job.owns_pid(pid):
                        continue
                    try:
                        rss += psutil.Process(pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if peak > memory_mib * 2**20:
                    guard = 'memory_budget'
                    break
                if time.monotonic() >= deadline:
                    guard = 'timeout'
                    break
                if job.finished():
                    break
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            if guard:
                job.terminate()
            exit_code = job.poll()
            members = job.member_evidence()
            lifecycle = list(job.lifecycle)
        if guard:
            result = {'status': guard, 'passed': False, 'saved_solutions': [],
                      'coverage': 'Worker stopped; no partial reference pass inferred'}
        elif worker_output.exists() and exit_code == 0:
            result = json.loads(worker_output.read_text(encoding='utf-8'))
        else:
            result = {'status': 'error', 'passed': False, 'saved_solutions': [],
                      'error': (folder / 'worker.log').read_text(encoding='utf-8', errors='replace')[-2000:]}
    require(tool_identity() == tools_before, 'CPU reference tool changed during execution')
    result.update(schema='graph_chunk_cpu_reference.v1', cpu_only=True, tools=tools_before,
                  performance_certified=False, physics_certified=False)
    result['resource_guard'] = {
        'timeout_seconds': seconds, 'memory_budget_mib': memory_mib,
        'peak_sampled_aggregate_rss_mib': peak / 2**20, 'rss_sampling_seconds': .05,
        'elapsed_seconds_including_owned_cleanup': time.monotonic() - started,
        'ownership': 'Windows Job Object; suspended launch then assignment; owned tree termination only',
        'members': members, 'launch_lifecycle': lifecycle, 'worker_exit_code': exit_code,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_output:
        try:
            result = solve(args.prefix)
        except Exception as exc:
            result = {'status': 'error', 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
        write_new(args.worker_output, result)
        return 0
    if args.output is None:
        parser.error('--output is required')
    if args.output.exists():
        raise FileExistsError('Preserve existing CPU reference evidence: ' + str(args.output))
    result = bounded_worker(args.prefix)
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'passed': result['passed'],
                      'reference': result.get('reference'),
                      'solutions': len(result.get('saved_solutions', [])),
                      'output': str(args.output.resolve())}, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
