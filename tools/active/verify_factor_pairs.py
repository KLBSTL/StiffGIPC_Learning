"""CPU-only verification of exported, protected factor-action PCG pairs.

The rho stopping threshold is recorded verbatim. No absolute true-residual
acceptance threshold is imposed on default PCG solutions. `passed` validates
evidence integrity, finiteness and agreement with the GPU's reported residual;
it does not certify candidate convergence quality, trajectory or performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import ROOT, read, sha
from verify_systems import matrix_from_snapshot


MODES = {'host': 0, 'graph': 1}
ARMS = {'triangular': 'triangular', 'factor_inverse': 'inverse'}


def relative_difference(values, reference):
    delta = float(np.linalg.norm(values - reference))
    scale = float(np.linalg.norm(reference))
    return delta / scale if scale > 0 else delta


def bounds(values):
    return [min(values), max(values)]


def verify(path):
    study = read(path)
    prefix = Path(str(path)[:-len('_study.json')])
    meta, matrix, rhs = matrix_from_snapshot(prefix)
    pairs = study['factor_action_pairs']
    expected = {(mode, arm, pair) for mode in MODES for arm in ARMS for pair in (1, 2, 3)}
    keys = [(r['mode'], r['factor_action'], r['pair']) for r in pairs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Expected exactly 12 unique host/Graph, triangular/inverse, three-repeat records')
    if matrix.shape != (rhs.size, rhs.size) or rhs.size != meta['dofs']:
        raise ValueError('Snapshot A/b dimensions disagree')
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise FloatingPointError('Nonfinite saved A/b')
    inputs = [path] + [Path(str(prefix) + suffix) for suffix in
                      ('_meta.json', '_rows.bin', '_cols.bin', '_values.bin', '_rhs.bin')]
    solutions = {}
    files = {}
    for mode, arm, pair in expected:
        output = Path(f'{prefix}_factor_{ARMS[arm]}_m{MODES[mode]}_r{pair}_x.bin')
        values = np.fromfile(output, dtype='<f8')
        if values.shape != rhs.shape or not np.isfinite(values).all():
            raise ValueError(f'Missing, wrong-size or nonfinite exported solution: {output}')
        solutions[mode, arm, pair] = values
        files[mode, arm, pair] = output
        inputs.append(output)

    norm_b = float(np.linalg.norm(rhs))
    denominator = norm_b if norm_b > 0 else 1.0
    absolute_matrix = abs(matrix)
    maximum_row_terms = int(np.diff(matrix.indptr).max(initial=0))
    epsilon = np.finfo(np.float64).eps
    rows = []
    for record in pairs:
        mode, arm, pair = record['mode'], record['factor_action'], record['pair']
        values = solutions[mode, arm, pair]
        actual = float(np.linalg.norm(rhs - matrix @ values) / denominator)
        reported = float(record['true_relative_residual'])
        # CPU and GPU SpMV sum terms in different orders. This scale-dependent
        # envelope checks reporting agreement; it is NOT a solver tolerance.
        arithmetic_scale = float(np.linalg.norm(absolute_matrix @ np.abs(values) + np.abs(rhs)) / denominator)
        agreement_limit = 32 * epsilon * max(maximum_row_terms, 1) * max(arithmetic_scale, 1.0)
        disagreement = abs(actual - reported)
        reference = solutions[mode, 'triangular', 1]
        difference = relative_difference(values, reference)
        recorded_difference = float(record['triangular_difference']['relative'])
        difference_limit = 512 * epsilon * max(1.0, difference, recorded_difference)
        finite = bool(np.isfinite([actual, reported, arithmetic_scale, difference, recorded_difference]).all())
        if not finite:
            raise FloatingPointError('Nonfinite CPU/GPU residual or solution comparison')
        iterations = int(record['iterations'])
        rho_tolerance = float(record['rho_tolerance'])
        rows.append({
            'mode': mode, 'factor_action': arm, 'pair': pair,
            'solution_file': str(files[mode, arm, pair].resolve()),
            'iterations': iterations, 'rho_tolerance': rho_tolerance,
            'solve_ms': float(record['solve_ms']),
            'cpu_true_relative_residual': actual, 'gpu_true_relative_residual': reported,
            'residual_absolute_disagreement': disagreement,
            'residual_reporting_agreement_limit': agreement_limit,
            'residual_reporting_agrees': bool(finite and disagreement <= agreement_limit),
            'relative_difference_to_first_triangular': difference,
            'reported_difference_to_first_triangular': recorded_difference,
            'difference_reporting_agrees': bool(finite and abs(difference-recorded_difference) <= difference_limit),
            'finite': finite,
            'passed': bool(finite and iterations >= 0 and rho_tolerance > 0
                           and disagreement <= agreement_limit
                           and abs(difference-recorded_difference) <= difference_limit),
        })

    groups = []
    for mode in MODES:
        arm_summaries = {}
        for arm in ARMS:
            selected = [r for r in rows if r['mode'] == mode and r['factor_action'] == arm]
            pairwise = [relative_difference(solutions[mode, arm, j], solutions[mode, arm, i])
                        for i in (1, 2, 3) for j in (1, 2, 3) if j > i]
            arm_summaries[arm] = {
                'iteration_range': bounds([r['iterations'] for r in selected]),
                'cpu_true_relative_residual_range': bounds([r['cpu_true_relative_residual'] for r in selected]),
                'rho_tolerances': sorted({r['rho_tolerance'] for r in selected}),
                'max_repeat_pairwise_relative_solution_difference': max(pairwise),
                'max_relative_difference_to_first_triangular': max(r['relative_difference_to_first_triangular'] for r in selected),
            }
        comparisons = []
        for pair in (1, 2, 3):
            old = next(r for r in rows if r['mode'] == mode and r['factor_action'] == 'triangular' and r['pair'] == pair)
            new = next(r for r in rows if r['mode'] == mode and r['factor_action'] == 'factor_inverse' and r['pair'] == pair)
            if old['rho_tolerance'] != new['rho_tolerance']:
                raise ValueError('A paired comparison changed the PCG rho stopping threshold')
            comparisons.append({
                'pair': pair,
                'candidate_relative_solution_difference': relative_difference(
                    solutions[mode, 'factor_inverse', pair], solutions[mode, 'triangular', pair]),
                'candidate_iteration_delta': new['iterations'] - old['iterations'],
                'candidate_true_residual_delta': new['cpu_true_relative_residual'] - old['cpu_true_relative_residual'],
                'candidate_true_residual_ratio': (new['cpu_true_relative_residual'] / old['cpu_true_relative_residual']
                                                  if old['cpu_true_relative_residual'] > 0 else None),
                'candidate_residual_within_triangular_repeat_range': bool(
                    arm_summaries['triangular']['cpu_true_relative_residual_range'][0]
                    <= new['cpu_true_relative_residual']
                    <= arm_summaries['triangular']['cpu_true_relative_residual_range'][1]),
            })
        groups.append({'mode': mode, 'arms': arm_summaries, 'paired_comparisons': comparisons})
    unchanged = study.get('system_unchanged') is True
    restored = study.get('primary_restored_bitwise') is True
    metadata_matches = study['system'] == meta
    return {
        'study': str(path.resolve()), 'frame': study['frame'], 'direction': study['direction'],
        'outer': study.get('outer'), 'inner': study.get('inner'),
        'linear_system_id': study.get('linear_system_id'), 'dofs': int(rhs.size),
        'maximum_matrix_row_terms': maximum_row_terms,
        'system_unchanged': unchanged, 'primary_restored_bitwise': restored,
        'study_snapshot_metadata_matches_file': metadata_matches,
        'rows': rows, 'comparisons': groups,
        'input_sha256': {str(p.resolve()): sha(p) for p in sorted(set(inputs))},
        'passed': bool(unchanged and restored and metadata_matches and all(r['passed'] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', help='Fixed-system directories containing *_study.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({p for folder in args.directories for p in (ROOT/folder).glob('*_study.json')})
    if not paths:
        parser.error('No fixed-system studies found')
    output = ROOT / args.output
    with output.open('x', encoding='utf-8') as stream:
        stream.write('{}\n')
    result = {
        'scope': 'Independent FP64 A/b residual reporting and relative comparison to stable triangular PCG',
        'cpu_only': True, 'physical_certified': False, 'performance_certified': False,
        'default_pcg_absolute_true_residual_threshold': None,
        'passed_meaning': 'Evidence integrity, finite exported solutions and CPU/GPU reporting agreement only',
        'reporting_agreement_rule': '32 * FP64 epsilon * max row nnz * max(norm(abs(A)abs(x)+abs(b))/norm(b),1)',
        'repeat_envelope_note': 'Strict observed stable range is reported descriptively, without widening it or using it as an automatic acceptance gate',
        'verifier_sha256': sha(Path(__file__)), 'systems': [], 'passed': True,
    }
    for path in paths:
        try:
            row = verify(path)
        except Exception as error:
            row = {'study': str(path.resolve()), 'passed': False,
                   'error': f'{type(error).__name__}: {error}'}
        result['systems'].append(row)
        result['passed'] &= row['passed']
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'study': path.name, 'passed': row['passed'], 'error': row.get('error')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
