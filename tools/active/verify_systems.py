"""CPU-only A/b and stable MAS checks for historical and newly frozen systems.

No GPU execution. A small direct-solve residual certifies a linear reference,
not nonlinear convergence or physical correctness of the simulated trajectory.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

from config import ROOT, read
from fixtures import CASES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import unpack


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def accurate_reference(matrix, rhs):
    matrix = matrix.tocsc()
    matrix.sum_duplicates()
    lu = splu(matrix)
    solution = lu.solve(rhs)
    refinements = 0
    for _ in range(4):
        residual = rhs - matrix @ solution
        error = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300))
        if error <= 1e-10:
            break
        solution += lu.solve(residual)
        refinements += 1
    error = float(np.linalg.norm(rhs - matrix @ solution) / max(np.linalg.norm(rhs), 1e-300))
    if not np.isfinite(solution).all() or not np.isfinite(error):
        raise FloatingPointError('Nonfinite CPU reference or residual')
    return solution, {
        'method': 'FP64 sparse LU with at most four iterative refinements',
        'refinements': refinements, 'true_relative_residual': error,
        'passed': bool(np.isfinite(solution).all() and error <= 1e-8),
    }


def verify(prefix, fixture=None):
    required = []

    def binary(suffix, dtype='<f8'):
        path = Path(str(prefix) + suffix)
        required.append(path)
        return np.fromfile(path, dtype=dtype)

    meta, matrix, rhs = matrix_from_snapshot(prefix)
    required += [Path(str(prefix) + suffix) for suffix in
                 ('_meta.json', '_rows.bin', '_cols.bin', '_values.bin', '_rhs.bin')]
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise ValueError('Nonfinite A/b')
    if matrix.shape != (rhs.size, rhs.size) or rhs.size != meta['dofs']:
        raise ValueError('A/b dimensions disagree')
    solution, reference = accurate_reference(matrix, rhs)
    norm_a = float(np.linalg.norm(matrix.data))
    delta = matrix - matrix.T
    matrix_symmetry = float(np.linalg.norm(delta.data) / max(norm_a, 1e-300))
    report = {
        'prefix': str(prefix.resolve()), 'fixture': str(fixture.resolve()) if fixture else None,
        'dofs': int(rhs.size), 'blocks': meta['blocks'], 'cpu_reference': reference,
        'matrix_relative_asymmetry': matrix_symmetry,
        'metadata_preconditioner_complete': meta.get('preconditioner_export_complete', False),
        'physical_truth_certified': False, 'mas': [], 'abd': [], 'gpu_solution_checks': [],
    }
    for path in sorted(prefix.parent.glob(prefix.name + '*_x.bin')):
        if path.name.startswith(prefix.name + '_mas_'):
            continue
        x = np.fromfile(path, dtype='<f8')
        if x.shape != rhs.shape:
            continue
        report['gpu_solution_checks'].append({
            'file': str(path), 'true_relative_residual': float(np.linalg.norm(rhs - matrix @ x) /
                                                           max(np.linalg.norm(rhs), 1e-300)),
            'relative_error_to_cpu': relative(solution, x),
        })

    operators = []
    if 'diag_inverse' in meta.get('buffers', {}):
        diag = binary('_diag_inverse.bin').reshape(-1, 3, 3).transpose(0, 2, 1)
    else:
        diag = None
    missing_mas = False
    for local in meta.get('local_preconditioners', []):
        kind = local.get('kind')
        offset = 3 * local.get('offset', 0)
        if kind == 'abd':
            inverse = binary('_' + local['buffer'] + '.bin').reshape(-1, 12, 12).transpose(0, 2, 1)
            operators.append(('abd', offset, inverse))
            if len(inverse):
                asymmetry = np.linalg.norm(inverse - inverse.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(
                    np.linalg.norm(inverse, axis=(1, 2)), 1e-300)
                eigenvalues = np.linalg.eigvalsh((inverse + inverse.transpose(0, 2, 1)) / 2)
                report['abd'].append({'blocks': len(inverse), 'min_eigenvalue': float(eigenvalues.min()),
                                      'max_relative_asymmetry': float(asymmetry.max()),
                                      'passed': bool(eigenvalues.min() > 0 and asymmetry.max() <= 1e-8)})
            continue
        if kind != 'MAS_full_owned_buffers':
            missing_mas = True
            continue
        if fixture is None and not local.get('cholesky', False):
            report['mas'].append({'status': 'active_M_is_not_Cholesky',
                                  'hint': 'Do not substitute reconstructed stable M for the recorded operator'})
            missing_mas = True
            continue
        nodes, mapped, levels, _, clusters, *_ = local['dimensions']
        count = local['buffers']['d_inverseMatMas']['count']
        bank = clusters // count
        size = bank * 3
        if clusters % count or bank != 16:
            raise ValueError('Unexpected MAS layout; do not infer block size from kernel name')
        h_path = Path(str(prefix) + '_mas_d_inverseMatMas.bin')
        required.append(h_path)
        h = unpack(h_path, '<f8', count, bank)
        h = (h + h.transpose(0, 2, 1)) / 2
        d = np.arange(size)
        h[:, d, d] = np.where(h[:, d, d] == 0, 1, h[:, d, d])
        factor_path = fixture / 'factors.bin' if fixture else Path(str(prefix) + '_mas_d_cholesky.bin')
        if not factor_path.exists() or not factor_path.stat().st_size:
            report['mas'].append({'status': 'missing_active_cholesky_factors',
                                  'hint': 'Use a Cholesky run with GIPC_MAS_SNAPSHOT=1'})
            missing_mas = True
            continue
        required.append(factor_path)
        factor = np.fromfile(factor_path, '<f8').reshape(count, size, size)
        factor_error = np.linalg.norm(factor @ factor.transpose(0, 2, 1) - h, axis=(1, 2)) / np.maximum(
            np.linalg.norm(h, axis=(1, 2)), 1e-300)
        minimum_diagonal = float(np.diagonal(factor, axis1=1, axis2=2).min())
        mode = read(fixture/'fixture.json').get('factor_action','triangular') if fixture else local.get('factor_action','triangular')
        if mode not in ('triangular','factor_inverse'):
            raise ValueError('Unknown saved MAS action mode')
        inverse_path = fixture/'inverse_factor.bin' if fixture else Path(str(prefix)+'_mas_d_factor_inverse.bin')
        inverse_factor = None
        inverse_checks = None
        if inverse_path.exists() and inverse_path.stat().st_size:
            required.append(inverse_path)
            inverse_factor = np.fromfile(inverse_path,'<f8').reshape(count,size,size)
            if not np.isfinite(inverse_factor).all():raise FloatingPointError('Nonfinite saved inverse factor')
            identity = np.eye(size)
            bl = np.linalg.norm(inverse_factor@factor-identity,axis=(1,2))/np.sqrt(size)
            lb = np.linalg.norm(factor@inverse_factor-identity,axis=(1,2))/np.sqrt(size)
            diagonal = np.diagonal(inverse_factor,axis1=1,axis2=2)
            inverse_checks = {'left_identity_max_relative':float(bl.max()),
                              'right_identity_max_relative':float(lb.max()),
                              'minimum_diagonal':float(diagonal.min()),
                              'strict_lower_layout':bool(np.all(np.triu(inverse_factor,1)==0)),
                              'passed':bool(bl.max()<=1e-8 and lb.max()<=1e-8 and diagonal.min()>0
                                            and np.all(np.triu(inverse_factor,1)==0))}
        if mode=='factor_inverse' and inverse_factor is None:
            raise ValueError('Recorded inverse-factor action has no saved B')
        # Small independent 48x48 spectra only; never run a global eigensolver.
        eigenvalues = np.linalg.eigvalsh(h)
        positive_blocks = eigenvalues[:, 0] > 0
        conditions = eigenvalues[positive_blocks, -1] / eigenvalues[positive_blocks, 0]
        spectrum = {
            'method': 'FP64 eigvalsh of symmetric 48x48 local matrices',
            'min_eigenvalue': float(eigenvalues.min()),
            'max_eigenvalue': float(eigenvalues.max()),
            'nonpositive_numerical_eigenvalue_blocks': int((~positive_blocks).sum()),
            'condition_positive_blocks_quantiles': {
                str(q): float(np.quantile(conditions, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            } if len(conditions) else {},
        }
        part = binary('_mas_d_partId_map_real.bin', '<i4')[:mapped]
        real = binary('_mas_d_real_map_partId.bin', '<i4')[:nodes]
        coarse = binary('_mas_d_coarseTable.bin', '<i4').reshape(-1, 6)[:nodes, :levels - 1]
        valid = part >= 0
        fine_ids = np.flatnonzero(valid)
        in_range = bool(np.all(part[valid] < nodes) and np.all((coarse >= 0) & (coarse < clusters))
                        and np.all((real >= 0) & (real < mapped)))
        if not in_range:
            raise ValueError('MAS hierarchy index out of range')
        adjoint_structure = bool(np.all(np.bincount(part[valid], minlength=nodes) == 1)
                                 and np.array_equal(real[part[valid]], fine_ids))

        def apply_fem(vector, *, nodes=nodes, mapped=mapped, clusters=clusters, part=part,
                      real=real, coarse=coarse, factor=factor, count=count, size=size, valid=valid,
                      inverse_factor=inverse_factor, mode=mode, force_triangular=False, force_factor_inverse=False):
            fem = vector.reshape(nodes, 3)
            restricted = np.zeros((clusters, 3))
            restricted[np.flatnonzero(valid)] = fem[part[valid]]
            for level in range(coarse.shape[1]):
                np.add.at(restricted, coarse[:, level], fem)
            w = restricted.reshape(count, size)
            if (mode=='factor_inverse' or force_factor_inverse) and not force_triangular:
                assert inverse_factor is not None
                w = np.einsum('nji,nj->ni',inverse_factor,np.einsum('nij,nj->ni',inverse_factor,w))
            else:
                for j in range(size):
                    w[:, j] /= factor[:, j, j]
                    w[:, j + 1:] -= factor[:, j + 1:, j] * w[:, j, None]
                for j in range(size - 1, -1, -1):
                    w[:, j] /= factor[:, j, j]
                    w[:, :j] -= factor[:, j, :j] * w[:, j, None]
            prolonged = w.reshape(clusters, 3)[real].copy()
            for level in range(coarse.shape[1]):
                prolonged += w.reshape(clusters, 3)[coarse[:, level]]
            return prolonged.ravel()

        operators.append(('mas', offset, (nodes, apply_fem)))
        action_errors = []
        for k in range(10):
            vector_path = Path(str(prefix) + f'_v{k}.bin')
            gpu_path = fixture / f'p1_v{k}.bin' if fixture else Path(str(prefix) + f'_Mv{k}.bin')
            if not vector_path.exists() or not gpu_path.exists():
                continue
            vector = np.fromfile(vector_path, '<f8')[offset:offset + 3 * nodes]
            expected = apply_fem(vector)
            actual = np.fromfile(gpu_path, '<f8')
            if not fixture:
                actual = actual[offset:offset + 3 * nodes]
            action_errors.append(relative(expected, actual))
            required.extend([vector_path, gpu_path])
        rng_local=np.random.default_rng(20261004)
        local_vectors=[rhs[offset:offset+3*nodes]]+[rng_local.standard_normal(3*nodes) for _ in range(3)]
        inverse_vs_triangular=max(relative(apply_fem(v,force_triangular=True),apply_fem(v)) for v in local_vectors)
        prepared_inverse_vs_triangular=(max(relative(apply_fem(v,force_triangular=True),
                                                   apply_fem(v,force_factor_inverse=True)) for v in local_vectors)
                                       if inverse_factor is not None else None)
        report['mas'].append({
            'status': 'checked', 'operator_source': 'fixture stable Cholesky' if fixture else 'saved active Cholesky',
            'active_snapshot_cholesky': local.get('cholesky'), 'blocks': count, 'block_width': size,
            'factor_min_diagonal': minimum_diagonal,
            'factor_action':mode,'inverse_factor_checks':inverse_checks,
            'actual_action_vs_triangular_cpu_max_relative':inverse_vs_triangular,
            'prepared_inverse_action_vs_triangular_cpu_max_relative':prepared_inverse_vs_triangular,
            'factor_reconstruction_max_relative': float(factor_error.max()),
            'local_matrix_spectrum': spectrum,
            'restriction_prolongation_adjoint_structure': adjoint_structure,
            'gpu_action_checks': len(action_errors),
            'gpu_action_max_relative': max(action_errors) if action_errors else None,
            'passed': bool(np.isfinite(factor).all() and minimum_diagonal > 0 and
                           factor_error.max() <= 1e-8 and adjoint_structure and inverse_vs_triangular<=1e-8 and
                           (inverse_checks is None or inverse_checks['passed']) and
                           (prepared_inverse_vs_triangular is None or prepared_inverse_vs_triangular<=1e-8) and
                           (not action_errors or max(action_errors) <= 1e-8)),
        })

    def apply(vector):
        out = vector.copy() if diag is None else np.einsum('nij,nj->ni', diag, vector.reshape(-1, 3)).ravel()
        for kind, offset, data in operators:
            if kind == 'abd':
                length = 12 * len(data)
                out[offset:offset + length] = np.einsum(
                    'nij,nj->ni', data, vector[offset:offset + length].reshape(-1, 12)).ravel()
            else:
                nodes, function = data
                out[offset:offset + 3 * nodes] = function(vector[offset:offset + 3 * nodes])
        return out

    if missing_mas or not any(v['status'] == 'checked' for v in report['mas']):
        report['preconditioner_check'] = {'passed': False, 'status': 'incomplete',
                                        'hint': 'Capture GIPC_MAS_SNAPSHOT=1 with active Cholesky; do not substitute a different M'}
    else:
        rng = np.random.default_rng(20261004)
        vectors = [rhs, solution] + [rng.standard_normal(rhs.size) for _ in range(8)]
        mv = [apply(v) for v in vectors]
        if not all(np.isfinite(m).all() for m in mv):
            raise FloatingPointError('Nonfinite CPU preconditioner action')
        quadratics = [float(v @ m) for v, m in zip(vectors, mv)]
        bilinear = []
        for i in range(0, len(vectors), 2):
            u, v = vectors[i:i + 2]
            mu, m = mv[i:i + 2]
            scale = max(np.linalg.norm(u) * np.linalg.norm(m) + np.linalg.norm(v) * np.linalg.norm(mu), 1e-300)
            bilinear.append(float(abs(u @ m - v @ mu) / scale))
        if not all(np.isfinite(v) for v in quadratics + bilinear):
            raise FloatingPointError('Nonfinite preconditioner quadratic or symmetry check')
        positive = all(q > 0 if np.any(v) else q == 0 for q, v in zip(quadratics, vectors))
        report['preconditioner_check'] = {
            'status': 'checked', 'probe_count': len(vectors), 'seed': 20261004,
            'quadratic_forms': quadratics, 'all_nonzero_probe_quadratics_positive': positive,
            'scaled_bilinear_asymmetry_max': max(bilinear),
            'finite': bool(all(np.isfinite(m).all() for m in mv)),
            'proof_limit': 'Positive probe values are not an exhaustive spectral proof; factor and R/P structure checked separately',
            'passed': bool(positive and max(bilinear) <= 1e-8 and all(np.isfinite(m).all() for m in mv)),
        }
    report['input_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(set(required))}
    report['passed'] = bool(reference['passed'] and matrix_symmetry <= 1e-12 and
                            report['preconditioner_check']['passed'] and
                            all(v['passed'] for v in report['abd']) and
                            all(v.get('passed', False) for v in report['mas']))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-fixtures', type=Path)
    parser.add_argument('--prefix', action='append', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    tasks = []
    if args.historical_fixtures:
        for name, version, folder, kind in CASES:
            prefix = ROOT / f'downloads/autodl_perf_v{version}_20261003/runs/autodl/autodl_perf_v{version}_{folder}/mas_audit_{kind}'
            tasks.append((name, prefix, args.historical_fixtures / name))
    tasks += [(p.name, p, None) for p in args.prefix]
    if not tasks:
        parser.error('Provide --historical-fixtures and/or --prefix')
    with args.output.open('x') as output:
        output.write('{}\n')
    result = {'cpu_only': True, 'linear_reference_only': True,
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'systems': [], 'passed': True}
    for name, prefix, fixture in tasks:
        try:
            row = verify(prefix, fixture)
        except Exception as error:
            row = {'passed': False, 'prefix': str(prefix), 'error': f'{type(error).__name__}: {error}'}
        row['case'] = name
        result['systems'].append(row)
        result['passed'] &= row['passed']
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'case': name, 'passed': row['passed'],
                          'cpu_reference': row.get('cpu_reference'), 'error': row.get('error')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
