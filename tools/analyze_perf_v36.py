"""Read-only fixed-system evidence; write a new analysis file, never rerun GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy import sparse


def matrix_from_snapshot(prefix):
    meta = json.loads(Path(str(prefix) + '_meta.json').read_text())
    rows = np.fromfile(str(prefix) + '_rows.bin', dtype='<i4')
    cols = np.fromfile(str(prefix) + '_cols.bin', dtype='<i4')
    blocks = np.fromfile(str(prefix) + '_values.bin', dtype='<f8').reshape(-1, 3, 3).transpose(0, 2, 1)
    rr = np.broadcast_to(rows[:, None, None] * 3 + np.arange(3)[None, :, None], blocks.shape).ravel()
    cc = np.broadcast_to(cols[:, None, None] * 3 + np.arange(3)[None, None, :], blocks.shape).ravel()
    off = np.broadcast_to((rows != cols)[:, None, None], blocks.shape).ravel()
    matrix = sparse.coo_matrix((np.r_[blocks.ravel(), blocks.ravel()[off]],
                               (np.r_[rr, cc[off]], np.r_[cc, rr[off]])), shape=(meta['dofs'],) * 2).tocsr()
    rhs = np.fromfile(str(prefix) + '_rhs.bin', dtype='<f8')
    return meta, matrix, rhs


def analyze_system(path):
    study = json.loads(path.read_text())
    prefix = Path(str(path)[:-len('_study.json')])
    meta, matrix, rhs = matrix_from_snapshot(prefix)
    assert study['system_unchanged'] and study['primary_restored_bitwise']
    independent = []
    for phase in range(4):
        for mode, name in enumerate(['host', 'graph', 'graph_fused']):
            xp = Path(f'{prefix}_p{phase}_m{mode}_x.bin')
            if not xp.exists():
                continue
            x = np.fromfile(xp, dtype='<f8')
            residual = float(np.linalg.norm(rhs - matrix @ x) / max(np.linalg.norm(rhs), 1e-300))
            record = next(r for r in study['runs'] if r['mode'] == name and r['repeat'] == 1 and
                          r['phase'] == ('fixed_iterations' if phase == 0 else 'rho_tolerance') and
                          r['rho_tolerance'] == [1e-4, 1e-4, 1e-8, 1e-12][phase])
            independent.append({'phase_index': phase, 'mode': name, 'cpu_true_relative_residual': residual,
                                'gpu_true_relative_residual': record['true_relative_residual'],
                                'absolute_residual_disagreement': abs(residual - record['true_relative_residual'])})
    spectral = []
    for key in meta['buffers']:
        if 'inverse' not in key:
            continue
        width = 12 if key.startswith('abd') else 3
        blocks = np.fromfile(str(prefix) + '_' + key + '.bin', dtype='<f8').reshape(-1, width, width).transpose(0, 2, 1)
        ev = np.linalg.eigvalsh((blocks + blocks.transpose(0, 2, 1)) / 2)
        scale = np.maximum(np.max(np.abs(ev), axis=1), 1e-300)
        spectral.append({'buffer': key, 'min_eigenvalue': float(ev.min()) if ev.size else None,
                         'negative_blocks_relative_1e12': int(np.sum(ev[:, 0] < -1e-12 * scale)),
                         'max_relative_asymmetry': float(np.max(np.linalg.norm(blocks - blocks.transpose(0, 2, 1), axis=(1, 2)) /
                                                                              np.maximum(np.linalg.norm(blocks, axis=(1, 2)), 1e-300), initial=0))})
    groups = []
    for phase, tolerance in [('fixed_iterations', 1e-4), ('rho_tolerance', 1e-4), ('rho_tolerance', 1e-8), ('rho_tolerance', 1e-12)]:
        modes = {}
        for mode in sorted({r['mode'] for r in study['runs']}):
            rows = [r for r in study['runs'] if r['phase'] == phase and r['rho_tolerance'] == tolerance and r['mode'] == mode]
            modes[mode] = {'median_ms': statistics.median(r['solve_ms'] for r in rows),
                           'iteration_range': [min(r['iterations'] for r in rows), max(r['iterations'] for r in rows)],
                           'max_residual': max(r['true_relative_residual'] if r['true_relative_residual'] is not None else 1e300 for r in rows),
                           'max_repeat_difference': max(r['repeat_difference']['relative'] or 0 for r in rows),
                           'max_host_difference': max(r['host_difference']['relative'] or 0 for r in rows),
                           'failures': sum(bool(r['error']) or r['limit'] for r in rows)}
        groups.append({'phase': phase, 'rho_tolerance': tolerance, 'modes': modes,
                       'graph_speed_vs_host': modes['host']['median_ms'] / modes['graph']['median_ms'],
                       'fused_speed_vs_graph': modes['graph']['median_ms'] / modes['graph_fused']['median_ms'] if 'graph_fused' in modes else None})
    return {'file': str(path), 'frame': study['frame'], 'direction': study['direction'], 'dofs': meta['dofs'],
            'preconditioner_export_complete': meta['preconditioner_export_complete'],
            'snapshot_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in prefix.parent.glob(prefix.name + '_*.bin')},
            'groups': groups, 'operator_mean_us': study['operator_mean_us'], 'operator_repeats': study['operator_repeats'],
            'independent_residuals': independent, 'preconditioner_spectra': spectral}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier analysis'
    systems = [analyze_system(p) for p in sorted((args.run / 'fixed').glob('*_study.json'))]
    assert systems
    result = {'scope': 'Frozen in-process systems; CPU reconstruction independently checks exported solutions. Not simulation speedup.',
              'systems': systems}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    for phase, tol in [('fixed_iterations', 1e-4), ('rho_tolerance', 1e-4), ('rho_tolerance', 1e-8), ('rho_tolerance', 1e-12)]:
        groups = [g for s in systems for g in s['groups'] if g['phase'] == phase and g['rho_tolerance'] == tol]
        print(json.dumps({'systems': len(groups), 'phase': phase, 'rho_tolerance': tol,
                          'median_graph_speed': statistics.median(g['graph_speed_vs_host'] for g in groups),
                          'median_fused_speed': statistics.median(g['fused_speed_vs_graph'] for g in groups if g['fused_speed_vs_graph'] is not None)
                          if any(g['fused_speed_vs_graph'] is not None for g in groups) else None,
                          'max_residual': max(m['max_residual'] for g in groups for m in g['modes'].values()),
                          'max_repeat_diff': max(m['max_repeat_difference'] for g in groups for m in g['modes'].values()),
                          'failures': sum(m['failures'] for g in groups for m in g['modes'].values())}))


if __name__ == '__main__':
    main()
