"""Independent CPU analysis of v37 serialized systems and MAS operators."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot


def difference(a, b):
    if a.shape != b.shape:
        return {'same_shape': False, 'shapes': [a.shape, b.shape]}
    return {'same_shape': True, 'relative': float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300)),
            'max_abs': float(np.max(np.abs(a-b), initial=0)), 'unequal': int(np.count_nonzero(a != b))}


def unpack(prefix, dtype, count, bank):
    blocks = np.fromfile(prefix, dtype=dtype).reshape(count, bank*(bank+1)//2, 3, 3).transpose(0, 1, 3, 2)
    out = np.zeros((count, 3*bank, 3*bank))
    for r in range(bank):
        for c in range(r, bank):
            block = blocks[:, bank*r-r*(r+1)//2+c]
            out[:, 3*r:3*r+3, 3*c:3*c+3] = block
            if c != r:
                out[:, 3*c:3*c+3, 3*r:3*r+3] = block.transpose(0, 2, 1)
    return out


def audit(path):
    record = json.loads(path.read_text())
    prefix = Path(str(path)[:-len('_audit.json')])
    meta, matrix, rhs = matrix_from_snapshot(prefix)
    x = np.fromfile(str(prefix)+'_x.bin', '<f8')
    result = {'file': str(path), 'frame': record['frame'], 'dofs': len(rhs),
              'snapshot_complete': meta['preconditioner_export_complete'],
              'solution_unchanged': record['solution_unchanged_bitwise'],
              'cpu_true_relative_residual': float(np.linalg.norm(rhs-matrix@x)/max(np.linalg.norm(rhs), 1e-300)),
              'max_repeat_M_relative': max(max(p['repeat_relative']) for p in record['probes']),
              'max_scaled_bilinear_asymmetry': max(p['scaled_asymmetry'] for p in record['bilinear_pairs']),
              'probes': []}
    for p in record['probes']:
        k = p['index']
        v = np.fromfile(f'{prefix}_v{k}.bin', '<f8')
        mv = np.fromfile(f'{prefix}_Mv{k}.bin', '<f8')
        av = np.fromfile(f'{prefix}_Av{k}.bin', '<f8')
        result['probes'].append({'index': k, 'input': p['input'], 'v_Mv': float(v@mv),
                                 'v_Av': float(v@(matrix@v)), 'spmv_cpu_gpu': difference(av, matrix@v)})
    result['mas_blocks'] = []
    for local in meta['local_preconditioners']:
        if local['kind'] != 'MAS_full_owned_buffers':
            continue
        # Allocation may include padding: only the live clusters are operator blocks.
        count = local['buffers']['d_inverseMatMas']['count']
        bank = local['dimensions'][4] // count
        for name, dtype in [('d_inverseMatMas', '<f8'), ('d_precondMatMas', '<f4')]:
            allocation = local['buffers'][name]['count']
            blocks = unpack(f'{prefix}_mas_{name}.bin', dtype, allocation, bank)[:count]
            symmetric = (blocks+blocks.transpose(0, 2, 1))/2
            vals = np.linalg.eigvalsh(symmetric)
            scale = np.maximum(np.max(np.abs(vals), axis=1), 1e-300)
            bad = np.flatnonzero(vals[:, 0] < -1e-7*scale)
            result['mas_blocks'].append({'buffer': name, 'live_blocks': count,
                'finite': bool(np.isfinite(blocks).all()), 'min_eigenvalue': float(vals.min()),
                'min_relative_eigenvalue': float(np.min(vals[:, 0]/scale)),
                'negative_blocks_relative_1e7': len(bad), 'negative_block_ids': bad.tolist(),
                'nonpositive_blocks': int(np.count_nonzero(vals[:, 0] <= 0)),
                'max_asymmetry': float(np.max(np.linalg.norm(blocks-blocks.transpose(0, 2, 1), axis=(1, 2)) /
                                             np.maximum(np.linalg.norm(blocks, axis=(1, 2)), 1e-300)))})
    return result


def window(runs):
    result = []
    ref = runs[0] / 'state_window'
    keys = sorted(ref.glob('*_state.json'), key=lambda p: tuple(map(int, p.stem.replace('f','').replace('_n','_').split('_')[:2])))
    for run in runs[1:]:
        records = []
        for key in keys:
            stem = key.name[:-len('_state.json')]
            if not (run/'state_window'/key.name).exists():
                records.append({'key': stem, 'missing': True}); continue
            one = {'key': stem, 'buffers': {}}
            for field in ['vertices', 'q', 'safe', 'trial', 'safe_q', 'trial_q', 'friction_lambda', 'scalars', 'contact_ids', 'ground_ids', 'friction_ids']:
                dtype = '<i4' if 'ids' in field else '<f8'
                a = np.fromfile(ref/f'{stem}_state_{field}.bin', dtype=dtype)
                b = np.fromfile(run/'state_window'/f'{stem}_state_{field}.bin', dtype=dtype)
                one['buffers'][field] = difference(a.astype(float), b.astype(float))
                if field.endswith('ids') and field in ('contact_ids','friction_ids') and a.size % 4 == 0 and b.size % 4 == 0:
                    one['buffers'][field]['same_unordered_tuples'] = sorted(map(tuple,a.reshape(-1,4))) == sorted(map(tuple,b.reshape(-1,4)))
            for field in ['rows','cols','values','rhs','x']:
                dtype = '<i4' if field in ('rows','cols') else '<f8'
                a = np.fromfile(ref/f'{stem}_{field}.bin', dtype=dtype)
                b = np.fromfile(run/'state_window'/f'{stem}_{field}.bin', dtype=dtype)
                one['buffers'][field] = difference(a.astype(float), b.astype(float))
            records.append(one)
        result.append({'reference': runs[0].name, 'candidate': run.name, 'systems': records})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    runs = a.root / 'runs/autodl'
    result = {'mas': [audit(path) for path in sorted(runs.glob('*/mas_audit_*_audit.json'))],
              'window': window([runs/f'autodl_perf_v37_host_r{i}' for i in range(1,4)])}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'mas_systems': len(result['mas']), 'window_pairs': len(result['window'])}))
