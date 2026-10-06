"""Per-frame physical divergence diagnostic, executed on the AutoDL host."""
import argparse
import json
from pathlib import Path

import numpy as np


def states(run):
    return {int(path.stem[-4:]): path for path in (run / 'trace').glob('state_*.bin')}


def tet_signs(x, tets, initial_det):
    if not len(tets):
        return 0
    q = x[tets]
    det = np.einsum('ij,ij->i', q[:, 1] - q[:, 0],
                    np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]))
    return int(np.count_nonzero(det * initial_det <= 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref_files, candidate_files = states(args.reference), states(args.candidate)
    indices = sorted(ref_files.keys() & candidate_files.keys())
    masses = np.fromfile(args.reference / 'trace/masses.bin', dtype='<f8')
    topology = np.fromfile(args.reference / 'trace/topology.bin', dtype='<u4')
    _, faces, tets_n = topology[:3]
    tets = topology[3 + 3 * faces:].reshape(tets_n, 4)
    initial = np.fromfile(ref_files[indices[0]], dtype='<f8').reshape(-1, 3)
    candidate_initial = np.fromfile(candidate_files[indices[0]], dtype='<f8').reshape(-1, 3)
    if not np.array_equal(initial, candidate_initial):
        raise ValueError('Initial states differ')
    q = initial[tets]
    initial_det = np.einsum('ij,ij->i', q[:, 1] - q[:, 0],
                            np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]))
    scale = float(np.linalg.norm(np.ptp(initial, axis=0)))
    rows = []
    for frame in indices:
        x = np.fromfile(ref_files[frame], dtype='<f8').reshape(-1, 3)
        y = np.fromfile(candidate_files[frame], dtype='<f8').reshape(-1, 3)
        rms = float(np.sqrt(np.average(np.sum((x - y) ** 2, axis=1), weights=masses)))
        rows.append({'frame': frame, 'normalized_mass_rms_vs_base': rms / scale,
                     'base_inverted_tets': tet_signs(x, tets, initial_det),
                     'toi_inverted_tets': tet_signs(y, tets, initial_det)})
    result = {'reference': args.reference.name, 'candidate': args.candidate.name,
              'scene_scale_m': scale, 'first_toi_inversion_frame': next((r['frame'] for r in rows if r['toi_inverted_tets']), None),
              'first_rms_over_1e_6_frame': next((r['frame'] for r in rows if r['normalized_mass_rms_vs_base'] > 1e-6), None),
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'first_toi_inversion_frame': result['first_toi_inversion_frame'],
                      'first_rms_over_1e_6_frame': result['first_rms_over_1e_6_frame'],
                      'samples': [rows[i] for i in (0, 20, 40, 48, 60, 80, 100) if i < len(rows)]}))


if __name__ == '__main__':
    main()
