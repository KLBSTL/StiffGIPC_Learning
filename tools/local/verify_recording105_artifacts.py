"""Verify sealed observations, frame contracts and lossless rendered mesh exports."""
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ('tools/local', 'tools/bench')]
from config import read
from linux_runner import require, sha, write_new
from local_identity import verify, record


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    output = ROOT / 'reports/local_recording105_dt0p04_20261011'
    summary = read(output / 'summary.json')
    require(summary['complete'], 'Analysis incomplete')
    for name, digest in summary['output_sha256'].items():
        require(sha(output / name) == digest, 'CSV modified: ' + name)
    frame_rows = rows(output / 'per_frame.csv')
    frame100 = rows(output / 'frame100_actual.csv')
    require(len(frame_rows) == 7560 and len(frame100) == 72, 'Wrong frame counts')
    require([r for r in frame_rows if int(r['physical_frame']) == 100] == frame100, 'Frame100 table differs')
    counts = Counter(r['run'] for r in frame_rows)
    require(len(counts) == 72 and set(counts.values()) == {105}, 'Wrong trajectory counts')
    indices = Counter((r['run'], int(r['physical_frame'])) for r in frame_rows)
    require(len(indices) == 7560 and set(indices.values()) == {1}, 'Duplicate physical frames')
    directions, iterations, newton_counts = Counter(), Counter(), Counter()
    for r in rows(output / 'per_newton.csv'):
        key = r['run'], int(r['physical_frame'])
        newton_counts[key] += 1
        if r['P']:
            directions[key] += 1
            iterations[key] += int(r['P'])
    for r in frame_rows:
        key = r['run'], int(r['physical_frame'])
        require(int(r['physical_frame']) == int(r['local_frame']) and 1 <= key[1] <= 105, 'Wrong physical numbering')
        require(float(r['dt']) == .04 and math.isfinite(float(r['solver_ms'])), 'Invalid frame configuration')
        require(int(r['N']) == directions[key] and int(r['P']) == iterations[key]
                and int(r['newton_records']) == newton_counts[key], 'Newton/PCG count discrepancy')
    for r in rows(output / 'trajectory_errors.csv'):
        if int(r['frame']) == 0:
            require(all(float(r[k]) == 0 for k in ('position_rms_mm', 'position_max_mm',
                    'velocity_rms_m_per_s', 'velocity_max_m_per_s')), 'Non-identical initial error state')
    gate = read(output / 'checkpoint_gate.json')
    require(gate['passed'] and gate['exact_input_runs'] == 126
            and all(all(r['checks'].values()) for r in gate['runs']), 'Checkpoint input gate failed')
    initial = read(output / 'trajectory_input_gate.json')
    require(initial['passed'] and initial['exact_initial_runs'] == 72, 'Initial input gate failed')
    manifest = read(output / 'visuals/manifest.json')
    require(manifest['renderer']['depth_order_smoke_passed'], 'Depth smoke failed')
    evidence = manifest['source_vertices']
    require(len(evidence) == 324, 'Missing visual source meshes')
    for item in evidence:
        source = Path(item['state'])
        obj = output / 'visuals' / item['obj']
        topology = source.parent / 'topology.bin'
        require(sha(source) == item['state_sha256'] and sha(topology) == item['topology_sha256']
                and sha(obj) == item['obj_sha256'], 'Visual source identity changed')
        vertices, faces = [], []
        with obj.open(encoding='ascii') as stream:
            for line in stream:
                if line.startswith('v '):
                    vertices.append([float(z) for z in line.split()[1:]])
                elif line.startswith('f '):
                    faces.append([int(z) - 1 for z in line.split()[1:]])
        raw = np.fromfile(topology, dtype='<u4')
        nv, nf, _ = map(int, raw[:3])
        require(np.array_equal(np.asarray(vertices), np.fromfile(source, dtype='<f8').reshape(nv, 3))
                and np.array_equal(np.asarray(faces), raw[3:3 + 3 * nf].reshape(nf, 3)), 'OBJ changed geometry')
    original = read(ROOT / 'runs/recording_build_20261011/original_copy.json')
    source_root, copy_root = Path(original['source']), Path(original['copy'])
    verify([dict(item, path=str(source_root / Path(item['path']).relative_to(copy_root))) for item in original['files']])
    verify([summary['analysis_controller'], manifest['report_controller'], summary['identity']])
    result = dict(passed=True, actual_frames=7560, actual_frame100_rows=72,
        checkpoint_inputs=126, initial_inputs=72, lossless_obj_meshes=324,
        original_files_unchanged=len(original['files']), verifier=record(__file__))
    write_new(output / 'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
