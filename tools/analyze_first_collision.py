"""Inspect topology and geometry of the first independently flagged CCD primitive."""
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, default=ROOT / 'reports/autodl_v18_mass_bunny_first_collision.json')
parser.add_argument('--output', type=Path, default=ROOT / 'reports/autodl_v18_mass_bunny_first_collision_geometry.json')
parser.add_argument('--trace', type=Path, default=ROOT / 'runs/autodl/autodl_v18_mass_bunny_host100/trace')
args = parser.parse_args()
report = json.loads(args.report.read_text())
hit = report['first_collision']
trace = args.trace
topology = np.fromfile(trace / 'topology.bin', dtype='<u4')
vertices, face_count, tet_count = topology[:3]
faces = topology[3:3 + 3 * face_count].reshape(-1, 3)
tets = topology[3 + 3 * face_count:].reshape(tet_count, 4)
v, f0, f1, f2 = hit['ids']
ids = [v, f0, f1, f2]
body = np.fromfile(trace / 'body_ids.bin', dtype='<i4')
fixed = np.fromfile(trace / 'boundary_types.bin', dtype='<i4')
present = {index: (tets == index).any(axis=1) for index in ids}
shared_all = np.flatnonzero(np.logical_and.reduce([present[index] for index in ids]))
shared_face = np.flatnonzero(np.logical_and.reduce([present[index] for index in ids[1:]]))
shared_vertex_and_face = {str(index): int(np.count_nonzero(present[v] & present[index])) for index in ids[1:]}


def state(name):
    return np.fromfile(trace / 'substeps' / name, dtype='<f8').reshape(vertices, 3)


def geometry(x):
    p, a, b, c = x[ids]
    normal = np.cross(b - a, c - a)
    area2 = float(np.linalg.norm(normal))
    signed_distance = float(np.dot(p - a, normal) / area2)
    return {'signed_distance_m': signed_distance, 'face_double_area_m2': area2,
            'point': p.tolist(), 'triangle': [a.tolist(), b.tolist(), c.tolist()]}


result = {'first_collision': hit, 'body_ids': body[ids].tolist(), 'fixed_types': fixed[ids].tolist(),
          'same_tet_ids': shared_all[:20].tolist(), 'same_face_tet_count': len(shared_face),
          'point_face_vertex_shared_tet_counts': shared_vertex_and_face,
          'initial': geometry(state('safe_0000_0000.bin')) if (trace / 'substeps/safe_0000_0000.bin').exists() else None,
          'from': geometry(state(hit['from'])), 'to': geometry(state(hit['to']))}
args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('initial', 'from', 'to')} |
                 {'from_signed_distance_m': result['from']['signed_distance_m'],
                  'to_signed_distance_m': result['to']['signed_distance_m']}))
