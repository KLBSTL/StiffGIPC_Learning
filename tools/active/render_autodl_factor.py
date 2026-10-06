"""Offline raw-mesh views of downloaded AutoDL windows, using the established
Matplotlib Poly3DCollection renderer, orthographic camera and cloth colormap.

Default: one image per scene, three repeat rows x Stiff/triangular/inverse
columns, all at that scene's planned final frame. No GPU, SSH or reconstruction.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from analyze_autodl_factor import child, prepare_tasks, verify_download_provenance
from config import ROOT, digest, read, sha


DEFAULT_ROOT = ROOT / 'downloads/autodl_factor_20261004_4090'
DISPLAY_ROLES = ('stiff', 'tri_toi_graph', 'toi_graph')
LABELS = {'stiff': 'StiffGIPC', 'tri_toi_graph': 'TOI Graph: triangular', 'toi_graph': 'TOI Graph: inverse factor'}
CAMERA = {'elevation_degrees': 26, 'azimuth_degrees': -65, 'projection': 'ortho',
          'display_axis_order': [0, 2, 1], 'deformation_scale': 1.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry(trace, scene):
    raw = np.fromfile(trace / 'topology.bin', dtype='<u4')
    require(len(raw) >= 3, 'Missing topology header')
    nv, nf, nt = map(int, raw[:3])
    require(raw.size == 3 + 3 * nf + 4 * nt and nv > 0, 'Malformed topology')
    faces = raw[3:3 + 3 * nf].reshape(-1, 3)
    tets = raw[3 + 3 * nf:].reshape(-1, 4)
    require(not raw[3:].size or int(raw[3:].max()) < nv, 'Invalid topology vertex index')
    initial = np.fromfile(trace / 'state_0000.bin', dtype='<f8').reshape(nv, 3)
    require(np.isfinite(initial).all(), 'Nonfinite initial mesh')
    metadata = read(trace / 'metadata.json')
    abd = int(metadata['abd_point_num'])
    in_tet = np.zeros(nv, dtype=bool)
    in_tet[tets.ravel()] = True
    cloth_faces = ~in_tet[faces].any(axis=1)
    triangles = faces[cloth_faces]
    edges = np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    lengths = np.linalg.norm(initial[edges[:, :, 0]] - initial[edges[:, :, 1]], axis=2)
    require(np.all(lengths > 0), 'Degenerate cloth edge')
    fixed = np.fromfile(trace / 'boundary_types.bin', dtype='<i4') == 1
    require(len(fixed) == nv and 0 <= abd <= nv, 'Invalid boundary/ABD map')
    objects = [obj for obj in scene['objects'] if obj['dimension'] == 3]
    if objects and all(obj['body_type'] == 'ABD' and obj.get('fixed_mode') == 'all' for obj in objects):
        fixed_abd = in_tet & (np.arange(nv) < abd)
        require(int(fixed_abd.sum()) == abd, 'Fixed ABD mapping incomplete')
        fixed |= fixed_abd
    rest = volumes(initial, tets)
    require(np.all(rest != 0), 'Degenerate rest tetrahedron')
    return {'vertices': nv, 'faces': faces, 'tets': tets, 'initial': initial, 'cloth_faces': cloth_faces,
            'edges': edges, 'rest_lengths': lengths, 'rest_volumes': rest, 'fixed': fixed,
            'fem_tets': np.all(tets >= abd, axis=1), 'abd_tets': np.all(tets < abd, axis=1)}


def volumes(positions, tets):
    points = positions[tets]
    return np.einsum('ij,ij->i', points[:, 1] - points[:, 0],
                     np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0])) / 6


def raw_metrics(mesh, positions):
    require(positions.shape == mesh['initial'].shape and np.isfinite(positions).all(), 'Nonfinite/malformed final mesh')
    edges = mesh['edges']
    stretch = np.linalg.norm(positions[edges[:, :, 0]] - positions[edges[:, :, 1]], axis=2) / mesh['rest_lengths']
    ratio = volumes(positions, mesh['tets']) / mesh['rest_volumes']
    fem, abd, fixed = mesh['fem_tets'], mesh['abd_tets'], mesh['fixed']
    metrics = {'finite': True, 'cloth_max_stretch': float(stretch.max()) if stretch.size else None,
               'fem_min_J': float(ratio[fem].min()) if fem.any() else None,
               'fem_nonpositive': int((ratio[fem] <= 0).sum()) if fem.any() else None,
               'fem_negative_volume_m3': float(np.sum(np.maximum(-ratio[fem], 0) * abs(mesh['rest_volumes'][fem])))
                   if fem.any() else None,
               'abd_min_J': float(ratio[abd].min()) if abd.any() else None,
               'fixed_vertices': int(fixed.sum()),
               'fixed_max_drift_m': float(np.linalg.norm(positions[fixed] - mesh['initial'][fixed], axis=1).max())
                   if fixed.any() else 0.}
    return metrics, stretch.max(axis=1) if stretch.size else np.empty(0)


def prepare_scene(root, key, tasks, matrix_rows, repeats, provenance):
    selected = [task for repeat in repeats for role in DISPLAY_ROLES for task in tasks
                if task['scene_key'] == key and task['repeat'] == repeat and task['role'] == role]
    require(len(selected) == 3 * len(repeats), 'Missing displayed role/repeat: ' + key)
    steps = {task['expanded_config']['steps'] for task in selected}
    require(len(steps) == 1, 'Displayed windows end at different physical frames')
    frame = next(iter(steps))
    panels = []
    reference, mesh = None, None
    for task in selected:
        run = child(root, 'runs/active/' + task['name'])
        require(task['name'] in matrix_rows, 'Displayed run missing from batch: ' + run.name)
        result = read(run / 'result.json')
        request = read(run / 'requested.json')
        require(result == matrix_rows[task['name']]['result'] and result['status'] == 'completed'
                and result['recorded_frames'] == frame and result.get('finite') is True,
                'Displayed run did not complete the full finite window: ' + run.name)
        require(request['expanded_config'] == task['expanded_config']
                and request['config_sha256'] == digest(task['expanded_config'])
                and request['binary'] == task['binary'], 'Displayed plan/request mismatch: ' + run.name)
        require(sha(run / 'build_manifest.json') == provenance['canonical_build_manifest_sha256'],
                'Displayed run build differs from verified canonical build')
        binary = next(row for row in provenance['binaries']
                      if row['group'] == 'binaries' and row['name'] == task['binary'])
        require(binary['verified'] and request['exe_sha256'] == binary['actual_sha256'],
                'Displayed executable identity differs from verified downloaded bytes')
        scene = read(run / 'output/scene.json')
        identity = {name: sha(run / 'trace' / name) for name in
                    ('topology.bin', 'state_0000.bin', 'boundary_types.bin', 'masses.bin', 'body_ids.bin', 'metadata.json')}
        identity['scene.json'] = sha(run / 'output/scene.json')
        if reference is None:
            reference = identity
            mesh = geometry(run / 'trace', scene)
        require(identity == reference, 'Displayed meshes do not share scene/initial/topology/material identity')
        state = run / 'trace' / f'state_{frame:04d}.bin'
        positions = np.fromfile(state, dtype='<f8').reshape(mesh['vertices'], 3)
        metrics, face_stretch = raw_metrics(mesh, positions)
        panels.append({'run': run.name, 'role': task['role'], 'repeat': task['repeat'], 'frame': frame,
                       'state_sha256': sha(state), 'positions': positions, 'face_stretch': face_stretch,
                       'metrics': metrics})
    low = np.min([panel['positions'].min(axis=0) for panel in panels], axis=0)
    high = np.max([panel['positions'].max(axis=0) for panel in panels], axis=0)
    extent = high - low
    padding = np.maximum(extent * .04, max(float(extent.max()), 1e-6) * .005)
    return {'scene_key': key, 'scene': selected[0]['expanded_config']['scene'], 'frame': frame,
            'physical_time_seconds': frame * selected[0]['expanded_config']['dt'], 'mesh': mesh, 'panels': panels,
            'bounds_min_m': low - padding, 'bounds_max_m': high + padding, 'initial_identity_sha256': reference}


def text_number(value, digits='.5g'):
    return 'absent' if value is None else format(value, digits)


def render(root, plan_path, matrix_path, output, repeats):
    require(not output.exists(), 'A fresh output directory is required')
    require(repeats and len(set(repeats)) == len(repeats) and set(repeats) <= {1, 2, 3}, 'Choose distinct repeats from 1,2,3')
    plan, matrix, bundle = read(plan_path), read(matrix_path), read(root / 'autodl_bundle.json')
    require(matrix['plan_sha256'] == sha(plan_path), 'Window plan/batch identity mismatch')
    tasks = prepare_tasks(root, plan)
    matrix_rows = {row['name']: row for row in matrix['runs']}
    require(len(matrix_rows) == len(matrix['runs']), 'Duplicate matrix run name')
    provenance = verify_download_provenance(root, plan, matrix, bundle, plan_path)
    require(provenance['passed'], 'Downloaded source/executable/ordered-ledger identity did not pass: '
            + '; '.join(provenance['failures']))
    scenes, failures = [], []
    for key in sorted({task['scene_key'] for task in tasks}):
        try:
            scenes.append(prepare_scene(root, key, tasks, matrix_rows, repeats, provenance))
        except (ValueError, KeyError, OSError, AssertionError) as error:
            failures.append({'scene_key': key, 'failure': str(error)})
    output.mkdir(parents=True, exist_ok=False)
    # Same software renderer and camera used by render_factor_windows.py.
    # Backend imports are delayed so --self-test never draws or initializes GL.
    os.environ['MPLCONFIGDIR'] = str(output / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    color_upper = max([1.05] + [panel['metrics']['cloth_max_stretch'] for scene in scenes
                               for panel in scene['panels'] if panel['metrics']['cloth_max_stretch'] is not None])
    norm, cmap = Normalize(1., color_upper), plt.get_cmap('viridis')
    report = {'schema_version': 1, 'download_root': str(root), 'plan_sha256': sha(plan_path),
              'matrix_sha256': sha(matrix_path), 'renderer_sha256': sha(Path(__file__)),
              'renderer_pattern_source': 'tools/active/render_factor_windows.py',
              'renderer_pattern_sha256': sha(ROOT / 'tools/active/render_factor_windows.py'),
              'camera': CAMERA, 'selected_repeats': repeats, 'common_cloth_colormap': 'viridis',
              'common_cloth_color_range': [1., color_upper], 'noncloth_rgba': [.70, .70, .72, 1.],
              'color_scope': 'One cloth stretch ratio color range across every panel and scene. '
                             'A visual upper-range floor of 1.05 prevents amplifying roundoff; raw metrics remain unchanged.',
              'physical_certified': False, 'performance_certified': False,
              'scope': 'Actual final-frame exported positions, no displacement exaggeration or temporal interpolation. '
                       'Captions show final-frame observations, not full-window maxima. Images are not CCD or quality proof.',
              'download_provenance': provenance,
              'scenes': [], 'failures': failures}
    for scene in scenes:
        figure = plt.figure(figsize=(14, 3.8 * len(repeats)))
        try:
            low, high = scene['bounds_min_m'], scene['bounds_max_m']
            mesh = scene['mesh']
            for index, panel in enumerate(scene['panels']):
                colors = np.full((len(mesh['faces']), 4), [.70, .70, .72, 1.])
                colors[mesh['cloth_faces']] = cmap(norm(panel['face_stretch']))
                axis = figure.add_subplot(len(repeats), 3, index + 1, projection='3d')
                axis.add_collection3d(Poly3DCollection(panel['positions'][:, [0, 2, 1]][mesh['faces']],
                                                     facecolors=colors, edgecolors='none', rasterized=True))
                axis.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), zlim=(low[1], high[1]))
                axis.set_box_aspect((high - low)[[0, 2, 1]])
                axis.view_init(elev=CAMERA['elevation_degrees'], azim=CAMERA['azimuth_degrees'])
                axis.set_proj_type('ortho')
                axis.set_axis_off()
                m = panel['metrics']
                axis.set_title(f"{LABELS[panel['role']]} | repeat {panel['repeat']}\n"
                    f"cloth max {text_number(m['cloth_max_stretch'], '.8f')} | FEM min J {text_number(m['fem_min_J'])}\n"
                    f"FEM nonpos {text_number(m['fem_nonpositive'])} | negative volume {text_number(m['fem_negative_volume_m3'], '.3g')} m^3\n"
                    f"ABD min J {text_number(m['abd_min_J'])} | fixed drift {m['fixed_max_drift_m']:.3g} m", fontsize=8)
            figure.suptitle(f"{scene['scene']} | frame {scene['frame']} | t={scene['physical_time_seconds']:.2f} s\n"
                           'actual meshes; common camera and scale within scene; common colors across all scenes', fontsize=11)
            figure.subplots_adjust(left=.01, right=.88, bottom=.035, top=.88, hspace=.35, wspace=.01)
            bar_axis = figure.add_axes([.92, .22, .014, .52])
            colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis,
                                      label='cloth edge length / initial length')
            colorbar.formatter = ScalarFormatter(useOffset=False)
            colorbar.update_ticks()
            target = output / f"autodl_factor_{scene['scene_key']}_final.png"
            require(not target.exists(), 'Refusing to overwrite image')
            figure.savefig(target, dpi=150, bbox_inches='tight')
            record = {key: value for key, value in scene.items() if key not in ('mesh', 'panels', 'bounds_min_m', 'bounds_max_m')}
            record.update(image=str(target), image_sha256=sha(target), bounds_min_m=low.tolist(), bounds_max_m=high.tolist(),
                          panels=[{key: value for key, value in panel.items() if key not in ('positions', 'face_stretch')}
                                  for panel in scene['panels']])
            report['scenes'].append(record)
            print(json.dumps({'scene': scene['scene_key'], 'frame': scene['frame'], 'panels': len(scene['panels']),
                              'image': str(target)}), flush=True)
        except (ValueError, KeyError, OSError, AssertionError) as error:
            report['failures'].append({'scene_key': scene['scene_key'], 'failure': str(error)})
        finally:
            plt.close(figure)
    report['all_three_scenes_rendered'] = len(report['scenes']) == 3 and not report['failures']
    with (output / 'render_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def self_test():
    with tempfile.TemporaryDirectory(prefix='autodl_mesh_metrics_') as temporary:
        trace = Path(temporary)
        np.array([3, 1, 0, 0, 1, 2], dtype='<u4').tofile(trace / 'topology.bin')
        np.zeros(3, dtype='<i4').tofile(trace / 'boundary_types.bin')
        initial = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        initial.tofile(trace / 'state_0000.bin')
        (trace / 'metadata.json').write_text(json.dumps({'abd_point_num': 0}))
        mesh = geometry(trace, {'objects': [{'dimension': 2, 'body_type': 'FEM'}]})
        positions = initial.copy()
        positions[1, 0] = 1.2
        metrics, face = raw_metrics(mesh, positions)
        require(abs(metrics['cloth_max_stretch'] - 1.2) < 1e-12 and abs(face[0] - 1.2) < 1e-12,
                'Raw cloth stretch fixture failed')
        require(metrics['fem_min_J'] is None and metrics['abd_min_J'] is None, 'Absent material was misreported')
        positions[0, 0] = np.nan
        try:
            raw_metrics(mesh, positions)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite raw mesh was accepted')
    print(json.dumps({'synthetic_raw_metrics': 'passed', 'absent_groups_explicit': True,
                      'nonfinite_mesh_rejected': True, 'images_rendered': 0, 'gpu_or_remote_used': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--plan', default='configs/active/autodl_window.json')
    parser.add_argument('--matrix', default='reports/active/AUTODL_WINDOW_BATCH.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--repeats', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required outside --self-test')
    root = args.root.resolve()
    report = render(root, child(root, args.plan), child(root, args.matrix), args.output_dir.resolve(), args.repeats)
    return int(not report['all_three_scenes_rendered'])


if __name__ == '__main__':
    raise SystemExit(main())
