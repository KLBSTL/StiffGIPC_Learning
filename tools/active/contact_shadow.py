"""Offline AL-contact diagnostics from saved state windows; no simulation edits.

Contact-free FEM diagonal blocks are reconstructed by subtracting the complete
saved affine AL diagonal from total A, only when friction and reduced slack are
disabled. ABD compliance is explicitly pending without production vertex J.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_model_v48 import CONTACT
from analyze_perf_v36 import matrix_from_snapshot


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def summary(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic input')
    return {'count': int(len(values)), 'min': float(values.min()),
            'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': float(values.max())}


def contact_identity(contact):
    return (int(contact['kind']), *map(int, contact['ids']))


def contact_gram_shadow(ids, slots, gradients, weights, eligible, abd, free, inverses):
    """Bounded spectra of contact coupling in the exact reconstructed D0 norm."""
    selected = np.flatnonzero(eligible)
    if not len(selected):
        return {'available': False, 'reason': 'No eligible contacts'}
    rr, cc, vv = [], [], []
    used_vertices = np.unique(ids[selected][slots[selected]] - abd)
    for row, original in enumerate(selected):
        for j in range(4):
            if not slots[original, j]:
                continue
            vertex = ids[original, j] - abd
            if not free[vertex]:
                continue
            block = int(np.searchsorted(used_vertices, vertex))
            for axis in range(3):
                value = gradients[original, j, axis] * np.sqrt(weights[original])
                if value != 0:
                    rr.append(row); cc.append(3 * block + axis); vv.append(value)
    jac = sparse.coo_matrix((vv, (rr, cc)), shape=(len(selected), 3 * len(used_vertices))).tocsr()
    block_inverse = sparse.block_diag(list(inverses[used_vertices]), format='csr')
    gram = (jac @ block_inverse @ jac.T).tocsr()
    diagonal = gram.diagonal()
    adjacency = gram.copy().tocoo()
    meaningful = np.abs(adjacency.data) > 1e-12 * np.sqrt(np.maximum(
        diagonal[adjacency.row] * diagonal[adjacency.col], 0))
    adjacency = sparse.coo_matrix((np.ones(int(meaningful.sum())),
                                  (adjacency.row[meaningful], adjacency.col[meaningful])), shape=gram.shape).tocsr()
    count, component = connected_components(adjacency, directed=False)
    spectra = []
    for index in range(count):
        rows = np.flatnonzero(component == index)
        if len(rows) == 1:
            continue
        entry = {'contacts': int(len(rows)), 'contact_indices': selected[rows].tolist()}
        if len(rows) > 256:
            entry['spectrum_pending_reason'] = 'Bounded shadow caps dense connected-component spectra at 256 contacts'
        else:
            block = gram[rows][:, rows].toarray()
            eig = np.linalg.eigvalsh((block + block.T) / 2)
            scale = max(float(np.max(np.abs(eig))), 1e-300)
            entry.update({'min_eigenvalue': float(eig[0]), 'max_eigenvalue': float(eig[-1]),
                          'near_null_relative_1e10': int(np.sum(np.abs(eig) <= 1e-10 * scale)),
                          'negative_beyond_roundoff_relative_1e10': int(np.sum(eig < -1e-10 * scale))})
        spectra.append(entry)
    return {'available': True, 'operator': 'sqrt(gamma*mu) J_FEM D0_inverse J_FEM^T sqrt(gamma*mu)',
            'not_full_A_condition_number': True, 'contacts': len(selected),
            'components': int(count), 'largest_component_contacts': int(np.bincount(component).max()),
            'off_diagonal_component_cutoff_relative': 1e-12,
            'diagonal': summary(diagonal), 'nontrivial_component_spectra': spectra}


def analyze(run, stem):
    prefix = run / 'state_window' / stem
    inputs = []

    def load_state(name, dtype='<f8'):
        path = Path(str(prefix) + '_state_' + name + '.bin')
        inputs.append(path)
        expected = state['buffers'][name]
        if path.stat().st_size != expected:
            raise ValueError('Partial or inconsistent state buffer: ' + str(path))
        return np.fromfile(path, dtype=dtype)

    state_path = Path(str(prefix) + '_state.json')
    state = read(state_path); inputs.append(state_path)
    frame, direction_index = map(int, re.fullmatch(r'f(\d+)_n(\d+)', stem).groups())
    if frame != state['frame']:
        raise ValueError('Frame identity mismatch')
    c = load_state('contacts', CONTACT)
    vertices = load_state('vertices').reshape(-1, 3)
    safe = load_state('safe').reshape(-1, 3)
    scalars = load_state('scalars')
    if len(scalars) != 5:
        raise ValueError('Unknown scalar layout; expected mu, initial_mu, delta, friction rates')
    mu, initial_mu, delta, friction_rate, ground_friction_rate = scalars
    if min(mu, initial_mu, delta) <= 0 or not np.isfinite(scalars).all():
        raise ValueError('Invalid contact parameters')
    if not np.isin(c['kind'], [0, 1, 2]).all():
        raise ValueError('Unknown contact kind or raw layout')
    slots = np.arange(4)[None, :] < np.where(c['kind'] == 2, 1, 4)[:, None]
    ids = np.where(slots, c['ids'], 0)
    if np.any(c['ids'][slots] < 0) or np.any(c['ids'][slots] >= len(vertices)):
        raise ValueError('Contact vertex indices out of range')
    gradients = np.where(slots[:, :, None], c['grad'], 0.)
    local_mu = np.where(c['penalty_mu'] > 0, c['penalty_mu'] * mu / initial_mu, mu)
    value = c['offset'] + np.einsum('nki,nki->n', gradients, vertices[ids] - c['anchor'])
    slack = np.maximum(0., value - c['lambda'] / local_mu)
    effective_weight = c['gamma'] * local_mu
    if (effective_weight < 0).any() or not np.isfinite(effective_weight).all():
        raise ValueError('Invalid effective penalty weight')

    trace_meta_path = run / 'trace' / 'metadata.json'
    scene_path = run / 'output' / 'scene.json'
    topology_path = run / 'trace' / 'topology.bin'
    boundary_path = run / 'trace' / 'boundary_types.bin'
    initial_path = run / 'trace' / 'state_0000.bin'
    inputs += [trace_meta_path, scene_path, topology_path, boundary_path, initial_path]
    trace_meta, scene = read(trace_meta_path), read(scene_path)
    abd = int(trace_meta['abd_point_num'])
    top = np.fromfile(topology_path, '<u4'); nv, nf, nt = map(int, top[:3])
    if nv != len(vertices) or len(top) != 3 + 3 * nf + 4 * nt:
        raise ValueError('Topology dimensions mismatch')
    tets = top[3 + 3 * nf:].reshape(-1, 4)
    in_tet = np.zeros(nv, dtype=bool); in_tet[tets.ravel()] = True
    labels = np.where(in_tet, 'fem_tet', 'cloth').astype('<U12')
    labels[:abd] = 'abd'
    combinations = []
    for row, mask, kind in zip(ids, slots, c['kind']):
        names = sorted(set(labels[row[mask]]))
        if kind == 2:
            names.append('ground')
        combinations.append('+'.join(names))
    combinations = np.asarray(combinations, dtype=str)
    groups = {'all': np.ones(len(c), bool), 'positive_slack': c['slack'] > 0,
              'zero_slack': c['slack'] == 0, 'zero_lambda': c['lambda'] == 0,
              'age_0': c['release_age'] == 0, 'age_1_5': (c['release_age'] >= 1) & (c['release_age'] <= 5),
              'age_6_25': (c['release_age'] >= 6) & (c['release_age'] <= 25)}
    for kind, name in enumerate(('pt', 'ee', 'ground')):
        groups[name] = c['kind'] == kind
    for name in np.unique(combinations):
        groups['bodies:' + name] = combinations == name
    records = {}
    for name, mask in groups.items():
        records[name] = {
            'count': int(mask.sum()),
            'affine_c_over_delta': summary(value[mask] / delta),
            'negative_c_over_delta': summary(np.maximum(-value[mask], 0) / delta),
            'split_constraint_residual_over_delta': summary(np.abs(value[mask] - c['slack'][mask]) / delta),
            'lambda_over_mu_delta': summary(c['lambda'][mask] / (local_mu[mask] * delta)),
            'abs_lambda_c_over_mu_delta_squared': summary(np.abs(c['lambda'][mask] * value[mask]) / (local_mu[mask] * delta * delta)),
            'gamma': summary(c['gamma'][mask]), 'mu': summary(local_mu[mask]),
            'release_age': summary(c['release_age'][mask]),
        }
    full_stats_path = run / 'output' / 'stats.json'
    inputs.append(full_stats_path)
    complete_frames = read(full_stats_path)['frames']
    f = complete_frames[frame - 1] if frame <= len(complete_frames) else {}
    nr = f.get('newton', [])[direction_index - 1] if direction_index <= len(f.get('newton', [])) else {}
    if nr and (nr.get('toi_outer') != state['outer'] or nr.get('toi_inner') != state['inner']):
        raise ValueError('Saved state does not match nonlinear iteration')
    result = {'system': stem, 'frame': frame, 'outer': state['outer'], 'inner': state['inner'],
              'state_capture_stage': state['stage'], 'contacts': len(c), 'contact_bytes': CONTACT.itemsize,
              'mu': float(mu), 'initial_mu': float(initial_mu), 'delta_m': float(delta),
              'frame_statistics_complete': bool(nr),
              'pcg_iterations': nr.get('pcg', {}).get('iterations'),
              'trial_direction_axis_velocity_m_s': nr.get('trial_newton_axis_velocity_m_s'),
              'line_search_r': nr.get('line_search_r'),
              'energy_before': nr.get('trial_energy_before'), 'energy_after': nr.get('trial_energy_after'),
              'max_slack_reconstruction_error_over_delta': float(np.max(np.abs(slack - c['slack']), initial=0) / delta),
              'groups': records,
              'current_vertices_minus_safe_m': summary(np.linalg.norm(vertices - safe, axis=1)),
              'global_kkt_or_physical_quality_certified': False}

    # Geometric volume diagnostics cover movable FEM tetrahedra only, not ABD.
    fem_tets = tets[np.all(tets >= abd, axis=1)]
    if len(fem_tets):
        initial = np.fromfile(initial_path, '<f8').reshape(-1, 3)
        def det(x):
            p = x[fem_tets]
            return np.linalg.det(np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=-1))
        rest = det(initial)
        if np.any(rest == 0):
            raise ValueError('Degenerate rest tetrahedron')
        for name, x in [('current_trial', vertices), ('safe', safe)]:
            jac = det(x) / rest
            result[name + '_fem_j'] = summary(jac)
            result[name + '_nonpositive_fem_tets'] = int(np.sum(jac <= 0))

    compliance = {'method': 'diag3(A_total - H_all_saved_affine_AL), on free FEM DOFs only',
                  'total_A_used_as_contact_free': False, 'available': False,
                  'abd_compliance_available': False,
                  'missing_for_abd': ['production per-vertex ABD Jacobian (3x12) or independently verified equivalent',
                                      'ABD body index and fixed-DOF mask matching global A'],
                  'candidate_algorithm_modified': False}
    result['local_compliance_shadow'] = compliance
    friction = scene['effective_scalar_fields'].get('friction_compiled')
    reduced = f.get('toi_reduced_slack')
    if friction is not False or reduced is not False:
        compliance['unavailable_reason'] = 'Need friction_compiled=false and reduced_slack=false to isolate contact-free A'
    else:
        meta, matrix, rhs = matrix_from_snapshot(prefix)
        inputs += [Path(str(prefix) + suffix) for suffix in ('_meta.json', '_rows.bin', '_cols.bin', '_values.bin', '_rhs.bin')]
        q = load_state('q').reshape(-1, 12)
        offset = 12 * len(q)
        if meta['dofs'] != offset + 3 * (nv - abd):
            raise ValueError('Generalized ABD/FEM DOF layout mismatch')
        rows = np.fromfile(str(prefix) + '_rows.bin', '<i4')
        cols = np.fromfile(str(prefix) + '_cols.bin', '<i4')
        blocks = np.fromfile(str(prefix) + '_values.bin', '<f8').reshape(-1, 3, 3).transpose(0, 2, 1)
        diagonal = np.zeros((meta['dofs'] // 3, 3, 3))
        diag_entries = rows == cols
        np.add.at(diagonal, rows[diag_entries], blocks[diag_entries])
        d0 = diagonal[offset // 3:].copy()
        contact_diagonal = np.zeros_like(d0)
        # Primitive vertices are distinct in the current contact schema. Do
        # not silently omit cross terms for a malformed repeated vertex ID.
        for row, mask in zip(ids, slots):
            if len(np.unique(row[mask])) != int(mask.sum()):
                raise ValueError('Repeated vertex ID needs aggregated Jacobian before diagonal reconstruction')
        for j in range(4):
            select = slots[:, j] & (ids[:, j] >= abd)
            grad = gradients[select, j]
            np.add.at(contact_diagonal, ids[select, j] - abd,
                      effective_weight[select, None, None] * grad[:, :, None] * grad[:, None, :])
        d0 -= contact_diagonal
        fixed = np.fromfile(boundary_path, '<i4') == 1
        if len(fixed) != nv:
            raise ValueError('Boundary layout mismatch')
        free = ~fixed[abd:]
        eig = np.linalg.eigvalsh((d0 + d0.transpose(0, 2, 1)) / 2)
        asym = np.linalg.norm(d0 - d0.transpose(0, 2, 1), axis=(1, 2)) / np.maximum(np.linalg.norm(d0, axis=(1, 2)), 1e-300)
        good = free & (eig[:, 0] > 0) & (asym <= 1e-10)
        compliance['free_fem_blocks'] = int(free.sum())
        compliance['nonpositive_free_fem_blocks'] = int(np.sum(free & (eig[:, 0] <= 0)))
        compliance['asymmetric_free_fem_blocks'] = int(np.sum(free & (asym > 1e-10)))
        compliance['min_free_block_eigenvalue'] = float(eig[free, 0].min()) if free.any() else None
        compliance['contact_diagonal_norm_over_total_diagonal_norm'] = float(np.linalg.norm(contact_diagonal[free]) / max(np.linalg.norm(diagonal[offset // 3:][free]), 1e-300))
        inverses = np.zeros_like(d0); inverses[good] = np.linalg.inv(d0[good])
        fem_only = np.all(~slots | (ids >= abd), axis=1)
        eligible = fem_only.copy()
        qi = np.zeros(len(c))
        for j in range(4):
            selected = slots[:, j] & (ids[:, j] >= abd)
            vi = ids[selected, j] - abd
            eligible[selected] &= (~free[vi]) | good[vi]
            grad = gradients[selected, j].copy(); grad[~free[vi]] = 0
            qi[selected] += np.einsum('ni,nij,nj->n', grad, inverses[vi], grad)
        eligible &= qi > 0
        compliance['available'] = bool(eligible.any())
        compliance['eligible_fem_only_contacts'] = int(eligible.sum())
        compliance['contacts_with_abd_pending'] = int((~fem_only).sum())
        compliance['coverage_note'] = 'All normal AL contact diagonals removed; per-contact compliance is complete only for FEM-only contacts with SPD free blocks'
        compliance['by_group'] = {}
        for name, mask in groups.items():
            use = mask & eligible
            compliance['by_group'][name] = {'count': int(use.sum()),
                'compliance': summary(qi[use]), 'current_mu_times_compliance': summary(local_mu[use] * qi[use]),
                'current_gamma_mu_times_compliance': summary(effective_weight[use] * qi[use]),
                'shadow_mu_0p1_over_compliance': summary(.1 / qi[use]),
                'shadow_mu_over_current_mu': summary(.1 / (qi[use] * local_mu[use]))}
        compliance['contact_gram'] = contact_gram_shadow(ids, slots, gradients, effective_weight,
                                                        eligible, abd, free, inverses)
        solution_path = Path(str(prefix) + '_x.bin')
        if solution_path.exists():
            inputs.append(solution_path); x = np.fromfile(solution_path, '<f8')
            if x.shape != rhs.shape:
                raise ValueError('Saved linear solution size mismatch')
            displacement = np.zeros_like(vertices); displacement[abd:] = x[offset:].reshape(-1, 3)
            directional = np.einsum('nki,nki->n', gradients, displacement[ids])
            contribution = effective_weight * directional ** 2
            total_curvature = float(x @ (matrix @ x))
            result['direction_curvature'] = {
                'assembled_total_xAx': total_curvature,
                'fem_only_contact_xHcontactx': float(contribution[fem_only].sum()),
                'fem_only_positive_slack_contact_xHcontactx': float(contribution[fem_only & (c['slack'] > 0)].sum()),
                'full_contact_curvature_available': bool(fem_only.all()),
                'fem_only_fraction_of_total': float(contribution[fem_only].sum() / total_curvature) if total_curvature != 0 else None,
                'partial_contribution_is_not_a_bound': not bool(fem_only.all()),
                'cpu_true_relative_residual': float(np.linalg.norm(rhs - matrix @ x) / max(np.linalg.norm(rhs), 1e-300))}
    result['inputs_sha256'] = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(inputs))}
    return result, c


def compare_contacts(left, right, left_name, right_name):
    a = {contact_identity(c): c for c in left}
    b = {contact_identity(c): c for c in right}
    shared = sorted(a.keys() & b.keys())
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError('Duplicate contact identity prevents unambiguous update comparison')
    return {'from': left_name, 'to': right_name, 'shared': len(shared),
            'added': len(b.keys() - a.keys()), 'removed': len(a.keys() - b.keys()),
            'observed_lambda_abs_delta': summary([abs(b[k]['lambda'] - a[k]['lambda']) for k in shared]),
            'observed_gamma_abs_delta': summary([abs(b[k]['gamma'] - a[k]['gamma']) for k in shared]),
            'observed_slack_abs_delta_m': summary([abs(b[k]['slack'] - a[k]['slack']) for k in shared]),
            'note': 'Observed adjacent captures; this is not an isolated multiplier-update derivative'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--systems', nargs='+', default=['f25_n7', 'f25_n8'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    if args.output.exists():
        raise FileExistsError('Refusing to replace prior shadow report')
    if not (run / 'result.json').exists():
        raise ValueError('Run result marker missing; do not inspect partially written snapshots')
    systems, contacts = [], []
    for name in args.systems:
        result, c = analyze(run, name); systems.append(result); contacts.append(c)
    report = {'schema_version': 1, 'run': str(run), 'run_result': read(run / 'result.json'),
              'systems': systems,
              'contact_changes': [compare_contacts(contacts[i], contacts[i + 1], args.systems[i], args.systems[i + 1]) for i in range(len(contacts) - 1)],
              'conclusion': 'Observational diagnostics only; no penalty, solver, or physical parameters changed',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'systems': len(systems), 'output': str(args.output.resolve()),
                      'contacts': [s['contacts'] for s in systems],
                      'fem_compliance_available': [s['local_compliance_shadow']['available'] for s in systems]}))


if __name__ == '__main__':
    main()
