"""Reopen native banks and solutions; audit actual A-energy, actions and costs."""
import json
from pathlib import Path

# Import this before NumPy so the existing fixed-reference thread policy applies.
from m3_numeric import ROOT, FrozenSystem, need, rel, residual_metrics
import numpy as np
from cgstats_numeric import coefficients
from qrfree_numeric import CoordinateBalanced, scaled_coefficients
from cpu_fixed_reference import decimal_residual
from local_identity import record, verify
from linux_runner import write_new
from m0_analyze import csv_write

RAW = ROOT / 'runs/innovation_p5_native_history_20261016'
REPORT = ROOT / 'reports/innovation_p5_native_history_20261016'
RESTORE = ('full_system_restored', 'primary_restored_bitwise', 'workspace_restored',
           'production_graph_restored', 'operator_signature_restored',
           'production_info_restored', 'solve_context_restored', 'config_restored',
           'cost_trace_restored')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    destination = REPORT / 'REPRESENTATIVE_AUDIT.json'
    need(not destination.exists(), 'Do not overwrite representative audit')
    tables = REPORT / 'representative_tables'
    need(not tables.exists(), 'Use a fresh audit table directory')
    tables.mkdir()
    ledger = read(RAW / 'representative_ledger.json')
    analysis = read(RAW / 'analysis/summary.json')
    need(ledger['complete'] and analysis['implementation_checks_passed'], 'Incomplete inputs')
    verify(analysis['inputs'] + [analysis['analyzer']])
    stage1 = REPORT / 'BUILD_AND_SMOKE_SEAL.json'
    verify(read(stage1)['files'])
    identity = read(RAW / 'compiled_identity.json')
    verify(identity['sources'] + [identity['exe']] + identity['dlls'])
    costs, setups, banks, actions, solutions, native = [], [], [], [], [], []
    bindings = [record(stage1), record(RAW / 'representative_ledger.json'),
                record(RAW / 'analysis/summary.json'), record(Path(__file__))]
    for label, scene, frame in (('bunny', 'Bunny', 14), ('mixed', 'Mixed', 15)):
        folder = RAW / label
        result = read(folder / 'result.json')
        validation = read(folder / 'config_validation.json')
        power = read(folder / 'power_samples.json')
        need(result['status'] == 'completed' and result['recorded_frames'] == frame
             and result['finite'] and result['cleanup_owned_job_empty'], 'Native run incomplete')
        need(validation['passed'] and power['samples']
             and power['after'] == power['expected']
             and all(r['guid'] == power['expected'] for r in power['samples']), 'Power/config mismatch')
        systems = {d: FrozenSystem(folder / 'fixed' / f'f{frame}_n{d}') for d in (1, 2)}
        need(np.array_equal(systems[1].active, systems[2].active), 'Changed active coordinates')
        for direction, system in systems.items():
            study = read(str(system.prefix) + '_native_history_study.json')
            need(study['passed'] and not study['restoration_error']
                 and all(study[k] for k in RESTORE), 'Restoration failure')
            need(study['rho_tolerance'] == 1e-4 and study['mas'] == 'legacy'
                 and study['native_execution'] == 'host' and study['zero_initial_guess'], 'Native protocol changed')
            need(len(study['runs']) == 6, 'Missing native runs')
            setups.append(dict(scene=scene, direction=direction,
                private_setup_host_ms=study['private_setup_host_ms'], restore_host_ms=study['restore_host_ms'],
                bank_bytes=study['history_bank_bytes'], **{k: study[k] for k in RESTORE}))
            for row in study['runs']:
                x = np.fromfile(row['solution_file'], dtype='<f8')
                need(x.shape == system.full_b.shape and np.isfinite(x).all(), 'Bad native x')
                full_res = np.linalg.norm(system.full_b - system.full_A @ x) / np.linalg.norm(system.full_b)
                need(abs(full_res - row['true_relative_residual']) < 1e-8, 'Native residual mismatch')
                costs.append(dict(scene=scene, direction=direction, **{k: row[k] for k in
                    ('pair', 'order', 'capture', 'iterations', 'reset_host_ms', 'solve_host_ms',
                     'solve_event_ms', 'export_and_residual_host_ms')}))
                native_row = dict(scene=scene, direction=direction, pair=row['pair'], capture=row['capture'],
                                  iterations=row['iterations'], original_A_relative_l2=float(full_res),
                                  Decimal50_relative_l2=None)
                if row['capture'] and row['pair'] == 1:
                    _, dec = decimal_residual(system.full_A, system.full_b, x)
                    need(abs(dec['relative_l2'] - full_res) < 1e-9, 'Native Decimal50 mismatch')
                    native_row['Decimal50_relative_l2'] = dec['relative_l2']
                native.append(native_row)
                if not row['capture']:
                    continue
                steps, width = row['iterations'], row['history_width']
                ids = np.fromfile(row['indices_file'], dtype='<i4')
                history = np.fromfile(row['history_file'], dtype='<f8').reshape(len(x), width, order='F')
                scalar = np.fromfile(row['scalars_file'], dtype='<f8').reshape(steps, 3)
                need(np.isfinite(history).all() and np.isfinite(scalar).all(), 'Nonfinite bank')
                expected = np.r_[np.arange(min(16, steps)), np.arange(max(16, steps-112), steps)]
                need(np.array_equal(ids, expected) and np.count_nonzero(history[:, 0]) == 0,
                     'Bank index/initial iterate mismatch')
                # Verify actual retained consecutive increments, including the final update.
                # No formula or recorder slot logic is used to compute their A-energy.
                extended = np.column_stack([history, x])
                next_ids = np.r_[ids[1:], steps]
                contiguous = np.flatnonzero(next_ids == ids + 1)
                delta = extended[:, contiguous + 1] - history[:, contiguous]
                energy = np.einsum('ij,ij->j', delta, system.full_A @ delta)
                alpha, rho, pap = scalar.T
                energy_error = rel(energy, (alpha * rho)[ids[contiguous]])
                need(energy_error < 1e-6, 'Actual saved increment energy mismatch')
                context_error = float(np.max(np.abs(alpha * pap - rho) / rho))
                need(context_error < 1e-10 and np.all(rho[:-1] > 1e-4 * rho[0])
                     and rho[-1] <= 1e-4 * rho[0], 'Scalar context/previous-rho stop mismatch')
                S = (x[:, None] - history)[system.active]
                direct = S.T @ (system.A @ S)
                direct = .5 * (direct + direct.T)
                # Independently form each suffix sum, rather than call statistical_gram.
                suffix = np.array([np.sum(alpha[i:] * rho[i:], dtype=np.float64) for i in ids])
                H = suffix[np.maximum.outer(np.arange(width), np.arange(width))]
                us, info = coefficients(H, 8)
                ud, dinfo = coefficients(direct, 8)
                need(info['actual_rank'] == dinfo['actual_rank'] == 8, 'Rank-eight unavailable')
                banks.append(dict(scene=scene, direction=direction, pair=row['pair'], steps=steps, width=width,
                    retained_increment_checks=len(contiguous), increment_energy_relative=energy_error,
                    alpha_pAp_rho_max_relative=context_error, Gram_relative=rel(H, direct)))
                if direction != 1:
                    continue  # n2 history is audited, never used to construct target basis.
                target = systems[2]
                stat = CoordinateBalanced(target.A, target.apply, S @ scaled_coefficients(us, info['selected_eigenvalues']))
                exact = CoordinateBalanced(target.A, target.apply, S @ scaled_coefficients(ud, dinfo['selected_eigenvalues']))
                rng = np.random.default_rng(20261016)
                probes = [target.b.copy(), rng.normal(size=len(target.b)), rng.normal(size=len(target.b))]
                for i, probe in enumerate(probes):
                    stat_value, direct_value = stat(probe), exact(probe)
                    need(np.isfinite(stat_value).all() and np.isfinite(direct_value).all()
                         and probe @ stat_value > 0 and probe @ direct_value > 0, 'Invalid probe action')
                    actions.append(dict(scene=scene, pair=row['pair'], probe='target_b' if i == 0 else f'random_{i}',
                        statistics_vs_direct_relative=rel(stat_value, direct_value),
                        source_direction=1, target_direction=2, target_history_used=False))
        target = systems[2]
        for row in [r for r in analysis['methods'] if r['scene'] == scene]:
            need(row['converged'] and row.get('status') is None, 'Unexpected transfer failure')
            verify([row['solution']])
            x = np.load(row['solution']['path'], allow_pickle=False)
            need(x.shape == target.b.shape and np.isfinite(x).all(), 'Bad saved target x')
            metrics, _ = residual_metrics(target.A, target.b, x, target.apply)
            need(max(metrics.values()) <= .01 and all(abs(metrics[k] - row[k]) < 1e-9 for k in metrics),
                 'Saved target residual/metadata mismatch')
            solutions.append(dict(scene=scene, pair=row['pair'], method=row['method'], iterations=row['iterations'], **metrics))
    need(len(native) == 24 and len(banks) == 12 and len(actions) == 18 and len(solutions) == 14,
         'Incomplete representative audit')
    for name, rows in (('native_costs', costs), ('native_setups', setups), ('bank_audit', banks),
                       ('action_audit', actions), ('target_solution_audit', solutions), ('native_solution_audit', native)):
        csv_write(tables / (name + '.csv'), rows)
    verify(bindings + analysis['inputs'])
    write_new(destination, dict(passed=True, source=record(Path(__file__)), bindings=bindings,
        native_solutions=native, banks=banks, actions=actions, target_solutions=solutions,
        power_guid=power['expected'], source_direction=1, target_direction=2, target_history_used_for_basis=False,
        native_Decimal50_checks=4, original_implementation_checks_unchanged=True,
        native_balanced_PCG_implemented=False, performance_certified=False))
    print(json.dumps(dict(passed=True, native_solves=len(native), banks=len(banks),
        retained_increment_checks=sum(r['retained_increment_checks'] for r in banks),
        target_solutions=len(solutions), native_Decimal50_checks=4,
        max_increment_energy_relative=max(r['increment_energy_relative'] for r in banks),
        max_action_relative=max(r['statistics_vs_direct_relative'] for r in actions))), flush=True)


if __name__ == '__main__':
    main()
