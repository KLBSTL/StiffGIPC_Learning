"""Verify and catalogue this finite diagnosis without changing old decisions."""
import json
from pathlib import Path
from config import ROOT, read, sha, matches_requested
from ipc_benchmark import write

TAG = 'ipc_observer_causal_20261005'


def main():
    folder = ROOT / 'reports/active'
    protocol = read(folder / f'{TAG}_protocol.json')
    analysis = read(folder / f'{TAG}_analysis.json')
    records = {r['name']: r for r in analysis['runs']}
    manifest = read(ROOT / 'builds/active/manifest.json')
    identities = []
    for item in manifest['compiler_inputs']:
        for kind in ('source', 'object'):
            entry = item[kind]
            identities.append({'kind': kind, 'path': entry['path'],
                               'passed': sha(ROOT / entry['path']) == entry['sha256']})
    binaries = manifest['binaries']
    for entry in binaries.values():
        identities.append({'kind': 'binary', 'path': entry['path'],
                           'passed': sha(ROOT / entry['path']) == entry['sha256']})
    checks = []
    plans = [ROOT / f'configs/active/{TAG}.json',
             ROOT / 'configs/active/ipc_observer_fixed_probe_20261005.json']
    for plan in plans:
        for spec in read(plan)['runs']:
            run = ROOT / 'runs/active' / spec['name']
            requested = read(run / 'requested.json')
            result = read(run / 'result.json')
            c = requested['expanded_config']
            tests = {
                'requested_config': matches_requested(c, spec['config']),
                'binary_identity': sha(Path(requested['command'][0])) == requested['exe_sha256'],
                'completed': result['status'] == 'completed' and result['exit_code'] == 0,
                'frame_budget': result['recorded_frames'] == c['steps'],
                'finite': result['finite'],
                'legacy_stopping': c['ipc_termination'] == 'legacy'
                    and c['ipc_newton_tol'] == .01 and c['ipc_cumulative_tol'] == .01
                    and c['ipc_min_updates'] == 6 and c['pcg_rho_tol'] == 1e-4,
                'execution_scope': c['backend'] == 'ipc' and c['execution'] == 'host'
                    and c['mas'] == 'legacy' and not any(c[k] for k in ('refit', 'batch', 'reuse')),
            }
            resolved_path = run / 'resolved_config.json'
            if resolved_path.exists():
                resolved = read(resolved_path)
                stop = resolved['ipc_stopping']
                tests['resolved_config'] = (
                    resolved['contact_backend'] == 'ipc'
                    and resolved['configured_pcg_execution'] == 'host'
                    and resolved['pcg_rho_tol'] == 1e-4
                    and stop['termination'] == 'legacy' and stop['cumulative_tol'] == .01
                    and stop['min_updates'] == 6 and not stop['residual_shadow']
                    and not resolved['mas']['cholesky']
                    and not any(resolved['acceleration_features'].values()))
            if spec['name'] in records:
                r = records[spec['name']]
                tests['numerical_checks'] = not r['pcg_failures'] and r['finite'] and r['velocity_finite']
                tests['actual_velocity_files'] = r['velocity_frames'] == (101 if c['trace_velocity'] else 0)
            checks.append({'run': spec['name'], 'passed': all(tests.values()), 'checks': tests,
                           'resolved_config_available': resolved_path.exists()})
    probe = read(folder / 'ipc_observer_fixed_probe_20261005_analysis.json')
    report = {
        'scope': 'Evidence integrity and finite diagnostic completion only; no physical/performance certification',
        'run_checks': checks, 'build_input_identity': identities,
        'input_state_identity': analysis['identity'],
        'old_quality_protocol_unchanged': sha(folder / 'ipc_revision_20261005_quality_protocol.json')
            == protocol['quality_protocol_sha256'],
        'predeclared_protocol_unchanged': sha(folder / f'{TAG}_protocol.json') == analysis['protocol_sha256'],
        'fixed_operator_cpu_reference_and_restoration': probe['passed'],
        'gpu_lock_present': (ROOT / 'runs/active/.gpu.lock').exists(),
        'frozen_program_resolved_config_limitation':
            'Frozen raw/observed programs do not emit resolved_config; requested environment, '
            'binary identity, native logs, work and exported file counts are the available evidence.',
    }
    report['passed'] = (all(c['passed'] for c in checks)
        and all(i['passed'] for i in identities)
        and all(analysis['identity'].values())
        and report['old_quality_protocol_unchanged'] and report['predeclared_protocol_unchanged']
        and probe['passed'] and not report['gpu_lock_present'])
    target = folder / f'{TAG}_verification.json'
    if target.exists():
        if read(target) != report:
            raise ValueError('Existing verification evidence differs; preserve it: ' + str(target))
    else:
        write(target, report)
    if not report['passed']:
        raise ValueError('Evidence verification failed; see ' + str(target))
    index_path = folder / 'EXPERIMENT_INDEX.json'
    index = read(index_path)
    current = {c['run'] for c in checks}
    updates = {}
    evidence = ['reports/active/IPC_OBSERVER_CAUSAL_RESULTS_20261005.md',
                target.relative_to(ROOT).as_posix()]
    for row in index['entries']:
        if row['id'] in current:
            decision = {'status': 'pending', 'status_scope': 'Diagnostic evidence verified; promotion not certified',
                        'reason': 'Completed finite observation/operator diagnosis. Old quality bounds preserved.',
                        'performance_certified': False, 'quality_certified': False}
            if row['id'] in records:
                decision['old_protocol_material_passed'] = records[row['id']]['old_protocol_material_passed']
            else:
                decision['fixed_reference_restoration_passed'] = True
            row.update(decision)
            row['evidence'] = list(dict.fromkeys(row['evidence'] + evidence))
            updates[row['id']] = decision | {'decision_evidence': evidence[0]}
    if len(updates) != len(current):
        raise ValueError('Missing diagnosis entries in index')
    decisions_path = folder / 'EXPERIMENT_DECISIONS.json'
    decisions = read(decisions_path)
    decisions.update(updates)
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'runs_verified': len(checks),
                      'build_inputs_verified': len(identities), 'index_updated': len(updates),
                      'old_quality_protocol_unchanged': report['old_quality_protocol_unchanged']}))


if __name__ == '__main__':
    main()
