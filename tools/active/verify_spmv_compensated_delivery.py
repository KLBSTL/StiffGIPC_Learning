"""CPU-only delivery check of immutable local evidence; no candidate promotion.

All assertions run before appending to the catalogue. Historical entries and
decisions retain their original contents, including failed long experiments.
"""
import copy
import io
import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone

from config import ROOT, digest, matches_requested, read, sha
from ipc_benchmark import write
from spmv_compensated import TAG, audit_controller
from validate_run import validate
from verify_ipc_light_delivery import CountedResult
import test_contracts

IDENTITY = '5269bb3d4668f016f845a3f48af38fef5ed1538caba5f40057e9d15d79075af4'
INITIAL_IDENTITY = '290356b4f0d35bc7185baac374d407a8ddb3c9d6ef934dc0b2341650257854be'


def rejected_observation_mutations(frames, terminal_frames, cfg, terminal_cfg):
    """Prove missing/duplicate records and incorrect inactive gates fail audit."""
    mutations = ('missing_array', 'missing_row', 'duplicate', 'inactive_min', 'inactive_animation')
    count = 0
    for name in mutations:
        changed = copy.deepcopy(frames[:1])
        node = changed[0]['newton'][0]
        if name == 'missing_array':
            del node['ipc_residual_observations']
        elif name == 'missing_row':
            node['ipc_residual_observations'].clear()
        elif name == 'duplicate':
            node['ipc_residual_observations'].append(copy.deepcopy(node['ipc_residual']))
        else:
            key = 'minimum_updates_ready' if name == 'inactive_min' else 'animation_ready'
            for row in (node['ipc_residual'], node['ipc_residual_observations'][0]):
                row[key] = not row[key]
        try:
            audit_controller(changed, cfg)
        except (AssertionError, KeyError):
            count += 1
        else:
            raise AssertionError('Invalid observation accepted: ' + name)
    changed = copy.deepcopy(terminal_frames)
    node = next(n for f in changed for n in f['newton'] if 'ipc_terminal_residual' in n)
    node['ipc_residual_observations'].reverse()
    try:
        audit_controller(changed, terminal_cfg)
    except AssertionError:
        count += 1
    else:
        raise AssertionError('Reordered normal/terminal records accepted')
    assert count == 6
    return count


def check_study(record):
    path = ROOT / record['path']
    study = read(path)
    assert sha(path) == record['sha256'] and study == record['result'] and study['passed']
    assert study['cpu_long_double_digits'] == 53
    assert not any(study[k] for k in ('full_PCG_speed_or_quality_certified',
                                   'full_PCG_zero_rhs_reexecuted',
                                   'production_graph_growth_invalidations_tested'))
    assert all(study[k] for k in ('all_work_buffers_restored_bitwise', 'rhs_unchanged_bitwise',
                                 'system_and_MAS_scratch_unchanged', 'production_graph_unchanged'))
    assert len(study['cases']) == 3
    assert {c['input'] for c in study['cases']} == {'actual_solved_Newton_direction', 'zero', 'signed_tail_pattern'}
    for case in study['cases']:
        assert case['stored_blocks'] % 256 != 0 and case['lower_blocks'] == 0
        assert case['old_passed'] and case['old_Ap_reference']['passed']
        assert case['old_dot_cpu_error'] <= case['old_scalar_roundoff_bound']
        assert len(case['runs']) == 3
        assert {r['execution'] for r in case['runs']} == {'host_launch', 'graph_first', 'graph_replay'}
        for row in case['runs']:
            assert row['passed'] and row['Ap_pair_within_bounds'] and row['Ap_cpu_reference']['passed']
            assert row['Ap_cpu_reference']['max_error_over_own_bound'] <= 1
            assert row['quadratic_cpu_error'] <= case['fused_scalar_roundoff_bound']
            if case['input'] == 'zero':
                assert row['Ap_bitwise_equal'] and row['quadratic'] == 0
    empty = study['empty_matrix_fixture']
    assert len(empty) == 3 and {r['execution'] for r in empty} == {'host_launch', 'graph_first', 'graph_replay'}
    assert all(r['passed'] and r['Ap_and_quadratic_finite_zero'] and r['quadratic'] == 0 for r in empty)
    return {'path': record['path'], 'sha256': sha(path), 'input_checks': 9, 'empty_checks': 3,
            'restoration_checks_passed': True, 'gpu_lower_blocks_covered': False}


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    assert not target.exists(), 'Never overwrite delivery evidence or append twice'
    protocol_path = folder / f'{TAG}_protocol.json'
    protocol = read(protocol_path)
    analysis_path = folder / f'{TAG}_analysis.json'
    analysis = read(analysis_path)
    assert protocol['identity_at_prepare'] == analysis['exe_sha256'] == IDENTITY
    for item in (protocol, analysis):
        assert not any(item[k] for k in ('performance_certified', 'quality_certified', 'default_promoted'))
    quality_path = folder / 'ipc_revision_20261005_quality_protocol.json'
    assert sha(quality_path) == protocol['old_quality_sha256']
    assert protocol['old_quality_sha256'] == '1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f'
    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    exe = ROOT / 'builds/active/Release/gipc.exe'
    assert sha(exe) == IDENTITY == manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    assert manifest['status'] == 'completed' and len(manifest['compiler_inputs']) == 37
    assert len(manifest['changed_objects']) == 3
    archived = ROOT / 'builds/active/provenance/20261005T160145_988186Z_ipc_spmv_compensated_20261005_tail_guard/manifest.json'
    assert read(archived) == manifest
    initial_path = ROOT / 'builds/active/provenance/20261005T155443_023848Z_ipc_spmv_compensated_20261005/manifest.json'
    initial = read(initial_path)
    assert initial['status'] == 'completed' and len(initial['changed_objects']) == 11
    assert len(initial['compiler_inputs']) == 37
    assert initial['binaries'][exe.relative_to(ROOT).as_posix()]['sha256'] == INITIAL_IDENTITY
    link_commands = '\n'.join(c for row in manifest['link_evidence'] for c in row['commands']).upper()
    identity_checks = 0
    for item in manifest['files']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
        identity_checks += 1
    for unit in manifest['compiler_inputs']:
        for item in [unit['source'], unit['object'], *unit['project_include_inputs']]:
            assert sha(ROOT / item['path']) == item['sha256'], item['path']
            identity_checks += 1
        assert (ROOT / unit['object']['path']).name.upper() in link_commands
    assert not (ROOT / 'runs/active/.gpu.lock').exists()

    cpu_path = folder / f'{TAG}_cpu_checks.json'
    cpu = read(cpu_path)
    assert sha(cpu_path) == protocol['cpu_checks_sha256'] and cpu['passed']
    assert (cpu['config_tests'], cpu['config_subtests']) == (15, 88)
    for record in cpu['native']:
        cpu_exe = ROOT / f"builds/active/Release/{record['name']}.exe"
        assert sha(cpu_exe) == record['exe_sha256']
        result = subprocess.run([str(cpu_exe)], capture_output=True, text=True, check=True, timeout=30)
        assert result.stdout.strip() == record['stdout'] and record['exit_code'] == 0
    native_counts = json.loads(next(r['stdout'] for r in cpu['native'] if r['name'] == 'ipc_residual_controller_test'))
    assert native_counts == {'passed': True, 'property_checks': 2923, 'audited_stress_steps': 800, 'controller_groups': 3}
    assert sha(ROOT / 'sources/stiff_active/StiffGIPC/solver/ipc_residual_controller.h') == cpu['controller_source_sha256']
    assert sha(ROOT / 'sources/stiff_active/tests/ipc_residual_controller_test.cpp') == cpu['controller_test_source_sha256']
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, resultclass=CountedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful() and (tests.testsRun, tests.subtests) == (15, 88), stream.getvalue()
    cpu_operator = ROOT / 'sources/stiff_active/StiffGIPC/linear_system/utils/spmv_quadratic_cpu_check.py'
    operator = subprocess.run([sys.executable, str(cpu_operator)], capture_output=True, text=True, check=True, timeout=30)
    assert operator.stdout.startswith('PASS: 9 block-count ownership cases')

    rows, stages, artifacts, run_evidence = [], {}, [], []
    controller_totals = {k: 0 for k in ('observations', 'active_observations', 'audited_accepts', 'compensated_exits', 'terminal_diagnostic_observations')}
    enabled_directions = smoke_directions = 0
    saved_activation = {}
    analysis_audits = {r['name']: {k: v for k, v in r.items() if k != 'name'} for r in analysis['controller_audits']}
    for stage, count in (('guards', 2), ('smoke', 16), ('activation', 2)):
        plan_path = ROOT / f'configs/active/{TAG}_{stage}.json'
        batch_path = folder / f'{TAG}_{stage}_batch.json'
        plan, batch = read(plan_path), read(batch_path)
        assert sha(plan_path) == protocol['plan_sha256'][plan_path.relative_to(ROOT).as_posix()] == batch['plan_sha256']
        assert not batch['skipped'] and len(plan['runs']) == len(batch['runs']) == count
        assert not batch['performance_certified'] and not batch['quality_certified']
        stages[stage] = {'planned': count, 'completed': count, 'unattempted': 0}
        artifacts.extend((plan_path, batch_path))
        for planned, row in zip(plan['runs'], batch['runs']):
            assert all(row[k] == v for k, v in planned.items())
            run = ROOT / 'runs/active' / row['name']
            request, result = read(run / 'requested.json'), read(run / 'result.json')
            cfg = request['expanded_config']
            assert matches_requested(cfg, row['config']) and request['config_sha256'] == digest(cfg)
            assert request['exe_sha256'] == IDENTITY and request['binary'] == 'active'
            assert sha(ROOT / request['manifest']) == request['manifest_sha256']
            assert result == row['result'] and result['status'] == 'completed' and result['exit_code'] == 0
            assert row['checks']['passed'] and row['checks']['metrics']['finite'] and not row['checks']['metrics']['pcg_failures']
            assert result['recorded_frames'] == cfg['steps'] == {'guards': 2, 'smoke': 3, 'activation': 23}[stage]
            assert not result['performance_certified'] and result['timing_is_diagnostic']
            assert cfg['backend'] == 'ipc' and cfg['pcg_rho_tol'] == 1e-4 and cfg['legacy_restrict'] == 'atomic'
            assert cfg['ipc_newton_tol'] == .01 and cfg['ipc_min_updates'] == 6
            assert all(not cfg[k] for k in ('mas_fused_dot', 'discrete_bvh_refit', 'refit', 'batch', 'reuse'))
            assert cfg['ipc_cumulative_tol'] == (.001 if cfg['ipc_termination'] == 'compensated' else .01)
            assert validate(run)['passed']
            frames = read(run / 'output/stats.json')['frames']
            assert len(frames) == cfg['steps']
            audit = audit_controller(frames, cfg)
            assert audit == row['checks']['controller'] == analysis_audits[row['name']]
            for key in controller_totals:
                controller_totals[key] += audit[key]
            pcg = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            assert all(p['spmv_fused_quadratic_requested'] == cfg['spmv_fused_quadratic'] for p in pcg)
            if cfg['spmv_fused_quadratic']:
                assert pcg and all(p['spmv_fused_quadratic_effective'] and not p['spmv_fused_quadratic_fallback_reason'] for p in pcg)
            else:
                assert all(not p['spmv_fused_quadratic_effective'] and p['spmv_fused_quadratic_fallback_reason'] == 'disabled_by_request' for p in pcg)
            if stage == 'smoke':
                expected = (129, 9) if row['scene_key'] == 'hang' else (75, 7)
                assert (row['checks']['metrics']['pcg'], len(pcg)) == expected
                smoke_directions += len(pcg)
                if cfg['spmv_fused_quadratic']:
                    enabled_directions += len(pcg)
            if stage == 'activation':
                saved_activation[row['variant']] = (frames, cfg)
            assert row['checks']['requested_sha256'] == sha(run / 'requested.json')
            assert row['checks']['stats_sha256'] == sha(run / 'output/stats.json')
            run_evidence.append({'name': row['name'], 'stage': stage, 'recorded_frames': len(frames),
                                 'result_sha256': sha(run / 'result.json'), 'requested_sha256': sha(run / 'requested.json'),
                                 'stats_sha256': sha(run / 'output/stats.json'), 'config_sha256': request['config_sha256'],
                                 'resolved_sha256': sha(run / 'resolved_config.json'), 'controller': audit})
            rows.append(row | {'stage': stage})
    assert (smoke_directions, enabled_directions) == (128, 64)
    assert controller_totals == {'observations': 265, 'active_observations': 5, 'audited_accepts': 3,
                                 'compensated_exits': 1, 'terminal_diagnostic_observations': 1}
    mutation_checks = rejected_observation_mutations(*saved_activation['activation'][:1],
                         saved_activation['terminal'][0], saved_activation['activation'][1], saved_activation['terminal'][1])
    assert analysis['attempted'] == analysis['completed'] == len(rows) == 20
    assert not analysis['failures'] and not analysis['unattempted'] and len(analysis['short_comparisons']) == 16
    studies = [s for row in rows for s in row['checks']['studies']]
    assert studies == analysis['studies'] and len(studies) == 2
    study_checks = [check_study(s) for s in studies]
    fixture_path = ROOT / f'runs/active/{TAG}_fixtures/result.json'
    fixture = read(fixture_path)
    assert fixture['exe_sha256'] == IDENTITY and fixture['passed']
    assert len(fixture['checks']) == 8 and all(c['passed'] for c in fixture['checks'])

    index_path, decisions_path = folder / 'EXPERIMENT_INDEX.json', folder / 'EXPERIMENT_DECISIONS.json'
    index, decisions = read(index_path), read(decisions_path)
    prior, old_decisions = index['entries'], dict(decisions)
    assert len(prior) == 460
    prior_digest, decisions_digest = digest(prior), digest(old_decisions)
    report_path = folder / 'IPC_SPMV_COMPENSATED_RESULTS_20261006.md'
    debug_path = folder / 'IPC_SPMV_COMPENSATED_DEBUG_20261006.md'
    evidence = [p.relative_to(ROOT).as_posix() for p in (report_path, debug_path, target, analysis_path, protocol_path)]
    additions = []

    def add(name, origin, status, reason, **fields):
        assert name not in decisions and name not in {r['id'] for r in prior + additions}
        decision = {'status': status, 'status_scope': 'Implementation and bounded local diagnostics only',
                    'reason': reason, 'performance_certified': False, 'quality_certified': False,
                    'default_promoted': False, 'decision_evidence': evidence[0]}
        decisions[name] = decision
        additions.append({'id': name, 'origin': origin, 'evidence': evidence, **decision, **fields})

    for row in rows:
        req = read(ROOT / 'runs/active' / row['name'] / 'requested.json')
        add(row['name'], 'active_run', 'accepted', 'Finite bounded local diagnostic completed; no long-scene certification',
            stage=row['stage'], scene=req['expanded_config']['scene'], frames=row['result']['recorded_frames'],
            completion='completed', solver_seconds=row['result']['solver_seconds'],
            exe_sha256=IDENTITY, config_sha256=req['config_sha256'])
    add(TAG + '_fixtures', 'active_fixture', 'accepted', 'Eight existing GPU regression fixtures passed', exe_sha256=IDENTITY)
    add(TAG + '_cpu_checks', 'active_tooling_check', 'accepted', 'Configuration/native/CPU operator and observation completeness checks passed')
    add(TAG + '_spmv_candidate', 'active_candidate_decision', 'pending',
        'Implemented and local operator/replay checked; full-PCG net benefit, growth and complete quality pending', implemented=True, promotion_pending=True)
    add(TAG + '_compensated_entry', 'active_candidate_decision', 'pending',
        'Named report formula and controller validated, real exit observed; independent quality and history benefit unproved', implemented=True, promotion_pending=True)
    add(TAG + '_delivery', 'active_delivery', 'accepted', 'Two components, debugging docs and bounded evidence delivered; no default promotion', exe_sha256=IDENTITY)
    assert len(additions) == 25
    index['entries'] = prior + additions
    index['updated_utc'] = datetime.now(timezone.utc).isoformat()
    assert digest(index['entries'][:460]) == prior_digest
    assert digest({key: decisions[key] for key in old_decisions}) == decisions_digest
    assert len({r['id'] for r in index['entries']}) == len(index['entries']) == 485
    artifacts += [protocol_path, analysis_path, cpu_path, fixture_path, manifest_path, archived, initial_path,
                  quality_path, report_path, debug_path, ROOT / 'STATUS.md', ROOT / 'tools/active/spmv_compensated.py',
                  ROOT / 'tools/active/verify_spmv_compensated_delivery.py', cpu_operator,
                  ROOT / 'sources/stiff_active/StiffGIPC/solver/IPC_COMPENSATED_TOI.md',
                  ROOT / 'sources/stiff_active/StiffGIPC/linear_system/utils/SPMV_FUSED_QUADRATIC.md']
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}
    verification = {
        'delivery_verified': True, 'scope': 'Implemented components and bounded local checks; no complete-scene certification',
        'exe_sha256': IDENTITY, 'initial_build_sha256': INITIAL_IDENTITY, 'initial_binary_used_for_GPU': False,
        'compiler_units': 37, 'changed_objects_after_tail_repair': 3,
        'source_object_include_checks': identity_checks, 'all_unit_objects_present_in_link_command': True,
        'stages': stages, 'attempted_runs': 20, 'completed_runs': 20, 'failed_runs': 0, 'unattempted_runs': 0,
        'gpu_fixtures_passed': 8, 'config_tests_passed': 15, 'config_subtests_passed': 88,
        'native_controller_checks': native_counts, 'cpu_operator_output': operator.stdout.strip(),
        'controller_totals': controller_totals, 'observation_negative_checks': mutation_checks,
        'smoke_PCG_directions': 128, 'smoke_enabled_spmv_directions': 64, 'studies': study_checks,
        'old_quality_protocol_unchanged': True, 'default_legacy_thresholds_changed': False,
        'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
        'no100_300_or_AutoDL_this_round': True, 'GPU_graph_growth_covered': False, 'GPU_lower_fixture_covered': False,
        'full_zero_RHS_PCG_reexecuted': False, 'gpu_lock_present': False, 'run_evidence': run_evidence,
        'prior_index_count': 460, 'new_index_count': 25, 'total_index_count': 485,
        'prior_index_entries_digest': prior_digest, 'prior_decisions_digest': decisions_digest,
        'historical_entries_and_decisions_preserved': True, 'evidence_sha256': hashes,
    }
    # All checks and artifact reads precede catalogue writes. Do not rebuild it.
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    assert digest(read(index_path)['entries'][:460]) == prior_digest
    assert digest({key: read(decisions_path)[key] for key in old_decisions}) == decisions_digest
    write(target, verification)
    print(json.dumps({key: verification[key] for key in ('delivery_verified', 'completed_runs', 'failed_runs',
        'gpu_fixtures_passed', 'observation_negative_checks', 'total_index_count',
        'historical_entries_and_decisions_preserved', 'performance_certified', 'quality_certified', 'default_promoted')}))


if __name__ == '__main__':
    main()
