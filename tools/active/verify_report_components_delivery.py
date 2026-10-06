"""CPU-only delivery audit; preserve resource failures and historical decisions.

The coverage ledger derives unattempted rows from frozen plans, not from missing
speed measurements. No GPU, certification, or automatic candidate promotion.
"""
import io
import json
import unittest
from datetime import datetime, timezone

from config import ROOT, read, sha, digest, matches_requested
from ipc_benchmark import metrics, write
from report_components import TAG, protocol_path
from validate_run import validate
from verify_ipc_light_delivery import CountedResult
import test_contracts


def execution_coverage(protocol):
    stages = {}
    actual = []
    unattempted = []
    batches = {}
    plans = {}
    for stage in ('smoke', 'guards', 'performance', 'quality'):
        suffix = '_v2' if stage == 'guards' else ''
        plan_path = ROOT / f'configs/active/{TAG}_{stage}{suffix}.json'
        batch_path = ROOT / f'reports/active/{TAG}_{stage}_batch.json'
        assert sha(plan_path) == protocol['plan_sha256'][plan_path.relative_to(ROOT).as_posix()]
        plan = read(plan_path)
        batch = read(batch_path)
        assert batch['plan_sha256'] == sha(plan_path)
        planned = {r['name']: r for r in plan['runs']}
        attempted = {r['name']: r for r in batch['runs']}
        assert len(planned) == len(plan['runs']) and len(attempted) == len(batch['runs'])
        assert set(attempted) <= set(planned)
        for name, row in attempted.items():
            assert all(row[key] == planned[name][key] for key in planned[name])
        batches[stage] = batch
        plans[stage] = plan
        missing = [row for row in plan['runs'] if row['name'] not in attempted]
        stored_skips = {r['name'] for r in batch['skipped']}
        assert stored_skips <= {r['name'] for r in missing}
        if stage != 'performance':
            assert stored_skips == {r['name'] for r in missing}
        for row in missing:
            if stage == 'performance':
                cause = next(r for r in batch['runs'] if r['arm'] == row['arm'] and r['hard_failures'])
                reason = 'Earlier same-scene arm resource failure; later repetition not executed'
            else:
                prior = batches['smoke']['runs'] + batches.get('performance', {}).get('runs', [])
                cause = next(r for r in prior if r['scene_key'] == row['scene_key']
                             and r['result']['status'] == 'memory_budget')
                reason = 'Earlier resource failure in scene; no retry'
            assert cause['result']['status'] == 'memory_budget'
            unattempted.append(row | {'stage': stage, 'completion': 'not_executed',
                                     'reason': reason, 'blocking_run': cause['name']})
        for row in batch['runs']:
            actual.append(row | {'stage': stage})
        stages[stage] = {
            'planned': len(planned), 'attempted': len(attempted),
            'completed': sum(r['result']['status'] == 'completed' for r in batch['runs']),
            'resource_failed': sum(r['result']['status'] == 'memory_budget' for r in batch['runs']),
            'unattempted': len(missing), 'stored_skip_rows': len(batch['skipped']),
            'plan_sha256': sha(plan_path), 'original_batch_sha256': sha(batch_path),
        }
    assert [(v['planned'], v['attempted'], v['completed'], v['resource_failed'], v['unattempted'])
            for v in stages.values()] == [(12, 12, 10, 2, 0), (6, 5, 4, 1, 1),
                                         (30, 10, 0, 10, 20), (12, 0, 0, 0, 12)]
    return {'stages': stages, 'actual_runs': actual, 'unattempted': unattempted,
            'scope': 'Frozen plans and immutable original batches; missing performance skips reconstructed from preceding failures',
            'original_batches_unchanged': True, 'performance_certified': False,
            'quality_certified': False, 'default_promoted': False}, plans


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    coverage_path = folder / f'{TAG}_execution_coverage.json'
    assert not target.exists() and not coverage_path.exists(), 'Do not overwrite delivery evidence'
    protocol_file = protocol_path()
    protocol = read(protocol_file)
    analysis_file = folder / f'{TAG}_analysis_v2.json'
    analysis = read(analysis_file)
    assert analysis['prior_analysis_sha256'] == sha(folder / f'{TAG}_analysis.json')
    assert all(not analysis[k] for k in ('performance_certified', 'quality_certified', 'default_promoted'))
    assert len(analysis['runs']) == 14 and len(analysis['failures']) == 13
    assert not analysis['position_velocity']
    assert all(not values for values in analysis['summary'].values())
    assert all(not item['paired_ratios'] and item['median'] is None
               for variants in analysis['paired_speed'].values()
               for ratios in variants.values() for item in ratios.values())
    assert sha(folder / 'ipc_revision_20261005_quality_protocol.json') == protocol['old_quality_protocol_sha256']
    assert sha(folder / f'{TAG}_protocol.json') == protocol['prior_protocol_sha256']
    coverage, plans = execution_coverage(protocol)

    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    exe = ROOT / 'builds/active/Release/gipc.exe'
    identity = sha(exe)
    assert identity == '9ae1c09394e9f36aed90499dae578064f28921bd96467f6c5a45206cea2a8df0'
    assert manifest['status'] == 'completed'
    assert identity == manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    assert len(manifest['compiler_inputs']) == 37 and len(manifest['changed_objects']) == 18
    link_commands = '\n'.join(c for row in manifest['link_evidence'] for c in row['commands']).upper()
    identity_checks = 0
    for item in manifest['files']:
        assert sha(ROOT / item['path']) == item['sha256']
        identity_checks += 1
    for unit in manifest['compiler_inputs']:
        for item in [unit['source'], unit['object'], *unit['project_include_inputs']]:
            assert sha(ROOT / item['path']) == item['sha256']
            identity_checks += 1
        assert (ROOT / unit['object']['path']).name.upper() in link_commands
    assert not (ROOT / 'runs/active/.gpu.lock').exists()

    run_evidence = []
    for row in coverage['actual_runs']:
        run = ROOT / 'runs/active' / row['name']
        result = read(run / 'result.json')
        request = read(run / 'requested.json')
        assert result == row['result'] and matches_requested(request['expanded_config'], row['config'])
        assert request['config_sha256'] == digest(request['expanded_config'])
        config = request['expanded_config']
        assert config['backend'] == 'ipc' and config['ipc_termination'] == 'legacy'
        assert config['ipc_cumulative_tol'] == .01 and config['ipc_min_updates'] == 6
        assert config['pcg_rho_tol'] == 1e-4 and config['legacy_restrict'] == 'atomic'
        assert result['status'] in ('completed', 'memory_budget')
        if result['status'] == 'completed':
            assert not row['hard_failures'] and result['recorded_frames'] == config['steps']
            assert validate(run)['passed']
        else:
            assert row['hard_failures'] == ['run_memory_budget']
            assert not result['performance_certified']
        count = result['recorded_frames']
        assert count > 0
        m = metrics(run, frame_limit=count)
        assert m['finite'] and not m['pcg_failures']
        frames = read(run / 'output/stats.json')['frames'][:count]
        assert len(frames) == count and all('phase_ms' in f and 'newton_exit' in f for f in frames)
        if row['binary'] == 'active':
            assert request['exe_sha256'] == identity
            resolved = read(run / 'resolved_config.json')
            assert resolved['report_components']['mas_fused_dot_requested'] == config['mas_fused_dot']
            assert resolved['report_components']['discrete_bvh_refit'] == config['discrete_bvh_refit']
            pcg = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            assert all(p['mas_fused_dot_requested'] == config['mas_fused_dot'] for p in pcg)
            if config['mas_fused_dot'] and row['scene_key'] != 'mixed':
                assert pcg and all(p['mas_fused_dot_effective'] for p in pcg)
        run_evidence.append({'name': row['name'], 'status': result['status'],
                             'recorded_frames': count, 'finite_exported_prefix': True,
                             'pcg_failures_on_exported_prefix': 0,
                             'requested_sha256': sha(run / 'requested.json'),
                             'stats_sha256': sha(run / 'output/stats.json'),
                             'result_sha256': sha(run / 'result.json')})

    for scene, group in analysis['incomplete_prefix_diagnosis'].items():
        assert group['frames'] == (40 if scene == 'hang' else 22) and len(group['rows']) == 5
        assert not group['performance_certified'] and not group['quality_certified']
        assert len({row['pcg'] for row in group['rows']}) == 1
        for row in group['rows']:
            run = ROOT / 'runs/active' / row['run']
            assert row['requested_run_status'] == 'memory_budget'
            assert sha(run / 'requested.json') == row['requested_sha256']
            assert sha(run / 'output/stats.json') == row['stats_sha256']
            assert row['discrete_bvh_counters_available'] == (row['variant'] != 'stiff')
            if row['variant'] in ('discrete', 'both'):
                for counters in row['discrete_bvh_counters'].values():
                    assert counters['production_refits'] == 0
                    assert counters['production_rebuilds'] == counters['swept_rebuilds'] == counters['construct_calls']

    fixture_path = ROOT / f'runs/active/{TAG}_fixtures/result.json'
    fixture = read(fixture_path)
    assert fixture['exe_sha256'] == identity and fixture['passed']
    assert len(fixture['checks']) == 8 and all(c['passed'] for c in fixture['checks'])
    guard_path = folder / f'{TAG}_guard_checks.json'
    guards = read(guard_path)
    assert guards['passed'] and guards['exe_sha256'] == identity and not guards['full_guard_windows_completed']
    for record in guards['MAS_dot_audits']:
        study = ROOT / 'runs/active' / record['run'] / 'fixed/f2_n1_mas_dot_study.json'
        assert sha(study) == record['study_sha256'] and read(study)['passed']
    contact = next(r for r in guards['BVH_same_state_audits'] if r['steps'] == 59)
    assert contact['completed_prefix_frames'] == 23 and not contact['full_requested_window_passed']
    assert sum(c['validation_passed'] for c in contact['counts'].values()) == 130
    assert sum(c['diagnostic_pairs_compared'] for c in contact['counts'].values()) == 68
    assert sum(c['diagnostic_overflow_retries'] for c in contact['counts'].values()) == 28
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, resultclass=CountedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful() and tests.testsRun == 13 and tests.subtests == 68, stream.getvalue()

    index_path = folder / 'EXPERIMENT_INDEX.json'
    decisions_path = folder / 'EXPERIMENT_DECISIONS.json'
    index = read(index_path)
    prior = index['entries']
    decisions = read(decisions_path)
    old_decisions = dict(decisions)
    prior_digest = digest(prior)
    decisions_digest = digest(old_decisions)
    assert len(prior) == 396
    report_path = folder / 'IPC_REPORT_COMPONENTS_RESULTS_20261005.md'
    debug_path = folder / 'IPC_COMPONENT_DEBUG_GUIDE_20261005.md'
    evidence = [p.relative_to(ROOT).as_posix() for p in
                (report_path, debug_path, target, coverage_path, analysis_file, guard_path)]
    additions = []

    def add(name, origin, status, reason, **fields):
        assert name not in decisions and name not in {r['id'] for r in prior + additions}
        decision = {'status': status, 'status_scope': 'Component/local diagnostic scope only; complete gates pending',
                    'reason': reason, 'performance_certified': False, 'quality_certified': False,
                    'default_promoted': False, 'decision_evidence': evidence[0]}
        decisions[name] = decision
        additions.append({'id': name, 'origin': origin, 'evidence': evidence, **decision, **fields})

    for row in coverage['actual_runs']:
        run = ROOT / 'runs/active' / row['name']
        request = read(run / 'requested.json')
        result = row['result']
        add(row['name'], 'active_run', 'accepted' if result['status'] == 'completed' else 'failed',
            'Finite local diagnostic completed; long performance/quality pending' if result['status'] == 'completed'
            else 'Existing GPU memory reserve stopped requested window; failure retained',
            stage=row['stage'], scene=request['expanded_config']['scene'], frames=result['recorded_frames'],
            completion=result['status'], solver_seconds=result.get('solver_seconds'),
            exe_sha256=request['exe_sha256'], config_sha256=request['config_sha256'])
    for row in coverage['unattempted']:
        assert not (ROOT / 'runs/active' / row['name'] / 'result.json').exists()
        add(row['name'], 'active_unattempted', 'pending', row['reason'],
            stage=row['stage'], scene=row['config']['scene'], completion='not_executed',
            requested_frames=row['config']['steps'], blocking_run=row['blocking_run'])
    add(TAG + '_fixtures', 'active_fixture', 'accepted', 'Eight existing regression fixtures passed; no whole-scene certification',
        exe_sha256=identity)
    add(TAG + '_mas_fused_dot_candidate', 'active_candidate_decision', 'pending',
        'Implemented and local collect/zero-RHS checked; stable net benefit, full quality and mixed coverage pending',
        implemented=True, promotion_pending=True)
    add(TAG + '_discrete_bvh_candidate', 'active_candidate_decision', 'pending',
        'Implemented and same-state query checked on saved prefix; no production refit hits observed; full gates pending',
        implemented=True, promotion_pending=True)
    add(TAG + '_component_delivery', 'active_delivery', 'accepted',
        'Code/docs and local evidence delivered; full testing blocked by unchanged resource gate',
        exe_sha256=identity)
    assert len(additions) == 64
    index['entries'] = prior + additions
    index['updated_utc'] = datetime.now(timezone.utc).isoformat()
    assert digest(index['entries'][:len(prior)]) == prior_digest
    assert digest({key: decisions[key] for key in old_decisions}) == decisions_digest
    assert len({r['id'] for r in index['entries']}) == len(index['entries'])
    artifacts = [analysis_file, protocol_file, manifest_path, fixture_path, guard_path,
                 report_path, debug_path, ROOT / 'STATUS.md', ROOT / 'tools/active/report_components.py',
                 ROOT / 'tools/active/verify_report_components_delivery.py']
    artifacts += [ROOT / f'configs/active/{TAG}_{stage}{"_v2" if stage == "guards" else ""}.json'
                  for stage in plans]
    artifact_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}
    verification = {
        'delivery_verified': True, 'scope': 'Implemented components and local guards; complete windows not certified',
        'exe_sha256': identity, 'compiler_units': 37, 'changed_objects': 18,
        'source_object_include_checks': identity_checks, 'all_unit_objects_present_in_link_command': True,
        'attempted_runs': 27, 'completed_runs': 14, 'resource_failed_runs': 13, 'unattempted_runs': 33,
        'gpu_fixtures_passed': 8, 'config_tests_passed': tests.testsRun, 'config_subtests_passed': tests.subtests,
        'old_quality_protocol_unchanged': True, 'full_guard_windows_completed': False,
        'production_thresholds_changed': False, 'default_promoted': False,
        'performance_certified': False, 'quality_certified': False, 'gpu_lock_present': False,
        'run_evidence': run_evidence, 'prior_index_count': len(prior), 'new_index_count': len(additions),
        'total_index_count': len(index['entries']), 'prior_index_entries_digest': prior_digest,
        'prior_decisions_digest': decisions_digest, 'historical_entries_and_decisions_preserved': True,
        'evidence_sha256': artifact_hashes,
    }
    # All checks precede evidence/index writes. Never rebuild the old catalogue.
    write(coverage_path, coverage)
    verification['evidence_sha256'][coverage_path.relative_to(ROOT).as_posix()] = sha(coverage_path)
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    assert digest(read(index_path)['entries'][:len(prior)]) == prior_digest
    assert digest({key: read(decisions_path)[key] for key in old_decisions}) == decisions_digest
    write(target, verification)
    print(json.dumps({key: verification[key] for key in (
        'delivery_verified', 'attempted_runs', 'completed_runs', 'resource_failed_runs', 'unattempted_runs',
        'config_tests_passed', 'config_subtests_passed', 'gpu_fixtures_passed', 'total_index_count',
        'historical_entries_and_decisions_preserved', 'performance_certified', 'quality_certified')}))


if __name__ == '__main__':
    main()
