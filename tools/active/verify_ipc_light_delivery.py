"""Close observation delivery without promoting optimization or changing history."""
import io
import json
import unittest
from datetime import datetime, timezone
from config import ROOT, read, sha, digest
from ipc_benchmark import write
import test_contracts

TAG = 'ipc_light_cost_20261005'


class CountedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtests = 0

    def addSubTest(self, test, subtest, err):
        self.subtests += 1
        super().addSubTest(test, subtest, err)


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    assert not target.exists(), 'Use a new delivery name; do not overwrite evidence'
    analysis_path = folder / f'{TAG}_analysis_v2.json'
    analysis = read(analysis_path)
    protocol_path = folder / f'{TAG}_protocol.json'
    protocol = read(protocol_path)
    plan_path = ROOT / f'configs/active/{TAG}.json'
    plan = read(plan_path)
    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    exe = ROOT / 'builds/active/Release/gipc.exe'
    identity = sha(exe)
    assert identity == manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256'] == analysis['exe_sha256']
    assert analysis['delivery_verified'] and len(analysis['run_records']) == 12 and len(analysis['profiles']) == 6
    assert analysis['protocol_sha256'] == sha(protocol_path) and analysis['plan_sha256'] == sha(plan_path)
    assert all(not analysis[k] for k in ('performance_certified', 'quality_certified', 'default_promoted', 'old_failures_reclassified'))
    assert analysis['candidate']['status'] == 'pending, not implemented'
    assert sha(folder/'ipc_revision_20261005_quality_protocol.json') == protocol['old_quality_protocol_sha256']
    assert len(manifest['compiler_inputs']) == 37 and len(manifest['changed_objects']) == 7
    source_checks = 0
    for unit in manifest['compiler_inputs']:
        for item in [unit['source'], unit['object'], *unit['project_include_inputs']]:
            assert sha(ROOT/item['path']) == item['sha256']
            source_checks += 1
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    records = {r['name']: r for r in analysis['run_records']}
    assert set(records) == {r['name'] for r in plan['runs']}
    for name, record in records.items():
        run = ROOT/'runs/active'/name
        result = read(run/'result.json')
        assert result['status'] == 'completed' and result['finite'] and result['recorded_frames'] == record['steps']
        assert sha(run/'requested.json') == record['requested_sha256']
        assert sha(run/'resolved_config.json') == record['resolved_sha256']
        resolved = read(run/'resolved_config.json')
        assert resolved['legacy_restrict'] == 'atomic' and resolved['pcg_rho_tol'] == 1e-4
        assert resolved['ipc_stopping']['termination'] == 'legacy'
        assert resolved['ipc_stopping']['cumulative_tol'] == .01 and resolved['ipc_stopping']['min_updates'] == 6
    native_events = []
    for profile in analysis['profiles']:
        run = ROOT/'runs/active'/profile['run']
        assert sha(run/'nsight.sqlite') == profile['sqlite_sha256']
        assert sha(run/'nsight.nsys-rep') == profile['nsight_report_sha256']
        assert sha(run/'activity_analysis.json') == profile['activity_analysis_sha256']
        frame = read(run/'output/stats.json')['frames'][profile['frame']-1]
        rows = frame['newton']
        directions = sum('pcg' in row for row in rows)
        creates = 6*len(rows)+2
        records_count = 6*directions+2*(len(rows)-directions)+2
        actual = profile['runtime_api']['cuda_events']
        assert actual['create']['count'] == creates and actual['record']['count'] == records_count
        assert records[profile['run']]['observation']['mode'] == 'nvtx_cpu_only'
        native_events.append({'run': profile['run'], 'frame': profile['frame'],
                              'create': creates, 'record': records_count, 'matches_native_only': True})
    for group in analysis['short_prefix_neutrality']:
        for comparison in group['comparisons']:
            assert comparison['same_pcg_accept_count_exit_and_contact_work']
            assert max(comparison['position_component_max_by_frame_m']) <= protocol['checks']['prefix_position_component_difference_m_max']
    fixture_path = ROOT/f'runs/active/{TAG}_fixtures/result.json'
    fixtures = read(fixture_path)
    assert fixtures['passed'] and all(c['passed'] for c in fixtures['checks']) and len(fixtures['checks']) == 8
    assert fixtures['exe_sha256'] == identity
    test_output = io.StringIO()
    tests = unittest.TextTestRunner(stream=test_output, resultclass=CountedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful() and tests.testsRun == 12 and tests.subtests == 58, test_output.getvalue()
    index_path = folder/'EXPERIMENT_INDEX.json'
    decisions_path = folder/'EXPERIMENT_DECISIONS.json'
    index = read(index_path)
    prior = index['entries']
    assert len(prior) == 381
    prior_hash = digest(prior)
    decisions = read(decisions_path)
    old_decisions = dict(decisions)
    old_decisions_hash = digest(old_decisions)
    evidence = ['reports/active/IPC_LIGHT_COST_RESULTS_20261005.md', target.relative_to(ROOT).as_posix(), analysis_path.relative_to(ROOT).as_posix()]
    additions = []
    names = [r['name'] for r in plan['runs']] + [TAG+'_fixtures']
    names += [TAG+'_observation_delivery', TAG+'_ee_hypothesis']
    assert all(name not in decisions and name not in {r['id'] for r in prior} for name in names)
    for name in names:
        hypothesis = name.endswith('_ee_hypothesis')
        aggregate = name.endswith('_observation_delivery')
        fixture = name.endswith('_fixtures')
        decision = {'status': 'pending' if hypothesis else 'accepted',
                    'status_scope': 'Unimplemented hypothesis; no promotion evidence' if hypothesis else 'Diagnostic/tooling completion only',
                    'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
                    'reason': 'No frozen-tree net benefit or whole-scene 5% evidence; not implemented.' if hypothesis else 'Finite observation, identity and regression checks passed; no long-trajectory or speed certification.',
                    'decision_evidence': evidence[0]}
        entry = {'id': name, 'origin': 'active_candidate_decision' if hypothesis else 'active_delivery' if aggregate else 'active_fixture' if fixture else 'active_run',
                 'evidence': evidence, **decision}
        if not hypothesis and not aggregate:
            run = ROOT/'runs/active'/name
            result = read(run/'result.json')
            req_path = run/'requested.json'
            req = read(req_path) if req_path.exists() else {}
            cfg = req.get('expanded_config', {})
            entry.update(scene=cfg.get('scene'), frames=result.get('recorded_frames'), completion=result.get('status'),
                         solver_seconds=result.get('solver_seconds'), exe_sha256=identity, config_sha256=req.get('config_sha256'),
                         diagnostics=cfg.get('diagnostics'))
            entry['evidence'] = [p.relative_to(ROOT).as_posix() for p in [req_path, run/'result.json'] if p.exists()] + evidence
        additions.append(entry)
        decisions[name] = decision
    index['entries'] = prior+additions
    index['updated_utc'] = datetime.now(timezone.utc).isoformat()
    assert len({row['id'] for row in index['entries']}) == len(index['entries'])
    assert digest(index['entries'][:len(prior)]) == prior_hash
    assert digest({key: decisions[key] for key in old_decisions}) == old_decisions_hash
    report_path = folder/'IPC_LIGHT_COST_RESULTS_20261005.md'
    assert report_path.exists()
    verification = {'delivery_verified': True, 'scope': 'Light observation and cost diagnosis only',
                    'completed_runs': 12, 'profiles': 6, 'gpu_fixtures_passed': 8, 'config_tests_passed': tests.testsRun,
                    'config_subtests_passed': tests.subtests, 'exe_sha256': identity, 'native_event_checks': native_events,
                    'source_object_dependency_checks': source_checks, 'old_quality_protocol_unchanged': True,
                    'production_thresholds_changed': False, 'default_promoted': False,
                    'performance_certified': False, 'quality_certified': False, 'gpu_lock_present': False,
                    'prior_index_count': len(prior), 'prior_index_entries_digest': prior_hash,
                    'prior_decisions_digest': old_decisions_hash, 'historical_entries_and_decisions_preserved': True,
                    'new_index_count': len(additions), 'total_index_count': len(index['entries']),
                    'candidate_status': 'pending, not implemented',
                    'evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [analysis_path, protocol_path, plan_path, manifest_path, fixture_path, report_path, ROOT/'STATUS.md', ROOT/'tools/active/verify_ipc_light_delivery.py']}}
    # All validation precedes generated index writes; historical rows are unchanged.
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    assert digest(read(index_path)['entries'][:len(prior)]) == prior_hash
    assert digest({key: read(decisions_path)[key] for key in old_decisions}) == old_decisions_hash
    write(target, verification)
    print(json.dumps({key: verification[key] for key in ['delivery_verified', 'completed_runs', 'profiles', 'gpu_fixtures_passed', 'config_tests_passed', 'config_subtests_passed', 'prior_index_count', 'new_index_count', 'total_index_count', 'historical_entries_and_decisions_preserved', 'candidate_status']}))


if __name__ == '__main__':
    main()
