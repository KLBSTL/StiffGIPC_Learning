"""Final read-only assertions, then append this experiment to the old index.

No GPU run, threshold adjustment, or historical decision rewrite is permitted.
"""
import json
import io
import unittest
from datetime import datetime, timezone

from config import ROOT, digest, matches_requested, read, sha
from component_tuning import TAG
from ipc_benchmark import write
from validate_run import validate
from verify_ipc_light_delivery import CountedResult
import test_contracts


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    assert not target.exists(), 'Immutable evidence; do not append twice'
    protocol_path = folder / f'{TAG}_protocol.json'
    selection_path = folder / f'{TAG}_selection.json'
    analysis_path = folder / f'{TAG}_analysis.json'
    protocol, selection, analysis = read(protocol_path), read(selection_path), read(analysis_path)
    assert sha(protocol_path) == analysis['protocol_sha256'] and sha(selection_path) == analysis['selection_sha256']
    detailed_path = folder / f'{TAG}_evidence.json'
    detailed = read(detailed_path)
    assert detailed['analysis_sha256'] == sha(analysis_path)
    assert len(detailed['periodic_checks']) == 40 and all(r['passed'] for r in detailed['periodic_checks'])
    assert not detailed['baseline_material_protocol_available'] and not detailed['stiff_actual_velocity_export_available']
    missing_velocity = [d for d in analysis['distances'] if d['position_velocity']['velocity'].get('available') is False]
    assert len(missing_velocity) == 12 and all('_stiff_' in d['run'] for d in missing_velocity)
    test_output = io.StringIO()
    tests = unittest.TextTestRunner(stream=test_output, resultclass=CountedResult).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    assert tests.wasSuccessful() and (tests.testsRun, tests.subtests) == (15, 88), test_output.getvalue()
    assert sha(folder / 'ipc_revision_20261005_quality_protocol.json') == protocol['old_quality_sha256']
    assert protocol['old_quality_sha256'] == '1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f'
    assert sha(ROOT / 'manifests/perf_v34.json') == protocol['base_manifest_sha256']
    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    identity = sha(ROOT / 'builds/active/Release/gipc.exe')
    assert sha(manifest_path) == protocol['build_manifest_sha256']
    assert manifest['status'] == 'completed' and len(manifest['compiler_inputs']) == 37
    assert manifest['binaries']['builds/active/Release/gipc.exe']['sha256'] == identity == protocol['identity']
    link = '\n'.join(c for row in manifest['link_evidence'] for c in row['commands']).upper()
    checks = 0
    for row in manifest['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
        checks += 1
    for unit in manifest['compiler_inputs']:
        for row in (unit['source'], unit['object'], *unit['project_include_inputs']):
            assert sha(ROOT / row['path']) == row['sha256'], row['path']
            checks += 1
        assert (ROOT / unit['object']['path']).name.upper() in link
    assert not (ROOT / 'runs/active/.gpu.lock').exists()
    fixture_path = ROOT / f'runs/active/{TAG}_fixtures/result.json'
    fixture = read(fixture_path)
    assert fixture['passed'] and fixture['exe_sha256'] == identity
    assert len(fixture['checks']) == 8 and all(c['passed'] for c in fixture['checks'])
    artifacts = [protocol_path, selection_path, analysis_path, detailed_path, manifest_path, fixture_path]
    attempts, skipped, evidence = [], [], []
    for stage in ('guards', 'screen', 'full'):
        batch_path = folder / f'{TAG}_{stage}_batch.json'
        if not batch_path.exists():
            assert stage == 'full' and not selection['eligible']
            continue
        plan_path = ROOT / f'configs/active/{TAG}_{stage}.json'
        plan, batch = read(plan_path), read(batch_path)
        expected_sha = selection['full_plan_sha256'] if stage == 'full' else protocol['plan_sha256'][plan_path.relative_to(ROOT).as_posix()]
        assert sha(plan_path) == batch['plan_sha256'] == expected_sha
        planned = {r['name']: r for r in plan['runs']}
        assert len(planned) == len(plan['runs'])
        assert {r['name'] for r in batch['runs']} | {r['name'] for r in batch['skipped']} == set(planned)
        assert len(batch['runs']) + len(batch['skipped']) == len(planned)
        for row in batch['runs']:
            assert all(row[k] == value for k, value in planned[row['name']].items())
            run = ROOT / 'runs/active' / row['name']
            request, result = read(run / 'requested.json'), read(run / 'result.json')
            assert result == row['result'] and matches_requested(request['expanded_config'], row['config'])
            assert request['config_sha256'] == digest(request['expanded_config'])
            assert request['manifest_sha256'] == sha(run / 'build_manifest.json')
            if row['binary'] == 'active':
                assert request['exe_sha256'] == identity
            if row['checks']['passed']:
                assert result['status'] == 'completed' and result['recorded_frames'] == request['expanded_config']['steps']
                assert row['checks']['metrics']['finite'] and not row['checks']['metrics']['pcg_failures']
                if row['binary'] == 'active':
                    assert validate(run)['passed']
                enabled = request['expanded_config']['discrete_bvh_refit']
                for c in row['checks']['bvh_counters'].values():
                    if enabled:
                        assert c['production_refits'] and c['ordinary_cache_restores']
                        assert c['swept_cache_capacity_bytes_peak'] > 0 and c['swept_rebuilds'] == 0
                        assert c['construct_calls'] == c['production_rebuilds'] + c['production_refits']
                    else:
                        assert c['production_refits'] == 0 and c['swept_cache_capacity_bytes_peak'] == 0
            else:
                assert row['checks']['hard_failures'], 'Failure may not be silently reclassified'
            attempts.append(row | {'stage': stage})
            evidence.append({'run': row['name'], 'requested_sha256': sha(run / 'requested.json'),
                             'result_sha256': sha(run / 'result.json'), 'build_manifest_sha256': sha(run / 'build_manifest.json'),
                             'status': result['status'], 'checks_passed': row['checks']['passed']})
        skipped.extend({'stage': stage, **r} for r in batch['skipped'])
        artifacts.extend((plan_path, batch_path))
    assert all(not obj[k] for obj in (protocol, selection, analysis)
               for k in ('performance_certified', 'quality_certified', 'default_promoted'))
    report_path = folder / 'IPC_COMPONENT_TUNING_RESULTS_20261006.md'
    artifacts.extend((report_path, folder / 'IPC_COMPONENT_TUNING_PLAN_20261006.md', ROOT / 'STATUS.md',
                      ROOT / 'tools/active/component_tuning.py', ROOT / 'tools/active/verify_component_tuning.py',
                      ROOT / 'tools/active/component_tuning_evidence.py',
                      ROOT / 'sources/stiff_active/StiffGIPC/collision/DISCRETE_BVH_REFIT.md'))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts}
    index_path, decisions_path = folder / 'EXPERIMENT_INDEX.json', folder / 'EXPERIMENT_DECISIONS.json'
    index, decisions = read(index_path), read(decisions_path)
    prior, prior_decisions = index['entries'], dict(decisions)
    assert len(prior) == 485
    prior_digest, decisions_digest = digest(prior), digest(prior_decisions)
    additions = []

    def add(name, origin, status, reason, **fields):
        assert name not in decisions and name not in {r['id'] for r in prior + additions}
        decision = {'status': status, 'status_scope': 'Bounded local component diagnostic only', 'reason': reason,
                    'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
                    'decision_evidence': report_path.relative_to(ROOT).as_posix()}
        decisions[name] = decision
        additions.append({'id': name, 'origin': origin, 'evidence': [report_path.relative_to(ROOT).as_posix(),
                          target.relative_to(ROOT).as_posix()], **decision, **fields})

    for row in attempts:
        add(row['name'], 'active_run', 'accepted' if row['checks']['passed'] else 'failed',
            'Complete local diagnostic; whole-scene certification pending' if row['checks']['passed'] else str(row['checks']['hard_failures']),
            stage=row['stage'], scene=row['config']['scene'], completion=row['result']['status'],
            frames=row['result']['recorded_frames'], solver_seconds=row['result'].get('solver_seconds'),
            material_100f_passed=row['checks'].get('material_100f_passed'),
            exe_sha256=read(ROOT / 'runs/active' / row['name'] / 'requested.json')['exe_sha256'])
    for row in skipped:
        add(row['name'], 'active_unattempted', 'pending', row['reason'], stage=row['stage'], completion='not_executed')
    add(TAG + '_fixtures', 'active_fixture', 'accepted', 'Eight GPU regression checks passed', exe_sha256=identity)
    add(TAG + '_bvh_revision', 'active_candidate_decision', 'pending',
        'Split cache verified; 100f +8.9% hang/+0.5% fixed vs old combination; retained opt-in, full quality and controlled load pending', implemented=True,
        screen_selection_eligible=selection['eligible'], promotion_pending=True)
    add(TAG + '_mas_screen', 'active_candidate_decision', 'rejected',
        'Local promotion screen: only 1.01x geometric mean vs old combination; no expansion or tuning')
    add(TAG + '_spmv_screen', 'active_candidate_decision', 'rejected',
        'Local promotion screen: both cloth scenes slower, geometric mean .978x; default stays disabled')
    add(TAG + '_fused_screen', 'active_candidate_decision', 'rejected',
        'Local promotion screen: all three components 1.030x, weaker than BVH alone, below 1.05x full-test threshold')
    add(TAG + '_delivery', 'active_delivery', 'accepted', 'Revision and bounded evidence delivered; no default promotion', exe_sha256=identity)
    index['entries'] = prior + additions
    index['updated_utc'] = datetime.now(timezone.utc).isoformat()
    assert digest(index['entries'][:485]) == prior_digest
    assert digest({key: decisions[key] for key in prior_decisions}) == decisions_digest
    assert len({r['id'] for r in index['entries']}) == len(index['entries'])
    verification = {'delivery_verified': True, 'exe_sha256': identity, 'compiler_units': 37,
                    'changed_objects': len(manifest['changed_objects']), 'source_object_include_checks': checks,
                    'gpu_fixtures_passed': 8, 'attempted': len(attempts),
                    'config_tests_passed': tests.testsRun, 'config_subtests_passed': tests.subtests,
                    'periodic_run_tree_checks_passed': 40, 'missing_stiff_velocity_comparisons_explicit': 12,
                    'completed': sum(r['result']['status'] == 'completed' for r in attempts),
                    'failed': sum(not r['checks']['passed'] for r in attempts), 'unattempted': len(skipped),
                    'prior_index_count': 485, 'added_index_count': len(additions), 'total_index_count': len(index['entries']),
                    'prior_index_entries_digest': prior_digest, 'prior_decisions_digest': decisions_digest,
                    'historical_entries_and_decisions_preserved': True, 'old_quality_protocol_unchanged': True,
                    'performance_certified': False, 'quality_certified': False, 'default_promoted': False,
                    'scope': 'Local code/runtime/evidence verification; shared load and complete quality limits retained',
                    'run_evidence': evidence, 'evidence_sha256': hashes}
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    assert digest(read(index_path)['entries'][:485]) == prior_digest
    assert digest({key: read(decisions_path)[key] for key in prior_decisions}) == decisions_digest
    write(target, verification)
    print(json.dumps({k: verification[k] for k in ('delivery_verified', 'attempted', 'completed', 'failed',
          'unattempted', 'total_index_count', 'historical_entries_and_decisions_preserved', 'default_promoted')}))


if __name__ == '__main__':
    main()
