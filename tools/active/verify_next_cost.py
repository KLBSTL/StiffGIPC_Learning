"""Verify immutable cost evidence, then append this finite round to the index."""
import json
from datetime import datetime, timezone
import numpy as np
from config import ROOT, read, sha, digest, matches_requested
from ipc_benchmark import write
from validate_run import validate

TAG = 'ipc_next_cost_20261006'
IDENTITY = 'fb3b46246a710febecd8badb9e5b1fcf82125fc5deebefa0b9b2c19cd6017d1d'


def main():
    folder = ROOT / 'reports/active'
    target = folder / f'{TAG}_verification.json'
    assert not target.exists(), 'Do not append an experiment twice'
    analysis = read(folder / f'{TAG}_analysis.json')
    plan_path = ROOT / f'configs/active/{TAG}.json'
    plan = read(plan_path)
    batch = read(ROOT / plan['report'])
    assert analysis['plan_sha256'] == batch['plan_sha256'] == sha(plan_path)
    assert len(analysis['runs']) == len(batch['runs']) == len(plan['runs']) == 6
    assert len(analysis['profiles']) == 4
    assert not any(analysis[k] for k in ('performance_certified', 'quality_certified', 'default_promoted'))
    assert not (ROOT / 'runs/active/.gpu.lock').exists()
    quality_path = folder / 'ipc_revision_20261005_quality_protocol.json'
    assert sha(quality_path) == '1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f'
    manifest_path = ROOT / 'builds/active/manifest.json'
    manifest = read(manifest_path)
    assert sha(ROOT / 'builds/active/Release/gipc.exe') == IDENTITY
    assert manifest['binaries']['builds/active/Release/gipc.exe']['sha256'] == IDENTITY
    assert manifest['status'] == 'completed' and len(manifest['compiler_inputs']) == 37
    identity_checks = 0
    for record in manifest['files']:
        assert sha(ROOT / record['path']) == record['sha256'], record['path']
        identity_checks += 1
    for unit in manifest['compiler_inputs']:
        for record in (unit['source'], unit['object'], *unit['project_include_inputs']):
            assert sha(ROOT / record['path']) == record['sha256'], record['path']
            identity_checks += 1
    evidence = []
    for expected, actual, metrics in zip(plan['runs'], batch['runs'], analysis['runs']):
        assert expected == {k: actual[k] for k in expected}
        run = ROOT / 'runs/active' / expected['name']
        req, result = read(run / 'requested.json'), read(run / 'result.json')
        assert actual['result'] == result and result['status'] == 'completed'
        assert req['exe_sha256'] == metrics['exe_sha256'] == IDENTITY
        assert req['manifest_sha256'] == sha(run / 'build_manifest.json') == sha(manifest_path)
        c = req['expanded_config']
        assert matches_requested(c, expected['config']) and req['config_sha256'] == digest(c)
        assert c['ipc_termination'] == 'legacy' and c['ipc_cumulative_tol'] == .01 and c['pcg_rho_tol'] == 1e-4
        assert c['discrete_bvh_refit'] and not c['mas_fused_dot'] and not c['spmv_fused_quadratic']
        assert c['timeout_seconds'] == 120 and not c['ipc_residual_shadow'] and not c['ipc_terminal_audit_frame']
        assert result['recorded_frames'] == c['steps'] == metrics['frames']
        assert validate(run)['passed'] and metrics['finite'] and metrics['pcg_failures'] == 0
        for name, value in metrics['evidence'].items():
            assert sha(ROOT / name) == value
        if c['diagnostics']:
            rows = [json.loads(s) for s in (run / 'cost.jsonl').read_text().splitlines()]
            assert rows and {r['frame'] for r in rows} == {c['steps']}
            assert all(not r['operator_probe_enabled'] and not r['gpu_events_enabled'] and r['gpu_interval_ms'] is None for r in rows)
        if c['profile'] != 'none':
            profile = next(r for r in analysis['profiles'] if r['name'] == run.name)
            assert sha(run / 'nsight.nsys-rep') == profile['capture_sha256']
            assert sha(run / 'nsight.sqlite') == profile['sqlite_sha256']
            export = read(run / 'cost_export_identity.json')
            assert export['plan_sha256'] == sha(plan_path) and export['capture_sha256'] == profile['capture_sha256']
            assert export['exporter_sha256'] == sha(ROOT / 'tools/active/export_cost_plan.py')
            assert export['analyzer_sha256'] == sha(ROOT / 'tools/active/analyze_ipc_light_cost.py')
        evidence.append({'name': run.name, 'requested_sha256': sha(run / 'requested.json'), 'result_sha256': sha(run / 'result.json')})
    assert analysis['analyzer_sha256'] == sha(ROOT / 'tools/active/analyze_next_cost.py')
    # Recompute pre-window differences, independent of the offline summary.
    off = ROOT / 'runs/active' / f'{TAG}_fixed39_off/trace'
    cpu = ROOT / 'runs/active' / f'{TAG}_fixed39_cpu/trace'
    abd = read(off / 'metadata.json')['abd_point_num']
    differences = {}
    for kind in ('state', 'velocity'):
        values = []
        for frame in range(40):
            x = np.fromfile(off / f'{kind}_{frame:04d}.bin', dtype='<f8').reshape(-1, 3)[abd:]
            y = np.fromfile(cpu / f'{kind}_{frame:04d}.bin', dtype='<f8').reshape(-1, 3)[abd:]
            assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            values.append(float(np.linalg.norm(x-y, axis=1).max()))
        differences[kind] = {'before_observation_window_max': max(values[:39]),
                             'before_max_frame': int(np.argmax(values[:39])), 'selected_frame_max': values[39]}
        assert differences[kind]['before_observation_window_max'] > differences[kind]['selected_frame_max']
    report_path = folder / 'IPC_NEXT_COST_RESULTS_20261006.md'
    artifact_paths = [plan_path, folder / f'{TAG}_analysis.json', ROOT / plan['report'], report_path,
                      folder / 'IPC_NEXT_COST_PLAN_20261006.md', manifest_path, quality_path, ROOT / 'STATUS.md',
                      ROOT / 'tools/active/export_cost_plan.py', ROOT / 'tools/active/analyze_next_cost.py', ROOT / 'tools/active/verify_next_cost.py']
    index_path, decisions_path = folder / 'EXPERIMENT_INDEX.json', folder / 'EXPERIMENT_DECISIONS.json'
    index, decisions = read(index_path), read(decisions_path)
    prior, old_decisions = index['entries'], dict(decisions)
    assert len(prior) == 561
    prior_digest, decisions_digest = digest(prior), digest(old_decisions)
    additions = []

    def add(name, origin, status, reason, **fields):
        assert name not in decisions and name not in {r['id'] for r in prior + additions}
        decision = {'status': status, 'status_scope': 'Finite local cost diagnostic; no certification',
                    'reason': reason, 'performance_certified': False, 'quality_certified': False,
                    'default_promoted': False, 'decision_evidence': report_path.relative_to(ROOT).as_posix()}
        decisions[name] = decision
        additions.append({'id': name, 'origin': origin, 'evidence': [report_path.relative_to(ROOT).as_posix(), target.relative_to(ROOT).as_posix()], **decision, **fields})

    for row in analysis['runs']:
        add(row['name'], 'active_run', 'accepted', 'Complete cost diagnostic; finite and no PCG hard failure',
            frames=row['frames'], completion='completed', exe_sha256=IDENTITY)
    add(TAG + '_initialization_candidate', 'active_candidate_decision', 'rejected',
        'Three diagnostic initialization inactivity envelopes below local 5% gate, including capture/scheduling; not a global exclusion', implemented=False)
    add(TAG + '_subtree_query_candidate', 'active_candidate_decision', 'pending',
        'Query cost supports further bounded examination; prune ratio, same-state equivalence and net saving unmeasured', implemented=False)
    add(TAG + '_delivery', 'active_delivery', 'accepted', 'Cost attribution delivered; solver unchanged and no new speed claim', exe_sha256=IDENTITY)
    index['entries'] = prior + additions
    index['updated_utc'] = datetime.now(timezone.utc).isoformat()
    assert digest(index['entries'][:561]) == prior_digest
    assert digest({k: decisions[k] for k in old_decisions}) == decisions_digest
    verification = {'delivery_verified': True, 'exe_sha256': IDENTITY, 'solver_unchanged': True,
                    'source_object_identity_checks': identity_checks, 'completed': 6, 'failed': 0, 'profiles': 4,
                    'pre_window_actual_state_differences': differences, 'old_quality_protocol_unchanged': True,
                    'new_gpu_fixtures_run': False, 'reused_prior_fixture_evidence': 'reports/active/ipc_component_tuning_20261006_verification.json',
                    'prior_index_count': 561, 'added_index_count': len(additions), 'total_index_count': len(index['entries']),
                    'prior_index_digest': prior_digest, 'prior_decisions_digest': decisions_digest,
                    'historical_entries_and_decisions_preserved': True, 'performance_certified': False,
                    'quality_certified': False, 'default_promoted': False, 'runs': evidence,
                    'evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in artifact_paths}}
    # All validation precedes the index mutation; a prior round is never reclassified.
    decisions_path.write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    assert digest(read(index_path)['entries'][:561]) == prior_digest
    assert digest({k: read(decisions_path)[k] for k in old_decisions}) == decisions_digest
    write(target, verification)
    print(json.dumps({k: verification[k] for k in ('delivery_verified', 'completed', 'failed', 'profiles', 'solver_unchanged', 'total_index_count', 'historical_entries_and_decisions_preserved')}))


if __name__ == '__main__':
    main()
