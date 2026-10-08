"""Index closed local evidence and publish only experiment statistics/identities."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from run_stage1 import ROOT, REPORT, SESSION
for folder in ('bench', 'diagnostic', 'full_eval', 'local'):
    sys.path.insert(0, str(ROOT / 'tools' / folder))
from local_identity import record, verify

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    identity = json.loads((SESSION / 'PROBE_IDENTITY.json').read_text())
    verify(identity['sources'] + identity['build_evidence'] + identity['objects'] +
           [identity['exe']] + identity['dlls'] + [r['binary'] for r in identity['compilers']])
    ctest = SESSION / 'validation_full'
    receipt = json.loads((ctest / 'RESULT.json').read_text())
    assert receipt['status'] == 'passed' and receipt['commands'][0]['exit_code'] == 0
    log = (ctest / 'ctest.log').read_text()
    assert '100% tests passed, 0 tests failed out of 17' in log
    shutil.copyfile(ctest / 'ctest.log', REPORT / 'FULL_CTEST.log')
    write(REPORT / 'NATIVE_VALIDATION.json', {
        'status': 'passed', 'tests': 17, 'commands': [
            {k: row[k] for k in ('name', 'argv', 'timeout_seconds', 'status', 'exit_code', 'wall_seconds')}
            for row in receipt['commands']], 'binaries': receipt['binaries'],
        'local_raw_receipt': record(ctest / 'RESULT.json'),
        'scope': 'Correctness checks only; no diagnostic neutrality, physical quality or speed certification'})
    write(REPORT / 'PROBE_PROGRAM_IDENTITY.json', identity)
    registries = [record(p) for p in sorted(SESSION.glob('*.json'))]
    run_folders = []
    for stage in ('reference', 'baseline_review', 'observer_off', 'probe'):
        run_folders += [p for p in sorted((SESSION / stage).iterdir())
                        if p.is_dir() and (p / 'result.json').is_file()]
    run_folders += [SESSION / p for p in ('profile_fixed_f40_graph',
                                         'profile_sphere_f49_node', 'query_counters_f49')]
    names = ('requested.json', 'resolved_config.json', 'result.json', 'evidence.json',
        'config_validation.json', 'build_manifest.json', 'capture_validation.json',
        'trace/frames.csv', 'output/stats.json', 'cost.jsonl', 'bvh_query_probe.jsonl',
        'nsight.nsys-rep', 'nsight.sqlite', 'counter.csv', 'query.ncu-rep')
    runs = []
    for folder in run_folders:
        result = json.loads((folder / 'result.json').read_text())
        evidence = json.loads((folder / 'evidence.json').read_text())
        # Verify even the private trajectory/process inventory without publishing it.
        for item in evidence['files']:
            path = folder / item['path']
            assert path.stat().st_size == item['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        runs.append({'run': str(folder.relative_to(SESSION)),
            'status': result['status'], 'recorded_frames': result.get('recorded_frames'),
            'records': [record(folder / name) for name in names if (folder / name).is_file()]})
    write(REPORT / 'LOCAL_EVIDENCE_INDEX.json', {'schema': 'report56.closed_local_evidence.v1',
        'registries': registries, 'runs': runs, 'solver_runs': len(runs),
        'scope': 'Hashes/paths of closed local evidence only. No raw trajectories, binaries or process telemetry published.',
        'ncu_parser_failure_preserved': True, 'ncu_offline_analysis': record(REPORT / 'QUERY_COUNTER_ANALYSIS.json')})
    old = json.loads((ROOT / 'reports/report5_report6_audit_20261008/ARTIFACT_INDEX.json').read_text())
    verify(old['artifacts'])
    public = [p for p in sorted(REPORT.iterdir()) if p.is_file() and
              p.name not in ('BASELINE_REVIEW.json', 'ARTIFACT_INDEX.json', 'seal_execution.log')]
    write(REPORT / 'ARTIFACT_INDEX.json', {'schema': 'report56.round_artifacts.v1',
        'artifacts': [record(p) for p in public],
        'original_sealed_artifacts_unchanged': len(old['artifacts']),
        'performance_certified': False, 'physical_quality_certified': False,
        'diagnostic_neutrality_certified': False, 'target_2x_achieved': False})
    print(json.dumps({'status': 'sealed', 'solver_runs': len(runs),
        'public_artifacts': len(public), 'native_tests': 17,
        'probe_exe_sha256': identity['exe']['sha256']}))

if __name__ == '__main__': main()
