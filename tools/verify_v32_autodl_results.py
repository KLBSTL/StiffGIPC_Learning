"""Verify archive bytes, extract inside a dedicated folder, and recheck reports."""
import hashlib
import csv
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'downloads/v32_20261002_records.tar.gz'
expected = (ROOT / 'downloads/v32_20261002_records.sha256').read_text().split()[0]
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
assert digest.hexdigest() == expected, 'Incomplete or changed download'
destination = ROOT / 'downloads/autodl_v32_20261002'
existing = any(destination.iterdir())
with tarfile.open(archive, 'r:gz') as tf:
    for member in tf.getmembers():
        assert member.isfile() or member.isdir(), member.name
        target = (destination / member.name).resolve()
        assert target.is_relative_to(destination.resolve()), member.name
    if existing:
        for member in tf.getmembers():
            if member.isfile():
                target = destination / member.name
                assert target.is_file(), target
                assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(tf.extractfile(member).read()).digest(), target
    else:
        tf.extractall(destination, filter='data')
sources = {}
for filename in ['IMPLEMENTATION_MANIFEST.json', 'manifests/robust_port_v32.json']:
    manifest = json.loads((ROOT / filename).read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    sources[filename] = {'files_verified': len(manifest['files']), 'source_digest': manifest['source_digest']}
for name, expected_sha in {
    'builds/local-base/Release/gipc.exe': '1c175da5a664bbc98657ddb6690a9902d821732c19e9b18f1f3ce0e868e05205',
    'builds/local-fused-v31/Release/gipc.exe': '7ff7e39ce10b3efa3f229ac6c2af707bb0f11da12321100df69301f297dcd9eb',
}.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_sha, name
report = json.loads((ROOT / 'reports/AUTODL_V32_SMALL_SCENES_20261002.json').read_text())
archived_report = json.loads((destination / 'reports/AUTODL_V32_SMALL_SCENES_20261002.json').read_text())
assert report == archived_report
manifest = json.loads((destination / 'manifests/robust_port_v32_autodl.json').read_text())
runner_sha = hashlib.sha256((ROOT / 'tools/run_robust_port.py').read_bytes()).hexdigest()
times = {}
for row in report['runs']:
    run = destination / row['run']
    request = json.loads((run / 'requested.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    assert request['source_digest'] == manifest['source_digest'] == report['source_digest']
    assert request['runner_sha256'] == runner_sha
    binary = 'stiff_base' if request['arm'] == 'base' else 'stiff_robust_port'
    assert request['exe_sha256'] == manifest['binaries'][f'builds/autodl-{binary}/gipc']['sha256']
    for k, v in {'dt': .01, 'steps': 100, 'suite': '0', 'tol': .01, 'pcg_tol': .0001,
                 'platform': 'autodl', 'profile': False, 'substeps': False,
                 'quality_only': False, 'trace_stride': 1}.items():
        assert request[k] == v, (run, k)
    stats_path = run / 'output/stats.json'
    assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == row['stats_sha256']
    frames = json.loads(stats_path.read_text())['frames']
    pcg = [n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    limits = sum(n.get('iteration_limit', False) for n in pcg)
    assert limits == row['work']['pcg_limit_hits']
    if request['execution'] == 'conditional_graph':
        assert all(n.get('execution') == 'conditional_graph' for n in pcg)
    if row['usable_timing']:
        assert result['status'] == 'completed' and result['finite'] and result['recorded_frames'] == 100
        assert len(frames) == 100 and limits == 0 and not row['work']['outer_limit_hits']
        assert row['quality']['all_frames_finite']
        table = list(csv.DictReader((run / 'trace/frames.csv').open()))
        assert [int(t['frame']) for t in table] == list(range(1, 101))
        assert math.isclose(sum(float(t['solver_ms']) for t in table) / 1000, result['solver_seconds'], rel_tol=1e-12)
        times[(row['scene'], row['arm'], row['repeat'])] = table
    else:
        assert result['status'] == 'failed' and result['exit_code'] == 4 and limits == 1
for scene in report['protocol']['scenes'].split(','):
    baseline = times[(scene, 'base', 1)]
    mask = [int(t['candidate_pairs']) + int(t['ground_candidates']) > 0 for t in baseline]
    for row in [r for r in report['runs'] if r['scene'] == scene and r['usable_timing']]:
        table = times[(scene, row['arm'], row['repeat'])]
        for phase in ['total', 'noncontact', 'contact']:
            values = [float(t['solver_ms']) for t, contact in zip(table, mask) if phase == 'total' or (phase == 'contact') == contact]
            assert math.isclose(sum(values) / 1000, row['seconds'][phase], abs_tol=1e-12)
    for summary in [s for s in report['summary'] if s['scene'] == scene]:
        selected = [r for r in report['runs'] if r['scene'] == scene and r['arm'] == summary['arm'] and r['usable_timing']]
        base = [r for r in report['runs'] if r['scene'] == scene and r['arm'] == 'base' and r['usable_timing']]
        assert len(selected) == summary['repeats'] == 3
        for phase in ['total', 'noncontact', 'contact']:
            median = statistics.median(r['seconds'][phase] for r in selected)
            assert math.isclose(median, summary['seconds'][phase], abs_tol=1e-12)
            speed = statistics.median(r['seconds'][phase] for r in base) / median if median else None
            recorded = summary['raw_speedup_vs_base'][phase]
            assert recorded is None if speed is None else math.isclose(recorded, speed, rel_tol=1e-12)
component = json.loads((destination / 'builds/toi_components.json').read_text())
assert component['passed'] and len(component['port_tests']) == 7 and all(t['passed'] for t in component['port_tests'])
assert len(report['runs']) == 46 and sum(r['usable_timing'] for r in report['runs']) == 42
assert len(report['skipped']) == 8 and report['component_fixtures_passed'] == 7
record = {'archive_sha256_verified': expected, 'archive_bytes': archive.stat().st_size,
          'extraction_root': str(destination),
          'formal_attempts': 46, 'full_100_frame_verified_runs': 42,
          'controlled_pcg_failures': 4, 'skipped_repeats': 8,
          'component_fixtures': 7, 'frozen_sources': sources,
          'old_local_base_and_v31_binaries_unchanged': True,
          'quality_checks_executed_on_autodl_full_trajectories': True,
          'local_checks_scope': 'archive hashes, frozen source preservation, all run metadata/stats, all frame times and median speed ratios',
          'full_raw_archive_retained_on_autodl': '/root/autodl-tmp/stiff_toi_cudagraph_20260929/v32_20261002_results.tar.gz',
          'full_raw_archive_sha256': (ROOT / 'downloads/v32_20261002_results.sha256').read_text().split()[0],
          'local_full_trajectory_archive_download_complete': False}
(ROOT / 'reports/AUTODL_V32_VERIFICATION_20261002.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
