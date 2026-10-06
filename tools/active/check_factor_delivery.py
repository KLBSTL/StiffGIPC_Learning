"""Verify the finished local evidence and the unopened AutoDL archive."""
import hashlib
import json
import tarfile
from pathlib import PurePosixPath
from config import ROOT, read, sha, digest


def main():
    archive = ROOT / 'packages/autodl_factor_diagnostic_20261004.tar.gz'
    target = ROOT / 'reports/active/FACTOR_DELIVERY_CHECK.json'
    if target.exists():
        raise FileExistsError(target)
    checks = {}
    for name in ('FACTOR_DEFAULT_IDENTITY', 'FACTOR_CPU_HISTORICAL', 'FACTOR_CPU_WINDOWS',
                 'FACTOR_CPU_DIFFICULT', 'FACTOR_PAIR_RESIDUALS', 'FACTOR_FIXED_CONFIG_CHECKS',
                 'FACTOR_COST_CONFIG_CHECKS'):
        data = read(ROOT / 'reports/active' / (name + '.json'))
        assert data['passed'] is True, name
        checks[name] = {'passed': True, 'sha256': sha(ROOT / 'reports/active' / (name + '.json'))}
    matrix = read(ROOT / 'reports/active/FACTOR_WINDOW_BATCH.json')
    quality = read(ROOT / 'reports/active/FACTOR_WINDOW_QUALITY_TIMING.json')
    assert len(matrix['runs']) == quality['runs_checked'] == 27
    assert all(r['result']['status'] == 'completed' and r['result']['finite'] for r in matrix['runs'])
    assert all(s['pcg_limit_or_breakdown_hits'] == 0 for scene in quality['scenes'] for s in scene['summary'])
    ccd = read(ROOT / 'reports/active/FACTOR_WINDOW_CCD.json')
    assert ccd['all_six_audits_passed'] and ccd['collision_flag_totals_cover_all_six_runs']
    assert ccd['actual_validator_paths_checked'] == 857 and ccd['actual_validator_collision_flags'] == 0
    assert not (ROOT / 'runs/active/.gpu.lock').exists()
    build = read(ROOT / 'builds/active/manifest.json')
    assert not build['changed_objects']
    for record in build['files']:
        assert sha(ROOT / record['path']) == record['sha256'], record['path']
    exe = ROOT / 'builds/active/Release/gipc.exe'
    assert sha(exe) == build['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names))
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                   and '..' not in PurePosixPath(m.name).parts for m in members)
        bundle = json.load(package.extractfile('autodl_bundle.json'))
        assert bundle['remote_tested'] is False and bundle['triangular_reference'] is True
        assert bundle['candidate_config']['mas_factor_action'] == 'factor_inverse'
        assert digest(bundle['files']) == bundle['source_digest']
        assert digest(bundle['candidate_config']) == bundle['candidate_config_sha256']
        for record in bundle['files']:
            content = package.extractfile(record['path']).read()
            assert len(content) == record['bytes']
            assert hashlib.sha256(content).hexdigest() == record['sha256'], record['path']
            assert sha(ROOT / record['path']) == record['sha256'], record['path']
        stages = {}
        for stage, expected in bundle['plan_digests'].items():
            plan = json.load(package.extractfile(f'configs/active/autodl_{stage}.json'))
            assert digest(plan) == expected
            stages[stage] = {'runs': len(plan['runs']),
                             'frame_counts': sorted({r['config']['steps'] for r in plan['runs']})}
        expected_names = {r['path'] for r in bundle['files']} | {'autodl_bundle.json'} \
            | {f'configs/active/autodl_{s}.json' for s in stages}
        assert set(names) == expected_names
    result = {'passed': True, 'scope': 'Delivery integrity and completed local checks only',
              'candidate_promoted': False, 'trajectory_quality_passed': False,
              'performance_certified': False, 'remote_tested': False,
              'binary_sha256': sha(exe), 'fixed_evidence': checks, 'window_runs': 27,
              'ccd_paths_checked': 857, 'ccd_flags': 0, 'archive': str(archive),
              'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
              'archive_source_files_verified': len(bundle['files']), 'archive_members_verified': len(names),
              'plans': stages, 'verifier_sha256': sha(__file__)}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'fixed_evidence'}))


if __name__ == '__main__':
    main()
