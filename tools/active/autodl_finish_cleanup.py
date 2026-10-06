"""Audit the completed explicit cleanup and remove three obsolete parent scripts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / 'cleanup_20261005'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert sys.platform == 'linux' and ROOT.name == 'autodl_factor_retest_20261004_659cef89'
    output = CLEAN / 'final_cleanup_audit.json'
    assert not output.exists()
    plan = read(CLEAN / 'plan_v2.json')
    tree = read(CLEAN / 'tree_apply.json')
    top = read(CLEAN / 'archives_pruned.json')
    deferred = read(CLEAN / 'deferred_pruned.json')
    assert tree['completed'] and top['status'] == deferred['status'] == 'completed'
    assert sha(CLEAN / 'plan_v2.json') == tree['plan_sha256']
    assert len(top['deleted']) == 12 and len(deferred['deleted']) == 47
    checked_reports = checked_important = 0
    for row in plan['files']:
        path = Path(row['path'])
        if row['action'] == 'delete' or row['reason'] == 'archive_external_report_gate':
            assert not path.exists() and not path.is_symlink(), str(path)
            continue
        status = path.lstat()
        assert (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns) == (
            row['device'], row['inode'], row['bytes'], row['mtime_ns']), str(path)
        if row['preserve_report'] and row['kind'] == 'file':
            assert sha(path) == row['sha256'], str(path)
            checked_reports += 1
        if row['reason'] == 'important_exact_prefix':
            checked_important += 1
    parent = Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929')
    assert parent.resolve() == parent and not parent.is_symlink()
    targets = [parent / name for name in ('cleanup_autodl_history.py',
                                        'cleanup_history_20261003.py', 'dedup_factors_v42.py')]
    records = []
    for path in targets:
        assert path.parent == parent and path.is_file() and not path.is_symlink()
        records.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    with (CLEAN / 'obsolete_parent_scripts.json').open('x') as stream:
        json.dump(records, stream, indent=2)
    for path, record in zip(targets, records):
        assert sha(path) == record['sha256']
        path.unlink()
    disks = {name: dict(zip(('total', 'used', 'free'), shutil.disk_usage(name)))
             for name in ('/', '/root/autodl-tmp')}
    result = {'completed': True, 'reports_verified_in_place': checked_reports,
              'important_files_identity_verified': checked_important,
              'tree_deleted_files': len(tree['deleted_files']),
              'tree_deleted_logical_bytes': tree['deleted_bytes'],
              'archives_deleted': 59, 'obsolete_parent_scripts_deleted': records,
              'disks_immediate': disks,
              'disk_before': (CLEAN / 'disk_before.txt').read_text()}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    subprocess.run(['df', '-B1', '--output=source,size,used,avail,target', '/', '/root/autodl-tmp'], check=True)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
