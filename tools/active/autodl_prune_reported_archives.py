"""Verify a report-only backup locally, then prune exactly 12 approved Linux archives.
No SSH, extraction, recursive deletion, or wildcard deletion. All targets pass
one complete preflight before the first unlink; every completed unlink is saved.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile

TOP = '/root/autodl-tmp'
ALLOWED = {TOP + '/' + name for name in (
    'autodl-full-four-arm-20260927-results.tar.gz', 'autodl_quick_final_20260926_r3_results.tar.gz',
    'autodl_resolution5_5f_20260926_results.tar.gz', 'autodl_resolution5_overlay_20260926.tar.gz',
    'autodl_stiff_base_graph_20260926.tar.gz', 'autodl_stiff_three_arm_20260927.tar.gz',
    'paper_yasps_fig14_20260928_bundle.tar.gz')}
ALLOWED |= {TOP + '/stiff_toi_cudagraph_20260929/' + name for name in (
    'v32_20261002_results.tar.gz', 'autodl_perf_v43_20261003.tar.gz',
    'autodl_perf_v41_20261003.tar.gz', 'mas_replay_v42a.tar.gz', 'mas_replay_v42.tar.gz')}
SCHEMA = 'autodl.reported_archive_backup.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(), 'Symlink/junction rejected: ' + str(part))
    return path


def safe_name(name):
    require(isinstance(name, str) and name and not any(c in name for c in ('\\', ':', '\0')), 'Unsafe tar/report name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.parts, 'Unsafe tar/report path')
    require(all(not p.endswith((' ', '.')) and not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', p.split('.')[0])
                for p in path.parts), 'Nonportable report path')
    return path.as_posix()


def stream_sha(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def sha(path):
    path = no_links(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'Expected regular file: ' + str(path))
        return stream_sha(stream)


def read(path):
    no_links(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_manifest(manifest):
    require(manifest.get('schema') == 'autodl.archive_reports.v1' and manifest.get('status') == 'completed'
            and not manifest.get('errors') and manifest.get('deletion_performed') is False, 'Incomplete report-extraction manifest')
    archives = manifest['archives']
    require(len(archives) == 12 and {r['archive'] for r in archives} == ALLOWED, 'Archive set must match the exact 12 approved paths')
    for row in archives:
        require(row['basename'] == PurePosixPath(row['archive']).name and row['status'] == 'completed'
                and row.get('archive_stat_unchanged') is True, 'Incomplete archive record')
        require(re.fullmatch('[0-9a-f]{64}', row['archive_sha256']) is not None, 'Invalid source archive SHA')
    expected, counts = {}, Counter()
    for row in manifest['selected_files']:
        relative = safe_name(row['output_path'])
        require(PurePosixPath(relative).parts[0] == row['archive_basename']
                and row['archive_basename'] in {r['basename'] for r in archives}, 'Report points outside its source archive')
        name = 'archive_reports/' + relative
        require(name not in expected and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Duplicate/invalid selected report')
        expected[name] = (row['bytes'], row['sha256'])
        counts[row['archive_basename']] += 1
    require(all(counts[r['basename']] == r['selected_count'] for r in archives)
            and sum(v[0] for v in expected.values()) == manifest['selected_bytes'], 'Report count/size mismatch')
    return expected


def verify_backup(archive, expected_sha, receipt):
    require(re.fullmatch('[0-9a-fA-F]{64}', expected_sha) is not None, 'Expected backup SHA256 is required')
    no_links(receipt)
    require(not receipt.exists() and sha(archive) == expected_sha.lower(), 'Existing receipt or incorrect backup SHA')
    found, manifest_bytes, seen = {}, None, set()
    with archive.open('rb') as raw, gzip.GzipFile(fileobj=raw, mode='rb') as compressed:
        with tarfile.open(fileobj=compressed, mode='r|', ignore_zeros=True) as container:
            for member in container:
                name = safe_name(member.name)
                require(name == 'archive_reports' or name.startswith('archive_reports/'), 'Tar member outside archive_reports')
                require(name not in seen and (member.isdir() or member.isfile()), 'Duplicate/link/special tar member')
                seen.add(name)
                if member.isdir():
                    continue
                with container.extractfile(member) as stream:
                    if name == 'archive_reports/manifest.json':
                        require(0 <= member.size <= 64 * 1024**2, 'Unreasonable manifest size')
                        manifest_bytes = stream.read()
                        require(len(manifest_bytes) == member.size, 'Truncated manifest')
                    else:
                        found[name] = (member.size, stream_sha(stream))
        while compressed.read(1024 * 1024):
            pass  # Validate gzip trailer/CRC, including data after tar padding.
    require(sha(archive) == expected_sha.lower(), 'Backup changed during verification')
    require(manifest_bytes is not None, 'Missing archive_reports/manifest.json')
    manifest = json.loads(manifest_bytes)
    require(found == validate_manifest(manifest), 'Backup omitted/changed reports or contains extra regular files')
    record = {'schema': SCHEMA, 'complete': True, 'archive_count': 12, 'files_verified': len(found),
              'backup_sha256': expected_sha.lower(), 'backup_bytes': archive.stat().st_size,
              'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    return record


def source_identity(path):
    no_links(path)
    status = path.lstat()
    require(stat.S_ISREG(status.st_mode), 'Source archive is not a regular file')
    return {'size': status.st_size, 'mtime_ns': status.st_mtime_ns, 'device': status.st_dev, 'inode': status.st_ino}


def reject_active_users(targets):
    require(Path('/proc').is_dir(), 'Cannot inspect active Linux processes')
    ancestors, pid = set(), os.getpid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        match = re.search(r'^PPid:\s*(\d+)', Path(f'/proc/{pid}/status').read_text(), re.M)
        pid = int(match[1]) if match else 0
    identities = {(r['archive_stat']['device'], r['archive_stat']['inode']) for r in targets}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) in ancestors:
            continue
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            require(not any(r['archive'] in command for r in targets), 'Active process names target archive: PID ' + process.name)
            for descriptor in (process / 'fd').iterdir():
                try:
                    status = descriptor.stat()
                    require((status.st_dev, status.st_ino) not in identities, 'Archive open by PID ' + process.name)
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass


def preflight(manifest_path, receipt_path, backup):
    require(manifest_path.parent.name == 'archive_reports', 'Manifest must reside in archive_reports')
    manifest, receipt = read(manifest_path), read(receipt_path)
    reports = validate_manifest(manifest)
    require(receipt.get('schema') == SCHEMA and receipt.get('complete') is True and receipt.get('archive_count') == 12
            and receipt.get('files_verified') == len(reports) and receipt['manifest_sha256'] == sha(manifest_path)
            and receipt['backup_sha256'] == sha(backup) and receipt['backup_bytes'] == backup.stat().st_size,
            'Receipt/actual backup/manifest identity mismatch')
    for relative, (size, digest) in reports.items():
        path = manifest_path.parent.joinpath(*PurePosixPath(relative).parts[1:])
        require(sha(path) == digest and path.stat().st_size == size, 'Retained remote report differs: ' + str(path))
    for row in manifest['archives']:
        path = Path(row['archive'])
        require(source_identity(path) == row['archive_stat'] and sha(path) == row['archive_sha256']
                and source_identity(path) == row['archive_stat'], 'Source archive changed: ' + str(path))
    reject_active_users(manifest['archives'])
    require(all(source_identity(Path(r['archive'])) == r['archive_stat'] for r in manifest['archives']), 'Source changed after full preflight')
    return manifest, receipt


def prune(manifest_path, receipt_path, backup, output):
    require(sys.platform == 'linux', 'Prune is Linux-only')
    no_links(output)
    require(not output.exists(), 'Output must be new')
    manifest, receipt = preflight(manifest_path, receipt_path, backup)
    record = {'status': 'in_progress', 'all_12_preflight_passed': True, 'receipt': receipt, 'deleted': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        def save():
            stream.seek(0); json.dump(record, stream, indent=2); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        save()
        try:
            for row in manifest['archives']:
                path = Path(row['archive'])
                require(str(path) in ALLOWED and source_identity(path) == row['archive_stat'], 'Target changed before unlink')
                path.unlink()
                record['deleted'].append({'archive': str(path), 'bytes': row['archive_stat']['size'], 'sha256': row['archive_sha256']})
                save()
            record['status'] = 'completed'
        except BaseException as error:
            record.update(status='failed', error=str(error))
            raise
        finally:
            save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify-backup')
    verify.add_argument('--archive', type=Path, required=True); verify.add_argument('--expected-sha', required=True)
    verify.add_argument('--receipt', type=Path, required=True)
    delete = commands.add_parser('prune')
    for option in ('manifest', 'receipt', 'backup', 'output'):
        delete.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = verify_backup(args.archive, args.expected_sha, args.receipt) if args.command == 'verify-backup' else \
        prune(args.manifest, args.receipt, args.backup, args.output)
    print(json.dumps(result if args.command == 'verify-backup' else {'status': result['status'], 'deleted_count': len(result['deleted'])}))


if __name__ == '__main__':
    main()
