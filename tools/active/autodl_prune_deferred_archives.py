"""Report-gated deletion of exact deferred archives from a reviewed cleanup plan.
Standalone entry point; keep autodl_prune_reported_archives.py beside it for
shared safe-path/hash/process helpers. The original 12-archive tool is unchanged.
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

import autodl_prune_reported_archives as common
from autodl_prune_reported_archives import no_links, read, reject_active_users, require, safe_name, sha, stream_sha

SCHEMA = 'autodl.deferred_archive_backup.v1'
IDENTITY = ('kind', 'device', 'inode', 'mode', 'bytes', 'mtime_ns', 'ctime_ns', 'link_target')


def planned_targets(plan_path, expected_sha):
    require(re.fullmatch('[0-9a-fA-F]{64}', expected_sha), 'Expected reviewed plan SHA is required')
    no_links(plan_path)
    descriptor = os.open(plan_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream: data = stream.read()
    require(hashlib.sha256(data).hexdigest() == expected_sha.lower(), 'Reviewed plan SHA mismatch')
    plan = json.loads(data)
    require(plan.get('schema') == 1 and re.fullmatch('history_[0-9a-f]{32}', plan['plan_id']), 'Invalid cleanup plan identity')
    trees = {tree['path']: tree for tree in plan['policy']['trees']}
    candidates = [row for row in plan['files'] if row.get('reason') == 'archive_external_report_gate']
    require(all(row.get('action') == 'retain' and row.get('preserve_report') is False
                and row.get('kind') == 'file' and row.get('link_target') is None for row in candidates),
            'Deferred archive rows must be regular, retained, non-report files')
    excluded = [row['path'] for row in candidates if row['path'].lower().endswith('.log.gz')]
    rows = [row for row in candidates if row['path'] not in excluded]
    require(rows and len({row['path'] for row in rows}) == len(rows), 'Empty or duplicate deferred archive selection')
    for row in rows:
        path, root = PurePosixPath(row['path']), PurePosixPath(row['tree'])
        historical = (PurePosixPath('/root/stiff_toi_cudagraph_20260929'), PurePosixPath('/root/paper_builds_20260929'))
        require(str(root) == row['tree'] and '..' not in root.parts and
                (any(root == allowed or allowed in root.parents for allowed in historical)
                 or PurePosixPath('/root/autodl-tmp') in root.parents), 'Tree is outside approved historical locations')
        require(path.is_absolute() and str(path) == row['path'] and '..' not in path.parts
                and not any(c in row['path'] for c in ('\\', '\0', ':')) and len(path.parts) > 3,
                'Noncanonical/broad planned path')
        require(row['tree'] in trees and root in path.parents and str(path.relative_to(root)) == row['relative_path']
                and row['kind'] == 'file' and row['link_target'] is None, 'Archive is not a regular file inside its exact planned tree')
        for prefix in trees[row['tree']].get('keep_prefixes', []):
            relative = PurePosixPath(safe_name(prefix))
            require(str(relative) == prefix and not (root / relative == path or root / relative in path.parents),
                    'Archive is inside an important keep prefix')
        protected = [PurePosixPath(p) for p in plan['policy'].get('protected_paths', [])]
        require(not any(p == path or p in path.parents for p in protected)
                and not any(part.lower().startswith('autodl_factor') for part in path.parts), 'Protected path in archive selection')
        require(all(field in row for field in IDENTITY), 'Incomplete planned archive identity')
        require(path.name.lower().endswith(('.tar.gz', '.tgz', '.tar', '.zip')), 'Unsupported deferred archive kind')
    return plan, rows, excluded


def validate_manifest(manifest, plan, rows, excluded, plan_sha):
    source = manifest.get('source_plan', {})
    require(manifest.get('schema') == 'autodl.archive_reports.v2' and manifest.get('status') == 'completed'
            and not manifest.get('errors') and manifest.get('deletion_performed') is False, 'Incomplete v2 archive report manifest')
    require(source.get('sha256') == plan_sha.lower() and source.get('plan_id') == plan['plan_id']
            and source.get('policy_sha256') == plan['policy_sha256'] and source.get('tool_sha256') == plan['tool_sha256'],
            'Extraction manifest is bound to another reviewed plan')
    require(manifest.get('excluded_compressed_logs') == excluded
            and [row['archive'] for row in manifest['archives']] == [row['path'] for row in rows], 'Extraction selection/order differs from plan')
    labels = {}
    for index, (archive, planned) in enumerate(zip(manifest['archives'], rows)):
        basename = PurePosixPath(planned['path']).name
        label = f'{index:03d}_{hashlib.sha256(planned["path"].encode()).hexdigest()[:12]}_{basename}'
        identity = {key: planned[key] for key in IDENTITY}
        require(archive['basename'] == basename and archive['archive_label'] == label and archive['status'] == 'completed'
                and archive.get('archive_stat_unchanged') is True and archive['plan_identity'] == identity,
                'Archive label/completion/planned identity mismatch')
        kind = 'tar.gz' if basename.lower().endswith(('.tar.gz', '.tgz')) else ('zip' if basename.lower().endswith('.zip') else 'tar')
        require(archive['archive_kind'] == kind, 'Extracted archive kind differs from planned filename')
        expected = {key: identity[source] for key, source in [('size', 'bytes'), ('mtime_ns', 'mtime_ns'),
                    ('device', 'device'), ('inode', 'inode'), ('mode', 'mode'), ('ctime_ns', 'ctime_ns')]}
        require(archive['archive_stat'] == expected and archive['compressed_bytes'] == identity['bytes']
                and re.fullmatch('[0-9a-f]{64}', archive['archive_sha256']), 'Invalid archive stat/hash record')
        labels[label] = archive
    expected, counts = {}, Counter()
    for row in manifest['selected_files']:
        name = safe_name(row['output_path']); label = row['archive_label']
        require(label in labels and PurePosixPath(name).parts[0] == label
                and row['archive_basename'] == labels[label]['basename'], 'Report is outside its exact archive label')
        name = 'archive_reports/' + name
        require(name not in expected and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid/duplicate selected report')
        expected[name] = (row['bytes'], row['sha256']); counts[label] += 1
    require(all(counts[label] == row['selected_count'] for label, row in labels.items())
            and sum(row[0] for row in expected.values()) == manifest['selected_bytes'], 'Selected report count/size mismatch')
    return expected


def verify_backup(plan_path, plan_sha, backup, expected_sha, receipt):
    plan, rows, excluded = planned_targets(plan_path, plan_sha)
    require(re.fullmatch('[0-9a-fA-F]{64}', expected_sha) and sha(backup) == expected_sha.lower(), 'Backup SHA mismatch')
    no_links(receipt); require(not receipt.exists(), 'Receipt must be new')
    found, data, seen = {}, None, set()
    with backup.open('rb') as raw, gzip.GzipFile(fileobj=raw, mode='rb') as compressed:
        with tarfile.open(fileobj=compressed, mode='r|', ignore_zeros=True) as archive:
            for member in archive:
                name = safe_name(member.name)
                require((name == 'archive_reports' or name.startswith('archive_reports/')) and name not in seen
                        and (member.isdir() or member.isfile()), 'Unsafe/duplicate/link/special backup member')
                seen.add(name)
                if member.isdir(): continue
                with archive.extractfile(member) as stream:
                    if name == 'archive_reports/manifest.json':
                        require(0 <= member.size <= 64 * 1024**2, 'Unreasonable manifest size')
                        data = stream.read(); require(len(data) == member.size, 'Truncated manifest')
                    else: found[name] = (member.size, stream_sha(stream))
        while compressed.read(1024 * 1024): pass
    require(data is not None and sha(backup) == expected_sha.lower(), 'Missing manifest or backup changed while reading')
    require(found == validate_manifest(json.loads(data), plan, rows, excluded, plan_sha), 'Backup reports omitted/changed or extra regular files present')
    record = {'schema': SCHEMA, 'complete': True, 'plan_sha256': plan_sha.lower(), 'plan_id': plan['plan_id'],
              'archive_count': len(rows), 'files_verified': len(found), 'backup_sha256': expected_sha.lower(),
              'backup_bytes': backup.stat().st_size, 'manifest_sha256': hashlib.sha256(data).hexdigest(),
              'verifier_sha256': sha(Path(__file__)), 'helper_sha256': sha(Path(common.__file__))}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream: json.dump(record, stream, indent=2)
    return record


def identity(path):
    no_links(path); status = path.lstat()
    require(stat.S_ISREG(status.st_mode), 'Deferred archive is not a regular file')
    return {'kind': 'file', 'device': status.st_dev, 'inode': status.st_ino, 'mode': status.st_mode,
            'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns, 'ctime_ns': status.st_ctime_ns, 'link_target': None}


def preflight(plan_path, plan_sha, manifest_path, receipt_path, backup):
    plan, rows, excluded = planned_targets(plan_path, plan_sha)
    manifest, receipt = read(manifest_path), read(receipt_path)
    reports = validate_manifest(manifest, plan, rows, excluded, plan_sha)
    require(receipt.get('schema') == SCHEMA and receipt.get('complete') is True and receipt.get('archive_count') == len(rows)
            and receipt.get('plan_sha256') == plan_sha.lower() and receipt.get('plan_id') == plan['plan_id']
            and receipt.get('files_verified') == len(reports) and receipt['manifest_sha256'] == sha(manifest_path)
            and receipt['backup_sha256'] == sha(backup) and receipt['backup_bytes'] == backup.stat().st_size
            and receipt['verifier_sha256'] == sha(Path(__file__)) and receipt['helper_sha256'] == sha(Path(common.__file__)),
            'Receipt/actual backup/manifest/plan/tool identity mismatch')
    for name, (size, digest) in reports.items():
        path = manifest_path.parent.joinpath(*PurePosixPath(name).parts[1:])
        require(sha(path) == digest and path.stat().st_size == size, 'Retained remote report changed: ' + str(path))
    for archive in manifest['archives']:
        path = Path(archive['archive'])
        require(identity(path) == archive['plan_identity'] and sha(path) == archive['archive_sha256']
                and identity(path) == archive['plan_identity'], 'Planned archive content/identity changed: ' + str(path))
    reject_active_users(manifest['archives'])
    require(all(identity(Path(row['archive'])) == row['plan_identity'] for row in manifest['archives']), 'Archive changed after full preflight')
    return manifest, receipt


def prune(plan_path, plan_sha, manifest_path, receipt_path, backup, output):
    require(sys.platform == 'linux', 'Prune is Linux-only'); no_links(output); require(not output.exists(), 'Output must be new')
    manifest, receipt = preflight(plan_path, plan_sha, manifest_path, receipt_path, backup)
    allowed = {row['archive'] for row in manifest['archives']}
    record = {'status': 'in_progress', 'all_archives_preflight_passed': True, 'plan_sha256': plan_sha.lower(),
              'receipt': receipt, 'deleted': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        def save():
            stream.seek(0); json.dump(record, stream, indent=2); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        save()
        try:
            for row in manifest['archives']:
                path = Path(row['archive'])
                require(str(path) in allowed and identity(path) == row['plan_identity'], 'Target changed before unlink')
                path.unlink()
                record['deleted'].append({'archive': str(path), 'bytes': row['compressed_bytes'], 'sha256': row['archive_sha256']}); save()
            record['status'] = 'completed'
        except BaseException as error:
            record.update(status='failed', error=str(error)); raise
        finally: save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    for name in ('verify-backup', 'prune'):
        command = commands.add_parser(name); command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--expected-plan-sha', '--expected-plan-sha256', dest='plan_sha', required=True)
        command.add_argument('--receipt', type=Path, required=True)
        if name == 'verify-backup':
            command.add_argument('--archive', type=Path, required=True); command.add_argument('--expected-sha', required=True)
        else:
            for option in ('manifest', 'backup', 'output'): command.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    result = verify_backup(args.plan, args.plan_sha, args.archive, args.expected_sha, args.receipt) if args.command == 'verify-backup' else \
        prune(args.plan, args.plan_sha, args.manifest, args.receipt, args.backup, args.output)
    print(json.dumps(result if args.command == 'verify-backup' else {'status': result['status'], 'deleted_count': len(result['deleted'])}))


if __name__ == '__main__': main()
