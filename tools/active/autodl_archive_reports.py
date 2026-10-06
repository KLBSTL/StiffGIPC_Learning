"""Read explicit or reviewed-plan archives into a fresh metadata-only report root.

No deletion, network, shell commands, tar extraction API, or silent truncation.
Only a completed manifest is evidence that every archive passed this policy.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import time
import zipfile

CODE_DIRS = {'sources', 'source', 'src', 'references', 'third_party', 'third-party',
             'vendor', 'vendors', 'external', 'externals', 'dependencies', 'deps', '.git',
             'build', 'builds', 'cmake', 'tools', 'scripts', 'include', 'includes',
             'toolsrc', 'stiffgipc', 'meshprocess', 'lib', 'libs', 'node_modules',
             '.venv', 'venv', '__pycache__', '.ssh', '.aws'}
RAW_DIRS = {'assets', 'dataset', 'datasets', 'raw', 'arrays', 'meshes', 'mesh',
            'states', 'snapshots', 'checkpoints', 'cache', 'caches', 'trace', 'traces',
            'state_window', 'fixed', 'frames'}
REPORT_DIRS = {'reports', 'report', 'results', 'metrics', 'logs', 'analysis',
               'analyses', 'configs', 'config', 'manifests'}
EXTENSIONS = {'.md', '.json', '.csv', '.log', '.txt', '.pdf'}
METADATA_NAMES = {'requested.json', 'resolved_config.json', 'result.json', 'build_manifest.json',
                  'process.json', 'config_validation.json', 'scene.json', 'stats.json',
                  'metrics.json', 'score_report.json', 'summary.json', 'summary.md', 'frames.csv'}
DOC_NAMES = {'readme.md', 'readme.txt', 'readme', 'agents.md', 'contributing.md',
             'license', 'license.txt', 'license.md', 'copying', 'cmakelists.txt',
             'requirements.txt', 'manifest.in'}
EXPERIMENT_NAME = re.compile(r'(report|summary|metric|result|config|stat|timing|benchmark|'
                             r'performance|experiment|quality|analysis|inventory|manifest)', re.I)
IDENTITY_FIELDS = ('kind', 'device', 'inode', 'mode', 'bytes', 'mtime_ns', 'ctime_ns', 'link_target')


class ArchivePolicyError(ValueError):
    pass


def no_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        try:
            if stat.S_ISLNK(part.lstat().st_mode):
                raise ArchivePolicyError('Symlink path rejected: ' + str(part))
        except FileNotFoundError:
            continue
    return path


def file_identity(status):
    return {'kind': 'file' if stat.S_ISREG(status.st_mode) else 'other',
            'device': status.st_dev, 'inode': status.st_ino, 'mode': status.st_mode,
            'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns,
            'ctime_ns': status.st_ctime_ns, 'link_target': None}


def descriptor_matches(status, expected):
    actual = file_identity(status)
    # Windows Python exposes creation time through path stat but change time
    # through descriptor stat. Linux production checks every field unchanged;
    # Windows offline tests still check ctime using path lstat before/after.
    if os.name == 'nt':
        actual['ctime_ns'] = expected['ctime_ns']
    return actual == expected


def archive_kind(path):
    name = path.name.lower()
    if name.endswith(('.tar.gz', '.tgz')):
        return 'tar.gz'
    if name.endswith('.tar'):
        return 'tar'
    if name.endswith('.zip'):
        return 'zip'
    raise ArchivePolicyError('Unsupported deferred archive requires review: ' + str(path))


def load_reviewed_plan(plan_path, expected_sha256):
    """Only select exact, normalized Linux paths authorized by the reviewed plan."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256 or ''):
        raise ArchivePolicyError('Expected reviewed plan SHA256 is required')
    plan_path = no_symlinks(plan_path)
    descriptor = os.open(plan_path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as source:
        data = source.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256.lower():
        raise ArchivePolicyError('Reviewed plan SHA256 mismatch')
    plan = json.loads(data)
    if plan.get('schema') != 1 or not re.fullmatch(r'history_[0-9a-f]{32}', plan.get('plan_id', '')):
        raise ArchivePolicyError('Unexpected cleanup plan identity')
    rows, excluded = [], []
    for row in plan['files']:
        if row.get('reason') != 'archive_external_report_gate':
            continue
        path = PurePosixPath(row['path']); root = PurePosixPath(row['tree'])
        if (not path.is_absolute() or str(path) != row['path'] or '\\' in row['path']
                or '\x00' in row['path'] or '..' in path.parts or root not in path.parents
                or str(path.relative_to(root)) != row['relative_path']
                or row.get('action') != 'retain' or row.get('preserve_report') is not False
                or row.get('kind') != 'file' or row.get('link_target') is not None):
            raise ArchivePolicyError('Deferred archive row is not an exact regular retained file')
        if row['path'].lower().endswith('.log.gz'):
            excluded.append(row['path']); continue
        archive_kind(path)
        if not all(field in row for field in IDENTITY_FIELDS):
            raise ArchivePolicyError('Incomplete reviewed archive identity')
        rows.append(row)
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ArchivePolicyError('Nonempty unique deferred archive paths required')
    context = {'source_plan': {'path': str(plan_path), 'sha256': digest,
                              'plan_id': plan['plan_id'], 'policy_sha256': plan['policy_sha256'],
                              'tool_sha256': plan['tool_sha256']},
               'excluded_compressed_logs': excluded, 'rows': rows}
    return context


def safe_member_name(name, is_directory=False):
    if not isinstance(name, str) or '\\' in name or '\x00' in name or ':' in name:
        raise ArchivePolicyError('Unsafe archive path spelling')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        if is_directory and name in ('.', './'):
            return None
        raise ArchivePolicyError('Absolute, empty, or traversing archive member')
    # Avoid Win32 aliases when the same archive is verified on the local machine.
    for part in path.parts:
        stem = part.split('.')[0].upper()
        if part.endswith((' ', '.')) or stem in {'CON', 'PRN', 'AUX', 'NUL'} \
                or re.fullmatch(r'(COM|LPT)[1-9]', stem):
            raise ArchivePolicyError('Nonportable archive path component')
    return path


def selected_metadata(path, keep_compressed_logs=False):
    lower = tuple(part.lower() for part in path.parts)
    directories, name = set(lower[:-1]), lower[-1]
    suffix = PurePosixPath(name).suffix
    if directories & CODE_DIRS or name in DOC_NAMES:
        return False
    if keep_compressed_logs and name.endswith('.log.gz'):
        return True  # Preserve the original compressed bytes; never truncate.
    # Preserve frame timing/header metadata, never the binary trajectory files.
    if len(lower) >= 2 and lower[-2] in {'trace', 'traces'} and name in {'frames.csv', 'metadata.json'}:
        return not bool((directories - {'trace', 'traces'}) & RAW_DIRS)
    if directories & RAW_DIRS:
        return False
    if suffix == '.png':
        return bool(directories & {'reports', 'report'})
    if suffix not in EXTENSIONS:
        return False
    if directories & REPORT_DIRS:
        return True
    if suffix in {'.md', '.log'}:
        return True
    return name in METADATA_NAMES or bool(EXPERIMENT_NAME.search(name))


class HashingReader:
    def __init__(self, stream, progress):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.progress = progress

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        self.progress(self.bytes_read)
        return data


def save_manifest(destination, manifest):
    temporary = destination / '.manifest.pending'
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    temporary.replace(destination / 'manifest.json')


def extract_reports(archives, destination, max_entry_bytes=64 * 1024**2,
                    max_total_bytes=2 * 1024**3, progress_seconds=10, *, plan_context=None):
    destination = no_symlinks(destination).resolve()
    if destination.exists():
        raise FileExistsError('Destination must be new; no report is ever replaced: ' + str(destination))
    inputs = [no_symlinks(path).resolve() for path in archives]
    if not inputs or (plan_context is None and len({path.name.casefold() for path in inputs}) != len(inputs)):
        raise ArchivePolicyError('Explicit archives with distinct basenames are required')
    reviewed = {row['path']: row for row in plan_context['rows']} if plan_context else {}
    if plan_context and [str(path) for path in inputs] != [row['path'] for row in plan_context['rows']]:
        raise ArchivePolicyError('Input paths differ from reviewed plan order')
    initial = {}
    for path in inputs:
        value = file_identity(path.lstat())
        if value['kind'] != 'file':
            raise ArchivePolicyError('Expected an explicitly named regular archive')
        archive_kind(path)
        if plan_context and value != {field: reviewed[str(path)][field] for field in IDENTITY_FIELDS}:
            raise ArchivePolicyError('Archive identity differs from reviewed plan: ' + str(path))
        initial[str(path)] = value
        safe_member_name(path.name)
    if max_entry_bytes <= 0 or max_total_bytes <= 0:
        raise ArchivePolicyError('Positive extraction budgets are required')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'autodl.archive_reports.v2' if plan_context else 'autodl.archive_reports.v1', 'status': 'in_progress',
                'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selection_policy': 'experimental metadata only; no code, raw arrays, meshes or binary trajectories',
                'archives': [], 'selected_files': [], 'selected_bytes': 0,
                'max_entry_bytes': max_entry_bytes, 'max_total_bytes': max_total_bytes,
                'deletion_performed': False, 'errors': []}
    if plan_context:
        manifest.update({key: plan_context[key] for key in ('source_plan', 'excluded_compressed_logs')})
    save_manifest(destination, manifest)
    try:
        for index, archive in enumerate(inputs):
            no_symlinks(archive)
            before = archive.lstat()
            if file_identity(before) != initial[str(archive)]:
                raise ArchivePolicyError('Archive changed since complete preflight: ' + str(archive))
            label = (f'{index:03d}_{hashlib.sha256(str(archive).encode()).hexdigest()[:12]}_{archive.name}'
                     if plan_context else archive.name)
            kind = archive_kind(archive)
            entry = {'archive': str(archive), 'basename': archive.name, 'status': 'in_progress',
                     'compressed_bytes': before.st_size, 'members_seen': 0, 'selected_count': 0,
                     'archive_stat': {'size': before.st_size, 'mtime_ns': before.st_mtime_ns,
                                      'device': before.st_dev, 'inode': before.st_ino},
                     'skipped_regular_files': 0, 'directories_seen': 0, 'ignored_links': []}
            if plan_context:
                entry.update(archive_label=label, archive_kind=kind, plan_identity=initial[str(archive)])
                entry['archive_stat'].update(mode=before.st_mode, ctime_ns=before.st_ctime_ns)
            if kind == 'zip':
                entry['zip_payload_crc_scope'] = 'selected reports; all archive bytes included in SHA256'
            manifest['archives'].append(entry)
            output_root = destination / label
            output_root.mkdir(exist_ok=False)
            last_progress = [time.monotonic()]

            def progress(compressed_bytes):
                now = time.monotonic()
                if now - last_progress[0] >= progress_seconds:
                    print(json.dumps({'archive': str(archive), 'compressed_bytes_read': compressed_bytes,
                                      'members_seen': entry['members_seen'], 'selected_count': entry['selected_count']}),
                          flush=True)
                    last_progress[0] = now

            def save_selected(member_name, size, path, open_source):
                if size < 0:
                    raise ArchivePolicyError('Negative member size')
                if not selected_metadata(path, keep_compressed_logs=plan_context is not None):
                    entry['skipped_regular_files'] += 1
                    return
                if size > max_entry_bytes:
                    raise ArchivePolicyError(f'Oversize report requires review, never truncated: {member_name} ({size} bytes)')
                if manifest['selected_bytes'] + size > max_total_bytes:
                    raise ArchivePolicyError('Total report budget exceeded before member: ' + member_name)
                target = output_root.joinpath(*path.parts)
                if not target.resolve().is_relative_to(output_root.resolve()):
                    raise ArchivePolicyError('Resolved output escaped the archive report root')
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(); written = 0
                source = open_source()
                if source is None:
                    raise ArchivePolicyError('Regular member has no readable content')
                with source, target.open('xb') as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(chunk); digest.update(chunk); written += len(chunk)
                if written != size:
                    raise ArchivePolicyError('Report length differs from archive header: ' + member_name)
                record = {'archive_basename': archive.name, 'original_path': member_name,
                          'output_path': target.relative_to(destination).as_posix(),
                          'bytes': written, 'sha256': digest.hexdigest()}
                if plan_context:
                    record['archive_label'] = label
                manifest['selected_files'].append(record)
                manifest['selected_bytes'] += written
                entry['selected_count'] += 1

            def read_tar(stream):
                with tarfile.open(fileobj=stream, mode='r|') as container:
                    for member in container:
                        entry['members_seen'] += 1
                        path = safe_member_name(member.name, member.isdir())
                        if member.issym() or member.islnk():
                            entry['ignored_links'].append({'original_path': member.name,
                                'kind': 'symlink' if member.issym() else 'hardlink',
                                'target': member.linkname, 'extracted': False})
                            continue
                        if not (member.isdir() or member.isfile()):
                            raise ArchivePolicyError('Special member rejected: ' + member.name)
                        if member.isdir():
                            entry['directories_seen'] += 1
                            continue
                        save_selected(member.name, member.size, path, lambda: container.extractfile(member))

            descriptor = os.open(archive, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(descriptor, 'rb') as raw:
                if not descriptor_matches(os.fstat(raw.fileno()), initial[str(archive)]):
                    raise ArchivePolicyError('Opened archive differs from preflight identity')
                hashing = HashingReader(raw, progress)
                if kind == 'tar.gz':
                    # Drain true gzip EOF to validate CRC, including skipped data.
                    with gzip.GzipFile(fileobj=hashing, mode='rb') as compressed:
                        read_tar(compressed)
                        while compressed.read(1024 * 1024):
                            pass
                elif kind == 'tar':
                    read_tar(hashing)
                while hashing.read(1024 * 1024):
                    pass
                if kind == 'zip':
                    # ZIP needs random access. Hash every byte once, then seek on
                    # the same verified descriptor to read only selected entries.
                    raw.seek(0)
                    with zipfile.ZipFile(raw) as container:
                        for member in container.infolist():
                            entry['members_seen'] += 1
                            path = safe_member_name(member.filename, member.is_dir())
                            mode = member.external_attr >> 16
                            if stat.S_ISLNK(mode):
                                entry['ignored_links'].append({'original_path': member.filename,
                                    'kind': 'symlink', 'target': None, 'extracted': False})
                                continue
                            if member.is_dir():
                                entry['directories_seen'] += 1
                                continue
                            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                                raise ArchivePolicyError('Special ZIP member rejected: ' + member.filename)
                            save_selected(member.filename, member.file_size, path, lambda: container.open(member))
                entry['archive_sha256'] = hashing.digest.hexdigest()
                if hashing.bytes_read != before.st_size:
                    raise ArchivePolicyError('Compressed archive byte count changed during reading')
                if not descriptor_matches(os.fstat(raw.fileno()), initial[str(archive)]):
                    raise ArchivePolicyError('Opened archive changed while reading')
            no_symlinks(archive)
            if file_identity(archive.lstat()) != initial[str(archive)]:
                raise ArchivePolicyError('Archive changed during extraction')
            entry['archive_stat_unchanged'] = True
            entry['status'] = 'completed'
            save_manifest(destination, manifest)
            print(json.dumps({'archive': archive.name, 'status': 'completed',
                              'selected_count': entry['selected_count'], 'members_seen': entry['members_seen']}), flush=True)
        manifest['status'] = 'completed'
        save_manifest(destination, manifest)
        return manifest
    except BaseException as error:
        manifest['status'] = 'failed'
        if manifest['archives'] and manifest['archives'][-1]['status'] != 'completed':
            manifest['archives'][-1]['status'] = 'failed'
        manifest['errors'].append({'type': type(error).__name__, 'message': str(error)})
        save_manifest(destination, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--archive', nargs='+', type=Path)
    selection.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha')
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--max-entry-mib', type=int, default=64)
    parser.add_argument('--max-total-mib', type=int, default=2048)
    args = parser.parse_args()
    try:
        context = load_reviewed_plan(args.plan, args.expected_plan_sha) if args.plan else None
        if not args.plan and args.expected_plan_sha:
            raise ArchivePolicyError('--expected-plan-sha only applies with --plan')
        archives = [row['path'] for row in context['rows']] if context else args.archive
        result = extract_reports(archives, args.destination, args.max_entry_mib * 1024**2,
                                 args.max_total_mib * 1024**2, plan_context=context)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, RuntimeError, EOFError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error),
                          'manifest': str(args.destination / 'manifest.json')}), file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'selected_files': len(result['selected_files']),
                      'selected_bytes': result['selected_bytes'], 'manifest': str(args.destination / 'manifest.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
