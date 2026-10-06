"""Explicit, report-preserving cleanup of user-approved historical AutoDL trees.

Linux-only commands: plan -> preserve -> apply. No SSH, no automatic execution,
no recursive tree deletion. Archives stay for a separate report-extraction gate.
Delete payload identity uses type/dev/inode/size/mtime_ns/ctime_ns, not expensive
full raw-data hashing. Retained reports are copied and verified with SHA256.
"""
import argparse
import bisect
import datetime as dt
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import uuid


REPORT_EXTENSIONS = {'.md', '.json', '.csv', '.log', '.txt', '.pdf'}
REPORT_BRANCHES = {'report', 'reports', 'log', 'logs', 'manifest', 'manifests', 'analysis', 'analyses', 'results'}
BLOCKED_BRANCHES = {
    'source', 'sources', 'src', 'include', 'includes', 'build', 'builds', 'cmakefiles', 'cmake',
    'vendor', 'vendors', 'third_party', 'third-party', 'thirdparty', 'external', 'extern',
    'deps', 'dependencies', 'downloads', 'buildtrees', 'packages', 'ports', 'installed',
    'assets', 'asset', 'mesh', 'meshes', 'texture', 'textures',
    'dataset', 'datasets', 'trace', 'traces', 'substeps', 'raw', 'raw_data', 'rawtrace',
    'raw_trace', 'bin', 'lib', 'libraries', 'node_modules', '.git', '.venv', 'venv',
    '__pycache__', 'site-packages', 'doc', 'docs', 'documentation', 'examples', 'tests',
    'tools', 'scripts', 'references', 'externals', 'toolsrc', 'stiffgipc', 'meshprocess',
}
DEPENDENCY_ONLY_ROOTS = {
    '/root/autodl-tmp/.autodl_vcpkg_checkout_20260927',
    '/root/autodl-tmp/tinygltf-verify-20260927',
}
CODE_DOCUMENTS = {'readme', 'license', 'licence', 'copying', 'changelog', 'contributing',
                  'authors', 'maintainers', 'agents', 'skill', 'citation', 'code_of_conduct'}
CODE_FILENAMES = {'cmakelists.txt', 'cmakepresets.json', 'cmakeuserpresets.json',
                  'package.json', 'package-lock.json', 'vcpkg.json', 'vcpkg-configuration.json',
                  'requirements.txt', 'environment.yml', 'environment.yaml'}
RUN_METADATA = {'requested.json', 'result.json', 'resolved_config.json', 'config_validation.json',
                'stats.json', 'scene.json', 'metadata.json', 'manifest.json', 'build_manifest.json',
                'frames.csv', 'timing.csv', 'timings.csv', 'metrics.json', 'summary.json',
                'final.json', 'final_metadata.json', 'run.json', 'config.json', 'run.log',
                'stdout.log', 'stderr.log', 'run_meta.json', 'run_metadata.json'}
ARCHIVE_ENDINGS = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz',
                  '.zip', '.7z', '.rar', '.gz', '.bz2', '.xz', '.zst')
DEPENDENCY_ARCHIVE_BRANCHES = {
    'vendor', 'vendors', 'third_party', 'third-party', 'thirdparty', 'external', 'extern',
    'deps', 'dependencies', 'downloads', 'buildtrees', 'packages', 'ports', 'installed',
    'node_modules', '.venv', 'venv', 'site-packages', 'assets', 'asset', 'mesh', 'meshes',
    'texture', 'textures', 'dataset', 'datasets', '.git',
}
IDENTITY_FIELDS = ('kind', 'device', 'inode', 'mode', 'bytes', 'mtime_ns', 'ctime_ns', 'link_target')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assert_linux():
    require(platform.system() == 'Linux', 'History cleanup commands are Linux-only')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def sha(path):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def progress(event, **values):
    print(json.dumps({'event': event, **values}, ensure_ascii=True), file=sys.stderr, flush=True)


def normalized_absolute(text):
    require(isinstance(text, str) and text.startswith('/') and '\\' not in text and '\x00' not in text,
            'Expected an absolute Linux path: ' + str(text))
    pure = PurePosixPath(text)
    require(str(pure) == text.rstrip('/') and all(p not in ('.', '..') for p in text.split('/')[1:]),
            'Path must be normalized without dot components: ' + text)
    require(str(pure) not in ('/', '/root', '/root/autodl-tmp'), 'Broad root is forbidden')
    return str(pure)


def protected_factor(path):
    return any(part.lower().startswith('autodl_factor') for part in PurePosixPath(str(path)).parts)


def validate_tree_text(text):
    text = normalized_absolute(text)
    p = PurePosixPath(text)
    allowed = any(p == root or root in p.parents for root in
                  (PurePosixPath('/root/stiff_toi_cudagraph_20260929'), PurePosixPath('/root/paper_builds_20260929')))
    under_tmp = PurePosixPath('/root/autodl-tmp') in p.parents and len(p.parts) >= 4
    require(allowed or under_tmp, 'Tree is outside approved historical project locations: ' + text)
    require(not protected_factor(text), 'Current autodl_factor projects are always protected: ' + text)
    return text


def no_symlink_parents(path, include_leaf=True):
    target = path.absolute()
    parts = (target, *target.parents) if include_leaf else tuple(target.parents)
    for part in parts:
        try:
            require(not stat.S_ISLNK(part.lstat().st_mode), 'Symlink ancestor rejected: ' + str(part))
        except FileNotFoundError:
            continue
    return target


def relative_prefix(text):
    require(isinstance(text, str) and text and not text.startswith('/') and '\\' not in text,
            'keep_prefixes must be exact nonempty relative paths')
    p = PurePosixPath(text)
    require(str(p) == text and all(part not in ('', '.', '..') for part in text.split('/')),
            'Invalid keep prefix: ' + text)
    return text


def validate_policy(policy):
    require(policy.get('schema') == 1 and isinstance(policy.get('trees'), list) and policy['trees'],
            'Policy schema 1 and explicit nonempty trees list required')
    trees = []
    for item in policy['trees']:
        require(set(item) <= {'path', 'mode', 'keep_prefixes', 'note'}, 'Unknown tree policy field')
        path = Path(validate_tree_text(item['path']))
        no_symlink_parents(path)
        require(path.is_dir() and path.resolve() == path, 'Tree must be an existing real directory: ' + str(path))
        require(item['mode'] in ('reports_only', 'important'), 'Unknown cleanup mode')
        prefixes = [relative_prefix(v) for v in item.get('keep_prefixes', [])]
        require(len(set(prefixes)) == len(prefixes), 'Duplicate keep prefixes')
        require(item['mode'] == 'important' or not prefixes, 'reports_only cannot have arbitrary keep prefixes')
        for prefix in prefixes:
            target = path / prefix
            no_symlink_parents(target, include_leaf=False)
            require(target.exists() or target.is_symlink(), 'Important prefix does not exist: ' + str(target))
        trees.append({'path': str(path), 'mode': item['mode'], 'keep_prefixes': prefixes,
                      'note': item.get('note', '')})
    paths = [Path(t['path']) for t in trees]
    require(len(set(paths)) == len(paths) and not any(a in b.parents or b in a.parents
            for i, a in enumerate(paths) for b in paths[i + 1:]), 'Overlapping tree policies are forbidden')
    protected = [Path(normalized_absolute(p)) for p in policy.get('protected_paths', [])]
    require(not any(root == keep or keep in root.parents for root in paths for keep in protected),
            'A policy tree is inside an explicitly protected path')
    report_root = Path(normalized_absolute(policy['reports_root']))
    tmp_report = PurePosixPath('/root/autodl-tmp') in PurePosixPath(str(report_root)).parents
    protected_report = any(keep in report_root.parents for keep in protected)
    require(tmp_report or protected_report,
            'Central report root must be under /root/autodl-tmp or an explicitly protected output tree')
    no_symlink_parents(report_root)
    require(not any(report_root == root or root in report_root.parents or report_root in root.parents for root in paths),
            'Central reports must be separate from every cleanup tree')
    return {'schema': 1, 'trees': trees, 'reports_root': str(report_root),
            'protected_paths': [str(p) for p in protected]}


def classify(relative, tree):
    p = PurePosixPath(relative)
    parts = [part.lower() for part in p.parts]
    directories = set(parts[:-1])
    name = parts[-1]
    report_branch = bool(directories & REPORT_BRANCHES)
    blocked_parts = directories & BLOCKED_BRANCHES
    if report_branch:
        blocked_parts -= {'doc', 'docs', 'documentation'}
    blocked = bool(blocked_parts)
    report = not blocked and (p.suffix.lower() in REPORT_EXTENSIONS or name.endswith('.log.gz') or
                              (report_branch and p.suffix.lower() == '.png'))
    if len(parts) > 1 and parts[-2] in ('trace', 'traces') and name in ('frames.csv', 'metadata.json'):
        # Timing and topology headers are reports, unlike the neighboring raw
        # positions/velocities. Source/build/vendor precedence still applies.
        report = not bool(blocked_parts - {'trace', 'traces'})
    if not report_branch and p.stem.lower() in CODE_DOCUMENTS:
        report = False
    if not report_branch and name in CODE_FILENAMES:
        report = False
    if not report_branch and name not in RUN_METADATA and re.match(r'^(state|velocity|position|vertices|trajectory)[_\d.]', name):
        report = False
    dependency_only = tree.get('path') in DEPENDENCY_ONLY_ROOTS
    if dependency_only:
        report = False
    important = any(relative == prefix or prefix in [str(v) for v in p.parents]
                    for prefix in tree['keep_prefixes'])
    if important:
        return 'retain', 'important_exact_prefix', report
    if dependency_only:
        return 'delete', 'explicit_dependency_only_tree', False
    if report:
        return 'retain', 'experimental_report_or_metadata', True
    if name.endswith(ARCHIVE_ENDINGS) and not directories.intersection(DEPENDENCY_ARCHIVE_BRANCHES):
        return 'retain', 'archive_external_report_gate', False
    return 'delete', 'unselected_historical_payload', False


def stat_identity(status, link_target=None):
    kind = 'file' if stat.S_ISREG(status.st_mode) else 'symlink' if stat.S_ISLNK(status.st_mode) else 'directory' if stat.S_ISDIR(status.st_mode) else 'special'
    return {'kind': kind, 'device': status.st_dev, 'inode': status.st_ino, 'mode': status.st_mode,
            'bytes': status.st_size, 'mtime_ns': status.st_mtime_ns, 'ctime_ns': status.st_ctime_ns,
            'link_target': link_target if kind == 'symlink' else None}


def identity(path):
    status = path.lstat()
    return stat_identity(status, os.readlink(path) if stat.S_ISLNK(status.st_mode) else None)


def selected_identity(record):
    return {field: record[field] for field in IDENTITY_FIELDS}


def unlink_planned(path, row, own_unlink_ctimes):
    """Accept only ctime changes observed through our own verified unlink.

    O_PATH keeps the exact inode inspectable even when its last name is removed;
    O_NOFOLLOW also supports unlinking a symlink without opening its target.
    """
    key = (row['device'], row['inode'])
    expected = selected_identity(row)
    expected['ctime_ns'] = own_unlink_ctimes.get(key, expected['ctime_ns'])
    require(identity(path) == expected, 'File changed after preflight: ' + str(path))
    descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat_identity(before, row['link_target']) == expected,
                'Opened inode changed after preflight: ' + str(path))
        require(identity(path) == expected, 'Unlink path changed after opening: ' + str(path))
        os.unlink(path)
        after = os.fstat(descriptor)
        actual = stat_identity(after, row['link_target'])
        require(all(actual[field] == expected[field] for field in IDENTITY_FIELDS if field != 'ctime_ns')
                and after.st_nlink == before.st_nlink - 1,
                'Inode changed beyond this unlink: ' + str(path))
        own_unlink_ctimes[key] = actual['ctime_ns']
    finally:
        os.close(descriptor)


def is_protected(path, policy):
    return protected_factor(path) or any(path == Path(p) or Path(p) in path.parents for p in policy['protected_paths'])


def scan_policy(policy, hash_reports):
    files, directories, protected = [], [], []
    for tree in policy['trees']:
        root = Path(tree['path']); stack = [root]; count = 0
        progress('scan_start', root=str(root))
        while stack:
            directory = stack.pop()
            no_symlink_parents(directory)
            require(identity(directory)['kind'] == 'directory', 'Directory changed while scanning: ' + str(directory))
            directories.append({'path': str(directory), 'tree': str(root), **identity(directory)})
            with os.scandir(directory) as stream:
                children = sorted(stream, key=lambda entry: entry.name)
            for item in children:
                path = Path(item.path)
                if is_protected(path, policy):
                    protected.append(str(path)); continue
                entry = identity(path)
                if entry['kind'] == 'directory':
                    stack.append(path); continue
                require(entry['kind'] in ('file', 'symlink'), 'Special file makes cleanup unsafe: ' + str(path))
                relative = path.relative_to(root).as_posix()
                action, reason, report = classify(relative, tree)
                row = {'path': str(path), 'tree': str(root), 'relative_path': relative,
                       'action': action, 'reason': reason, 'preserve_report': report, **entry}
                if hash_reports and report:
                    row['sha256'] = sha(path) if entry['kind'] == 'file' else hashlib.sha256(entry['link_target'].encode()).hexdigest()
                    require(identity(path) == entry, 'Report changed while hashing: ' + str(path))
                files.append(row); count += 1
                if count % 10000 == 0:
                    progress('scan_progress', root=str(root), files=count)
        rows = [row for row in files if row['tree'] == str(root)]
        progress('scan_complete', root=str(root), files=count,
                 delete_bytes=sum(row['bytes'] for row in rows if row['action'] == 'delete'),
                 report_files=sum(row['preserve_report'] for row in rows))
    return {'files': sorted(files, key=lambda row: row['path']),
            'directories': sorted(directories, key=lambda row: row['path']),
            'protected_subtrees': sorted(protected)}


def write_new(path, value):
    no_symlink_parents(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))


def outside_trees(path, policy):
    no_symlink_parents(path)
    path = path.absolute()
    require(not any(path == Path(t['path']) or Path(t['path']) in path.parents for t in policy['trees']),
            'Control/receipt files must be outside cleanup trees')


def make_plan(policy_path, output):
    assert_linux(); original = read(policy_path); policy = validate_policy(original)
    outside_trees(output, policy); outside_trees(policy_path, policy)
    require(not output.exists(), 'Plan output already exists')
    snapshot = scan_policy(policy, hash_reports=True)
    summaries = []
    for tree in policy['trees']:
        rows = [row for row in snapshot['files'] if row['tree'] == tree['path']]
        summaries.append({'root': tree['path'], 'mode': tree['mode'],
                          'retain_files': sum(row['action'] == 'retain' for row in rows),
                          'retain_bytes': sum(row['bytes'] for row in rows if row['action'] == 'retain'),
                          'delete_files': sum(row['action'] == 'delete' for row in rows),
                          'delete_bytes': sum(row['bytes'] for row in rows if row['action'] == 'delete'),
                          'report_files': sum(row['preserve_report'] for row in rows),
                          'archive_files_deferred': sum(row['reason'] == 'archive_external_report_gate' for row in rows)})
    plan = {'schema': 1, 'plan_id': 'history_' + uuid.uuid4().hex,
            'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'policy_path': str(policy_path.absolute()), 'policy_sha256': sha(policy_path), 'policy': policy,
            'tool_sha256': sha(Path(__file__)), 'deletion_identity': list(IDENTITY_FIELDS),
            'archives_require_separate_report_extraction': True, 'summary': summaries, **snapshot}
    write_new(output, plan)
    return {'plan': str(output), 'plan_sha256': sha(output), 'summary': summaries,
            'protected_subtrees': snapshot['protected_subtrees']}


def load_plan(path):
    plan = read(path)
    require(plan.get('schema') == 1 and re.fullmatch(r'history_[0-9a-f]{32}', plan['plan_id']), 'Invalid plan identity')
    require(plan['tool_sha256'] == sha(Path(__file__)), 'Cleanup tool changed since planning')
    require(plan['policy_sha256'] == sha(Path(plan['policy_path'])), 'Policy changed since planning')
    require(validate_policy(read(plan['policy_path'])) == plan['policy'], 'Resolved policy changed')
    paths = [row['path'] for row in plan['files']]
    require(len(paths) == len(set(paths)), 'Duplicate planned file')
    by_root = {tree['path']: tree for tree in plan['policy']['trees']}
    for row in plan['files']:
        root = Path(row['tree']); file = Path(row['path'])
        require(row['tree'] in by_root and root in file.parents and file.relative_to(root).as_posix() == row['relative_path'],
                'Planned file is outside its exact tree')
        require(not is_protected(file, plan['policy']), 'Plan attempts to touch protected files')
        action, reason, report = classify(row['relative_path'], by_root[row['tree']])
        require((row['action'], row['reason'], row['preserve_report']) == (action, reason, report),
                'Plan classification differs from policy')
        require(row['kind'] in ('file', 'symlink'), 'Only regular files/symlinks are eligible')
    directory_paths = [row['path'] for row in plan['directories']]
    require(len(directory_paths) == len(set(directory_paths)), 'Duplicate planned directory')
    for row in plan['directories']:
        root = Path(row['tree']); directory = Path(row['path'])
        require(row['tree'] in by_root and (directory == root or root in directory.parents)
                and row['kind'] == 'directory' and not is_protected(directory, plan['policy']),
                'Planned directory is outside its exact tree or protected')
    return plan


def backup_destination(plan, row):
    source = Path(row['path'])
    kind = 'files' if row['kind'] == 'file' else 'symlink_metadata'
    destination = Path(plan['policy']['reports_root']) / plan['plan_id'] / kind / source.relative_to(source.anchor)
    if row['kind'] == 'symlink':
        destination = destination.with_name(destination.name + '.json')
    return destination


def symlink_report_metadata(row):
    return {'source': row['path'], 'link_target': row['link_target'], 'link_was_not_followed': True,
            'original_identity': selected_identity(row)}


def preserve(plan_path, receipt_path):
    assert_linux(); plan = load_plan(plan_path); outside_trees(receipt_path, plan['policy'])
    require(not receipt_path.exists(), 'Preservation receipt already exists')
    records = []
    for row in plan['files']:
        if not row['preserve_report']:
            continue
        source = Path(row['path']); no_symlink_parents(source, include_leaf=row['kind'] != 'symlink')
        require(identity(source) == selected_identity(row), 'Report identity changed: ' + str(source))
        destination = backup_destination(plan, row); no_symlink_parents(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row['kind'] == 'file':
            require(sha(source) == row['sha256'], 'Report content changed: ' + str(source))
            if not destination.exists():
                descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                with os.fdopen(descriptor, 'rb') as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            require(destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256'],
                    'Report backup verification failed: ' + str(destination))
        else:
            metadata = symlink_report_metadata(row)
            if not destination.exists():
                write_new(destination, metadata)
            require(destination.read_bytes() == encoded(metadata), 'Symlink metadata backup differs')
        require(identity(source) == selected_identity(row), 'Report changed while copying: ' + str(source))
        records.append({'source': row['path'], 'backup': str(destination), 'kind': row['kind'],
                        'bytes': destination.stat().st_size, 'sha256': sha(destination)})
        if len(records) % 1000 == 0:
            progress('preserve_progress', reports=len(records))
    receipt = {'schema': 1, 'plan_id': plan['plan_id'], 'plan_sha256': sha(plan_path),
               'reports_root': plan['policy']['reports_root'], 'complete': True,
               'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'reports': records}
    write_new(receipt_path, receipt)
    return {'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path),
            'plan_sha256': receipt['plan_sha256'], 'report_files': len(records),
            'report_bytes': sum(row['bytes'] for row in records)}


def verify_preservation(plan_path, plan, receipt_path):
    receipt = read(receipt_path)
    require(receipt.get('complete') is True and receipt['plan_id'] == plan['plan_id']
            and receipt['plan_sha256'] == sha(plan_path) and receipt['reports_root'] == plan['policy']['reports_root'],
            'Preservation receipt does not match this exact plan')
    expected = {row['path']: row for row in plan['files'] if row['preserve_report']}
    actual = {row['source']: row for row in receipt['reports']}
    require(len(actual) == len(receipt['reports']) and set(actual) == set(expected), 'Incomplete or duplicate report backups')
    for source, record in actual.items():
        destination = Path(record['backup']); no_symlink_parents(destination)
        require(destination == backup_destination(plan, expected[source]), 'Backup path does not mirror its planned source')
        require(destination.is_file() and destination.stat().st_size == record['bytes'] and sha(destination) == record['sha256'],
                'Report backup changed or missing: ' + str(destination))
        if expected[source]['kind'] == 'file':
            require(record['sha256'] == expected[source]['sha256'], 'Backup differs from original report SHA256')
        else:
            require(destination.read_bytes() == encoded(symlink_report_metadata(expected[source])),
                    'Backup differs from original symlink metadata')
    return receipt


def snapshot_unchanged(plan):
    fresh = scan_policy(plan['policy'], hash_reports=False)
    before = {row['path']: selected_identity(row) for row in plan['files']}
    after = {row['path']: selected_identity(row) for row in fresh['files']}
    require(before == after and fresh['protected_subtrees'] == plan['protected_subtrees'],
            'Tree file set or file identities changed since plan; regenerate the plan')
    # A new/removed empty directory is also a source-tree change. mtime changes
    # from directory reads are not relevant; inode/device/type must be stable.
    def dirs(rows):
        return {row['path']: (row['kind'], row['device'], row['inode']) for row in rows}
    require(dirs(fresh['directories']) == dirs(plan['directories']), 'Directory set/identity changed since planning')


def active_process_conflicts(plan):
    """Inspect live Linux cwd/exe/argv/fd/maps references without following tree links."""
    deleted = sorted(row['path'] for row in plan['files'] if row['action'] == 'delete')
    selected = set(deleted); roots = [Path(t['path']) for t in plan['policy']['trees']]
    conflicts = []

    def check(pid, source, raw, directory=False):
        if not raw or not raw.startswith('/'):
            return
        raw = raw.removesuffix(' (deleted)')
        path = Path(raw)
        if not any(path == root or root in path.parents for root in roots):
            return
        match = str(path) in selected
        if directory or path.is_dir():
            prefix = str(path).rstrip('/') + '/'; index = bisect.bisect_left(deleted, prefix)
            match |= index < len(deleted) and deleted[index].startswith(prefix)
        if match:
            conflicts.append({'pid': pid, 'source': source, 'path': str(path)})

    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cwd = os.readlink(proc / 'cwd')
            check(proc.name, 'cwd', cwd, directory=True)
            try:
                check(proc.name, 'exe', os.readlink(proc / 'exe'))
            except FileNotFoundError:
                pass  # Kernel thread or process that has just exited.
            for token in (proc / 'cmdline').read_bytes().split(b'\0'):
                value = os.fsdecode(token)
                if value and (value.startswith('/') or not value.startswith('-')):
                    check(proc.name, 'argv', value if value.startswith('/') else str(Path(cwd) / value))
            for descriptor in (proc / 'fd').iterdir():
                try:
                    check(proc.name, 'fd', os.readlink(descriptor))
                except FileNotFoundError:
                    continue
            for line in (proc / 'maps').read_text(errors='replace').splitlines():
                columns = line.split(maxsplit=5)
                if len(columns) == 6:
                    check(proc.name, 'maps', columns[5])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError('Cannot inspect live process safely: ' + str(proc)) from error
    return conflicts


def apply(plan_path, receipt_path, expected_plan_sha256, output):
    assert_linux()
    require(sha(plan_path) == expected_plan_sha256.lower(), 'Reviewed plan SHA256 mismatch')
    plan = load_plan(plan_path); outside_trees(output, plan['policy'])
    require(not output.exists(), 'Apply report already exists')
    verify_preservation(plan_path, plan, receipt_path)
    snapshot_unchanged(plan)  # Entire file set checked before the first unlink.
    conflicts = active_process_conflicts(plan)
    require(not conflicts, 'Active process references planned deletion: ' + json.dumps(conflicts[:20]))
    result = {'schema': 1, 'plan_id': plan['plan_id'], 'plan_sha256': expected_plan_sha256.lower(),
              'preservation_receipt_sha256': sha(receipt_path), 'completed': False,
              'deleted_files': [], 'removed_empty_directories': [], 'retained_nonempty_directories': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_name(output.name + '.journal.jsonl')
    require(not journal.exists(), 'Apply journal already exists')
    own_unlink_ctimes = {}
    with journal.open('x', encoding='utf-8') as audit:
        try:
            for tree in plan['policy']['trees']:
                conflicts = active_process_conflicts(plan)
                require(not conflicts, 'A process started using planned data: ' + json.dumps(conflicts[:20]))
                rows = [row for row in plan['files'] if row['tree'] == tree['path'] and row['action'] == 'delete']
                progress('delete_start', root=tree['path'], files=len(rows))
                for index, row in enumerate(rows, 1):
                    path = Path(row['path']); no_symlink_parents(path, include_leaf=row['kind'] != 'symlink')
                    require(Path(tree['path']) in path.parents and not is_protected(path, plan['policy']), 'Deletion escaped its explicit tree')
                    require(path.parent.resolve() == path.parent, 'Deletion parent no longer resolves exactly')
                    # No recursive deletion: unlink this exact lstat-verified
                    # regular file or symlink, never its referenced target.
                    result['pending_unlink'] = str(path)
                    unlink_planned(path, row, own_unlink_ctimes)
                    record = {'path': str(path), 'kind': row['kind'], 'bytes': row['bytes']}
                    result['deleted_files'].append(record)
                    result.pop('pending_unlink')
                    audit.write(json.dumps(record) + '\n'); audit.flush()
                    if index % 10000 == 0:
                        progress('delete_progress', root=tree['path'], files=index)
                progress('delete_complete', root=tree['path'], files=len(rows))
            for row in sorted(plan['directories'], key=lambda r: len(Path(r['path']).parts), reverse=True):
                directory = Path(row['path']); no_symlink_parents(directory)
                require(not is_protected(directory, plan['policy']), 'Protected directory in removal plan')
                tree = next(t for t in plan['policy']['trees'] if t['path'] == row['tree'])
                keeps = [Path(tree['path']) / prefix for prefix in tree['keep_prefixes']]
                if any(directory == keep or keep in directory.parents or directory in keep.parents for keep in keeps):
                    continue  # Preserve exact important prefixes even when empty.
                require(directory.resolve() == directory, 'Directory no longer resolves exactly')
                now = identity(directory)
                require((now['kind'], now['device'], now['inode']) == ('directory', row['device'], row['inode']),
                        'Directory changed before empty removal: ' + str(directory))
                try:
                    os.rmdir(directory)
                    result['removed_empty_directories'].append(str(directory))
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                        raise
                    result['retained_nonempty_directories'].append(str(directory))
            result['completed'] = True
        except Exception as error:
            result['failure'] = f'{type(error).__name__}: {error}'
            write_new(output, result)
            raise
    result['deleted_bytes'] = sum(row['bytes'] for row in result['deleted_files'])
    result['finished_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_new(output, result)
    return {'report': str(output), 'completed': True, 'deleted_files': len(result['deleted_files']),
            'deleted_bytes': result['deleted_bytes'], 'empty_directories': len(result['removed_empty_directories'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--policy', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('preserve'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('apply'); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--backup-receipt', type=Path, required=True); p.add_argument('--expected-plan-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = make_plan(args.policy, args.output)
    elif args.command == 'preserve':
        result = preserve(args.plan, args.output)
    else:
        result = apply(args.plan, args.backup_receipt, args.expected_plan_sha256, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
