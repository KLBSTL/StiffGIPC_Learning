"""Explicit export/receive/receipt-gated trace pruning for AutoDL batches.

No SSH, scheduling or automatic invocation. Run export/prune on the remote
package root and receive on the local host. Exported archives remain remote.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import tarfile
import uuid


ROOT = Path(__file__).resolve().parents[2]
CONTROL = 'TRANSFER_MANIFEST.json'
LEDGERS = {'reports/active/AUTODL_SMOKE_BATCH.json', 'reports/active/AUTODL_WINDOW_BATCH.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Unsafe archive path: ' + str(name))
    parts = PurePosixPath(name).parts
    require(not name.startswith('/') and all(part not in ('', '.', '..') for part in name.split('/')),
            'Unsafe archive path: ' + name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{kind}{n}' for kind in ('COM', 'LPT') for n in range(1, 10)}
    require(all(not part.endswith((' ', '.')) and part.split('.')[0].upper() not in reserved for part in parts),
            'Nonportable archive path: ' + name)
    return name


def within(root, name):
    name = safe_name(name)
    root = root.resolve()
    target = root.joinpath(*PurePosixPath(name).parts)
    current = root
    require(not root.is_symlink(), 'Root is a symlink')
    for component in PurePosixPath(name).parts:
        current = current / component
        require(not current.is_symlink(), 'Symlink encountered: ' + str(current))
    require(target.resolve().is_relative_to(root), 'Path escaped root: ' + name)
    return target


def reject_symlink_ancestors(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink ancestor rejected: ' + str(part))


def regular_files(directory):
    require(directory.is_dir() and not directory.is_symlink(), 'Expected real directory: ' + str(directory))
    result = []
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Symlink rejected: ' + str(path))
        if path.is_file():
            result.append(path)
        else:
            require(path.is_dir(), 'Special file rejected: ' + str(path))
    return result


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))


def package_root(root):
    reject_symlink_ancestors(root.absolute())
    root = root.resolve()
    require(root != Path(root.anchor) and root != Path.home().resolve(), 'Refusing broad root')
    require((root / 'autodl_bundle.json').is_file(), 'Not an unpacked AutoDL package root')
    return root


def export(root, stage, repeat, output, driver_file=None, jobs_root=None, include_binaries=False):
    root = package_root(root)
    require(stage in ('smoke', 'window'), 'Unsupported export stage')
    require((stage == 'smoke' and repeat in (None, 1)) or (stage == 'window' and repeat in (1, 2, 3)),
            'Window requires repeat 1, 2 or 3; smoke uses repeat 1')
    repeat = repeat or 1
    output = output.resolve()
    require(output.is_relative_to(root), 'Remote archive must remain inside this package root')
    output = within(root, output.relative_to(root).as_posix())
    sidecar = Path(str(output) + '.export.json')
    require(not output.exists() and not sidecar.exists(), 'Refusing to replace export files')
    ledger_name = f'reports/active/AUTODL_{stage.upper()}_BATCH.json'
    ledger_path = within(root, ledger_name)
    ledger = read(ledger_path)
    plan = read(within(root, f'configs/active/autodl_{stage}.json'))
    require(ledger['plan_sha256'] == sha(root / f'configs/active/autodl_{stage}.json'), 'Plan/ledger identity mismatch')
    planned = {row['name']: row for row in plan['runs']}
    rows = [row for row in ledger['runs'] if row['repeat'] == repeat]
    require(rows and len({row['name'] for row in rows}) == len(rows), 'Empty or duplicate export round')
    export_id = f'{stage}_r{repeat}_{uuid.uuid4().hex}'
    payload = {}

    def add(path, name=None):
        reject_symlink_ancestors(path)
        require(path.is_file() and not path.is_symlink(), 'Missing or unsafe export file: ' + str(path))
        name = safe_name(name or path.relative_to(root).as_posix())
        key = name.casefold()
        require(key not in {item.casefold() for item in payload}, 'Duplicate archive member: ' + name)
        payload[name] = path

    for row in rows:
        name = row['name']
        require(re.fullmatch(r'autodl_' + stage + r'_[A-Za-z0-9_.-]+', name) is not None, 'Invalid stage run name')
        require(name in planned and {k: v for k, v in row.items() if k != 'result'} == planned[name], 'Ledger run differs from plan')
        run = within(root, 'runs/active/' + name)
        require(read(run / 'result.json') == row['result'] and row['result']['status'] not in ('running', 'pending'),
                'Run is not a terminal ledger result: ' + name)
        for path in regular_files(run):
            add(path)
    for name in ('autodl_bundle.json', 'builds/autodl-active/manifest.json', ledger_name):
        add(within(root, name))
    # Additive orchestrators are outside the frozen bundle and need their own
    # transferred bytes so the analyzer can verify the recorded driver hash.
    for name in ('tools/active/autodl_job.py', 'tools/active/autodl_window_chunks.py',
                 'tools/active/autodl_transfer.py'):
        path = within(root, name)
        if path.is_file():
            add(path)
    for path in sorted((root / 'configs/active').glob('autodl_*.json')):
        add(path)
    build = within(root, 'builds/autodl-active')
    for path in regular_files(build):
        if path.suffix == '.log' or (path.parent == build and path.suffix == '.json' and path.name != 'manifest.json'):
            add(path)
    build_manifest = read(build / 'manifest.json')
    if stage == 'smoke' or include_binaries:
        for group in ('binaries', 'validator_binaries'):
            for record in build_manifest.get(group, {}).values():
                path = within(root, record['path'])
                require(sha(path) == record['sha256'], 'Built binary changed: ' + str(path))
                add(path)
    driver_file = driver_file or (root.parent / 'driver/job.py')
    jobs_root = jobs_root or (root.parent / 'jobs')
    external = {'driver_included': driver_file.is_file(), 'jobs_included': jobs_root.is_dir()}
    if driver_file.is_file():
        add(driver_file, f'transfers/{export_id}/external/driver/job.py')
    if jobs_root.is_dir():
        for path in regular_files(jobs_root):
            add(path, f'transfers/{export_id}/external/jobs/' + path.relative_to(jobs_root).as_posix())
    files = [{'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(payload.items())]
    manifest = {'schema': 1, 'export_id': export_id, 'export_root': str(root), 'stage': stage, 'repeat': repeat,
                'archive_relative_path': output.relative_to(root).as_posix(), 'ledger_path': ledger_name,
                'ledger_sha256': sha(ledger_path), 'selected_runs': rows, 'files': files, 'external': external}
    control = encoded(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation means interruption leaves an explicit incomplete file;
    # no existing archive is overwritten or silently treated as complete.
    with output.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1) as archive:
        info = tarfile.TarInfo(CONTROL); info.size = len(control)
        archive.addfile(info, io.BytesIO(control))
        for entry in files:
            path = payload[entry['path']]
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Input changed before export')
            info = tarfile.TarInfo(entry['path']); info.size = entry['bytes']; info.mode = path.stat().st_mode & 0o777
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    require(all(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
                for entry in files for path in [payload[entry['path']]]), 'Inputs changed during export; archive is invalid')
    result = {'archive': str(output), 'archive_sha256': sha(output), 'archive_bytes': output.stat().st_size,
              'transfer_manifest_sha256': hashlib.sha256(control).hexdigest(), 'transfer_manifest': manifest}
    write_new(sidecar, result)
    return {k: v for k, v in result.items() if k != 'transfer_manifest'} | {'export_record': str(sidecar), 'runs': len(rows)}


def verify_archive(archive_path, expected_sha256):
    require(re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256) is not None, 'Expected SHA256 is required')
    require(sha(archive_path) == expected_sha256.lower(), 'Archive SHA256 mismatch')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [safe_name(member.name) for member in members]
        require(len({name.casefold() for name in names}) == len(names), 'Duplicate/case-colliding archive members')
        require(all(member.isfile() for member in members), 'Non-regular tar member rejected')
        require(names.count(CONTROL) == 1, 'Missing unique transfer manifest')
        control = archive.extractfile(CONTROL).read()
        manifest = json.loads(control)
        expected = {safe_name(row['path']): row for row in manifest['files']}
        require(CONTROL not in expected and len(expected) == len(manifest['files'])
                and set(names) == set(expected) | {CONTROL}, 'Archive inventory mismatch')
        for member in members:
            if member.name == CONTROL:
                continue
            row = expected[member.name]
            require(member.size == row['bytes'] and stream_sha(archive.extractfile(member)) == row['sha256'],
                    'Payload byte count/SHA256 mismatch: ' + member.name)
    return manifest, hashlib.sha256(control).hexdigest()


def receive(archive_path, expected_sha256, destination):
    reject_symlink_ancestors(destination.absolute())
    archive_path = archive_path.resolve(); destination = destination.resolve()
    manifest, manifest_sha = verify_archive(archive_path, expected_sha256)
    export_id = safe_name(manifest['export_id'])
    require('/' not in export_id, 'Invalid export ID')
    receipt_path = within(destination, f'transfers/{export_id}/receipt.json')
    require(not receipt_path.exists(), 'This export already has a receipt')
    for row in manifest['selected_runs']:
        require(not within(destination, 'runs/active/' + safe_name(row['name'])).exists(), 'Run already exists; no run data may be overwritten')
    writes = []
    # Verify every member and preflight every destination before extracting any.
    with tarfile.open(archive_path, 'r:gz') as archive:
        for entry in manifest['files']:
            path = within(destination, entry['path'])
            if path.exists():
                require(path.is_file(), 'Destination is not a regular file')
                if sha(path) == entry['sha256']:
                    continue
                require(entry['path'] in LEDGERS, 'Existing non-ledger data differs: ' + entry['path'])
                old, new = read(path), json.load(archive.extractfile(entry['path']))
                require(all(old.get(k) == new.get(k) for k in ('source_digest', 'candidate_sha256', 'plan_sha256'))
                        and new['runs'][:len(old['runs'])] == old['runs'], 'Ledger is not a matching append-only update')
            writes.append((path, entry))
        for path, entry in writes:
            path.parent.mkdir(parents=True, exist_ok=True)
            within(destination, entry['path'])
            mode = 'wb' if entry['path'] in LEDGERS and path.exists() else 'xb'
            with path.open(mode) as target, archive.extractfile(entry['path']) as source:
                shutil.copyfileobj(source, target, 1024 * 1024)
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Post-extraction verification failed')
    saved_manifest = within(destination, f'transfers/{export_id}/transfer_manifest.json')
    write_new(saved_manifest, manifest)
    receipt = {'schema': 1, 'validation_complete': True, 'export_id': export_id,
               'archive_sha256': expected_sha256.lower(), 'transfer_manifest_sha256': manifest_sha,
               'transfer_manifest': manifest, 'destination': str(destination),
               'received_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'files_verified': len(manifest['files'])}
    write_new(receipt_path, receipt)
    return {'receipt': str(receipt_path), 'archive_sha256': expected_sha256.lower(), 'files_verified': len(manifest['files'])}


def prune(root, receipt_path):
    require(platform.system() == 'Linux', 'Pruning is remote Linux-only')
    root = package_root(root); receipt = read(receipt_path)
    require(receipt.get('validation_complete') is True, 'Receipt does not certify a complete receive')
    manifest = receipt['transfer_manifest']
    require(receipt.get('files_verified') == len(manifest['files']), 'Receipt inventory verification is incomplete')
    require(manifest['export_root'] == str(root) and manifest['export_id'] == receipt['export_id'], 'Receipt belongs to another export/root')
    archive = within(root, manifest['archive_relative_path'])
    export_record = read(Path(str(archive) + '.export.json'))
    require(export_record['transfer_manifest'] == manifest
            and export_record['archive_sha256'] == receipt['archive_sha256'] == sha(archive)
            and export_record['transfer_manifest_sha256'] == receipt['transfer_manifest_sha256']
            == hashlib.sha256(encoded(manifest)).hexdigest(), 'Receipt/export/archive identity mismatch')
    ledger = read(within(root, manifest['ledger_path']))
    current = {row['name']: row for row in ledger['runs']}
    targets = []
    active = within(root, 'runs/active').resolve()
    for row in manifest['selected_runs']:
        name = row['name']
        require(current.get(name) == row, 'Current ledger no longer matches exported run')
        trace = within(root, 'runs/active/' + safe_name(name) + '/trace')
        require(trace.resolve().parent.parent == active and trace.name == 'trace', 'Trace target is outside this active run')
        prefix = 'runs/active/' + name + '/trace/'
        expected = {entry['path']: entry for entry in manifest['files'] if entry['path'].startswith(prefix)}
        if not trace.exists():
            continue
        files = regular_files(trace)
        require({p.relative_to(root).as_posix() for p in files} == set(expected), 'Trace inventory changed after export')
        require(all(p.stat().st_size == expected[p.relative_to(root).as_posix()]['bytes']
                    and sha(p) == expected[p.relative_to(root).as_posix()]['sha256'] for p in files),
                'Trace data changed after export')
        targets.append(trace.resolve())
    lock = active / '.gpu.lock'
    if lock.exists():
        require(read(lock).get('name') not in {row['name'] for row in manifest['selected_runs']}, 'Selected run still owns GPU lock')
    # All targets and content are validated before the first deletion.
    removed = []
    for trace in targets:
        require(trace.parent.parent == active and trace.name == 'trace', 'Unsafe prune target')
        shutil.rmtree(trace)
        removed.append(str(trace))
    result = {'export_id': manifest['export_id'], 'archive_retained': str(archive), 'removed_trace_directories': removed,
              'receipt_sha256': sha(receipt_path), 'pruned_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_new(within(root, f'transfers/{manifest["export_id"]}/prune_{uuid.uuid4().hex}.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export'); exp.add_argument('--root', type=Path, default=ROOT)
    exp.add_argument('--stage', choices=['smoke', 'window'], required=True)
    exp.add_argument('--repeat', type=int); exp.add_argument('--output', type=Path, required=True)
    exp.add_argument('--driver-file', type=Path); exp.add_argument('--jobs-root', type=Path)
    exp.add_argument('--include-binaries', action='store_true')
    rec = sub.add_parser('receive'); rec.add_argument('--archive', type=Path, required=True)
    rec.add_argument('--expected-sha256', required=True); rec.add_argument('--destination', type=Path, required=True)
    delete = sub.add_parser('prune'); delete.add_argument('--root', type=Path, default=ROOT)
    delete.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        result = export(args.root, args.stage, args.repeat, args.output, args.driver_file, args.jobs_root, args.include_binaries)
    elif args.command == 'receive':
        result = receive(args.archive, args.expected_sha256, args.destination)
    else:
        result = prune(args.root, args.receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
