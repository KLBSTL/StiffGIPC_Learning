"""Export/verify/receive a bounded AutoDL evidence bundle, without SSH or GPU.

Only planned run metadata and named analysis artifacts are selected. Full raw
trajectories remain remote. Optional render inputs cover exactly the 27 panels;
they do not make this a complete CCD replay package or a certification.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import uuid

CONTROL = 'EVIDENCE_MANIFEST.json'
RUN_FILES = ('requested.json', 'build_manifest.json', 'config_validation.json', 'result.json',
             'process.json', 'run.log', 'timeCost.txt', 'output/scene.json', 'output/stats.json',
             'trace/frames.csv', 'trace/metadata.json')
CONFIGS = ('retest', 'smoke', 'window', 'audit', 'pilot', 'paired')
TOOLS = ('autodl_analysis_job.py', 'analyze_autodl_factor.py', 'analyze_autodl_factor_v2.py',
         'audit_autodl_factor.py', 'reaggregate_autodl_audit.py', 'render_autodl_factor.py',
         'analyze.py', 'config.py', 'validate_run.py', 'autodl_window_chunks.py',
         'autodl_linux.py', 'autodl_job.py', 'autodl_transfer.py', 'render_factor_windows.py',
         'autodl_collect_evidence.py')
TERMINAL_RUN = {'completed', 'failed', 'launcher_failed', 'timeout', 'disk_reserve',
                'memory_budget', 'foreign_gpu_load', 'incomplete_or_nonfinite',
                'binary_changed_during_run', 'configuration_failed'}
TERMINAL_JOB = {'completed', 'failed', 'completed_with_findings'}
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stream_sha(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\x00' not in name,
            'Unsafe member name: ' + str(name))
    require(not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')),
            'Unsafe member name: ' + name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{k}{n}' for k in ('COM', 'LPT') for n in range(1, 10)}
    require(all(not p.endswith((' ', '.')) and p.split('.')[0].upper() not in reserved
                for p in PurePosixPath(name).parts), 'Nonportable member name: ' + name)
    return name


def no_links(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink path rejected: ' + str(part))


def child(root, name):
    path = root.joinpath(*PurePosixPath(safe_name(name)).parts)
    no_links(path)
    require(all(not parent.exists() or parent.is_dir() for parent in path.parents),
            'Existing parent is not a directory: ' + str(path))
    require(path.resolve().is_relative_to(root.resolve()), 'Path escaped package root')
    return path


def identity(path):
    no_links(path)
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode), 'Expected regular file: ' + str(path))
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def stable_hash(path):
    before = identity(path); digest = sha(path)
    require(identity(path) == before, 'File changed while hashing: ' + str(path))
    return before, digest


def select(root, include_render_inputs):
    payload = {}; optional_logs = set(); runs = {}; source_checks = {}

    def add(name, log_optional=False, expected_sha=None):
        path = child(root, name); require(path.is_file(), 'Required evidence missing: ' + name)
        require(name.casefold() not in {p.casefold() for p in payload if p != name}, 'Case-colliding evidence names')
        payload[name] = path
        if log_optional: optional_logs.add(name)
        if expected_sha is not None:
            require(sha(path) == expected_sha, 'Recorded artifact identity differs: ' + name)
            source_checks[name] = expected_sha

    add('autodl_bundle.json'); add('builds/autodl-active/manifest.json')
    for stage, count in [('smoke', 18), ('window', 54)]:
        plan_name = f'configs/active/autodl_{stage}.json'
        ledger_name = f'reports/active/AUTODL_{stage.upper()}_BATCH.json'
        plan, ledger = read(child(root, plan_name)), read(child(root, ledger_name))
        require(plan['stage'] == stage and ledger['plan_sha256'] == sha(child(root, plan_name)), 'Plan/ledger SHA mismatch')
        planned = {r['name']: r for r in plan['runs']}; recorded = {r['name']: r for r in ledger['runs']}
        require(len(plan['runs']) == len(planned) == len(ledger['runs']) == len(recorded) == count
                and set(planned) == set(recorded), 'Incomplete or duplicate planned batch: ' + stage)
        add(plan_name); add(ledger_name)
        for name, task in planned.items():
            require(re.fullmatch(r'autodl_' + stage + r'_[A-Za-z0-9_.-]+', name) is not None, 'Invalid planned run name')
            row = recorded[name]
            require({k: v for k, v in row.items() if k != 'result'} == task, 'Ledger task differs from plan: ' + name)
            result = read(child(root, f'runs/active/{name}/result.json'))
            require(result == row['result'] and result.get('status') in TERMINAL_RUN, 'Nonterminal or divergent run result: ' + name)
            require(name not in runs, 'Run appears in multiple stages'); runs[name] = task
            for relative in RUN_FILES:
                add(f'runs/active/{name}/{relative}', log_optional=relative in ('run.log', 'timeCost.txt'))
            if task['binary'] == 'active': add(f'runs/active/{name}/resolved_config.json')
    for name in CONFIGS: add(f'configs/active/autodl_{name}.json')
    for name in ('components.json', 'guards.json', 'active_configure.log', 'active_build.log',
                 'base_configure.log', 'base_build.log', 'validator_configure.log',
                 'validator_build.log', 'validator_selftest.log', 'components.json.log', 'guards.json.log'):
        add('builds/autodl-active/' + name, log_optional=name.endswith('.log'))
    for name in TOOLS: add('tools/active/' + name)
    require(sha(payload['tools/active/autodl_collect_evidence.py']) == sha(Path(__file__)),
            'Executing collector differs from the collector included in the package')
    add('tools/validator/diagnose_first_path.cpp')
    for name in ('AUTODL_FACTOR_WINDOW_ANALYSIS.json', 'AUTODL_FACTOR_WINDOW_ANALYSIS_V2.json'):
        add('reports/active/' + name)
    audit_dir = child(root, 'reports/active/AUTODL_FACTOR_CCD')
    require(audit_dir.is_dir(), 'Native CCD evidence directory missing')
    for path in sorted(audit_dir.iterdir()):
        no_links(path)
        if path.is_file() and path.suffix in ('.json', '.log'):
            add(path.relative_to(root).as_posix())  # CPU logs are never omitted.
    require('reports/active/AUTODL_FACTOR_CCD/audit.json' in payload, 'Original audit is missing')
    original_audit = read(audit_dir / 'audit.json')
    audited = {row['run']: row for row in original_audit['runs']}
    windows = {name for name in runs if name.startswith('autodl_window_')}
    require(len(original_audit['runs']) == len(audited) == 54 and set(audited) == windows,
            'Native audit does not cover the exact 54 window records')
    for name in sorted(windows):
        for suffix in ('.record.json', '.identity.json', '.stdout.log'):
            add('reports/active/AUTODL_FACTOR_CCD/' + name + suffix)
        if audited[name].get('native_ccd_sha256'):
            add('reports/active/AUTODL_FACTOR_CCD/' + name + '.ccd.json',
                expected_sha=audited[name]['native_ccd_sha256'])
    # Reaggregation output can be kept next to the audit or at reports/active.
    reaggregated = [name for name in payload if 'CCD/' in name and name.endswith('.json')
                   and ('reaggregate' in name.lower() or 'v2' in Path(name).stem.lower())]
    for path in sorted(child(root, 'reports/active').glob('AUTODL_FACTOR_CCD*.json')):
        add(path.relative_to(root).as_posix()); reaggregated.append(path.relative_to(root).as_posix())
    require(reaggregated, 'Corrected CCD reaggregation report is missing')
    render_name = 'reports/active/AUTODL_FACTOR_FIGURES/render_manifest.json'
    add(render_name); rendered = read(child(root, render_name))
    require(rendered.get('all_three_scenes_rendered') is True and len(rendered['scenes']) == 3,
            'Three finalized scene images are required')
    require({scene['scene_key'] for scene in rendered['scenes']}
            == {task['scene_key'] for task in runs.values()}, 'Rendered scene set differs from the planned scenes')
    panels = []
    for scene in rendered['scenes']:
        image_name = f"reports/active/AUTODL_FACTOR_FIGURES/autodl_factor_{scene['scene_key']}_final.png"
        require(Path(scene['image']).name == Path(image_name).name, 'Unexpected rendered image filename')
        add(image_name, expected_sha=scene['image_sha256'])
        for panel in scene['panels']:
            name = panel['run']; require(name in runs and name.startswith('autodl_window_'), 'Unplanned render panel')
            require(runs[name]['scene_key'] == scene['scene_key'], 'Panel belongs to a different scene')
            config = runs[name]['config']
            if isinstance(config, str): config = read(child(root, config))
            require(panel['frame'] == scene['frame'] == config['steps'], 'Render panel frame differs from complete window')
            panels.append({'run': name, 'frame': panel['frame']})
            if include_render_inputs:
                prefix = f'runs/active/{name}/'
                require(set(scene['initial_identity_sha256']) == {'topology.bin', 'state_0000.bin', 'boundary_types.bin',
                        'masses.bin', 'body_ids.bin', 'metadata.json', 'scene.json'}, 'Incomplete render topology/initial identity')
                for filename, digest in scene['initial_identity_sha256'].items():
                    require(filename in ('topology.bin', 'state_0000.bin', 'boundary_types.bin', 'masses.bin',
                                         'body_ids.bin', 'metadata.json', 'scene.json'), 'Unknown render input')
                    add(prefix + ('output/' if filename == 'scene.json' else 'trace/') + filename, expected_sha=digest)
                add(prefix + f"trace/state_{panel['frame']:04d}.bin", expected_sha=panel['state_sha256'])
    require(len(panels) == 27 and len({p['run'] for p in panels}) == 27, 'Expected exactly 27 distinct render panels')
    jobs = child(root, 'jobs'); require(jobs.is_dir(), 'Jobs evidence missing')
    for name in ('build', 'smoke', 'window1', 'window2', 'window3', 'analysis'):
        job = read(child(root, f'jobs/{name}.json'))
        require(job.get('status') in TERMINAL_JOB and job.get('ended_utc'), 'Job has not finalized: ' + name)
    for path in sorted(jobs.iterdir()):
        if path.name.lower().startswith('cleanup') or path.suffix not in ('.json', '.log'): continue
        no_links(path)
        if path.suffix == '.json':
            job = read(path)
            require(job.get('status') not in ('running', 'pending', 'starting'), 'Job evidence is still being written')
        add(path.relative_to(root).as_posix(), log_optional=False)
    return payload, optional_logs, {'planned_runs': len(runs), 'render_panels': panels,
                                    'recorded_artifact_sha256_checks': source_checks,
                                    'reaggregation_reports': reaggregated}


def export(root, output_dir, include_render_inputs=False, max_file=MAX_FILE, max_total=MAX_TOTAL):
    no_links(root.absolute()); root = root.resolve()
    require(root != Path(root.anchor) and (root / 'autodl_bundle.json').is_file(), 'Expected an unpacked package root')
    require(0 < max_file <= MAX_FILE and 0 < max_total <= MAX_TOTAL, 'Evidence size limits can only be reduced')
    no_links(output_dir.absolute()); output_dir = output_dir.resolve()
    require(not output_dir.exists(), 'A new export directory is required')
    payload, optional_logs, selection = select(root, include_render_inputs)
    records, omitted, identities = [], [], {}
    for name, path in sorted(payload.items()):
        ident, digest = stable_hash(path); size = ident[3]; identities[name] = ident
        row = {'path': name, 'bytes': size, 'sha256': digest}
        if size > max_file:
            require(name in optional_logs, 'Critical evidence exceeds per-file limit: ' + name)
            omitted.append(row | {'reason': 'oversized_optional_runtime_or_build_log'}); continue
        records.append(row)
    require(sum(row['bytes'] for row in records) <= max_total, 'Evidence exceeds total size limit')
    manifest = {'schema': 'autodl.evidence.v1', 'export_id': uuid.uuid4().hex,
                'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'root': str(root),
                'collector_sha256': sha(Path(__file__)), 'files': records, 'omitted_files': omitted,
                'payload_bytes': sum(row['bytes'] for row in records), 'limits': {'file': max_file, 'total': max_total},
                'raw_retained_remote': True, 'complete_ccd_replay_input': False,
                'includes_exact_render_inputs': include_render_inputs,
                'physical_certified': False, 'performance_certified': False, 'selection': selection}
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / 'evidence.tar.gz'; manifest_bytes = encoded(manifest)
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped, \
            tarfile.open(fileobj=zipped, mode='w|') as tar:
        info = tarfile.TarInfo(CONTROL); info.size = len(manifest_bytes); info.mode = 0o644
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for row in records:
            path = payload[row['path']]
            require(identity(path) == identities[row['path']], 'Evidence changed before packing')
            info = tarfile.TarInfo(row['path']); info.size = row['bytes']; info.mode = 0o644
            with path.open('rb') as stream: tar.addfile(info, stream)
            require(identity(path) == identities[row['path']], 'Evidence changed while packing')
    for name, path in payload.items():
        require(identity(path) == identities[name] and sha(path) == next(r['sha256'] for r in records + omitted if r['path'] == name),
                'Evidence changed during export: ' + name)
    digest = sha(archive)
    verify(archive, digest, max_total=max_total)
    (output_dir / CONTROL).write_bytes(manifest_bytes)
    receipt = {'archive': str(archive), 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
               'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'members': len(records),
               'payload_bytes': manifest['payload_bytes'], 'export_id': manifest['export_id']}
    with (output_dir / 'export.json').open('xb') as stream: stream.write(encoded(receipt))
    return receipt


def verify(archive, expected_sha256, max_total=MAX_TOTAL):
    no_links(archive.absolute()); before = identity(archive)
    require(sha(archive) == expected_sha256.lower(), 'Archive SHA256 mismatch')
    with tarfile.open(archive, 'r|gz') as tar:
        control = tar.next()
        require(control is not None and control.name == CONTROL and control.isfile() and not control.issparse()
                and 0 <= control.size <= 8 * 1024 * 1024, 'Missing/invalid bounded first manifest')
        manifest = json.load(tar.extractfile(control)); require(manifest.get('schema') == 'autodl.evidence.v1', 'Wrong evidence schema')
        require(isinstance(manifest.get('export_id'), str) and re.fullmatch('[0-9a-f]{32}', manifest['export_id']) is not None,
                'Invalid export identity')
        rows = manifest['files']; planned = {safe_name(r['path']): r for r in rows}
        require(CONTROL not in planned and len(planned) == len(rows)
                and len({name.casefold() for name in planned}) == len(planned), 'Duplicate/invalid declared members')
        require(not any(PurePosixPath(name).parts[0].casefold() == 'evidence_receipts' for name in planned),
                'Archive members cannot occupy the local receipt namespace')
        folded = {name.casefold() for name in planned}
        require(not any(str(parent).casefold() in folded for name in planned
                        for parent in PurePosixPath(name).parents if str(parent) != '.'), 'File/directory member collision')
        total = sum(r['bytes'] for r in rows)
        require(total == manifest['payload_bytes'] and 0 <= total <= max_total, 'Payload byte budget mismatch')
        for row in rows:
            require(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_FILE
                    and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None, 'Malformed payload record')
        seen = set()
        while (member := tar.next()) is not None:
            name = safe_name(member.name)
            require(name in planned and name not in seen and member.isfile() and not member.issparse(),
                    'Unplanned/duplicate/non-regular archive member')
            row = planned[name]; require(member.size == row['bytes'], 'Member byte size mismatch')
            require(stream_sha(tar.extractfile(member)) == row['sha256'], 'Member SHA256 mismatch: ' + name)
            seen.add(name)
        require(seen == set(planned), 'Archive is missing declared members')
    require(identity(archive) == before, 'Archive changed while verifying')
    return manifest


def receive(archive, expected_sha256, destination):
    manifest = verify(archive, expected_sha256)
    no_links(destination.absolute()); destination = destination.resolve()
    require(destination != Path(destination.anchor) and destination != Path.home().resolve(), 'Broad receive destination refused')
    targets = []
    for row in manifest['files']:
        path = child(destination, row['path'])
        if path.exists():
            require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                    'Existing local file conflicts; receive into a fresh directory: ' + row['path'])
        else: targets.append((row, path))
    # Whole archive and every existing target verified before writing anything.
    receipt = child(destination, 'evidence_receipts/' + manifest['export_id'] + '.json')
    require(not receipt.exists(), 'This evidence export already has a receipt')
    archive_before = identity(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        for row, path in targets:
            no_links(path); path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(row['path']) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Received file verification failed')
    require(identity(archive) == archive_before, 'Archive changed while receiving')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    value = {'schema': 'autodl.evidence.receipt.v1', 'export_id': manifest['export_id'],
             'archive_sha256': expected_sha256.lower(), 'received_files': len(targets),
             'existing_identical_files': len(manifest['files']) - len(targets),
             'destination': str(destination), 'verified': True,
             'complete_ccd_replay_input': False, 'raw_retained_remote': True}
    with receipt.open('xb') as stream: stream.write(encoded(value))
    return value | {'receipt': str(receipt)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('export'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True); p.add_argument('--include-render-inputs', action='store_true')
    p.add_argument('--max-file-bytes', type=int, default=MAX_FILE); p.add_argument('--max-total-bytes', type=int, default=MAX_TOTAL)
    for name in ('verify', 'receive'):
        p = commands.add_parser(name); p.add_argument('--archive', type=Path, required=True); p.add_argument('--expected-sha256', required=True)
        if name == 'receive': p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export': result = export(args.root, args.output_dir, args.include_render_inputs, args.max_file_bytes, args.max_total_bytes)
    elif args.command == 'receive': result = receive(args.archive, args.expected_sha256, args.destination)
    else:
        manifest = verify(args.archive, args.expected_sha256)
        result = {'verified': True, 'export_id': manifest['export_id'], 'members': len(manifest['files']), 'payload_bytes': manifest['payload_bytes']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
