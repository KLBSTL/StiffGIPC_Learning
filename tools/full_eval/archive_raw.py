"""Archive finished run payloads; prune only after caller verifies local download."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from linux_runner import child, require, read, sha, write_new


def owned_run(root, name):
    root = Path(root).resolve()
    run = child(root, name)
    require(run.is_relative_to(root / 'runs') and len(run.relative_to(root).parts) == 3,
            'Select one completed run below runs/session/name')
    require((run / 'result.json').is_file() and (run / 'evidence.json').is_file(),
            'Runner has not finalized this output')
    require(read(run / 'result.json').get('status') not in ('running', None), 'Run still active')
    return run


def raw_rows(run):
    rows = read(run / 'evidence.json')['files']
    require(len({r['path'] for r in rows}) == len(rows), 'Duplicate evidence paths')
    return [r for r in rows if r['path'] == 'final.bin' or
            (r['path'].startswith('trace/') and Path(r['path']).suffix == '.bin')]


def verify_payload(archive, rows):
    expected = {r['path']: r for r in rows}
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        require(len(members) == len(expected) and {m.name for m in members} == expected.keys(),
                'Archive payload list differs from bound raw evidence')
        for member in members:
            row = expected[member.name]
            require(member.isfile() and member.size == row['bytes'], 'Unsafe/incomplete archive member')
            with tf.extractfile(member) as f:
                digest = hashlib.sha256()
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == row['sha256'], 'Archive member hash differs')


def prepare(root, run_name, archive_name):
    root = Path(root).resolve()
    run = owned_run(root, run_name)
    archive = child(root, archive_name)
    require(archive.parent == run.parent / 'archives' and archive.name == run.name + '.tar.gz',
            'Archive must be this session archives/run-name.tar.gz')
    require(not archive.exists() and not (run / 'archive_ready.json').exists(), 'Archive already exists')
    rows = raw_rows(run)
    require(rows, 'No bound raw payload to archive')
    for row in rows:
        p = child(run, row['path'])
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                'Raw changed before archive')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tf:
        for row in rows:
            tf.add(child(run, row['path']), arcname=row['path'], recursive=False)
    verify_payload(archive, rows)
    value = {'schema': 'full_eval_raw_archive_ready.v1', 'run_path': run_name,
             'archive_path': archive_name, 'archive_sha256': sha(archive),
             'archive_bytes': archive.stat().st_size, 'evidence_sha256': sha(run / 'evidence.json'),
             'raw_paths': [r['path'] for r in rows], 'raw_bytes': sum(r['bytes'] for r in rows)}
    write_new(run / 'archive_ready.json', value)
    return value


def persist(root, run_name, downloaded_sha256, local_archive_name):
    root = Path(root).resolve()
    run = owned_run(root, run_name)
    require(not (run / 'archive_receipt.json').exists(), 'Receipt already exists; no repeated pruning')
    ready = read(run / 'archive_ready.json')
    require(ready['schema'] == 'full_eval_raw_archive_ready.v1' and ready['run_path'] == run_name,
            'Archive preparation identity differs')
    require(isinstance(local_archive_name, str) and local_archive_name
            and '/' not in local_archive_name and '\\' not in local_archive_name,
            'Expected a local archive filename')
    archive = child(root, ready['archive_path'])
    require(archive.parent == run.parent / 'archives' and archive.name == run.name + '.tar.gz',
            'Prepared archive outside selected session')
    require(archive.is_file() and archive.stat().st_size == ready['archive_bytes']
            and sha(archive) == ready['archive_sha256'] == downloaded_sha256,
            'Caller download/remote archive hashes do not agree')
    require(sha(run / 'evidence.json') == ready['evidence_sha256'], 'Evidence changed after archive')
    rows = raw_rows(run)
    require([r['path'] for r in rows] == ready['raw_paths'], 'Prepared raw removal list changed')
    verify_payload(archive, rows)
    # Revalidate every exact file before any mutation. The tar remains both on
    # the remote and at the caller; no metadata, source, old run or tree deleted.
    for row in rows:
        p = child(run, row['path'])
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                'Raw changed before pruning')
    receipt = {'schema': 'full_eval_raw_archive.v1', 'archive_path': ready['archive_path'],
               'archive_sha256': ready['archive_sha256'], 'archive_bytes': ready['archive_bytes'],
               'evidence_sha256': ready['evidence_sha256'], 'download_verified': True,
               'local_archive_name': local_archive_name, 'removed_raw_paths': ready['raw_paths']}
    write_new(run / 'archive_receipt.json', receipt)
    for row in rows:
        child(run, row['path']).unlink()
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--run', required=True); a.add_argument('--archive', required=True)
    a = sub.add_parser('persist'); a.add_argument('--run', required=True)
    a.add_argument('--downloaded-sha256', required=True); a.add_argument('--local-archive-name', required=True)
    args = p.parse_args()
    value = (prepare(args.root, args.run, args.archive) if args.command == 'prepare' else
             persist(args.root, args.run, args.downloaded_sha256, args.local_archive_name))
    print(json.dumps(value, allow_nan=False))


if __name__ == '__main__':
    main()
