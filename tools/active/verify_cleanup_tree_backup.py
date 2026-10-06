"""Verify the downloaded report-only tree backup without extracting old paths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--expected-plan-sha', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open('rb') as stream:
        assert digest(stream) == args.expected_sha
    found, controls, names = {}, {}, set()
    control_names = {'tree_backup_receipt.json', 'plan_v2.json.gz', 'plan_v2_summary.json'}
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
            assert member.isdir() or member.isfile()
            assert str(name) not in names
            names.add(str(name))
            if member.isdir():
                continue
            with archive.extractfile(member) as stream:
                if str(name) in control_names:
                    controls[str(name)] = stream.read()
                else:
                    found[str(name)] = (member.size, digest(stream))
    assert set(controls) == control_names
    receipt = json.loads(controls['tree_backup_receipt.json'])
    assert receipt['complete'] and receipt['plan_sha256'] == args.expected_plan_sha
    assert hashlib.sha256(gzip.decompress(controls['plan_v2.json.gz'])).hexdigest() == args.expected_plan_sha
    root = PurePosixPath(receipt['reports_root']).parent
    expected = {str(PurePosixPath(row['backup']).relative_to(root)): (row['bytes'], row['sha256'])
                for row in receipt['reports']}
    assert len(expected) == len(receipt['reports']) and found == expected
    output = {'complete': True, 'archive_sha256': args.expected_sha,
              'plan_sha256': args.expected_plan_sha, 'files_verified': len(found),
              'bytes_verified': sum(value[0] for value in found.values()),
              'remote_receipt_sha256': hashlib.sha256(controls['tree_backup_receipt.json']).hexdigest()}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
