"""Hash-check and extract a result archive into a fresh, bounded directory."""
import argparse
import hashlib
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('target', type=Path)
parser.add_argument('sha256')
args = parser.parse_args()
assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == args.sha256
target = args.target.resolve()
assert not target.exists()
with tarfile.open(args.archive) as tar:
    members = tar.getmembers()
    assert all(m.isfile() and (target / m.name).resolve().is_relative_to(target) for m in members)
    target.mkdir(parents=True)
    tar.extractall(target, members=members, filter='data')
print(f'Hash verified; extracted {len(members)} files into {target}')
