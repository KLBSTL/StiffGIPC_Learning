"""Verified archive extraction with contained regular files and directories only."""
import argparse
import hashlib
from pathlib import Path
import tarfile
p=argparse.ArgumentParser()
p.add_argument('archive', type=Path)
p.add_argument('destination', type=Path)
p.add_argument('--sha256', required=True)
a=p.parse_args()
assert hashlib.file_digest(a.archive.open('rb'), 'sha256').hexdigest()==a.sha256
root=a.destination.resolve();root.mkdir(parents=True, exist_ok=True)
written=0
with tarfile.open(a.archive, 'r:gz') as archive:
    for member in archive:
        target=(root/member.name).resolve()
        assert target.is_relative_to(root)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True);continue
        assert member.isfile(), member.name
        data=archive.extractfile(member).read()
        if target.exists():
            assert target.read_bytes()==data, target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data);written+=1
print({'sha256_verified': True, 'new_files': written})
