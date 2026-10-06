"""Verify a complete archive; skip only byte-identical existing files."""
import argparse
import hashlib
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path);p.add_argument('--sha256',required=True)
a=p.parse_args();assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==a.sha256
root=a.destination.resolve();root.mkdir(parents=True,exist_ok=True)
written=matched=0
with tarfile.open(a.archive,'r:gz') as archive:
    for member in archive:
        assert member.isfile(), member.name
        target=(root/member.name).resolve();assert target.is_relative_to(root)
        data=archive.extractfile(member).read()
        if target.exists():
            assert target.read_bytes()==data, f'Existing file differs: {target}'
            matched+=1
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);written+=1
print({'archive_sha256_verified':True,'new_files':written,'identical_existing_files':matched})
