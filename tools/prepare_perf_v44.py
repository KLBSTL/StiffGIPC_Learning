"""Freeze and package only the isolated v44 source and helper scripts."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'bundles/autodl_perf_v44_20261004.tar.gz'
manifest_path = ROOT / 'manifests/perf_v44.json'
assert not target.exists() and not manifest_path.exists()
files = []
for tree in ['sources/stiff_base', 'sources/stiff_perf_v44']:
    for p in sorted((ROOT / tree).rglob('*')):
        if p.is_file() and not any(part in {'Output', 'sorted_mesh', '__pycache__', '.git'} for part in p.relative_to(ROOT).parts):
            files.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size})
manifest = {'implementation_version': 'v44-stage-diagnostic',
            'base_commit': 'bb2849a7b292099581907937860d96ecfdf42588', 'files': files,
            'source_digest': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'binaries': {}}
manifest_path.write_text(json.dumps(manifest, indent=2))
paths = [ROOT / f['path'] for f in files if f['path'].startswith('sources/stiff_perf_v44/')]
paths += [manifest_path] + list((ROOT / 'tools').glob('*.py')) + [ROOT / 'tools/autodl_v44_build.sh']
with tarfile.open(target, 'w:gz') as archive:
    for p in paths:
        relative = p.relative_to(ROOT).as_posix()
        if p.suffix == '.sh':
            data = p.read_bytes().replace(b'\r\n', b'\n')
            entry = tarfile.TarInfo(relative); entry.size = len(data); entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(data))
        else:
            archive.add(p, arcname=relative, recursive=False)
print(json.dumps({'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size}))
