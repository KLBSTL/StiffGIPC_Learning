"""Create a separate v34 identity without modifying older manifests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'manifests/perf_v34.json'
assert not target.exists(), 'Preserve the existing v34 manifest'
files = []
for tree in ['sources/stiff_base', 'sources/stiff_perf_v34']:
    for path in sorted((ROOT / tree).rglob('*')):
        if path.is_file() and not any(part in {'Output', 'sorted_mesh', '__pycache__', '.git'} for part in path.relative_to(ROOT).parts):
            files.append({'path': path.relative_to(ROOT).as_posix(),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
manifest = {'implementation_version': 'v34-fused-diag-update',
            'base_commit': 'bb2849a7b292099581907937860d96ecfdf42588', 'files': files,
            'source_digest': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'binaries': {}}
for tag in ['local-base', 'local-fused-v34']:
    path = ROOT / 'builds' / tag / 'Release/gipc.exe'
    manifest['binaries'][path.relative_to(ROOT).as_posix()] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'files': len(files), 'source_digest': manifest['source_digest']}))
