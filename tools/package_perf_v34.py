"""Package frozen source trees and minimal AutoDL helpers, preserving history."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
files = {}
for name in ['robust_port_v32', 'perf_v34']:
    manifest = json.loads((ROOT / f'manifests/{name}.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        files[entry['path']] = path
for relative in ['manifests/robust_port_v32.json', 'manifests/perf_v34.json',
                 'tools/run_robust_port.py', 'tools/run_perf_v34.py', 'tools/preflight.py',
                 'tools/benchmark_perf_v34.py', 'tools/benchmark_local_preconditioners_v33.py',
                 'tools/autodl_v34_build.sh']:
    files[relative] = ROOT / relative
target = ROOT / 'bundles/autodl_perf_v34_20261003.tar.gz'
assert not target.exists(), 'Preserve existing upload'
with tarfile.open(target, 'w:gz', compresslevel=6) as archive:
    for relative, path in sorted(files.items()):
        if relative.endswith('.sh'):
            data = path.read_bytes().replace(b'\r\n', b'\n')
            entry = tarfile.TarInfo(relative); entry.size = len(data); entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(data))
        else:
            archive.add(path, arcname=relative, recursive=False)
print(json.dumps({'path': str(target), 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'files': len(files)}))
