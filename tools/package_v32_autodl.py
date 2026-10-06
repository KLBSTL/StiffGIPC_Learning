"""Package only the frozen sources and necessary benchmark helpers."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'manifests/robust_port_v32.json').read_text())
selected = []
for entry in manifest['files']:
    path = ROOT / entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
    selected.append(path)
selected += [ROOT / p for p in [
    'manifests/robust_port_v32.json', 'tools/run_robust_port.py',
    'tools/preflight.py', 'tools/check_robust_port_components.py',
    'tools/autodl_v32_build.sh', 'tools/benchmark_v32_autodl.py']]
out = ROOT / 'bundles/autodl_v32_20261002.tar.gz'
assert not out.exists(), 'Preserve existing upload; choose a new archive identity'
with tarfile.open(out, 'w:gz', compresslevel=6) as archive:
    for path in selected:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(json.dumps({'path': str(out), 'files': len(selected), 'bytes': out.stat().st_size,
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
