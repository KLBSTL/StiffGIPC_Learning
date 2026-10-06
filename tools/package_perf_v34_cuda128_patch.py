"""Preserve Windows freeze; package a standalone fixture syntax fix for CUDA12.8."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'manifests/perf_v34.json').read_text())
for entry in manifest['files']:
    path = entry['path'].replace('sources/stiff_perf_v34/', 'sources/stiff_perf_v34_cuda128/')
    data = (ROOT / path).read_bytes()
    if path != 'sources/stiff_perf_v34_cuda128/tests/diag_fused_update_tests.cu':
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    entry.update(path=path, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
manifest['source_digest'] = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
manifest['binaries'] = {}
manifest['revision_note'] = 'Standalone fixture uses explicit std::array for CUDA12.8; numerical sources identical to Windows v34.'
target = ROOT / 'manifests/perf_v34_cuda128.json'
assert not target.exists()
target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
files = ['sources/stiff_perf_v34_cuda128/tests/diag_fused_update_tests.cu',
         'manifests/perf_v34_cuda128.json', 'tools/autodl_v34_build.sh',
         'tools/run_perf_v34.py', 'tools/benchmark_perf_v34.py']
bundle = ROOT / 'bundles/autodl_perf_v34_cuda128_patch.tar.gz'
assert not bundle.exists()
with tarfile.open(bundle, 'w:gz') as archive:
    for relative in files:
        path = ROOT / relative
        if relative.endswith('.sh'):
            data = path.read_bytes().replace(b'\r\n', b'\n')
            entry = tarfile.TarInfo(relative); entry.size = len(data); entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(data))
        else:
            archive.add(path, arcname=relative, recursive=False)
print(json.dumps({'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                  'bytes': bundle.stat().st_size, 'source_digest': manifest['source_digest']}))
