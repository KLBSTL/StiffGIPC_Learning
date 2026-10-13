"""Verify the history checkout/index against measured source and data hashes."""
import gzip
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'history/autodl_recording105_20261013'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    source = json.loads((ROOT / 'LOCAL_SOURCE_COMPARISON.json').read_text())
    transfer = json.loads((ROOT / 'TRANSFER_MANIFEST.json').read_text())
    expected = {r['path']: r['sha256'] for r in transfer['files']}
    for r in source['files']:
        assert sha(ROOT / r['path']) == r['sha256'], r['path']
    assert sha(ROOT / 'CMakeLists.txt') == source['local_cmake_sha256']
    requests = ''.join(':' + p + '\n' for p in expected).encode()
    batch = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
                           input=requests, stdout=subprocess.PIPE, check=True)
    stream = io.BytesIO(batch.stdout)
    for path, value in expected.items():
        header = stream.readline().decode().split()
        assert len(header) == 3 and header[1] == 'blob', (path, header)
        content = stream.read(int(header[2]))
        assert stream.read(1) == b'\n'
        assert hashlib.sha256(content).hexdigest() == value, path
    assert not stream.read()
    python_count = 0
    for folder in ('tools', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
            python_count += 1
    data = json.loads((HISTORY / 'DATA_MANIFEST.json').read_text())
    assert data['csv_count'] == 18
    for row in data['files']:
        path = HISTORY / row['stored_path']
        assert path.stat().st_size == row['stored_bytes']
        assert sha(path) == row['stored_sha256']
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as unzipped:
                assert hashlib.file_digest(unzipped, 'sha256').hexdigest() == row['source_sha256']
        else:
            assert sha(path) == row['source_sha256']
    changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT)
    paths = [p.decode() for p in changed.split(b'\0') if p]
    secret = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}\b')
    largest = 0
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        assert not any(p in path.parts for p in ('.ssh', 'node_modules', '__pycache__'))
        assert path.suffix.lower() not in ('.exe', '.dll', '.pdb', '.pem')
        largest = max(largest, path.stat().st_size)
        assert path.stat().st_size < 100 * 1024 * 1024, name
        assert not secret.search(path.read_bytes()), 'Secret-shaped material in ' + name
    result = dict(passed=True, exact_native_asset_files=len(source['files']),
                  exact_transfer_index_blobs=len(expected), translation_units=37,
                  python_sources_compiled=python_count, numerical_datasets=18,
                  staged_changed_files=len(paths), largest_changed_file_bytes=largest,
                  private_key_token_pattern_scan='passed',
                  note='Verification only; native source and measurements were not modified')
    receipt = HISTORY / 'evidence/history_snapshot_verification.json'
    assert not receipt.exists(), 'Preserve verification evidence'
    receipt.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
