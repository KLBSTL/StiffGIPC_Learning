"""AutoDL project-only cleanup: verified duplicate bundles and lossless old traces."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
REPORT = ROOT / 'cleanup_20261003.json'

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def sha(path):
    with path.open('rb') as f:
        return digest(f)

def checked(path):
    resolved = path.resolve(strict=True)
    assert ROOT in resolved.parents and not path.is_symlink(), str(path)
    return resolved

def main():
    assert not REPORT.exists(), 'Preserve prior cleanup receipt'
    result = {'root': str(ROOT), 'free_before': shutil.disk_usage(ROOT).free,
              'deleted_duplicates': [], 'archived_traces': [], 'deleted_objects': []}
    def checkpoint():
        result['free_after'] = shutil.disk_usage(ROOT).free
        REPORT.write_text(json.dumps(result, indent=2))
    known = json.loads((ROOT / 'cleanup_local_copies_20261003.json').read_text())
    for p in sorted(ROOT.glob('*.tar.gz')) + sorted(ROOT.glob('*/*.tar.gz')):
        checked(p)
        candidates = known.get(p.name, [])
        if not candidates:
            continue
        value = sha(p)
        match = next((x for x in candidates if x['sha256'] == value), None)
        if match:
            result['deleted_duplicates'].append({'path': str(p), 'bytes': p.stat().st_size,
                                                 'sha256': value, 'retained_local': match['path']})
            p.unlink()
            checkpoint()
    # Retain executables, libraries, source, manifests, logs and all current experiments.
    for version in ['v26', 'v27', 'v32_20261002', 'v34_20261003', 'v36_20261003', 'v37_20261003']:
        for p in (ROOT / version / 'builds').rglob('*.o'):
            checked(p)
            result['deleted_objects'].append({'path': str(p), 'bytes': p.stat().st_size})
            p.unlink()
    checkpoint()
    # Old binary trajectories remain recoverable byte-for-byte. Keep CSV/JSON/logs in place.
    for version in ['v17', 'v18']:
        for run in sorted((ROOT / 'history_runs' / version / 'autodl').iterdir()):
            trace = run / 'trace'
            if not trace.is_dir():
                continue
            files = sorted(trace.rglob('*.bin'))
            total = sum(p.stat().st_size for p in files)
            if total < 100 * 1024 * 1024:
                continue
            for p in files:
                checked(p)
            archive = trace / 'historical_binary_trace.tar.gz'
            assert not archive.exists(), str(archive)
            assert shutil.disk_usage(ROOT).free > total + 768 * 1024 * 1024
            records = [{'path': p.relative_to(trace).as_posix(), 'bytes': p.stat().st_size,
                        'sha256': sha(p)} for p in files]
            with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
                for p in files:
                    tar.add(p, arcname=p.relative_to(trace).as_posix(), recursive=False)
            with tarfile.open(archive, 'r:gz') as tar:
                members = tar.getmembers()
                assert len(members) == len(records)
                for member, entry in zip(members, records):
                    assert member.isfile() and member.name == entry['path'] and member.size == entry['bytes']
                    with tar.extractfile(member) as f:
                        assert digest(f) == entry['sha256']
            receipt = {'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size,
                       'original_bytes': total, 'files': records,
                       'restore': 'From this trace directory: tar -xzf historical_binary_trace.tar.gz'}
            (trace / 'historical_binary_trace.manifest.json').write_text(json.dumps(receipt, indent=2))
            # Re-check originals after archive verification; no recursive directory deletion.
            for p, entry in zip(files, records):
                checked(p)
                assert sha(p) == entry['sha256']
            for p in files:
                p.unlink()
            result['archived_traces'].append({'run': str(run), 'original_bytes': total,
                                               'archive_bytes': archive.stat().st_size,
                                               'sha256': receipt['sha256'], 'files': len(files)})
            checkpoint()
            print(json.dumps(result['archived_traces'][-1]), flush=True)
    checkpoint()
    print(json.dumps({'free_before': result['free_before'], 'free_after': result['free_after'],
                      'archives': len(result['archived_traces']), 'duplicates': len(result['deleted_duplicates'])}), flush=True)

if __name__ == '__main__':
    main()
