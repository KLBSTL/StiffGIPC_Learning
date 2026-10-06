"""Clean only this task's verified redundant files; preserve unique run evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SYSTEM = Path('/root/stiff_toi_cudagraph_20260929')
DATA = Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked(path, root):
    resolved = path.resolve(strict=True)
    boundary = root.resolve(strict=True)
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise RuntimeError(f'Cleanup target outside task: {path}')
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-archives', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    backups = json.loads(args.local_archives.read_text())['archives']
    report = {'apply': args.apply, 'before': {}, 'archives': [], 'objects': [], 'migrations': []}
    for root in (SYSTEM, DATA):
        report['before'][str(root)] = shutil.disk_usage(root)._asdict()
        candidates = list(root.glob('*.tar.gz')) + list(root.glob('*/*.tar.gz'))
        for path in candidates:
            checked(path, root)
            digest = sha256(path)
            if digest in backups and path.stat().st_size == backups[digest]['bytes']:
                report['archives'].append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
    # Keep the latest v26/v27 build caches. Historical executables and logs stay.
    versions = [SYSTEM] + [SYSTEM / ('v' + str(n)) for n in (15, 17, 18, 19, 20, 21, 22, 23, 24, 25)]
    for version in versions:
        build = version / 'builds'
        if not build.is_dir():
            continue
        for path in build.rglob('*.o'):
            checked(path, SYSTEM)
            if 'CMakeFiles' in path.parts and not path.is_symlink():
                report['objects'].append({'path': str(path), 'bytes': path.stat().st_size})
    for version in ('v17', 'v18'):
        source = SYSTEM / version / 'runs'
        destination = DATA / 'history_runs' / version
        if source.is_symlink():
            if source.resolve() != destination.resolve():
                raise RuntimeError('Unexpected existing history link')
            continue
        checked(source, SYSTEM)
        if destination.exists():
            raise RuntimeError(f'Archive destination already exists: {destination}')
        files = [p for p in source.rglob('*') if p.is_file()]
        report['migrations'].append({'source': str(source), 'destination': str(destination),
                                     'files': len(files), 'bytes': sum(p.stat().st_size for p in files)})
    if args.apply:
        required = sum(item['bytes'] for item in report['migrations']) + 3 * 1024**3
        if shutil.disk_usage(DATA).free < required:
            raise RuntimeError('Insufficient data disk reserve for verified migration')
        for group in ('archives', 'objects'):
            for item in report[group]:
                path = Path(item['path'])
                root = DATA if path.is_relative_to(DATA) else SYSTEM
                checked(path, root)
                if group == 'archives' and sha256(path) != item['sha256']:
                    raise RuntimeError('Archive changed since inventory')
                path.unlink()
        for item in report['migrations']:
            source, destination = Path(item['source']), Path(item['destination'])
            destination.mkdir(parents=True)
            checked(destination, DATA)
            subprocess.run(['rsync', '-a', str(source) + '/', str(destination) + '/'], check=True)
            verification = subprocess.run(['rsync', '-a', '--checksum', '--delete', '--dry-run',
                                           '--itemize-changes', str(source) + '/', str(destination) + '/'],
                                          check=True, capture_output=True, text=True)
            if verification.stdout.strip():
                raise RuntimeError('History copy verification failed; original retained')
            item['rsync_checksum_verified'] = True
            retired = source.parent / 'runs_cleanup_original'
            if retired.exists():
                raise RuntimeError('Unexpected retired history path')
            source.rename(retired)
            try:
                source.symlink_to(destination, target_is_directory=True)
            except BaseException:
                retired.rename(source)
                raise
            checked(retired, SYSTEM)
            if retired.name != 'runs_cleanup_original' or source.resolve() != destination.resolve():
                raise RuntimeError('Final history path verification failed')
            shutil.rmtree(retired)
            item['old_path_preserved_by_symlink'] = True
    report['after'] = {str(root): shutil.disk_usage(root)._asdict() for root in (SYSTEM, DATA)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({'apply': args.apply, 'backed_up_archives': len(report['archives']),
                      'archive_bytes': sum(i['bytes'] for i in report['archives']),
                      'object_files': len(report['objects']),
                      'object_bytes': sum(i['bytes'] for i in report['objects']),
                      'history_moves': len(report['migrations']), 'after': report['after']}))


if __name__ == '__main__':
    main()
