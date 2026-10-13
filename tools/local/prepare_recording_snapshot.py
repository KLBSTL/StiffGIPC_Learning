"""Copy an immutable current source/input snapshot for logging-only edits."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/local'), str(ROOT / 'tools/bench')]
from local_identity import record, tree

DEST = ROOT / 'runs/recording_build_20261011/source'

if __name__ == '__main__':
    assert not DEST.exists(), 'Never overwrite a source snapshot'
    DEST.mkdir(parents=True)
    for name in ('StiffGIPC', 'Assets', 'MeshProcess', 'tests'):
        shutil.copytree(ROOT / name, DEST / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'CMakeLists.txt', DEST / 'CMakeLists.txt')
    (DEST / 'tools').mkdir()
    shutil.copyfile(ROOT / 'tools/build_windows.py', DEST / 'tools/build_windows.py')
    files = sum((tree(DEST / name) for name in ('StiffGIPC', 'Assets', 'MeshProcess', 'tests')), []) + [record(DEST / 'CMakeLists.txt')]
    for item in files:
        relative = Path(item['path']).relative_to(DEST)
        original = record(ROOT / relative)
        assert original['sha256'] == item['sha256'] and original['bytes'] == item['bytes']
    (DEST.parent / 'original_copy.json').write_text(json.dumps(dict(source=str(ROOT), copy=str(DEST), files=files), indent=2), encoding='utf8')
    print(json.dumps(dict(copied_files=len(files), destination=str(DEST))))
