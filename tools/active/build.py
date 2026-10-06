"""Isolated active build and source/object/link provenance; never edits frozen trees.

Examples: build.py configure; build.py build --label clean --clean.
For a content-change check, use apply_patch to append a comment to one active
translation unit, build, revert the comment, build, and build once more with no
changes. Then verify-incremental validates those four immutable manifests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from build_support import (actual_compile_commands, compiler_dependencies, digest,
                           file_record, link_evidence, object_inventory, sha, source_map,
                           toolchain_evidence, project_include_closure)

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / 'sources/stiff_active'
BUILD = ROOT / 'builds/active'
HISTORY = BUILD / 'provenance'
CMAKE = Path('D:/computer/cmake/bin/cmake.exe')
MSBUILD = Path('D:/vs2022/MSBuild/Current/Bin/amd64/MSBuild.exe')
CONFIG = 'Release'


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checked(command: list[str], log: Path) -> None:
    with log.open('w', encoding='utf-8') as stream:
        stream.write('ARGV ' + json.dumps(command) + '\n'); stream.flush()
        status = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if status.returncode:
        lines = log.read_text(errors='replace').splitlines()
        print('\n'.join(lines[-45:]), file=sys.stderr)
        raise RuntimeError(f'command failed ({status.returncode}); log={log}')


def history_dir(label: str) -> Path:
    if not label or Path(label).name != label or label in ('.', '..'):
        raise ValueError('label must be a single path component')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = HISTORY / f'{stamp}_{label}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def frozen_check() -> dict:
    manifest = ROOT / 'manifests/perf_v54_local.json'
    old = json.loads(manifest.read_text())
    for record in old['files']:
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise RuntimeError(f'frozen source changed: {path}')
    for path, record in old['binaries'].items():
        if sha(ROOT / path) != record['sha256']:
            raise RuntimeError(f'frozen binary changed: {path}')
    return {'manifest': file_record(manifest, ROOT), 'source_digest': old['source_digest']}


def input_snapshot() -> list[dict]:
    paths = set(ACTIVE.rglob('*')) | set((ROOT / 'tools/active').glob('build*.py'))
    return [file_record(p, ROOT) for p in sorted(paths)
            if p.is_file() and '__pycache__' not in p.parts
            and (p.suffix in ('.cu', '.cpp', '.h', '.cuh', '.hpp', '.inl', '.inc', '.py')
                 or p.name == 'CMakeLists.txt')]


def archive_inputs(records: list[dict], exe: Path) -> dict:
    blobs = BUILD / 'blobs'
    blobs.mkdir(exist_ok=True)
    archive = blobs / f'overlay_{digest(records)}.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
            for record in records:
                package.write(ROOT / record['path'], record['path'])
    binary_blobs = []
    for binary in [exe, *sorted(exe.parent.glob('*.dll'))]:
        content_hash = sha(binary)
        saved = blobs / f'{content_hash}_{binary.name}'
        if not saved.exists():
            shutil.copyfile(binary, saved)
        if sha(saved) != content_hash:
            raise RuntimeError(f'content-addressed binary blob mismatch: {saved}')
        binary_blobs.append({'original': str(binary.relative_to(ROOT)), **file_record(saved, ROOT)})
    return {'overlay': file_record(archive, ROOT), 'overlay_source_digest': digest(records),
            'binaries': binary_blobs}


def configuration_inputs() -> dict:
    extensions = ('.cu', '.cpp', '.h', '.cuh', '.hpp', '.inl', '.inc')
    return {'cmake': file_record(ACTIVE / 'CMakeLists.txt', ROOT),
            'driver_sha256': sha(Path(__file__)),
            'active_paths': sorted(p.relative_to(ACTIVE).as_posix() for p in ACTIVE.rglob('*')
                                   if p.is_file() and p.suffix in extensions)}


def configure(label: str) -> Path:
    frozen = frozen_check()
    history = history_dir(label)
    initial_configuration_inputs = configuration_inputs()
    command = [str(CMAKE), '-S', str(ACTIVE), '-B', str(BUILD), '-G', 'Visual Studio 17 2022',
               '-A', 'x64', '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
               '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
               '-DVCPKG_TARGET_TRIPLET=x64-windows', '-DCMAKE_CUDA_ARCHITECTURES=86']
    write(history / 'request.json', {'kind': 'configure', 'command': command, 'frozen': frozen})
    checked(command, history / 'configure.log')
    if initial_configuration_inputs != configuration_inputs():
        raise RuntimeError('configure inputs changed while CMake was running; configure again before building')
    entries = source_map(BUILD, CONFIG)
    objects = [r['object_relative_path'].lower() for r in entries if r['object_relative_path']]
    if len(objects) != len(set(objects)):
        raise RuntimeError('effective sources have colliding object paths')
    write(history / 'result.json', {'status': 'completed', 'sources': len(entries),
                                   'translation_units': len(objects)})
    write(BUILD / 'configure_inputs.json', initial_configuration_inputs)
    print(json.dumps({'configure': 'completed', 'source_entries': len(entries), 'log': str(history)}))
    return history


def build(label: str, clean: bool, jobs: int) -> dict:
    stamp = BUILD / 'configure_inputs.json'
    # A running MSBuild project is not re-evaluated after its ZERO_CHECK child
    # regenerates vcxproj metadata. Configure in a separate process first.
    if not (BUILD / 'CMakeCache.txt').exists() or not stamp.exists() or json.loads(stamp.read_text()) != configuration_inputs():
        configure(label + '_configure')
    history = history_dir(label)
    frozen = frozen_check()
    before_sources = input_snapshot()
    before_objects = object_inventory(BUILD)
    previous_path=BUILD/'manifest.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    previous_units={r['logical_path']:r for r in previous.get('compiler_inputs',[])}
    invalidated=[]
    for row in source_map(BUILD,CONFIG):
        old=previous_units.get(row['logical_path'])
        if not old or not row['object_relative_path']:continue
        obj=(BUILD/row['object_relative_path']).resolve()
        if not obj.is_relative_to(BUILD.resolve()):raise RuntimeError('Object outside isolated build')
        if not obj.exists():continue
        closure=project_include_closure(Path(row['source_path']),old['command'],ROOT)
        current=[file_record(p,ROOT) for p in closure]
        stored=old.get('project_include_inputs')
        # Legacy manifests lacked the closure fence: force dependent objects
        # whose header timestamp is newer, then bind all closures going forward.
        stale=(digest(current)!=digest(stored)) if stored is not None else any(p.stat().st_mtime_ns>obj.stat().st_mtime_ns for p in closure)
        if stale:
            invalidated.append({'source':row['logical_path'],'object':str(obj),
                                'reason':'project_include_content_changed' if stored is not None else 'legacy_dependency_timestamp_stale',
                                'inputs':current})
            obj.unlink()  # Exact verified generated object only; no recursive removal.
    project = BUILD / 'inherited-v50/gipc.vcxproj'
    command = [str(MSBUILD), str(project), '/p:Configuration=Release', '/p:Platform=x64',
               f'/m:{jobs}', '/v:normal', '/t:Rebuild' if clean else '/t:Build',
               f'/bl:{history / "build.binlog"}']
    write(history / 'request.json', {'kind': 'build', 'clean': clean, 'command': command,
                                    'frozen': frozen, 'active_inputs_before': before_sources,
                                    'explicit_object_invalidations':invalidated})
    checked(command, history / 'build.log')
    after_sources = input_snapshot()
    if before_sources != after_sources:
        write(history / 'source_race.json', {'before': before_sources, 'after': after_sources})
        raise RuntimeError('active inputs changed during the build; refusing to freeze mixed provenance')
    entries = source_map(BUILD, CONFIG)
    commands = actual_compile_commands(BUILD, HISTORY)
    compiled = []
    for row in entries:
        if not row['object_relative_path']:
            continue
        source = Path(row['source_path']).resolve()
        obj = (BUILD / row['object_relative_path']).resolve()
        if not obj.is_relative_to(BUILD) or not obj.is_file():
            raise RuntimeError(f'expected isolated object missing: {obj}')
        command_record = commands.get(str(source).lower())
        if command_record is None:
            raise RuntimeError(f'actual compiler command missing for {source}')
        compiled.append({'logical_path': row['logical_path'], 'layer': row['layer'],
                         'source': file_record(source, ROOT), 'object': file_record(obj, ROOT),
                         'project_include_inputs':[file_record(p,ROOT) for p in project_include_closure(source,command_record['command'],ROOT)],
                         **command_record})
    links = link_evidence(BUILD, ROOT)
    if not links or not all(r['commands'] and r['inputs'] for r in links):
        raise RuntimeError('actual link command or tracked link inputs missing')
    linked_objects = {str((ROOT / f['path']).resolve()).lower()
                      for r in links for f in r['inputs'] if f['path'].lower().endswith('.obj')}
    missing = [r['object']['path'] for r in compiled
               if str((ROOT / r['object']['path']).resolve()).lower() not in linked_objects]
    if missing:
        raise RuntimeError(f'compiled source objects missing from link tracking: {missing}')
    for path in linked_objects:
        if not Path(path).is_relative_to(BUILD):
            raise RuntimeError(f'link reused an object outside builds/active: {path}')
    exe = BUILD / 'Release/gipc.exe'
    if not exe.is_file():
        raise RuntimeError('active executable not produced')
    dependencies, dependency_evidence = compiler_dependencies(BUILD, ROOT, commands, history)
    # Dependency probes read the same sources after MSBuild. Extend the original
    # content fence across these compiler calls before freezing the executable.
    after_sources = input_snapshot()
    if before_sources != after_sources:
        write(history / 'source_race.json', {'before': before_sources, 'after': after_sources,
                                            'stage': 'compiler_dependency_proof'})
        raise RuntimeError('active inputs changed during dependency verification; refusing to freeze mixed provenance')
    # Quoted includes search their original folder first. Never silently accept
    # an inherited header when an active header with the same logical name exists.
    for record in dependencies:
        path = (ROOT / record['path']).resolve()
        for provider in ('stiff_perf_v50', 'stiff_perf_v54'):
            base = ROOT / 'sources' / provider / 'StiffGIPC'
            if path.is_relative_to(base):
                logical = path.relative_to(base)
                if logical.suffix.lower() in ('.h', '.cuh', '.hpp', '.inl', '.inc') and (ACTIVE / 'StiffGIPC' / logical).exists():
                    raise RuntimeError(f'quoted include bypassed active overlay: {path}; overlay its including source')
    inherited_records = json.loads((ROOT / 'manifests/perf_v54_local.json').read_text())['files']
    effective_paths = {str(Path(r['source_path']).resolve()).lower() for r in entries}
    shadowed = set()
    for row in entries:
        for provider in ('stiff_perf_v50', 'stiff_perf_v54'):
            prior = ROOT / 'sources' / provider / 'StiffGIPC' / row['logical_path']
            if str(prior.resolve()).lower() not in effective_paths:
                shadowed.add(prior.relative_to(ROOT).as_posix())
    inherited_records = [r for r in inherited_records if r['path'] not in shadowed]
    files = {r['path']: r for r in inherited_records + dependencies + after_sources}
    for row in entries:
        item = file_record(Path(row['source_path']), ROOT); files[item['path']] = item
    files = sorted(files.values(), key=lambda item: item['path'])
    after_objects = object_inventory(BUILD)
    changed = [v['path'] for k, v in after_objects.items()
               if k not in before_objects or before_objects[k]['mtime_ns'] != v['mtime_ns']]
    manifest = {'schema': 'gipc-active-build-1', 'implementation_version': 'active-v54-overlay',
                'source_digest': digest(files), 'files': files, 'inherited': frozen,
                'file_scope': 'effective inherited frozen dependencies plus active and actually read project sources',
                'configuration': CONFIG, 'generator': 'Visual Studio 17 2022',
                'compiler_inputs': compiled, 'link_evidence': links,
                'compiler_dependency_evidence': dependency_evidence,
                'build_command': command, 'build_log': file_record(history / 'build.log', ROOT),
                'build_binlog': file_record(history / 'build.binlog', ROOT),
                'binaries': {exe.relative_to(ROOT).as_posix(): file_record(exe, ROOT)},
                'runtime_dependencies': [file_record(p, ROOT) for p in sorted(exe.parent.glob('*.dll'))],
                'build_tools': [file_record(CMAKE, ROOT), file_record(MSBUILD, ROOT)],
                'toolchain': toolchain_evidence(BUILD, ROOT),
                'cache': file_record(BUILD / 'CMakeCache.txt', ROOT),
                'changed_objects': changed, 'clean_requested': clean,
                'explicit_object_invalidations':invalidated,
                'status': 'completed', 'gpu_tests_run': False}
    manifest['archive'] = archive_inputs(after_sources, exe)
    frozen_check()
    write(history / 'manifest.json', manifest)
    write(BUILD / 'manifest.json', manifest)
    print(json.dumps({'build': 'completed', 'compiled_units': len(compiled),
                      'changed_objects': len(changed), 'exe_sha256': sha(exe),
                      'manifest': str(history / 'manifest.json')}))
    return manifest


def verify_incremental(source: str, label: str, paths: list[str | None]) -> None:
    active_source = (ACTIVE / 'StiffGIPC' / Path(source)).resolve()
    if not active_source.is_relative_to(ACTIVE / 'StiffGIPC') or not active_source.is_file():
        raise ValueError('single-file check requires an existing active source')
    if active_source.suffix not in ('.cu', '.cpp'):
        raise ValueError('single-file check must name a translation unit')
    if not all(paths):
        raise ValueError('supply --before, --modified, --restored, and --noop manifest paths')
    manifests = [json.loads(Path(p).read_text()) for p in paths]
    baseline, modified, restored, noop = manifests
    rows = [next(r for r in m['compiler_inputs'] if r['logical_path'] == source) for m in manifests]
    if rows[0]['source']['sha256'] == rows[1]['source']['sha256']:
        raise RuntimeError('the modified build did not contain a source content change')
    if rows[0]['source']['sha256'] != rows[2]['source']['sha256'] or rows[2]['source']['sha256'] != rows[3]['source']['sha256']:
        raise RuntimeError('restored/no-op source does not match the original content')
    if sha(active_source) != rows[0]['source']['sha256']:
        raise RuntimeError('the active source has not been restored')
    for result, row in [(modified, rows[1]), (restored, rows[2])]:
        changed = {str(Path(p).resolve()).lower() for p in result['changed_objects']}
        expected = str((ROOT / row['object']['path']).resolve()).lower()
        translation_objects = {str((ROOT / r['object']['path']).resolve()).lower()
                               for r in result['compiler_inputs']}
        if changed & translation_objects != {expected}:
            raise RuntimeError('content change/restore did not rebuild exactly its mapped translation unit')
        if Path(row['evidence_log']).resolve() != (ROOT / result['build_log']['path']).resolve():
            raise RuntimeError('modified source has no compiler invocation in its build log')
    if noop['changed_objects']:
        raise RuntimeError('no-op build unexpectedly rebuilt objects')
    if restored['binaries'] != noop['binaries']:
        raise RuntimeError('no-op build modified the executable')
    history = history_dir(label + '_verified')
    write(history / 'verification.json', {'no_op_passed': True, 'single_file_content_change_passed': True,
          'source': str(active_source), 'restored_source_hash': sha(active_source),
          'modified_source_hash': rows[1]['source']['sha256'],
          'rebuilt_translation_unit': rows[1]['object']['path'],
          'manifests': [file_record(Path(p), ROOT) for p in paths],
          'allowed_extra_objects': 'CUDA device-link object only; no unrelated translation unit rebuilt'})
    print(json.dumps({'incremental_verification': 'passed', 'report': str(history / 'verification.json')}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('configure', 'build', 'verify-incremental'))
    parser.add_argument('--label', default='active')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--source', default='solver/toi_solver.cu')
    for key in ('before', 'modified', 'restored', 'noop'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.jobs not in (1, 2, 3, 4):
        parser.error('--jobs must be 1 through 4')
    if args.action == 'configure': configure(args.label)
    elif args.action == 'build': build(args.label, args.clean, args.jobs)
    else: verify_incremental(args.source, args.label, [args.before, args.modified, args.restored, args.noop])


if __name__ == '__main__':
    main()
