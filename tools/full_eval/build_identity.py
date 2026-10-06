"""Read-only clean-build identity audit; writes only fresh requested JSON reports.

snapshot must precede both builds. verify never compiles, loads CUDA, runs the
program, or edits a build. Ninja's `-t commands gipc` only prints build commands.
System headers/shared libraries are not exhaustively frozen by this identity.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

SOURCE_DIRS = ('StiffGIPC', 'Assets', 'MeshProcess',
               'baseline/StiffGIPC', 'baseline/Assets', 'baseline/MeshProcess')
SOURCE_FILES = ('CMakeLists.txt', 'baseline/CMakeLists.txt')
EXPECTED = {'active': {'tu': 37, 'cuda': 33, 'cxx': 4},
            'base': {'tu': 35, 'cuda': 31, 'cxx': 4}}
SELECTED_SCENES = ('cloth_hang_l', 'cloth_fixed_bunny_l', 'bunny_cloth_bunny_l')
# Exact reviewed baseline-only caches, never a directory or suffix exception.
UNUSED_BASE_CACHES = frozenset({
    'sorted_mesh/cipc_table_sorted.16.obj', 'sorted_mesh/cipc_table_sorted.16.part',
    'sorted_mesh/cloth_high_sorted.16.obj', 'sorted_mesh/cloth_high_sorted.16.part',
    'sorted_mesh/cube_sorted.16.msh', 'sorted_mesh/cube_sorted.16.part',
    'sorted_mesh/high_mat_sorted.16.msh', 'sorted_mesh/high_mat_sorted.16.part'})


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(data)
    return value.hexdigest()


def linked(path):
    return path.is_symlink() or bool(getattr(path, 'is_junction', lambda: False)())


def inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root / path
    require(path.resolve().is_relative_to(root), 'Path outside root: ' + str(path))
    for part in (path, *path.parents):
        if part == root:
            break
        require(not linked(part), 'Linked path rejected: ' + str(part))
    return path.resolve()


def file_record(root, path, external=False):
    path = Path(path)
    if not external:
        path = inside(root, path)
    else:
        path = path.resolve()
    require(path.is_file(), 'Missing evidence file: ' + str(path))
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing: ' + str(path))
    name = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    return {'path': name, 'bytes': after.st_size, 'sha256': value}


def source_inventory(root):
    root = Path(root).resolve()
    paths = []
    for relative in SOURCE_DIRS:
        folder = inside(root, relative)
        require(folder.is_dir(), 'Missing source root: ' + relative)
        for directory, names, files in os.walk(folder, followlinks=False):
            for name in names + files:
                require(not linked(Path(directory) / name), 'Linked source entry rejected')
            names[:] = [name for name in names if name != '__pycache__']
            paths.extend(Path(directory) / name for name in files if not name.endswith('.pyc'))
    paths.extend(inside(root, name) for name in SOURCE_FILES)
    return [file_record(root, path) for path in sorted(paths)]


def normalized_before(packet):
    rows = packet if isinstance(packet, list) else next(
        (packet[key] for key in ('sources', 'records', 'files') if key in packet), None)
    require(isinstance(rows, list) and rows, 'Before snapshot needs a nonempty source record list')
    result = []
    for row in rows:
        name = next((row[key] for key in ('path', 'relative_path', 'relative') if key in row), None)
        size = row.get('bytes', row.get('size'))
        value = row.get('sha256')
        require(isinstance(name, str) and not Path(name).is_absolute() and
                not re.match(r'^[A-Za-z]:', name) and '\\' not in name and
                all(p not in ('', '.', '..') for p in name.split('/')), 'Invalid relative source path')
        require(type(size) is int and size >= 0 and isinstance(value, str) and
                re.fullmatch(r'[a-fA-F0-9]{64}', value), 'Invalid source hash/size')
        result.append({'path': name, 'bytes': size, 'sha256': value.lower()})
    require(len({row['path'] for row in result}) == len(result), 'Duplicate source record')
    return sorted(result, key=lambda row: row['path'])


def snapshot(root):
    root = Path(root).resolve()
    before = {kind: (root / 'build' / kind).exists() for kind in EXPECTED}
    require(not any(before.values()), 'Snapshot must precede creation of both clean build directories')
    return {'schema': 'full_eval_prebuild_sources.v1', 'created_utc': utc(),
            'root': str(root), 'build_directories_existed': before,
            'sources': source_inventory(root)}


def assets_equivalence(root, sources):
    def inventory(prefix):
        return {r['path'][len(prefix):]: {'path': r['path'][len(prefix):],
                'bytes': r['bytes'], 'sha256': r['sha256']}
                for r in sources if r['path'].startswith(prefix)}
    active, base = inventory('Assets/'), inventory('baseline/Assets/')
    require(not (active.keys() - base.keys()), 'Active-only Assets are forbidden')
    require(all(row == base[name] for name, row in active.items()), 'Common Assets differ')
    extras = base.keys() - active.keys()
    require(extras <= UNUSED_BASE_CACHES, 'Unapproved baseline-only Assets')
    references, scenes = set(), []
    for case in SELECTED_SCENES:
        relative = 'benchmark_scenes/' + case + '.json'
        require(relative in active, 'Selected scene missing from common Assets')
        scene = json.loads(inside(root, 'Assets/' + relative).read_text(encoding='utf-8-sig'))
        require(scene.get('case_id') == case and isinstance(scene.get('objects'), list)
                and scene['objects'], 'Selected scene identity/objects invalid')
        references.add(relative)
        meshes = []
        for obj in scene['objects']:
            name = obj['stiff_mesh']
            require(isinstance(name, str) and name in active, 'Selected mesh missing from common Assets')
            mesh = inside(root / 'Assets', name)
            require(active[name]['sha256'] == obj['stiff_mesh_sha256'], 'Selected frozen mesh identity differs')
            caches = ['sorted_mesh/' + mesh.stem + '_sorted.16' + mesh.suffix,
                      'sorted_mesh/' + mesh.stem + '_sorted.16.part']
            require(all(name in active for name in caches), 'Selected sorted-mesh/cache missing from common Assets')
            references.update([name, *caches])
            meshes.append({'stiff_mesh': name, 'sha256': active[name]['sha256'], 'derived_sorted_cache': caches})
        scenes.append({'scene': case, 'scene_sha256': active[relative]['sha256'], 'meshes': meshes})
    require(not (references & extras), 'Ignored baseline cache is referenced by selected plan')
    return {'passed': True, 'roots_identical': not extras, 'common_files': len(active),
            'common_files_all_equal': True, 'active_only': [],
            'ignored_baseline_only': [base[name] | {'reason': 'Exact reviewed unused cache; no selected stiff_mesh or derived sorting-cache reference.'}
                                      for name in sorted(extras)],
            'selected_scene_references': scenes,
            'cache_rule': 'MeshProcess/metis_partition/src/metis_sort.cpp: stem + _sorted.16 + original extension, and .part',
            'scope': 'Three explicitly selected full-evaluation scenes only; every common asset byte-identical; baseline exceptions remain hashed in sources.'}


def temporal_freshness(root, before_path, builds):
    # Ninja start_ms is relative to its own invocation. The supervisor begins
    # earlier (before configure), so this is only a conservative UTC lower
    # bound, not a reconstructed exact timestamp for an individual compile.
    path = inside(root, 'build_start.json')
    record = file_record(root, path)
    start = json.loads(path.read_text(encoding='utf-8-sig'))
    value = start.get('time_unix')
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Missing valid prebuild supervisor timestamp')
    start_ns = int(value * 1_000_000_000)
    before_ns = before_path.stat().st_mtime_ns
    require(before_ns < start_ns, 'Source snapshot was not written before build supervisor start')
    require(start_ns <= int(dt.datetime.now(dt.timezone.utc).timestamp() * 1_000_000_000),
            'Build supervisor timestamp is in the future')
    rows = {}
    for kind, build in builds.items():
        require(all(log['mtime_ns'] >= start_ns for log in build['build_logs'].values()),
                'Build command log predates supervisor start')
        objects = []
        for row in build['objects']:
            mtime = inside(root, row['object']['path']).stat().st_mtime_ns
            require(mtime >= start_ns, 'Actual object predates current build supervisor')
            lower = start_ns + row['execution']['start_ms'] * 1_000_000
            require(before_ns < lower, 'Source snapshot does not precede compile lower bound')
            objects.append({'path': row['object']['path'], 'mtime_ns': mtime,
                            'ninja_start_ms': row['execution']['start_ms'],
                            'compile_start_lower_bound_unix_ns': lower})
        rows[kind] = {'objects': objects, 'all_snapshot_before_compile_lower_bound': True}
    return {'passed': True, 'supervisor_start': record, 'time_unix': value,
            'before_file_mtime_ns': before_ns, 'builds': rows,
            'exact_compile_start_utc_available': False,
            'scope': 'Snapshot mtime strictly precedes supervisor start; object and log mtimes follow it. Ninja relative starts are recorded, not mistaken for absolute time. This relies on the task supervisor clock and its documented ordering, not a cryptographic provenance attestation.'}


def read_cache(path):
    return {line.split('=', 1)[0].split(':', 1)[0]: line.split('=', 1)[1]
            for line in path.read_text().splitlines()
            if '=' in line and not line.startswith(('#', '//'))}


def resolve_argument(cwd, value):
    p = Path(value)
    return (p if p.is_absolute() else cwd / p).resolve()


def expand_response(argv, cwd, root, records, depth=0):
    require(depth < 8, 'Recursive response file')
    result = []
    for token in argv:
        if token.startswith('@'):
            path = inside(root, resolve_argument(cwd, token[1:]))
            require(path.is_file(), 'Missing response file; clean build with Ninja -d keeprsp: ' + str(path))
            records[path] = file_record(root, path)
            result.extend(expand_response(shlex.split(path.read_text()), cwd, root, records, depth + 1))
        else:
            result.append(token)
    return result


def command_segments(text, cwd, root, responses):
    """Split actual Ninja shell wrappers, retaining cd-dependent working dirs."""
    lex = shlex.shlex(text, posix=True, punctuation_chars=';&|')
    lex.whitespace_split = True
    lex.commenters = ''
    segment = []
    pieces = []
    for token in list(lex) + ['&&']:
        if token in ('&&', ';'):
            if segment:
                pieces.append(segment)
                segment = []
        elif token in ('|', '||', '&'):
            raise ValueError('Unsupported shell control in Ninja command: ' + text)
        else:
            segment.append(token)
    for argv in pieces:
        if argv[0] == 'cd':
            require(len(argv) == 2, 'Unexpected Ninja cd command')
            cwd = inside(root, resolve_argument(cwd, argv[1]))
        elif argv != [':']:
            yield cwd, expand_response(argv, cwd, root, responses)


def option(argv, name):
    at = [i for i, value in enumerate(argv) if value == name]
    require(len(at) <= 1, 'Duplicate option ' + name)
    if not at:
        return None
    require(at[0] + 1 < len(argv), 'Missing option value ' + name)
    return argv[at[0] + 1]


def require_compile_source(argv, cwd, source):
    # CMake separable CUDA compilation emits -dc rather than host-style -c.
    # Accept only one explicit compile marker with the metadata source as its
    # immediate operand; never infer a source from other positional arguments.
    markers = [(i, value) for i, value in enumerate(argv)
               if value in ('-c', '-dc', '--device-c')]
    require(len(markers) == 1, 'Expected exactly one explicit compile source marker')
    index, marker = markers[0]
    require(marker == '-c' or source.suffix == '.cu', 'CUDA device compilation marker on non-CUDA source')
    require(index + 1 < len(argv) and not argv[index + 1].startswith('-')
            and resolve_argument(cwd, argv[index + 1]) == source,
            'Compile command source argument differs from metadata')


def without_dependencies(argv):
    # compile_commands commonly omits these Ninja-only depfile options.
    result = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in ('-MD', '-MMD', '--generate-dependencies-with-compile'):
            index += 1
        elif value in ('-MF', '-MT', '-MQ', '--dependency-output'):
            require(index + 1 < len(argv), 'Missing dependency option value')
            index += 2
        else:
            result.append(value)
            index += 1
    return result


def ninja_commands(program, build):
    argv = [str(program), '-C', str(build), '-t', 'commands', 'gipc']
    p = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=60)
    return {'argv': argv, 'stdout': p.stdout, 'stderr': p.stderr, 'returncode': p.returncode}


def parse_ninja_log(path, build):
    text = path.read_text()
    require(text.startswith('# ninja log v'), 'Unrecognized Ninja execution log')
    result = {}
    for line in text.splitlines()[1:]:
        fields = line.split('\t')
        require(len(fields) == 5, 'Malformed Ninja log row')
        start, end, mtime = map(int, fields[:3])
        require(0 <= start <= end and mtime >= 0 and fields[4], 'Invalid Ninja execution record')
        result[resolve_argument(build, fields[3])] = {
            'start_ms': start, 'end_ms': end, 'output_mtime': mtime, 'command_hash': fields[4]}
    return result


def build_log(root, path):
    record = file_record(root, path)
    with path.open(encoding='utf-8', errors='replace') as stream:
        first = stream.readline().strip()
    require(first.startswith('ARGV '), 'Build log must preserve exact ARGV: ' + str(path))
    argv = json.loads(first[5:])
    require(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv), 'Invalid build ARGV')
    record.update(argv=argv, mtime_ns=path.stat().st_mtime_ns)
    return record


def inspect_build(root, kind, reader=ninja_commands):
    source = root if kind == 'active' else root / 'baseline'
    build = inside(root, 'build/' + kind)
    native = {p.resolve() for p in (source / 'StiffGIPC').rglob('*') if p.suffix in ('.cu', '.cpp')}
    counts = {'tu': len(native), 'cuda': sum(p.suffix == '.cu' for p in native),
              'cxx': sum(p.suffix == '.cpp' for p in native)}
    require(counts == EXPECTED[kind], f'{kind} source counts differ: {counts}; expected {EXPECTED[kind]}')
    cache_path = build / 'CMakeCache.txt'
    cache = read_cache(cache_path)
    require(Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve() == source, 'Wrong CMake source root')
    require(cache.get('CMAKE_GENERATOR') == 'Ninja' and cache.get('CMAKE_BUILD_TYPE') == 'Release',
            'Expected Ninja Release build')
    require(cache.get('CMAKE_CUDA_ARCHITECTURES') == '89', 'Expected explicit CUDA architecture 89')
    compiler_records = []
    for key in ('CMAKE_CUDA_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER', 'CMAKE_MAKE_PROGRAM'):
        if key == 'CMAKE_C_COMPILER' and key not in cache:
            continue
        require(key in cache and Path(cache[key]).is_absolute(), 'Missing absolute compiler/tool path: ' + key)
        compiler_records.append({'cache_key': key, 'requested_path': cache[key],
                                 'resolved_file': file_record(root, Path(cache[key]), external=True)})
    responses = {}
    commands_path = build / 'compile_commands.json'
    commands = json.loads(commands_path.read_text())
    mapped = {}
    for row in commands:
        cwd = Path(row['directory']).resolve()
        argv = row.get('arguments') or shlex.split(row['command'])
        argv = expand_response(argv, cwd, root, responses)
        value = option(argv, '-o')
        if value is None:
            continue
        obj = resolve_argument(cwd, value)
        if 'gipc.dir' not in obj.parts:
            continue
        obj = inside(root, obj)
        src = resolve_argument(cwd, row['file'])
        require(src in native, 'gipc compiles outside selected native tree: ' + str(src))
        require(src not in mapped and all(v['object'] != obj for v in mapped.values()),
                'Duplicate TU or object output')
        require_compile_source(argv, cwd, src)
        require(obj.suffix == '.o' and obj.is_relative_to(build), 'Expected object inside this build')
        mapped[src] = {'object': obj, 'argv': argv, 'cwd': cwd}
    require(set(mapped) == native, 'Actual compile command/native source coverage mismatch')

    dump = reader(Path(cache['CMAKE_MAKE_PROGRAM']).resolve(), build)
    require(dump['returncode'] == 0 and dump['stdout'].strip(), 'Ninja command inspection failed')
    actual = {}
    for line in dump['stdout'].splitlines():
        for cwd, argv in command_segments(line, build, root, responses):
            value = option(argv, '-o')
            if value is not None:
                obj = resolve_argument(cwd, value)
                require(obj not in actual, 'Duplicate output in Ninja command graph: ' + str(obj))
                actual[obj] = {'cwd': cwd, 'argv': argv}
    execution = parse_ninja_log(build / '.ninja_log', build)
    object_rows = []
    for src, info in sorted(mapped.items()):
        obj = info['object']
        require(obj in actual and obj in execution, 'Object lacks actual Ninja command/execution: ' + str(obj))
        observed = actual[obj]
        require(observed['cwd'] == info['cwd'] and
                without_dependencies(observed['argv']) == without_dependencies(info['argv']),
                'Ninja actual compile command differs from compilation database: ' + str(src))
        object_rows.append({'source': src.relative_to(root).as_posix(),
                            'object': file_record(root, obj), 'execution': execution[obj],
                            'actual_argv': observed['argv']})
    exe = build / 'gipc'
    require(exe in actual and exe in execution, 'Executable lacks actual link command/execution')
    link = actual[exe]
    linked_objects = {resolve_argument(link['cwd'], arg) for arg in link['argv'] if arg.endswith('.o')}
    expected_objects = {v['object'] for v in mapped.values()}
    device_links = [p for p in linked_objects if p in actual and
                    any(a in ('-dlink', '--device-link') for a in actual[p]['argv'])]
    require(len(device_links) == 1, 'Expected exactly one linked CUDA device-link object')
    dlink = device_links[0]
    require(linked_objects == expected_objects | {dlink}, 'Host link object set differs from verified TU objects')
    require(dlink in execution, 'Device-link output has no actual execution record')
    device_inputs = {resolve_argument(actual[dlink]['cwd'], a)
                     for a in actual[dlink]['argv'] if a.endswith('.o') and
                     resolve_argument(actual[dlink]['cwd'], a) != dlink}
    cuda_objects = {info['object'] for src, info in mapped.items() if src.suffix == '.cu'}
    require(cuda_objects <= device_inputs and device_inputs <= expected_objects,
            'Device link misses CUDA objects or includes unknown objects')
    require(all(execution[p]['end_ms'] <= execution[exe]['start_ms']
                for p in expected_objects | {dlink}), 'Object/link execution order inconsistent')
    archives = sorted({resolve_argument(link['cwd'], arg) for arg in link['argv'] if arg.endswith('.a')})
    logs = {name: build_log(root, root / 'build_logs' / kind / (name + '.log'))
            for name in ('configure', 'build')}
    configure_argv, build_argv = logs['configure']['argv'], logs['build']['argv']
    require(option(configure_argv, '-S') is not None and option(configure_argv, '-B') is not None
            and resolve_argument(root, option(configure_argv, '-S')) == source
            and resolve_argument(root, option(configure_argv, '-B')) == build,
            'Configure log targets a different source/build directory')
    require(option(build_argv, '--build') is not None and
            resolve_argument(root, option(build_argv, '--build')) == build,
            'Build log targets a different build directory')
    evidence = [file_record(root, p) for p in
                (commands_path, cache_path, build / 'build.ninja', build / 'CMakeFiles/rules.ninja', build / '.ninja_log')]
    return {'kind': kind, 'counts': counts, 'objects': object_rows,
            'device_link': {'file': file_record(root, dlink), 'actual_argv': actual[dlink]['argv'],
                            'input_object_count': len(device_inputs), 'execution': execution[dlink]},
            'exe': file_record(root, exe), 'link_argv': link['argv'], 'link_execution': execution[exe],
            'explicit_link_archives': [file_record(root, p, external=True) for p in archives],
            'unresolved_library_options': [a for a in link['argv'] if a.startswith('-l')],
            'compiler_binaries': compiler_records, 'build_evidence': evidence,
            'response_files': list(responses.values()), 'build_logs': logs,
            'ninja_command_inspection': dump,
            'system_headers_and_shared_libraries_exhaustively_frozen': False}


def verify(root, before_path, reader=ninja_commands):
    root = Path(root).resolve()
    started = utc()
    before_path = inside(root, before_path)
    packet = json.loads(before_path.read_text(encoding='utf-8-sig'))
    expected = normalized_before(packet)
    current = sorted(source_inventory(root), key=lambda row: row['path'])
    require(expected == current, 'Prebuild/postbuild native/input inventory differs')
    builds = {kind: inspect_build(root, kind, reader) for kind in EXPECTED}
    require(current == sorted(source_inventory(root), key=lambda row: row['path']), 'Source changed during verification')
    freshness = packet.get('build_directories_existed') if isinstance(packet, dict) else None
    if freshness is not None:
        require(freshness == {'active': False, 'base': False}, 'Prebuild snapshot was not fresh')
    assets = assets_equivalence(root, current)
    temporal = temporal_freshness(root, before_path, builds)
    result = {'schema': 'full_eval_linux_build_identity.v1', 'passed': True,
            'started_utc': started, 'completed_utc': utc(), 'root': str(root),
            'before': file_record(root, before_path), 'sources_unchanged': True,
            'sources': current, 'assets_identical': assets['roots_identical'],
            'assets_equivalence': assets, 'temporal_freshness': temporal, 'builds': builds,
            'auditor': file_record(root, Path(__file__), external=True),
            'fresh_build_directory_prestate_recorded': freshness is not None,
            'freshness_note': ('Both build directories absent at snapshot.' if freshness is not None else
                               'Common source-only snapshot accepted; directory freshness requires owner execution evidence.'),
            'scope': 'Source/input stability, actual TU/object/device-link/host-link/executable identity; no GPU test or speed certification.'}
    _verify_identity_files(root, result)
    return result


def _verify_identity_files(root, packet):
    records = [packet['before'], packet['auditor'], *packet['sources'],
               packet['temporal_freshness']['supervisor_start']]
    require(inside(root, packet['before']['path']).stat().st_mtime_ns ==
            packet['temporal_freshness']['before_file_mtime_ns'], 'Sealed prebuild snapshot mtime changed')
    for kind, build in packet['builds'].items():
        require(kind in EXPECTED and build['counts'] == EXPECTED[kind], 'Unexpected sealed source counts')
        records.extend([build['exe'], build['device_link']['file'], *build['build_evidence'],
                        *build['response_files'], *build['build_logs'].values(),
                        *build['explicit_link_archives']])
        records.extend(row['object'] for row in build['objects'])
        for compiler in build['compiler_binaries']:
            row = compiler['resolved_file']
            sealed = Path(row['path']) if Path(row['path']).is_absolute() else root / row['path']
            require(Path(compiler['requested_path']).resolve() == sealed.resolve(), 'Compiler/tool symlink changed')
            records.append(row)
    for row in records:
        path = Path(row['path'])
        absolute = path if path.is_absolute() else root / path
        observed = file_record(root, absolute, external=path.is_absolute())
        require(all(observed[k] == row[k] for k in ('path', 'bytes', 'sha256')),
                'Sealed build identity changed: ' + row['path'])


def verify_identity(root, path, expected_sha=None):
    """Read-only session recheck of a sealed attachment; no Ninja subprocess."""
    root = Path(root).resolve()
    path = inside(root, path)
    if expected_sha is not None:
        require(sha(path) == expected_sha, 'Build identity attachment changed')
    packet = json.loads(path.read_text(encoding='utf-8-sig'))
    require(packet['schema'] == 'full_eval_linux_build_identity.v1' and packet['passed'] is True
            and packet['sources_unchanged'] is True and packet['assets_equivalence']['passed'] is True
            and packet['temporal_freshness']['passed'] is True
            and set(packet['builds']) == set(EXPECTED), 'Invalid build identity attachment')
    require(sorted(source_inventory(root), key=lambda r: r['path']) == packet['sources'],
            'Source/input changed since build identity seal')
    _verify_identity_files(root, packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('snapshot', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = inside(root, args.out)
    require(not out.exists(), 'Output exists; no overwrite')
    require((args.action == 'verify') == (args.before is not None), '--before is required only for verify')
    result = snapshot(root) if args.action == 'snapshot' else verify(root, args.before)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'action': args.action, 'out': str(out), 'sources': len(result['sources']),
                      'counts': {k: b['counts'] for k, b in result.get('builds', {}).items()},
                      'passed': result.get('passed'), 'gpu_run': False}))


if __name__ == '__main__':
    main()
