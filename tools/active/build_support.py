"""Build evidence readers. These inspect only the isolated active build tree."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def project_include_closure(source: Path, command: str, root: Path) -> list[Path]:
    """Conservative lexical closure using the actual compiler's include order.

    All conditional branches are traversed, so this may rebuild extra objects.
    SDK/system headers remain covered by the compiler; project headers get a
    content fence because CUDA MSBuild tracking missed changed inline layouts.
    """
    roots=[]
    for match in re.finditer(r'(?:^|\s)(?:-I|/I)(?:"([^"]+)"|([^\s]+))',command):
        roots.append(Path(match.group(1) or match.group(2)).resolve())
    visited=set()
    def visit(path):
        path=path.resolve()
        if path in visited or not path.is_relative_to(root):return
        visited.add(path)
        content=read_text(path)
        for mode,name in re.findall(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]',content,re.M):
            search=([path.parent] if mode=='"' else [])+roots
            for directory in search:
                child=(directory/name).resolve()
                if child.is_file():
                    if child.is_relative_to(root):visit(child)
                    break
    visit(source)
    return sorted(visited)


def file_record(path: Path, root: Path) -> dict:
    path = path.resolve()
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = str(path)
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}


def digest(records: list[dict]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    return raw.decode('utf-8', errors='replace')


def tlog_records(path: Path) -> list[dict]:
    records = []
    for line in read_text(path).splitlines():
        if line.startswith('^'):
            records.append({'inputs': line[1:].split('|'), 'lines': []})
        elif line.strip() and records:
            records[-1]['lines'].append(line.strip())
    return records


def source_map(build: Path, config: str) -> list[dict]:
    with (build / f'effective_sources_{config}.tsv').open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def object_inventory(build: Path) -> dict[str, dict]:
    return {str(p.resolve()).lower(): {'path': str(p.resolve()), 'mtime_ns': p.stat().st_mtime_ns,
                                     'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in build.rglob('*.obj')}


def actual_compile_commands(build: Path, history: Path) -> dict[str, dict]:
    commands = {}
    # CudaCompile tracking does not retain a command tlog, so keep the actual
    # nvcc invocations from normal-verbosity MSBuild logs across incremental runs.
    for log in sorted(history.glob('*/build.log')):
        for line in read_text(log).splitlines():
            lower = line.lower()
            if 'nvcc.exe' not in lower or (' -c ' not in lower and ' -x cu ' not in lower):
                continue
            matches = re.findall(r'"([^"\r\n]+\.cu)"|(?<!\S)([^\s"]+\.cu)(?=\s|$)', line,
                                 flags=re.IGNORECASE)
            if matches:
                source = next((a or b for a, b in reversed(matches)), '')
                if Path(source).is_absolute():
                    invocation = re.search(r'("[^"\r\n]*nvcc\.exe".*)', line, flags=re.IGNORECASE)
                    commands[str(Path(source).resolve()).lower()] = {
                        'command': invocation.group(1) if invocation else line.strip(),
                        'evidence_log': str(log.resolve()), 'kind': 'nvcc'}
    for tlog in build.rglob('CL.command.1.tlog'):
        for record in tlog_records(tlog):
            for source in record['inputs']:
                if Path(source).suffix.lower() in ('.cpp', '.c'):
                    commands[str(Path(source).resolve()).lower()] = {
                        'command': 'cl.exe ' + '\n'.join(record['lines']),
                        'evidence_log': str(tlog.resolve()), 'kind': 'cl'}
    return commands


def link_evidence(build: Path, root: Path) -> list[dict]:
    result = []
    for tlog in sorted(build.rglob('link.command.1.tlog')):
        if 'gipc.tlog' not in tlog.parts:
            continue
        reads = tlog.with_name('link.read.1.tlog')
        inputs = set()
        if reads.exists():
            for record in tlog_records(reads):
                for name in record['inputs'] + record['lines']:
                    p = Path(name)
                    if p.suffix.lower() in ('.obj', '.lib', '.res', '.def') and p.is_file():
                        inputs.add(p.resolve())
        records = tlog_records(tlog)
        # Link may have multiple target records after a failed build. Select
        # only commands producing the active executable.
        commands = ['link.exe ' + '\n'.join(r['lines']) for r in records
                    if any('gipc.exe' in s.lower() for s in r['lines'])]
        result.append({'tracking_log': str(tlog.resolve()), 'commands': commands,
                       'inputs': [file_record(p, root) for p in sorted(inputs)]})
    return result


def overlay_header_collision(path: Path, root: Path) -> Path | None:
    """Return the active header shadowed by this inherited dependency."""
    path = path.resolve()
    for provider in ('stiff_perf_v50', 'stiff_perf_v54'):
        base = root / 'sources' / provider / 'StiffGIPC'
        if path.is_relative_to(base):
            logical = path.relative_to(base)
            active = root / 'sources/stiff_active/StiffGIPC' / logical
            if logical.suffix.lower() in ('.h', '.cuh', '.hpp', '.inl', '.inc') and active.is_file():
                return active.resolve()
    return None


def nvcc_dependency_command(command: str, output: Path) -> str:
    """Keep the actual compile flags; change only phase, local-env and output."""
    command = re.sub(r'(?<!\S)--use-local-env(?=\s|$)', '', command)
    command, phases = re.subn(r'(?<!\S)--compile(?=\s|$)', '--generate-dependencies', command)
    command, outputs = re.subn(r'(?<!\S)-o\s+(?:"[^"]+"|\S+)',
                              lambda _: '-o "' + output.as_posix() + '"', command)
    if phases != 1 or outputs != 1:
        raise RuntimeError('Cannot derive a unique NVCC dependency command from the recorded compile')
    return command


def nvcc_dependency_paths(path: Path, cwd: Path) -> list[Path]:
    """Parse one NVCC make rule, including escaped spaces and drive colons."""
    content = re.sub(r'\\\r?\n', ' ', read_text(path))
    rule = re.search(r'(?<!\\):\s', content)
    if rule is None:
        raise RuntimeError(f'NVCC dependency output is not a make rule: {path}')
    tokens = re.findall(r'(?:\\.|[^\s])+', content[rule.end():])
    result = set()
    for token in tokens:
        name = re.sub(r'\\([ \t#:\\])', r'\1', token)
        item = Path(name)
        result.add((item if item.is_absolute() else cwd / item).resolve())
    return sorted(result)


def compiler_dependencies(build: Path, root: Path, commands: dict[str, dict],
                          provenance: Path) -> tuple[list[dict], dict]:
    """Keep native tracking unless a CUDA scanner conflicts with the overlay.

    CUDA MSBuild GenerateDeps is a separate scanner from NVCC. For a conflicting
    TU only, use NVCC's own dependency phase with the actual compiler invocation.
    Do not discard a frozen header just because an active counterpart exists.
    """
    paths = set()
    proof_dir = provenance / 'compiler_dependencies'
    proof_dir.mkdir(exist_ok=False)
    evidence = {'method': 'native tlogs, with fresh NVCC dependency phase for conflicting CUDA TUs',
                'raw_tracking_logs': [], 'nvcc_overrides': []}
    resolved = {}
    for pattern in ('CL.read.*.tlog', 'CudaCompile.read.*.tlog'):
        for tlog in sorted(build.rglob(pattern)):
            # The provenance copy is immutable even when a later build rewrites
            # its tracking log. Distinct directories can have the same basename.
            suffix = hashlib.sha256(str(tlog.resolve()).encode()).hexdigest()[:12]
            saved = proof_dir / (suffix + '_' + tlog.name)
            shutil.copyfile(tlog, saved)
            evidence['raw_tracking_logs'].append({'original': str(tlog.resolve()),
                                                  'copy': file_record(saved, root)})
            for record in tlog_records(saved):
                project_paths = {Path(name).resolve() for name in record['inputs'] + record['lines']
                                 if Path(name).is_file() and Path(name).resolve().is_relative_to(root / 'sources')}
                collisions = [p for p in sorted(project_paths) if overlay_header_collision(p, root)]
                if not collisions or not tlog.name.startswith('CudaCompile.'):
                    paths.update(project_paths)
                    continue
                sources = [Path(s).resolve() for s in record['inputs'] if Path(s).suffix.lower() == '.cu']
                if len(sources) != 1:
                    raise RuntimeError(f'Ambiguous conflicting CUDA tracking record: {record["inputs"]}')
                source = sources[0]
                key = str(source).lower()
                if key not in resolved:
                    original = commands.get(key)
                    if original is None or original.get('kind') != 'nvcc':
                        raise RuntimeError(f'Actual NVCC invocation missing for conflicting TU: {source}')
                    identity = source.stem + '_' + hashlib.sha256(key.encode()).hexdigest()[:12]
                    dep = proof_dir / (identity + '.d')
                    log = proof_dir / (identity + '.log')
                    command_file = proof_dir / (identity + '.command.json')
                    command = nvcc_dependency_command(original['command'], dep)
                    cwd = build / 'inherited-v50'
                    command_file.write_text(json.dumps({'source': str(source), 'cwd': str(cwd),
                        'command': command, 'source_compile': original,
                        'raw_tracking_log': file_record(saved, root),
                        'raw_collisions': [file_record(p, root) for p in collisions]}, indent=2), encoding='utf-8')
                    with log.open('w', encoding='utf-8') as stream:
                        status = subprocess.run(command, cwd=cwd, stdout=stream,
                                                stderr=subprocess.STDOUT, timeout=120)
                    if status.returncode or not dep.is_file():
                        raise RuntimeError(f'NVCC dependency phase failed ({status.returncode}): {log}')
                    fresh = nvcc_dependency_paths(dep, cwd)
                    fresh_project = {p for p in fresh if p.is_relative_to(root / 'sources')}
                    if source not in fresh_project or any(not p.is_file() for p in fresh_project):
                        raise RuntimeError(f'Incomplete fresh NVCC project dependencies: {dep}')
                    still_shadowed = [str(p) for p in fresh_project if overlay_header_collision(p, root)]
                    proof = {'source': file_record(source, root),
                             'command': file_record(command_file, root),
                             'dependency_file': file_record(dep, root), 'log': file_record(log, root),
                             'project_dependencies': [file_record(p, root) for p in sorted(fresh_project)],
                             'frozen_active_collisions_remaining': still_shadowed}
                    evidence['nvcc_overrides'].append(proof)
                    (proof_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
                    if still_shadowed:
                        raise RuntimeError(f'NVCC confirms an inherited header bypasses the active overlay: {still_shadowed}')
                    resolved[key] = fresh_project
                paths.update(resolved[key])
    (proof_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    evidence['summary_file'] = file_record(proof_dir / 'evidence.json', root)
    return [file_record(p, root) for p in sorted(paths)], evidence


def toolchain_evidence(build: Path, root: Path) -> dict:
    values = {}
    evidence = []
    for pattern in ('CMakeCXXCompiler.cmake', 'CMakeCUDACompiler.cmake'):
        for path in sorted((build / 'CMakeFiles').glob('*/' + pattern)):
            evidence.append(file_record(path, root))
            for name, value in re.findall(r'set\((CMAKE_[A-Z_]+) "([^"\r\n]*)"\)', read_text(path)):
                if name.endswith(('_COMPILER', '_COMPILER_VERSION', '_HOST_COMPILER_VERSION', '_HOST_LINK_LAUNCHER')):
                    values[name] = value
    executables = {Path(v).resolve() for v in values.values() if v and v.lower().endswith('.exe') and Path(v).is_file()}
    return {'cmake_values': values, 'compiler_metadata': evidence,
            'executables': [file_record(p, root) for p in sorted(executables)]}
