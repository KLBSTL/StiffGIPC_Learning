"""Fresh Windows build with compiler, source and executable provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CMAKE = Path('D:/computer/cmake/bin/cmake.exe')
MSBUILD = Path('D:/vs2022/MSBuild/Current/Bin/amd64/MSBuild.exe')

def record(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': h.hexdigest()}

def snapshot(source: Path) -> list[dict]:
    folders = [source / name for name in ('StiffGIPC', 'Assets', 'MeshProcess', 'tests')]
    paths = {source / 'CMakeLists.txt', Path(__file__)}
    for folder in folders:
        paths.update(p for p in folder.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts)
    return [record(p) for p in sorted(paths) if p.exists()]

def checked(argv: list[str], log: Path) -> None:
    with log.open('x', encoding='utf-8') as out:
        out.write('ARGV ' + json.dumps(argv) + '\n')
        out.flush()
        result = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]))
        raise RuntimeError(f'Build command failed ({result.returncode}): {log}')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('active', 'base'), required=True)
    parser.add_argument('--label', required=True, help='New build directory name; never reused')
    parser.add_argument('--jobs', type=int, default=2, choices=range(1, 9))
    parser.add_argument('--configured', action='store_true',
                        help='Use a completed configure-only directory; refuse prior build logs')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        parser.error('label must be one simple path component')
    source = ROOT if args.kind == 'active' else ROOT / 'baseline'
    build = ROOT / 'build' / args.label
    if args.configured:
        if not (build / 'gipc.vcxproj').is_file() or (build / 'build.log').exists():
            raise RuntimeError('Only an unbuilt configured directory can be reused')
        cache = (build / 'CMakeCache.txt').read_text(errors='replace')
        if f'CMAKE_HOME_DIRECTORY:INTERNAL={source.as_posix()}' not in cache:
            raise RuntimeError('Configured directory has a different source root')
    else:
        build.mkdir(parents=True, exist_ok=False)
    before = snapshot(source)
    configure = [str(CMAKE), '-S', str(source), '-B', str(build), '-G',
        'Visual Studio 17 2022', '-A', 'x64', '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
        '-DVCPKG_TARGET_TRIPLET=x64-windows', '-DCMAKE_CUDA_ARCHITECTURES=86']
    if not args.configured:
        checked(configure, build / 'configure.log')
    project = ET.parse(build / 'gipc.vcxproj')
    ns = {'m': 'http://schemas.microsoft.com/developer/msbuild/2003'}
    entries = [n for tag in ('CudaCompile', 'ClCompile')
               for n in project.findall(f'.//m:{tag}', ns) if n.attrib.get('Include')]
    units = [str((build / n.attrib['Include']).resolve()) for n in entries]
    expected = {str(p.resolve()).lower() for p in (source / 'StiffGIPC').rglob('*')
                if p.suffix.lower() in ('.cu', '.cpp')}
    if set(p.lower() for p in units) != expected:
        raise RuntimeError('Generated project does not cover exact native source set')
    outputs = []
    for entry in entries:
        field = 'CompileOut' if entry.tag.endswith('CudaCompile') else 'ObjectFileName'
        names = [n.text for n in entry.findall(f'm:{field}', ns)
                 if not n.attrib.get('Condition') or 'Release|x64' in n.attrib['Condition']]
        outputs.append(names[-1].lower() if names else Path(entry.attrib['Include']).stem.lower() + '.obj')
    if len(outputs) != len(set(outputs)):
        raise RuntimeError('Generated Windows project has colliding object outputs')
    (build / 'build_inputs_before.json').write_text(
        json.dumps(before, indent=2), encoding='utf-8')
    command = [str(MSBUILD), str(build / 'StiffGIPC.sln'), '/p:Configuration=Release',
        '/p:Platform=x64', f'/m:{args.jobs}', '/v:normal',
        f'/bl:{build / "build.binlog"}']
    checked(command, build / 'build.log')
    if before != snapshot(source):
        raise RuntimeError('Source changed during build; refusing mixed identity')
    exe = build / 'Release' / 'gipc.exe'
    log = (build / 'build.log').read_text(errors='replace')
    cuda_commands = [line.strip() for line in log.splitlines()
                     if 'nvcc.exe' in line.lower()
                     and any(token in line.lower() for token in (' -c ', ' --compile ', ' -x cu '))
                     and ' -dlink ' not in line.lower()]
    cuda_units = [p for p in units if p.lower().endswith('.cu')]
    for unit in cuda_units:
        if not any(unit.lower().replace('\\', '/') in line.lower().replace('\\', '/')
                   for line in cuda_commands):
            raise RuntimeError(f'Actual CUDA compile command missing: {unit}')
    objects = [record(p) for p in sorted(build.rglob('*.obj'))
               if 'gipc.dir' in p.parts]
    manifest = {'kind': args.kind, 'status': 'completed', 'source_root': str(source),
        'build_dir': str(build), 'source_inventory': before,
        'translation_units': units, 'cuda_commands': cuda_commands,
        'objects': objects, 'exe': record(exe),
        'dlls': [record(p) for p in sorted(exe.parent.glob('*.dll'))],
        'configure_command': configure, 'build_command': command,
        'cache': record(build / 'CMakeCache.txt'),
        'build_log': record(build / 'build.log'),
        'tracking_logs': [record(p) for p in sorted(build.rglob('*.tlog'))
                          if p.name in ('link.command.1.tlog', 'link.read.1.tlog',
                                        'CL.command.1.tlog')]}
    (build / 'windows_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'kind': args.kind,
        'translation_units': len(units), 'cuda_commands': len(cuda_commands),
        'exe_sha256': manifest['exe']['sha256'], 'manifest': str(build / 'windows_manifest.json')}))

if __name__ == '__main__':
    main()
