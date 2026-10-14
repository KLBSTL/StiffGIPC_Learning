"""Fresh frozen-B0 build; only object filenames are changed by a CMake wrapper."""
from pathlib import Path
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from build_windows import checked,record,snapshot,CMAKE,MSBUILD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--configured',action='store_true',help='Continue a configure-only directory, never an existing build')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.label):
        raise ValueError('Simple fresh build label required')
    source=ROOT/'baseline'
    wrapper=ROOT/'tools/bench/m0_base_wrapper'
    build=ROOT/'build'/args.label
    if args.configured:
        if not (build/'frozen_baseline/gipc.vcxproj').is_file() or (build/'build.log').exists():
            raise ValueError('Only an unbuilt configure-only directory can be continued')
        cache=(build/'CMakeCache.txt').read_text()
        if 'CMAKE_HOME_DIRECTORY:INTERNAL='+wrapper.as_posix() not in cache:
            raise ValueError('Configured source root differs')
    else:
        build.mkdir(parents=True,exist_ok=False)
    before=snapshot(source)+[record(wrapper/'CMakeLists.txt'),record(Path(__file__))]
    before_path=build/('build_inputs_before_build.json' if args.configured else 'build_inputs_before.json')
    with before_path.open('x',encoding='utf-8') as stream:json.dump(before,stream,indent=2)
    configure=[str(CMAKE),'-S',str(wrapper),'-B',str(build),'-G','Visual Studio 17 2022',
        '-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
        '-DVCPKG_TARGET_TRIPLET=x64-windows','-DCMAKE_CUDA_ARCHITECTURES=86']
    if not args.configured:checked(configure,build/'configure.log')
    project=build/'frozen_baseline/gipc.vcxproj'
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    tree=ET.parse(project)
    entries=[n for tag in ('CudaCompile','ClCompile') for n in tree.findall('.//m:'+tag,ns) if n.attrib.get('Include')]
    units=[str((project.parent/Path(n.attrib['Include'])).resolve()) for n in entries]
    expected={str(p.resolve()).casefold() for p in (source/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
    if {u.casefold() for u in units}!=expected or len(units)!=len(expected):
        raise ValueError('Baseline TU coverage differs')
    outputs=[]
    for n in entries:
        field='CompileOut' if n.tag.endswith('CudaCompile') else 'ObjectFileName'
        names=[v.text for v in n.findall('m:'+field,ns) if not v.attrib.get('Condition') or 'Release|x64' in v.attrib['Condition']]
        if not names:raise ValueError('Missing baseline object override')
        outputs.append(names[-1].casefold())
    if len(outputs)!=len(set(outputs)):raise ValueError('Baseline object outputs collide')
    command=[str(MSBUILD),str(build/'StiffM0FrozenBaseline.sln'),'/t:gipc',
        '/p:Configuration=Release','/p:Platform=x64','/m:2','/v:normal',f'/bl:{build/"build.binlog"}']
    checked(command,build/'build.log')
    if before!=snapshot(source)+[record(wrapper/'CMakeLists.txt'),record(Path(__file__))]:
        raise ValueError('Baseline/wrapper source changed during build')
    exe=build/'frozen_baseline/Release/gipc.exe'
    log=(build/'build.log').read_text(errors='replace')
    commands=[s.strip() for s in log.splitlines() if 'nvcc.exe' in s.lower() and ' --compile ' in s.lower()]
    if not all(any(u.replace('\\','/').casefold() in s.replace('\\','/').casefold() for s in commands) for u in units if u.endswith('.cu')):
        raise ValueError('Actual baseline CUDA commands unavailable')
    objects=[record(p) for p in sorted((build/'frozen_baseline/gipc.dir/Release').glob('*.obj'))
             if re.search(r'_[0-9a-f]{16}\.obj$',p.name)]
    if len(objects)!=len(units):raise ValueError('Baseline object coverage differs')
    tracking=build/'frozen_baseline/gipc.dir/Release/gipc.tlog'
    link=tracking/'link.read.1.tlog'
    linked=link.read_text(encoding='utf-16').replace('\\','/').casefold()
    if not all(o['path'].replace('\\','/').casefold() in linked for o in objects):
        raise ValueError('Baseline object missing from actual link')
    identity=dict(kind='base',status='completed',source_root=str(source),
        source_inventory=before,translation_units=units,objects=objects,cuda_commands=commands,
        exe=record(exe),dlls=[record(p) for p in sorted(exe.parent.glob('*.dll'))],
        configure_command=configure,build_command=command,
        build_evidence=[record(p) for p in [project,build/'CMakeCache.txt',before_path,
            build/'configure.log',build/'build.log',*tracking.glob('*.tlog')]],
        scope='Frozen baseline native/material source unchanged. Build-only per-source object-name adapter; no velocity/resolved-config/checkpoint capability added.')
    with (build/'identity.json').open('x',encoding='utf-8') as stream:json.dump(identity,stream,indent=2)
    print(json.dumps(dict(status='completed',units=len(units),exe_sha256=identity['exe']['sha256'])))


if __name__=='__main__':main()
