"""Separate read-only baseline frontend overlay; frozen Stiff source is untouched.

Only endpoint velocity export is added outside the existing solver timer.
This build is quality evidence, never the official performance denominator.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify
from linux_runner import require,read,write_new

def main():
    seal_existing='--seal-existing' in sys.argv
    frozen=read(ROOT/'reports/report_execution_20261007/FROZEN_REFERENCE.json')['programs']['base']
    verify(frozen['sources']+[frozen['exe']]+frozen['dlls'])
    source=Path(frozen['source_root']);original=source/'StiffGIPC/app/gl_main.cu'
    raw=original.read_bytes();text=raw.decode('utf-8')
    anchor='            if(!output) throw std::runtime_error("Failed to write frame trace");'
    require(text.count(anchor)==1,'Baseline export anchor changed')
    addition='''
            // Observation only: actual device velocity, outside solver timing.
            checkCudaErrors(cudaMemcpy(state.data(),d_tetMesh.velocities,
                state.size()*sizeof(double3),cudaMemcpyDeviceToHost));
            std::snprintf(filename,sizeof(filename),"/velocity_%04d.bin",frame);
            std::ofstream velocity_output(std::string(trace_dir)+filename,std::ios::binary);
            velocity_output.write(reinterpret_cast<char*>(state.data()),state.size()*sizeof(double3));
            if(!velocity_output) throw std::runtime_error("Failed to write velocity trace");'''
    build=ROOT/'build/report_base_observer_20261007_v2'
    if not seal_existing:build.mkdir(exist_ok=False)
    else:require((build/'build.log').is_file(),'No completed build to inspect')
    overlay=build/'source'
    if not seal_existing:overlay.mkdir()
    frontend=overlay/'gl_main_observed.cu'
    expected_frontend=text.replace(anchor,anchor+addition).encode('utf-8')
    if not seal_existing:frontend.write_bytes(expected_frontend)
    require(frontend.read_bytes()==expected_frontend,'Observed frontend differs')
    require(frontend.read_bytes().decode('utf-8').replace(anchor+addition,anchor).encode('utf-8')==raw,
            'Overlay changed more than the observation block')
    cmake=f'''cmake_minimum_required(VERSION 3.18)
project(StiffBaselineObserver LANGUAGES CXX CUDA)
add_subdirectory("{source.as_posix()}" frozen_base)
get_target_property(observer_sources gipc SOURCES)
list(REMOVE_ITEM observer_sources "{original.as_posix()}")
list(APPEND observer_sources "{frontend.as_posix()}")
set_property(TARGET gipc PROPERTY SOURCES "${{observer_sources}}")
target_include_directories(gipc PRIVATE "{(source/'StiffGIPC/app').as_posix()}")
foreach(observer_source IN LISTS observer_sources)
  string(SHA256 observer_key "${{observer_source}}")
  string(SUBSTRING "${{observer_key}}" 0 16 observer_key)
  if(observer_source MATCHES "\\\\.cu$")
    set_property(SOURCE "${{observer_source}}" TARGET_DIRECTORY gipc PROPERTY VS_SETTINGS "CompileOut=$(IntDir)observer_${{observer_key}}.obj")
  elseif(observer_source MATCHES "\\\\.cpp$")
    set_property(SOURCE "${{observer_source}}" TARGET_DIRECTORY gipc PROPERTY VS_SETTINGS "ObjectFileName=$(IntDir)observer_${{observer_key}}.obj")
  endif()
endforeach()
'''
    if not seal_existing:(overlay/'CMakeLists.txt').write_text(cmake,encoding='utf-8')
    require((overlay/'CMakeLists.txt').read_text(encoding='utf-8')==cmake,'Observed CMake differs')
    sys.path.insert(0,str(ROOT/'tools'))
    from build_windows import checked,CMAKE,MSBUILD
    configure=[str(CMAKE),'-S',str(overlay),'-B',str(build),'-G','Visual Studio 17 2022',
        '-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
        '-DVCPKG_TARGET_TRIPLET=x64-windows','-DCMAKE_CUDA_ARCHITECTURES=86']
    if not seal_existing:
        checked(configure,build/'configure.log')
        checked([str(MSBUILD),str(build/'StiffBaselineObserver.sln'),'/t:gipc',
            '/p:Configuration=Release','/p:Platform=x64','/m:2','/v:normal'],build/'build.log')
    exe=build/'frozen_base/Release/gipc.exe';require(exe.is_file(),'Observed executable missing')
    for dll in frozen['dlls']:
        dest=exe.parent/Path(dll['path']).name
        if not dest.exists():shutil.copyfile(dll['path'],dest)
    import xml.etree.ElementTree as ET
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    project=build/'frozen_base/gipc.vcxproj'
    tree=ET.parse(project)
    units=[str((project.parent/n.attrib['Include']).resolve()) for tag in ('CudaCompile','ClCompile')
        for n in tree.findall(f'.//m:{tag}',ns) if n.attrib.get('Include')]
    expected={p.resolve() for p in (source/'StiffGIPC').rglob('*') if p.suffix in ('.cu','.cpp')}
    expected.remove(original.resolve());expected.add(frontend.resolve())
    require({Path(u).resolve() for u in units}==expected,'Observed translation unit coverage changed')
    log=(build/'build.log').read_text(errors='replace')
    commands=[s.strip() for s in log.splitlines() if 'nvcc.exe' in s.lower() and ' --compile ' in s.lower()]
    require(all(any(str(p).replace('\\','/').lower() in s.replace('\\','/').lower() for s in commands)
                for p in expected if p.suffix=='.cu'),'Observed actual CUDA command missing')
    objects=[record(p) for p in sorted((build/'frozen_base/gipc.dir/Release').glob('observer_*.obj'))]
    require(len(objects)==len(units),'Observed object coverage mismatch')
    link=build/'frozen_base/gipc.dir/Release/gipc.tlog/link.read.1.tlog'
    linked=link.read_text(encoding='utf-16').replace('\\','/').lower()
    require(all(o['path'].replace('\\','/').lower() in linked for o in objects),'Observed native object not linked')
    verify(frozen['sources']+[frozen['exe']]+frozen['dlls'])
    identity=dict(frozen,kind='base_observed',exe=record(exe),
        dlls=[record(p) for p in sorted(exe.parent.glob('*.dll'))],
        sources=frozen['sources']+[record(frontend),record(overlay/'CMakeLists.txt')],
        build_evidence=[record(project),record(link),record(build/'CMakeCache.txt'),record(build/'configure.log'),record(build/'build.log')],
        objects=objects,source_digest=hashlib.sha256(frontend.read_bytes()).hexdigest(),
        translation_units=units,cuda_commands=commands,
        capabilities={'resolved_config':False,'actual_velocity':True},
        scope='Observation frontend overlay only; original source and solver unchanged. Not the timing denominator.')
    write_new(build/'identity.json',identity)
    print(json.dumps(dict(status='completed',units=len(units),exe_sha256=identity['exe']['sha256'])))

if __name__=='__main__':main()
