"""Narrow contracts before the immutable native quality-screen build."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from build_windows import CMAKE,MSBUILD,checked,snapshot
from local_identity import record,verify
from linux_runner import require,write_new

def main():
    build=ROOT/'build/native_quality_screen_20261016';require(not build.exists(),'Fresh build required');build.mkdir()
    sources=snapshot(ROOT)
    command=[str(CMAKE),'-S',str(ROOT),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_TOOLCHAIN_FILE=E:/university_class/my_includes/vcpkg/scripts/buildsystems/vcpkg.cmake',
        '-DVCPKG_TARGET_TRIPLET=x64-windows','-DCMAKE_CUDA_ARCHITECTURES=86']
    checked(command,build/'configure_contracts.log')
    compile_command=[str(MSBUILD),str(build/'StiffGIPC.sln'),'/t:native_balanced_math_test;native_balanced_gpu_test',
        '/p:Configuration=Release','/p:Platform=x64','/m:2','/v:normal']
    checked(compile_command,build/'contracts_build.log');verify(sources)
    write_new(build/'contracts_identity.json',dict(sources=sources,source=record(Path(__file__)),
        configure=command,command=compile_command,executables=[record(build/'Release'/name) for name in
            ('native_balanced_math_test.exe','native_balanced_gpu_test.exe')]))
    print(json.dumps(dict(compiled=True,contracts=2)),flush=True)
if __name__=='__main__':main()
