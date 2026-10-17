"""Standalone direct-vs-CG-statistics cost probe; no production build changes."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/bench'),str(ROOT/'tools/local')]
from build_windows import checked,CMAKE,MSBUILD
from local_identity import record,verify,tree
from linux_runner import write_new,require


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='cgstats_gpu_20261015');args=p.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_-]+',args.label) is not None,'Simple build label required')
    source=ROOT/'tools/bench/cgstats_gpu';build=ROOT/'build'/args.label
    require(not build.exists(),'Fresh build required');build.mkdir()
    inputs=tree(source)+[record(Path(__file__))];write_new(build/'inputs.json',inputs)
    configure=[str(CMAKE),'-S',str(source),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86',
        '-Dnlohmann_json_DIR=E:/Anaconda/Library/share/cmake/nlohmann_json',
        '-DEigen3_DIR=E:/university_class/my_includes/vcpkg/installed/x64-windows/share/eigen3']
    checked(configure,build/'configure.log')
    command=[str(MSBUILD),str(build/'CGStatisticsCost.sln'),'/t:cgstats_cost','/p:Configuration=Release',
        '/p:Platform=x64','/m:2','/v:normal',f'/bl:{build/"build.binlog"}']
    checked(command,build/'build.log');verify(inputs);exe=build/'Release/cgstats_cost.exe'
    require(exe.is_file(),'Probe executable missing')
    write_new(build/'identity.json',dict(sources=inputs,exe=record(exe),configure=configure,build_command=command,
        evidence=tree(build/'cgstats_cost.dir/Release/cgstats_cost.tlog')+
            [record(build/n) for n in ('configure.log','build.log','CMakeCache.txt','cgstats_cost.vcxproj')],
        production_build_modified=False,translation_units=1))
    print(json.dumps(dict(built=True,exe=record(exe))),flush=True)


if __name__=='__main__':main()
