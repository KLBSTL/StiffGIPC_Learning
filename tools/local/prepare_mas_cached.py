"""Small private extension of the sealed native-PCG fixture; production untouched."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cache_pcg import CUDA,FROZEN,edit
from local_identity import record,verify,tree,compiler_records
from linux_runner import require,write_new
from build_windows import CMAKE,checked
RAW=ROOT/'runs/innovation_p5_mas_cached_20261017'
REPORT=ROOT/'reports/innovation_p5_mas_cached_20261017'
BUILD=ROOT/'build/mas_cached_20261017'
PROJECT=ROOT/'tools/bench/mas_cached'
PARENT=ROOT/'reports/innovation_p5_cache_pcg_20261017/DELIVERY_SEAL.json'
OLD=ROOT/'build/cache_pcg_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh paths required');verify(read(PARENT)['files'])
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir();generation=[]
    def generate(source,name,changes=()):
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(edit(source.read_text(encoding='utf-8'),changes),encoding='utf-8')
        generation.append(dict(source=record(source),target=record(target),edits=changes))
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc','graph_tiny_info.inl','native_kernels.inl',
                 'linear_system/solver/native_balanced_math.h','linear_system/solver/native_balanced_gpu.cuh']:
        generate(OLD/'overlay'/name,name)
    generate(PROJECT/'cost_profile.h','cost_profile.h')
    changes=[('action=="contracts"||action=="measure"','action=="contracts"||action=="measure"||action=="profile"',1),
        ('#include "graph_tiny_info.inl"','#include "graph_tiny_info.inl"\n#include "cost_profile.h"',1),
        ('        auto base=[&](const double* input,double* output){','        NativeCostProfile costs;\n        auto base=[&](const double* input,double* output){\n            costs.call("quality_native_MAS",[&]{',1),
        ('abd_action(input,output);m.preconditioning(reinterpret_cast<const double3*>(input+mas_offset),reinterpret_cast<double3*>(output+mas_offset));\n        };',
         'abd_action(input,output);m.preconditioning(reinterpret_cast<const double3*>(input+mas_offset),reinterpret_cast<double3*>(output+mas_offset));\n            });\n        };',1),
        ('            spmv.warp_reduce_sym_spmv(1.,values.value','            costs.call("SpMV",[&]{spmv.warp_reduce_sym_spmv(1.,values.value',1),
        ('            balanced_cuda(cudaGetLastError());\n        };\n        auto check=',
         '            balanced_cuda(cudaGetLastError());});\n        };\n        auto check=',1),
        ('return My_PCG_General_v_v_Reduction_Algorithm(dot_partials.value,a,c,reduction_result.value,n);',
         'return costs.call("dot_with_host_readback",[&]{return My_PCG_General_v_v_Reduction_Algorithm(dot_partials.value,a,c,reduction_result.value,n);});',1),
        ('            if(execution)balanced_cuda(cudaGraphLaunch(graphs[method]->exec,cudaStreamPerThread));else direct(method);',
         '            costs.call("recurrence_apply",[&]{if(execution)balanced_cuda(cudaGraphLaunch(graphs[method]->exec,cudaStreamPerThread));else direct(method);});',1),
        ('            Events events;Json output;','            Events events;Json output;costs.stages=Json::object();costs.enabled=action=="profile";',1),
        ('const double host_ms=elapsed(start),event_ms=events.ms();const auto export_start=Clock::now();',
         'const double host_ms=elapsed(start),event_ms=events.ms();costs.enabled=false;\n            balanced_require(costs.depth==0,"Profiler depth leak");\n            const Json profile={{"enabled",action=="profile"},{"stages",costs.stages}};const auto export_start=Clock::now();',1),
        ('return Json{{"method",methods[method]}','return Json{{"profile",profile},{"method",methods[method]}',1),
        ('method=comparison?label:0;','method=comparison?(label?1:2):2;',1)]
    generate(OLD/'overlay/fixture.inl','fixture.inl',changes)
    s=(OLD/'overlay/recurrence.inl').read_text(encoding='utf-8');changes=[]
    operations=[('copy_rhs','check(cudaMemcpyAsync(r.value,b.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));'),
        ('copy_initial_direction','check(cudaMemcpyAsync(p.value,z.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));'),
        ('vector_update_x_r','update_vector_dx_r<<<blocks,256,0,cudaStreamPerThread>>>(work_x.value,r.value,p.value,Ap.value,alpha,n);\n        check(cudaGetLastError());'),
        ('vector_update_direction','update_vector_c<<<blocks,256,0,cudaStreamPerThread>>>(p.value,z.value,beta,n);\n        check(cudaGetLastError());'),
        ('explicit_residual_vector','audit_residual<<<blocks,256,0,cudaStreamPerThread>>>(quality_r.value,b.value,residual_ax.value,n);\n    check(cudaGetLastError());')]
    for name,body in operations:changes.append((body,'costs.call("'+name+'",[&]{'+body+'});',1))
    generate(OLD/'overlay/recurrence.inl','recurrence.inl',changes)
    project=BUILD/'project';project.mkdir()
    for name in ['driver.cu','CMakeLists.txt']:
        source=OLD/'project'/name;s=source.read_text(encoding='utf-8');changes=[]
        if name=='CMakeLists.txt':changes=[('cache_pcg','mas_cached',s.count('cache_pcg'))]
        target=project/name;target.write_text(edit(s,changes),encoding='utf-8');generation.append(dict(source=record(source),target=record(target),edits=changes))
    manifest=read(ROOT/'runs/innovation_p5_cache_pcg_20261017/input_manifest.json')
    write_new(RAW/'input_manifest.json',manifest);write_new(BUILD/'generation.json',dict(files=generation))
    write_new(RAW/'production_before.json',tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')])
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),files=len(read(PARENT)['files'])))
    print(dict(prepared=True,cases=len(manifest['cases'])),flush=True)
def build():
    require(not (BUILD/'identity.json').exists(),'Already built')
    sources=tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+tree(FROZEN)+[record(Path(__file__))]
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(BUILD/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),'-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log');command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','mas_cached','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/mas_cached.exe'),runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],
        compilers=compiler_records(BUILD),configure=configure,command=command,build_evidence=[record(BUILD/p) for p in ['CMakeCache.txt','mas_cached.vcxproj','configure.log','build.log','sources_before.json','generation.json']]))
    print(dict(compiled=True,exe=record(BUILD/'Release/mas_cached.exe')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args();prepare() if a.action=='prepare' else build()
