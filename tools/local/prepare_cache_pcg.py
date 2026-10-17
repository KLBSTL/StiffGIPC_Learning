"""Private replay of the audited native recurrence on immutable A/b/M inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from linux_runner import require,write_new
from build_windows import CMAKE,checked
from prepare_tiny_cache import FROZEN,CUDA
RAW=ROOT/'runs/innovation_p5_cache_pcg_20261017'
REPORT=ROOT/'reports/innovation_p5_cache_pcg_20261017'
BUILD=ROOT/'build/cache_pcg_20261017'
PROJECT=ROOT/'tools/bench/cache_pcg'
PARENT=ROOT/'reports/innovation_p5_tiny_cache_20261017/DELIVERY_SEAL.json'
OLD=ROOT/'build/tiny_cache_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def edit(s,changes):
    for a,b,n in changes:
        require(s.count(a)==n,'Ambiguous native adaptation: '+a[:80]);s=s.replace(a,b)
    return s
def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh paths required');verify(read(PARENT)['files'])
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir();generation=[]
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc','graph_tiny_info.inl',
                 'linear_system/solver/native_balanced_math.h','linear_system/solver/native_balanced_gpu.cuh']:
        source=OLD/'overlay'/name;target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());generation.append(dict(source=record(source),target=record(target)))
    source=OLD/'overlay/tiny_fixture.inl';s=source.read_text(encoding='utf-8')
    boundary='        auto op=[&](int method){';require(s.count(boundary)==1,'Setup boundary changed')
    prefix=s.split(boundary)[0]
    changes=[('action=="contracts"||action=="measure"','action=="contracts"||action=="measure"',1),
        ('if(measure&&probe!="rhs")continue;','if(probe!="rhs")continue;',1),
        ('std::array<std::string,4> methods={"MAS","original","prefetch","cached"}',
         'std::array<std::string,3> methods={"original","cached","MAS"}',1)]
    target=overlay/'fixture.inl';target.write_text(edit(prefix,changes)+(PROJECT/'continuation.inl').read_text(),encoding='utf-8')
    generation.append(dict(source=record(source),target=record(target),boundary=boundary,edits=changes,continuation=record(PROJECT/'continuation.inl')))
    source=FROZEN/'linear_system/solver/pcg_solver.cu';s=source.read_text(encoding='utf-8')
    begin='__global__ void PCG_vdv_Reduction(';end='namespace gipc\n{\nPCGSolver::PCGSolver'
    require(s.count(begin)==s.count(end)==1,'Native kernels boundary changed')
    target=overlay/'native_kernels.inl';target.write_text(s[s.index(begin):s.index(end)],encoding='utf-8')
    generation.append(dict(source=record(source),target=record(target),begin=begin,end=end))
    source=FROZEN/'linear_system/solver/pcg_native_balanced_target.inl';s=source.read_text(encoding='utf-8')
    begin='auto quality_probe=';end='const int quality_order';require(s.count(begin)==s.count(end)==1,'Recurrence boundary changed')
    s=s[s.index(begin):s.index(end)]
    changes=[('r.buffer_view().copy_from(b.buffer_view());','check(cudaMemcpyAsync(r.value,b.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));',1),
        ('p.copy_from(z);','check(cudaMemcpyAsync(p.value,z.value,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice,cudaStreamPerThread));',1),
        ('if(augmented)space->apply(r.data(),z.data(),base,nullptr,fused?ApplyMode::FusedProjection:ApplyMode::Reference);\n        else base(r.data(),z.data());','op(current_method);',1)]
    s=edit(s,changes)
    for name in ['work_x','residual_ax','quality_r','quality_z','r','z','b','p','Ap']:
        old=name+'.data()';count=s.count(old)
        if count:s=s.replace(old,name+'.value');changes.append((old,name+'.value',count))
    target=overlay/'recurrence.inl';target.write_text(s,encoding='utf-8');generation.append(dict(source=record(source),target=record(target),begin=begin,end=end,edits=changes))
    project=BUILD/'project';project.mkdir()
    for name in ['driver.cu','CMakeLists.txt']:(project/name).write_bytes((PROJECT/name).read_bytes())
    manifest=read(ROOT/'runs/innovation_p5_tiny_cache_20261017/input_manifest.json')
    manifest['cases']=[r for r in manifest['cases'] if r['probe']=='rhs']
    extra=[]
    for item in manifest['cases']:
        for key in ['rows','cols','values','rhs']:extra.append(record(Path(item['prefix']+'_'+key+'.bin')))
    manifest['pcg_inputs']=extra;write_new(RAW/'input_manifest.json',manifest)
    write_new(BUILD/'generation.json',dict(files=generation));write_new(RAW/'production_before.json',tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')])
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),files=len(read(PARENT)['files'])))
    print(dict(prepared=True,cases=len(manifest['cases'])),flush=True)
def build():
    require(not (BUILD/'identity.json').exists(),'Already built')
    sources=tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+tree(FROZEN)+[record(Path(__file__))]
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(BUILD/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),'-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log');command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','cache_pcg','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/cache_pcg.exe'),runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],
        compilers=compiler_records(BUILD),configure=configure,command=command,build_evidence=[record(BUILD/p) for p in ['CMakeCache.txt','cache_pcg.vcxproj','configure.log','build.log','sources_before.json','generation.json']]))
    print(dict(compiled=True,exe=record(BUILD/'Release/cache_pcg.exe')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args();prepare() if a.action=='prepare' else build()
