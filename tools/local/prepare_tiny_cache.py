"""Generate finite three-arm cache probe from immutable audited fixtures."""
import argparse
import ast
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new
from prepare_tiny_prefetch import CUDA,FROZEN,read
RAW=ROOT/'runs/innovation_p5_tiny_cache_20261017'
REPORT=ROOT/'reports/innovation_p5_tiny_cache_20261017'
BUILD=ROOT/'build/tiny_cache_20261017'
PROJECT=ROOT/'tools/bench/tiny_cache'
PARENT=ROOT/'reports/innovation_p5_tiny_review_20261017/DELIVERY_SEAL.json'
OLD=ROOT/'build/tiny_prefetch_20261017_retry1'
GEN=[]

def transform(source,edits):
    for old,new,count in edits:
        require(source.count(old)==count,'Ambiguous generated patch: '+old[:80]);source=source.replace(old,new)
    return source

def generated(source,target,edits=(),suffix=''):
    data=Path(source).read_text(encoding='utf-8');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(transform(data,edits)+suffix,encoding='utf-8')
    GEN.append(dict(source=record(source),target=record(target),edits=[dict(old=a,new=b,count=c) for a,b,c in edits],suffix=suffix))

def functions(path,names):
    source=path.read_text(encoding='utf-8');lines=source.splitlines(keepends=True);nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    return '\n'.join(''.join(lines[nodes[name].lineno-1:nodes[name].end_lineno]) for name in names)

def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh paths required');verify(read(PARENT)['files'])
    old_identity=read(OLD/'identity.json');verify(old_identity['sources']);RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir()
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc','linear_system/solver/native_balanced_math.h']:
        generated(OLD/'overlay'/name,overlay/name)
    kernel=(PROJECT/'cached_kernel.cuh').read_text(encoding='utf-8')
    # Ignore only layout/pragmas and the two storage names; both original loops
    # must still occur once, with identical ascending/descending arithmetic.
    old_header=(FROZEN/'linear_system/solver/native_balanced_gpu.cuh').read_text(encoding='utf-8')
    old_body=old_header.split('static __global__ void balanced_tiny_solve(')[1].split('static __global__ void balanced_difference')[0]
    candidate=re.sub(r'#pragma unroll','',kernel).replace('lower[','L[').replace('work[','a[')
    candidate=re.sub(r'\s+','',candidate)
    for line in old_body.splitlines():
        if line.strip().startswith('for(int i='):require(candidate.count(re.sub(r'\s+','',line))==1,'Changed arithmetic order')
    edits=[('static __global__ void balanced_difference(',kernel+'\nstatic __global__ void balanced_difference(',1),
        ('bool prefetch=false','int tiny_mode=0',1)]
    for pointer in ['a','c']:
        old='if(prefetch)balanced_tiny_solve_prefetch<<<1,32,0,cudaStreamPerThread>>>(L.value,'+pointer+'.value);'
        new='if(tiny_mode==2)balanced_tiny_solve_cached<<<1,32,0,cudaStreamPerThread>>>(L.value,'+pointer+'.value);\n        else if(tiny_mode==1)balanced_tiny_solve_prefetch<<<1,32,0,cudaStreamPerThread>>>(L.value,'+pointer+'.value);'
        edits.append((old,new,1))
    generated(OLD/'overlay/linear_system/solver/native_balanced_gpu.cuh',overlay/'linear_system/solver/native_balanced_gpu.cuh',edits)
    generated(PROJECT/'graph_info.inl',overlay/'graph_tiny_info.inl')
    # Preserve all setup to the exact audited workspace boundary.
    template=OLD/'overlay/tiny_fixture.inl';s=template.read_text(encoding='utf-8');boundary='        unsigned char* workspace=workspaceStorage.value+guard_bytes;'
    require(s.count(boundary)==1,'Setup boundary differs');prefix=s.split(boundary)[0]
    prefix_edits=[('std::array<std::string,3> methods={"MAS","original","prefetch"}','std::array<std::string,4> methods={"MAS","original","prefetch","cached"}',1)]
    prefix=transform(prefix,prefix_edits)
    continuation=ROOT/'tools/bench/tiny_prefetch/continuation.inl'
    cont_edits=[('Mode::FusedProjection,method==2','Mode::FusedProjection,method-1',1),
        ('std::array<std::vector<double>,3> cpu=','std::array<std::vector<double>,4> cpu=',1),
        ('n),{}};cpu[2]=cpu[1];','n),{},{}};cpu[2]=cpu[3]=cpu[1];',1),
        ('std::array<std::vector<double>,3> host_reference','std::array<std::vector<double>,4> host_reference',1),
        ('std::array<std::array<std::unique_ptr<Graph>,2>,3> graphs','std::array<std::array<std::unique_ptr<Graph>,2>,4> graphs',1),
        ('method<3','method<4',1),('if(method==2)same(host_reference[1],host_reference[2]);','if(method>=2)same(host_reference[1],host_reference[method]);',1),
        ('method==2,&opaque','method?method-1:0,&opaque',1),('if(method==2)same(v,host_reference[1]);','if(method>=2)same(v,host_reference[1]);',2),
        ('comparison_order<2','comparison_order<3',1),('comparison=(comparison_order+quartet+round)%2','comparison=(comparison_order+quartet+round)%3',1),
        ('const int method=comparison?(label?2:1):0;','const int method=comparison?(label?3:comparison):0;',1),
        ('comparison?"AB":"AA"','comparison==0?"AA":comparison==1?"vs_original":"vs_prefetch"',1),
        ('execute(2,0,16)','execute(3,0,16)',1),('(measure?384:0)','(measure?576:0)',1)]
    generated(continuation,overlay/'continuation.inl',cont_edits)
    fixture=overlay/'tiny_fixture.inl';fixture.write_text(prefix+(overlay/'continuation.inl').read_text(encoding='utf-8'),encoding='utf-8')
    write_new(BUILD/'fixture_generation.json',dict(template=record(template),boundary=boundary,prefix_edits=[dict(old=a,new=b,count=c) for a,b,c in prefix_edits],
        continuation=record(overlay/'continuation.inl'),target=record(fixture)))
    # Standalone tiny paths, including same-object original A/A and two paired B comparisons.
    old_project=ROOT/'tools/bench/tiny_prefetch';small=old_project/'tiny_contracts.inl'
    small_edits=[('if(method)gipc::balanced_tiny_solve_prefetch','if(method==2)gipc::balanced_tiny_solve_cached<<<1,32,0,cudaStreamPerThread>>>(dl.value+16,da.value+16);\n            else if(method==1)gipc::balanced_tiny_solve_prefetch',1),
        ('std::array<std::array<std::unique_ptr<Graph>,2>,2> graphs','std::array<std::array<std::unique_ptr<Graph>,2>,3> graphs',1),
        ('std::array<std::vector<double>,2> host','std::array<std::vector<double>,3> host',1),
        ('for(int method=0;method<2;++method)','const char* methods[3]={"original","prefetch","cached"};\n        for(int method=0;method<3;++method)',1),
        ('method?"prefetch":"original"','methods[method]',4),('host[0].data(),host[1].data()','host[0].data(),host[method].data()',1),
        ('method==1);','method);',1),
        ('for(int position=0;position<4;++position){','for(int comparison_order=0;comparison_order<3;++comparison_order)for(int position=0;position<4;++position){',1),
        ('const int method=int(outer)^((round+quartet)%2),width=mode?16:1;',
         'const int label=int(outer)^((round+quartet)%2),comparison=(comparison_order+quartet+round)%3;\n                    const int method=comparison?(label?2:comparison-1):0,width=mode?16:1;',1),
        ('+"_p"+std::to_string(position+1);','+"_c"+std::to_string(comparison)+"_p"+std::to_string(position+1);',1),
        ('{"label",method?"B":"A"}','{"label",label?"B":"A"}',1),
        ('{"comparison","AB"}','{"comparison",comparison==0?"AA":comparison==1?"vs_original":"vs_prefetch"}',1),('(measure?192:0)','(measure?576:0)',1)]
    generated(small,overlay/'tiny_contracts.inl',small_edits)
    project=BUILD/'project';generated(old_project/'driver.cu',project/'driver.cu')
    cmake=(old_project/'CMakeLists.txt').read_text();generated(old_project/'CMakeLists.txt',project/'CMakeLists.txt',[('tiny_prefetch','tiny_cache',cmake.count('tiny_prefetch'))])
    # Generate only the audited numeric and strict metadata functions, not legacy CLI actions.
    a=ROOT/'tools/bench/analyze_tiny_prefetch.py';review=ROOT/'tools/bench/tiny_prefetch_review.py'
    numeric=functions(a,['csv_new','check_graphs','tiny_audit','full_audit'])
    numeric_edits=[('(4 if tiny else 6)','(6 if tiny else 8)',1),('(32 if method==\'prefetch\' else 1)','(1 if method==\'original\' else 32)',1),
        ('(512 if method==\'prefetch\' else 0)','(0 if method==\'original\' else 512)',1),
        ("len(c['warm'])==6","len(c['warm'])==9",1), # only tiny occurrence; full handled below with separate text
        ('dict(MAS=base,original=value,prefetch=value)','dict(MAS=base,original=value,prefetch=value,cached=value)',1),
        ("method!='prefetch'","method not in ['prefetch','cached']",1)]
    # The warm predicate appears twice; replace scoped function bodies instead.
    numeric_edits=[e for e in numeric_edits if e[0]!="len(c['warm'])==6"]
    numeric=transform(numeric,numeric_edits)
    numeric=numeric.replace("len(c['warm'])==6,'Wrong tiny warm count'","len(c['warm'])==9,'Wrong tiny warm count'").replace("len(c['warm'])==6,'Wrong full warm count'","len(c['warm'])==8,'Wrong full warm count'")
    strict=functions(review,['exact','finite','strict_metadata','independent_statistics'])
    strict_edits=[("['original','prefetch'] if tiny else ['MAS','original','prefetch']","['original','prefetch','cached'] if tiny else ['MAS','original','prefetch','cached']",1),
        ('v1.check_graphs(case,tiny)','check_graphs(case,tiny)',1),
        ("require(row['comparison']=='AB' if tiny else row['comparison'] in ['AA','AB'],'Wrong comparison')","require(row['comparison'] in ['AA','vs_original','vs_prefetch'],'Wrong comparison')",1),
        ("tag+=('_p'+str(row['position'])) if tiny else '_c'+str(int(row['comparison']=='AB'))+'_p'+str(row['position'])",
         "tag+='_c'+str(['AA','vs_original','vs_prefetch'].index(row['comparison']))+'_p'+str(row['position'])",1),
        ('tiny_measure=192,measure=384','tiny_measure=576,measure=576',1),
        ('    if measure:v1.paired_stats(data,tiny) # exact independent native-order reconstruction\n','',1),
        ("if cell[2]=='AB' else None","if cell[2]!='AA' else None",1)]
    strict=transform(strict,strict_edits)
    imports='from pathlib import Path\nimport csv,math,statistics as st\nfrom decimal import Decimal,localcontext\nfrom linux_runner import require\nfrom m3_numeric import FrozenSystem,rel\nimport numpy as np\nfrom scipy.linalg import cho_solve\n'
    validation=overlay/'validation_impl.py';validation.write_text(imports+numeric+'\n'+strict,encoding='utf-8');ast.parse(validation.read_text())
    write_new(BUILD/'validation_generation.json',dict(numeric_source=record(a),numeric_functions=['csv_new','check_graphs','tiny_audit','full_audit'],
        review_source=record(review),review_functions=['exact','finite','strict_metadata','independent_statistics'],numeric_edits=[dict(old=x,new=y,count=z) for x,y,z in numeric_edits],
        warm_replacements={'tiny':9,'full':8},strict_edits=[dict(old=x,new=y,count=z) for x,y,z in strict_edits],imports=imports,target=record(validation)))
    write_new(BUILD/'generation.json',dict(files=GEN,kernel=record(PROJECT/'cached_kernel.cuh')))
    manifest=read(ROOT/'runs/innovation_p5_tiny_prefetch_20261017_retry1/input_manifest.json')
    verify(manifest['original_inputs']+manifest['generated_inputs']);manifest.update(new_source=record(Path(__file__)),protocol=record(REPORT/'PLAN.md'))
    write_new(RAW/'input_manifest.json',manifest);write_new(RAW/'production_before.json',tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')])
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),files=len(read(PARENT)['files']),original_sources=read(OLD/'identity.json')['sources']))
    print(dict(prepared=True,parent_files=len(read(PARENT)['files'])),flush=True)

def build():
    require(not (BUILD/'identity.json').exists(),'Already built');parent=read(REPORT/'PARENT.json');verify([parent['parent']]+parent['original_sources'])
    sources=tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+[record(Path(__file__))]+parent['original_sources']
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(BUILD/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),'-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log');command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','tiny_cache','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/tiny_cache.exe'),runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],
        compilers=compiler_records(BUILD),configure=configure,command=command,build_evidence=[record(BUILD/p) for p in ['CMakeCache.txt','tiny_cache.vcxproj','configure.log','build.log','sources_before.json','generation.json','fixture_generation.json','validation_generation.json']]))
    print(dict(compiled=True,exe=record(BUILD/'Release/tiny_cache.exe')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args();prepare() if a.action=='prepare' else build()
