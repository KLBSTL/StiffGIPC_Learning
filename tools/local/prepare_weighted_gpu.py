"""Generate the minimal four-arm private extension of sealed native PCG."""
import argparse
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cache_pcg import CUDA,FROZEN,edit
from local_identity import record,verify,tree,compiler_records
from linux_runner import require,write_new
from build_windows import CMAKE,checked
RAW=ROOT/'runs/innovation_p5_weighted_gpu_20261017'
REPORT=ROOT/'reports/innovation_p5_weighted_gpu_20261017'
BUILD=ROOT/'build/weighted_gpu_20261017'
PROJECT=ROOT/'tools/bench/weighted_gpu'
PARENTS=[ROOT/'reports'/name/'DELIVERY_SEAL.json' for name in ['innovation_p5_mas_cached_20261017','innovation_p5_next_steps_20261017']]
OLD=ROOT/'build/mas_cached_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def function(source,name):
    s=source.read_text(encoding='utf-8');lines=s.splitlines(keepends=True)
    node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ''.join(lines[node.lineno-1:node.end_lineno])
def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh paths required')
    for p in PARENTS:verify(read(p)['files'])
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir();generation=[]
    def generate(source,target,changes=()):
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(edit(source.read_text(encoding='utf-8'),changes),encoding='utf-8')
        generation.append(dict(source=record(source),target=record(target),edits=changes))
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc','graph_tiny_info.inl','native_kernels.inl','cost_profile.h',
                 'linear_system/solver/native_balanced_math.h','linear_system/solver/native_balanced_gpu.cuh']:
        generate(OLD/'overlay'/name,overlay/name)
    generate(ROOT/'tools/bench/weighted_check/weighted_check_schedule.h',overlay/'weighted_check_schedule.h')
    for name in ['screen_block.inl','quality_event.inl']:generate(PROJECT/name,overlay/name)
    source=OLD/'overlay/fixture.inl';s=source.read_text(encoding='utf-8')
    start=s.index('        for(int mode=0;mode<2;++mode)for(int method=0;method<3;++method)\n            c["warm"]')
    end=s.index('        reset();check_state();guards();c["readonly_and_scratch_unchanged"]',start)
    changes=[('const int count=meta.at("blocks"),blocks=(n+255)/256,max_iter=2001;','const int count=meta.at("blocks"),blocks=(n+255)/256;int max_iter=2001;',1),
        ('{"graphs",Json::array()},{"warm",Json::array()}','{"graphs",Json::array()},{"budget_checks",Json::array()},{"warm",Json::array()}',1),
        ('int current_method=0,execution=0;','int current_method=0,execution=0;bool current_cooldown=false;',1),
        ('auto run=[&](int method,int mode,const std::string& tag){\n            current_method=method;execution=mode;',
         'auto run=[&](int arm,int mode,const std::string& tag){\n            const int method=arm%2?1:2;current_method=method;execution=mode;current_cooldown=arm>=2;',1),
        ('return Json{{"profile",profile},{"method",methods[method]}','return Json{{"arm",arm},{"policy",current_cooldown?"cooldown2":"legacy"},{"profile",profile},{"method",methods[method]}',1),
        (s[start:end],(PROJECT/'warm_measure.inl').read_text(encoding='utf-8'),1),('(measure?384:0)','(measure?1152:0)',0)]
    # Original uses the ternary without parentheses; match it exactly.
    changes[-1]=('measure?384:0','measure?1152:0',1)
    generate(source,overlay/'fixture.inl',changes)
    source=OLD/'overlay/recurrence.inl';s=source.read_text(encoding='utf-8')
    start=s.index('            if(screened) {');end=s.index('            if(due&&quality_probe',start)
    init='    policy_schedule=gipc::research::WeightedCheckSchedule{};\n    output["policy"]=current_cooldown?"cooldown2":"legacy";output["quality_events"]=Json::array();\n    output["schedule_queries"]=0;output["rhs_norm2"]=static_cast<double>(rhs_norm2);\n'
    changes=[('auto quality_probe=','gipc::research::WeightedCheckSchedule policy_schedule;\nauto quality_probe=',1),
        ('    point["quality_checked"]=true;','    #include "quality_event.inl"\n    point["quality_checked"]=true;',1),
        ('    costs.call("copy_rhs"',init+'    costs.call("copy_rhs"',1),
        ('    const int limit=std::min<int>(int(max_iter)-1,2000);','    const int limit=std::min<int>(int(max_iter)-1,2000);output["iteration_limit"]=limit;',1),
        ('{"recursive_relative_l2",nullptr}', '{"recursive_relative_l2",nullptr},{"recursive_norm_rr",nullptr},{"recursive_norm_evaluated",false},\n            {"policy_cooldown_before",current_cooldown&&policy_schedule.cooling_down()},{"policy_cooldown_after",current_cooldown&&policy_schedule.cooling_down()},{"schedule_due",false}',1),
        ('        if(quality_active) {','        if(quality_active||iteration==limit) {',1),
        (s[start:end],'            if(screened) {\n                #include "screen_block.inl"\n            }\n            point["schedule_due"]=due;\n',1)]
    generate(source,overlay/'recurrence.inl',changes)
    project=BUILD/'project';project.mkdir()
    for name in ['driver.cu','CMakeLists.txt']:
        source=OLD/'project'/name;s=source.read_text(encoding='utf-8')
        changes=[('mas_cached','weighted_gpu',s.count('mas_cached'))] if name=='CMakeLists.txt' else [('#include "fixture.inl"','#include "weighted_check_schedule.h"\n#include "fixture.inl"',1)]
        generate(source,project/name,changes)
    write_new(RAW/'input_manifest.json',read(ROOT/'runs/innovation_p5_mas_cached_20261017/input_manifest.json'))
    write_new(BUILD/'generation.json',dict(files=generation));write_new(RAW/'production_before.json',tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')])
    write_new(REPORT/'PARENTS.json',dict(parents=[record(p) for p in PARENTS]))
    print(dict(prepared=True,parents=len(PARENTS)),flush=True)
def build():
    require(not (BUILD/'identity.json').exists(),'Already built');sources=tree(PROJECT)+tree(BUILD/'overlay')+tree(BUILD/'project')+tree(FROZEN)+[record(Path(__file__))]
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(BUILD/'project'),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64','-DCMAKE_GENERATOR_INSTANCE=D:/vs2022',
        '-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),'-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log');command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','weighted_gpu','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/weighted_gpu.exe'),runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],
        compilers=compiler_records(BUILD),configure=configure,command=command,build_evidence=[record(BUILD/p) for p in ['CMakeCache.txt','weighted_gpu.vcxproj','configure.log','build.log','sources_before.json','generation.json']]))
    print(dict(compiled=True,exe=record(BUILD/'Release/weighted_gpu.exe')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args();prepare() if a.action=='prepare' else build()
