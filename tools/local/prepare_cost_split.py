"""Additive native cost/stability probe; no production or parent modifications."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new
RAW=ROOT/'runs/innovation_p5_cost_split_20261017'
REPORT=ROOT/'reports/innovation_p5_cost_split_20261017'
BUILD=ROOT/'build/cost_split_20261017'
PROJECT=ROOT/'tools/bench/cost_split'
PARENT=ROOT/'reports/innovation_p5_workspace_20261017/DELIVERY_SEAL.json'
FROZEN=ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC'
CUDA=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
NSYS=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh phase required')
    parent=read(PARENT);verify(parent['files'])
    old=ROOT/'build/workspace_apply_20261017';identity=read(old/'identity.json');verify(identity['sources'])
    template=ROOT/'tools/bench/full_apply_graph/fixture.inl';old_parent=read(ROOT/'build/full_apply_graph_20261017/identity.json')
    require(record(template) in old_parent['sources'],'Original setup is not compiled parent source')
    source=template.read_text(encoding='utf-8');anchor='        auto op=[&](int method,const Space::Profile* profile=nullptr){'
    require(source.count(anchor)==1,'Setup boundary ambiguous');prefix=source.split(anchor)[0]
    edits=[('int MASPreconditioner::full_apply_graph_fixture(const char* manifest_path,const char* raw_output,bool measure) {',
        'int MASPreconditioner::full_apply_cost_fixture(const char* manifest_path,const char* raw_output,const char* action_raw) {\n    const std::string action=action_raw;\n    balanced_require(action=="contracts"||action=="stability"||action=="profile"||action=="trace","Unknown action");\n    const bool measure=action!="contracts";bool capture_started=false;'),
        ('out/(measure?"measurements.json":"contracts.json")','out/"data.json"'),
        ('if(measure&&probe=="zero")continue;','if(measure&&probe!="rhs")continue;'),
        ('gipc::BalancedBuffer<double> r(n),z(n);Space space(n);',
        'gipc::BalancedBuffer<double> r(n),z(n);\n        constexpr size_t workspace_bytes=4*1024*1024,guard_bytes=256;\n        gipc::BalancedBuffer<unsigned char> workspaceStorage(workspace_bytes+2*guard_bytes);\n        Space space(n);')]
    for a,b in edits:require(prefix.count(a)==1,'Template edit ambiguous');prefix=prefix.replace(a,b)
    # The new entry is outside namespace full_apply_probe; qualify this one guard.
    prefix=prefix.replace('    balanced_require(action==','    gipc::balanced_require(action==')
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir()
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc']:
        p=old/'overlay'/name;require(record(p) in identity['sources'],'Parent overlay not sealed')
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes()
        if name.endswith('.cuh'):
            a=b'    static int full_apply_graph_fixture(const char*,const char*,bool);'
            require(data.count(a)==1,'Static declaration ambiguous');data=data.replace(a,b'    static int full_apply_cost_fixture(const char*,const char*,const char*);')
        target.write_bytes(data)
    continuation=PROJECT/'continuation.inl';generated=overlay/'cost_fixture.inl'
    generated.write_text(prefix+continuation.read_text(encoding='utf-8'),encoding='utf-8')
    write_new(BUILD/'generation.json',dict(template=record(template),boundary=anchor,edits=[dict(old=a,new=b) for a,b in edits],
        qualification=dict(old='    balanced_require(action==',new='    gipc::balanced_require(action=='),
        continuation=record(continuation),generated=record(generated)))
    manifest=read(ROOT/'runs/innovation_p5_workspace_20261017/input_manifest.json');verify(manifest['original_inputs'])
    write_new(RAW/'input_manifest.json',dict(cases=manifest['cases'],original_inputs=manifest['original_inputs'],generated_inputs=[],
        source=record(Path(__file__)),protocol=record(REPORT/'PLAN.md'),calls=[128,512,2048],rounds=3,quartets=4,profile_calls=32))
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),verified_files=len(parent['files']),original_source_files=read(ROOT/'reports/innovation_p5_workspace_20261017/PARENT.json')['original_source_files'],
        generation=record(BUILD/'generation.json'),existing_source_edits=[]))
    print(json.dumps(dict(prepared=True,parent_files=len(parent['files']))),flush=True)

def build():
    parent=read(REPORT/'PARENT.json');verify(parent['original_source_files']+[parent['parent'],parent['generation']])
    require(not (BUILD/'identity.json').exists(),'Already built')
    sources=tree(PROJECT)+tree(BUILD/'overlay')+parent['original_source_files']+[record(Path(__file__))]
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(PROJECT),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),
        '-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log');command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','cost_split','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    import subprocess
    version=subprocess.run([str(NSYS),'--version'],capture_output=True,check=True,timeout=15)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/cost_split.exe'),
        runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],compilers=compiler_records(BUILD),
        nsys=dict(binary=record(NSYS),version=(version.stdout+version.stderr).decode(errors='replace')),
        configure=configure,command=command,build_evidence=[record(BUILD/n) for n in ['CMakeCache.txt','cost_split.vcxproj','configure.log','build.log','sources_before.json','generation.json']],
        production_solver_modified=False))
    print(json.dumps(dict(compiled=True,binary=record(BUILD/'Release/cost_split.exe'))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.action=='prepare' else build()
