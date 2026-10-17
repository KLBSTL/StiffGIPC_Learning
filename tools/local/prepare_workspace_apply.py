"""Single-variable workspace experiment with an audited parent setup template."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new
RAW=ROOT/'runs/innovation_p5_workspace_20261017'
REPORT=ROOT/'reports/innovation_p5_workspace_20261017'
BUILD=ROOT/'build/workspace_apply_20261017'
PROJECT=ROOT/'tools/bench/workspace_apply'
PARENT=ROOT/'reports/innovation_p5_full_apply_graph_20261017/DELIVERY_SEAL.json'
FROZEN=ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC'
CUDA=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh phase required')
    parent=read(PARENT);verify(parent['files']);oldraw=ROOT/'runs/innovation_p5_full_apply_graph_20261017'
    oldbuild=ROOT/'build/full_apply_graph_20261017';identity=read(oldbuild/'identity.json');verify(identity['sources'])
    known={r['path']:r for r in parent['files']};fixture=ROOT/'tools/bench/full_apply_graph/fixture.inl'
    require(known.get(str(fixture.resolve()))==record(fixture),'Parent setup source differs')
    template=fixture.read_text(encoding='utf-8');anchor='        auto op=[&](int method,const Space::Profile* profile=nullptr){'
    require(template.count(anchor)==1,'Template boundary differs');prefix=template.split(anchor)[0]
    edits=[('    const auto manifest=read_json(manifest_path);const fs::path out=raw_output;',
            '    constexpr int calls=128;\n    const auto manifest=read_json(manifest_path);const fs::path out=raw_output;'),
        ('{"host16","graph16","graph1x16"}','{"host128","graph16x8","graph1x128"}'),
        ('gipc::BalancedBuffer<double> r(n),z(n);Space space(n);',
         'gipc::BalancedBuffer<double> r(n),z(n);\n        constexpr size_t workspace_bytes=4*1024*1024,guard_bytes=256;\n        gipc::BalancedBuffer<unsigned char> workspaceStorage(workspace_bytes+2*guard_bytes);\n        Space space(n);')]
    for old,new in edits:require(prefix.count(old)==1,'Template edit ambiguous');prefix=prefix.replace(old,new)
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir()
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc']:
        source=oldbuild/'overlay'/name;require(known.get(str(source.resolve()))==record(source),'Parent overlay differs')
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    continuation=(PROJECT/'continuation.inl').read_text(encoding='utf-8')
    (overlay/'workspace_fixture.inl').write_text(prefix+continuation,encoding='utf-8')
    write_new(BUILD/'generation.json',dict(parent_fixture=record(fixture),continuation=record(PROJECT/'continuation.inl'),
        edits=[dict(old=a,new=b) for a,b in edits],prefix_boundary=anchor,generated=record(overlay/'workspace_fixture.inl')))
    manifest=read(oldraw/'input_manifest.json');verify(manifest['original_inputs']+manifest['generated_inputs'])
    write_new(RAW/'input_manifest.json',dict(cases=manifest['cases'],original_inputs=manifest['original_inputs']+manifest['generated_inputs'],
        generated_inputs=[],source=record(Path(__file__)),protocol=record(REPORT/'PLAN.md'),calls=128,rounds=3,groups=6,
        methods=['MAS','reference','fused'],executions=['host128','graph16x8','graph1x128'],workspaces=['default','user4MiB']))
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),verified_files=len(parent['files']),
        original_source_files=read(ROOT/'reports/innovation_p5_full_apply_graph_20261017/PARENT.json')['original_source_files'],
        generation=record(BUILD/'generation.json'),existing_source_edits=[]))
    print(json.dumps(dict(prepared=True,parent_files=len(parent['files']),cases=6)),flush=True)

def build():
    parent=read(REPORT/'PARENT.json');verify(parent['original_source_files']+[parent['parent'],parent['generation']])
    require(not (BUILD/'identity.json').exists(),'Already built')
    sources=tree(PROJECT)+tree(BUILD/'overlay')+parent['original_source_files']+[record(Path(__file__))]
    write_new(BUILD/'sources_before.json',sources)
    configure=[str(CMAKE),'-S',str(PROJECT),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),
        '-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(configure,BUILD/'configure.log')
    command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','workspace_apply','--','/m:2','/v:normal']
    checked(command,BUILD/'build.log');verify(sources)
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/workspace_apply.exe'),
        runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],compilers=compiler_records(BUILD),
        configure=configure,command=command,parents=[dict(seal=parent['parent'])],production_solver_modified=False,
        build_evidence=[record(BUILD/n) for n in ['CMakeCache.txt','workspace_apply.vcxproj','configure.log','build.log','sources_before.json','generation.json']]))
    print(json.dumps(dict(compiled=True,binary=record(BUILD/'Release/workspace_apply.exe'))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.action=='prepare' else build()
