"""Fresh private L-prefetch probe. Parent and production are read-only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new
import numpy as np
from scipy.linalg import cho_solve
RAW=ROOT/'runs/innovation_p5_tiny_prefetch_20261017_retry1'
REPORT=ROOT/'reports/innovation_p5_tiny_prefetch_20261017_retry1'
BUILD=ROOT/'build/tiny_prefetch_20261017_retry1'
PROJECT=ROOT/'tools/bench/tiny_prefetch'
PARENT=ROOT/'reports/innovation_p5_cost_split_20261017/DELIVERY_SEAL.json'
FROZEN=ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC'
CUDA=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def edited(source,edits):
    for a,b in edits:
        require(source.count(a)==1,'Ambiguous literal patch: '+a[:80]);source=source.replace(a,b)
    return source

def tiny_inputs(cases):
    """Deterministic bounded inputs, including poisoned unused triangle."""
    folder=RAW/'tiny_inputs';folder.mkdir();items=[]
    def add(name,L,b,kind):
        L=np.array(L,dtype='<f8',order='F');b=np.array(b,dtype='<f8')
        lower=np.tril(L);require(np.isfinite(lower).all() and np.all(np.diag(lower)>0),'Invalid lower factor')
        cond=float(np.linalg.cond(lower)**2);require(cond<1e8,'Unbounded SPD condition')
        a=folder/(name+'_L.bin');rhs=folder/(name+'_b.bin')
        with a.open('xb') as f:f.write(L.tobytes(order='F'))
        with rhs.open('xb') as f:f.write(b.tobytes())
        items.append(dict(name=name,L=str(a),b=str(rhs),condition=cond,kind=kind,
            expected=cho_solve((lower,True),b,check_finite=True).tolist()))
    for c in cases:
        if c['probe']!='rhs':continue
        n=c['n'];L=np.fromfile(c['L'],dtype='<f8').reshape(8,8,order='F')
        Z=np.fromfile(c['Z'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(c['W'],dtype='<f8').reshape(n,8,order='F')
        r=np.fromfile(c['files']['r'],dtype='<f8');first=Z.T@r
        # An independent deterministic second projection exercises another RHS.
        base=np.fromfile(c['files']['MAS_expected'],dtype='<f8');second=W.T@base
        add(c['scene'].lower()+'_first',L,first,'actual_first_projection')
        add(c['scene'].lower()+'_second',L,second,'actual_W_projection_on_MAS_r')
        add(c['scene'].lower()+'_zero',L,np.zeros(8),'actual_zero')
    rng=np.random.default_rng(20261017)
    for k in range(64):
        diagonal=k%4==0;scale=[1e-6,1.,1e6,1.][k%4]
        L=np.diag(np.geomspace(1.,[1.,4.,16.,64.][k%4],8))
        if not diagonal:L+=np.tril(rng.normal(size=(8,8))*.15,-1)
        L*=scale;b=rng.normal(size=8)*[1e-12,1.,1e12,1.][k%4]
        if k%8==7:b[:]=0
        if k%2:L[np.triu_indices(8,1)]=np.nan
        add('synthetic_'+str(k),L,b,'diagonal' if diagonal else 'dense_lower_upper_nan' if k%2 else 'dense_lower')
    require(len(items)==70,'Unexpected tiny case count')
    return items,tree(folder)

def prepare():
    require(not RAW.exists() and not BUILD.exists(),'Fresh phase required')
    parent=read(PARENT);verify(parent['files']);old=ROOT/'build/cost_split_20261017'
    identity=read(old/'identity.json');verify(identity['sources'])
    template=ROOT/'tools/bench/full_apply_graph/fixture.inl'
    require(record(template) in read(ROOT/'build/full_apply_graph_20261017/identity.json')['sources'],'Unsealed template')
    anchor='        auto op=[&](int method,const Space::Profile* profile=nullptr){'
    source=template.read_text(encoding='utf-8');require(source.count(anchor)==1,'Boundary ambiguous')
    edits=[('int MASPreconditioner::full_apply_graph_fixture(const char* manifest_path,const char* raw_output,bool measure) {',
        '#include "graph_tiny_info.inl"\nint MASPreconditioner::tiny_prefetch_fixture(const char* manifest_path,const char* raw_output,const char* action_raw) {\n    const std::string action=action_raw;\n    gipc::balanced_require(action=="contracts"||action=="measure","Unknown action");\n    const bool measure=action=="measure";'),
        ('out/(measure?"measurements.json":"contracts.json")','out/"data.json"'),
        ('if(measure&&probe=="zero")continue;','if(measure&&probe!="rhs")continue;'),
        ('const std::array<std::string,3> methods={"MAS","reference","fused"};',
         'const std::array<std::string,3> methods={"MAS","original","prefetch"};'),
        ('gipc::BalancedBuffer<double> r(n),z(n);Space space(n);',
         'gipc::BalancedBuffer<double> r(n),z(n);\n        constexpr size_t workspace_bytes=4*1024*1024,guard_bytes=256;\n        gipc::BalancedBuffer<unsigned char> workspaceStorage(workspace_bytes+2*guard_bytes);Space space(n);')]
    prefix=edited(source.split(anchor)[0],edits)
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay';overlay.mkdir()
    for name in ['solver/MASPreconditioner.cuh','abd_kernel.inc','load_buffers.inc']:
        p=old/'overlay'/name;require(record(p) in identity['sources'],'Parent overlay absent from source identity')
        data=p.read_bytes()
        if name.endswith('.cuh'):
            a=b'    static int full_apply_cost_fixture(const char*,const char*,const char*);'
            require(data.count(a)==1,'Declaration ambiguous');data=data.replace(a,b'    static int tiny_prefetch_fixture(const char*,const char*,const char*);')
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    original=FROZEN/'linear_system/solver/native_balanced_gpu.cuh'
    target=overlay/'linear_system/solver/native_balanced_gpu.cuh';target.parent.mkdir(parents=True)
    kernel=(PROJECT/'tiny_kernel.cuh').read_text(encoding='utf-8')
    header_edits=[('static __global__ void balanced_difference(',kernel+'\nstatic __global__ void balanced_difference('),
        ('const Profile* profile=nullptr,ApplyMode mode=ApplyMode::Reference) {',
         'const Profile* profile=nullptr,ApplyMode mode=ApplyMode::Reference,bool prefetch=false) {')]
    for pointer in ['a','c']:
        call='balanced_tiny_solve<<<1,1,0,cudaStreamPerThread>>>(L.value,'+pointer+'.value);'
        header_edits.append((call,'if(prefetch)balanced_tiny_solve_prefetch<<<1,32,0,cudaStreamPerThread>>>(L.value,'+pointer+'.value);\n        else '+call))
    header=edited(original.read_text(encoding='utf-8'),header_edits);target.write_text(header,encoding='utf-8')
    # Quoted sibling include must resolve beside the private overlaid header.
    (target.parent/'native_balanced_math.h').write_bytes((original.parent/'native_balanced_math.h').read_bytes())
    # Algebraic loop bytes must match after substituting only the storage name.
    body=original.read_text(encoding='utf-8').split('static __global__ void balanced_tiny_solve(')[1].split('static __global__ void balanced_difference')[0]
    loops='\n'.join(line.strip() for line in body.splitlines() if line.strip().startswith('for(int i='))
    candidate='\n'.join(line.strip() for line in kernel.splitlines() if line.strip().startswith('for(int i='))
    require(loops==candidate.replace('lower[','L['),'Candidate reordered original arithmetic')
    continuation=PROJECT/'continuation.inl';generated=overlay/'tiny_fixture.inl'
    generated.write_text(prefix+continuation.read_text(encoding='utf-8'),encoding='utf-8')
    write_new(BUILD/'generation.json',dict(template=record(template),boundary=anchor,edits=[dict(old=a,new=b) for a,b in edits],
        header_original=record(original),header_edits=[dict(old=a,new=b) for a,b in header_edits],header_generated=record(target),
        kernel=record(PROJECT/'tiny_kernel.cuh'),arithmetic_loops_identical=True,continuation=record(continuation),generated=record(generated)))
    manifest=read(ROOT/'runs/innovation_p5_cost_split_20261017/input_manifest.json');verify(manifest['original_inputs'])
    tiny,generated_inputs=tiny_inputs(manifest['cases'])
    write_new(RAW/'input_manifest.json',dict(cases=manifest['cases'],original_inputs=manifest['original_inputs'],tiny_cases=tiny,
        generated_inputs=generated_inputs,source=record(Path(__file__)),protocol=record(REPORT/'PLAN.md'),calls=2048,rounds=3,quartets=4))
    write_new(REPORT/'PARENT.json',dict(parent=record(PARENT),verified_files=len(parent['files']),
        original_source_files=read(ROOT/'reports/innovation_p5_cost_split_20261017/PARENT.json')['original_source_files'],
        generation=record(BUILD/'generation.json'),existing_source_edits=[]))
    write_new(RAW/'production_before.json',tree(ROOT/'StiffGIPC')+[record(ROOT/'CMakeLists.txt')])
    print(dict(prepared=True,parent_files=len(parent['files']),tiny_cases=len(tiny)),flush=True)

def build(retry=False):
    parent=read(REPORT/'PARENT.json');verify(parent['original_source_files']+[parent['parent'],parent['generation']])
    require(not (BUILD/'identity.json').exists(),'Already built')
    if retry:
        require((BUILD/'build.log').is_file() and not (BUILD/'build_retry1.log').exists(),'Only explicit first build retry')
        p=BUILD/'overlay/linear_system/solver/native_balanced_math.h';require(not p.exists(),'Retry sibling already exists')
        p.write_bytes((FROZEN/'linear_system/solver/native_balanced_math.h').read_bytes())
        write_new(BUILD/'build_retry1_reason.json',dict(reason='Initial compile C1083: private header missing its quoted native_balanced_math.h sibling; copy original bytes, no math changes',
            failed_log=record(BUILD/'build.log'),old_sources=record(BUILD/'sources_before.json'),added=record(p)))
    sources=tree(PROJECT)+tree(BUILD/'overlay')+parent['original_source_files']+[record(Path(__file__))]
    write_new(BUILD/('sources_before_retry1.json' if retry else 'sources_before.json'),sources)
    configure=[str(CMAKE),'-S',str(PROJECT),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),
        '-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    if not retry:checked(configure,BUILD/'configure.log')
    command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','tiny_prefetch','--','/m:2','/v:normal']
    checked(command,BUILD/('build_retry1.log' if retry else 'build.log'));verify(sources)
    evidence=['CMakeCache.txt','tiny_prefetch.vcxproj','configure.log','build.log','sources_before.json','generation.json']
    if retry:evidence+=['build_retry1.log','sources_before_retry1.json','build_retry1_reason.json']
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/tiny_prefetch.exe'),
        runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],compilers=compiler_records(BUILD),
        configure=configure,command=command,build_evidence=[record(BUILD/n) for n in evidence],build_retry=retry,
        production_solver_modified=False))
    print(dict(compiled=True,binary=record(BUILD/'Release/tiny_prefetch.exe')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build','build_retry1']);a=p.parse_args()
    prepare() if a.action=='prepare' else build(a.action=='build_retry1')
