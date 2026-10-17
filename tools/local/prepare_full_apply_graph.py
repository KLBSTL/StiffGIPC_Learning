"""Additive, exact-source native MAS replay with fixed parent systems."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new
RAW=ROOT/'runs/innovation_p5_full_apply_graph_20261017'
REPORT=ROOT/'reports/innovation_p5_full_apply_graph_20261017'
BUILD=ROOT/'build/full_apply_graph_20261017'
PROJECT=ROOT/'tools/bench/full_apply_graph'
FROZEN=ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC'
CUDA=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def prepare():
    import numpy as np
    from scipy.linalg import cho_solve
    from m3_numeric import FrozenSystem,rel
    require(not RAW.exists() and not BUILD.exists(),'Fresh phase required')
    parents=[REPORT.parent/'innovation_p5_projection_probe_20261016/DELIVERY_SEAL.json',
             REPORT.parent/'innovation_p5_fused_apply_20261016/DELIVERY_SEAL.json']
    for p in parents:verify(read(p)['files'])
    RAW.mkdir();BUILD.mkdir();overlay=BUILD/'overlay/solver';overlay.mkdir(parents=True)
    frozen=tree(FROZEN);known={r['path']:r for r in read(parents[1])['files']}
    require(all(known.get(r['path'])==r for r in frozen),'Frozen original differs')
    header=(FROZEN/'solver/MASPreconditioner.cuh').read_bytes()
    anchor=b'    static int replay_fixture('
    require(header.count(anchor)==1,'Static declaration insertion ambiguous')
    addition=b'    static int full_apply_graph_fixture(const char*,const char*,bool);\n'
    modified=header.replace(anchor,addition+anchor)
    (overlay/'MASPreconditioner.cuh').write_bytes(modified)
    abd=(FROZEN/'linear_system/preconditioner/abd_preconditioner.cu').read_bytes()
    start=abd.index(b'namespace\n{') if b'namespace\n{' in abd else abd.index(b'namespace\r\n{')
    end=abd.index(b'}  // namespace',start)+len(b'}  // namespace')
    kernel=abd[start:end]
    require(b'abd_preconditioner_apply_kernel' in kernel,'ABD kernel extraction failed')
    (BUILD/'overlay/abd_kernel.inc').write_bytes(kernel+b'\n')
    names=re.findall(rb'MAS_SAVE\((\w+)\);',header)
    require(len(names)==len(set(names)) and len(names)>=30,'MAS buffer enumeration ambiguous')
    (BUILD/'overlay/load_buffers.inc').write_text('\n'.join('load(m.'+x.decode()+',"'+x.decode()+'");' for x in names)+'\n',encoding='utf-8')
    write_new(REPORT/'PARENT.json',dict(parents=[dict(seal=record(p),verified_files=len(read(p)['files'])) for p in parents],
        original_source_files=frozen,existing_source_edits=[],overlay_static_declaration_only=True))
    originals=[];cases=[];dest=RAW/'inputs';dest.mkdir()
    for label,scene,frame in [('bunny','Bunny',14),('mixed','Mixed',15)]:
        prefix=ROOT/'runs/innovation_p5_fused_apply_20261016'/label/'fixed'/f'f{frame}_n2'
        files=[record(p) for p in sorted(prefix.parent.glob(prefix.name+'*')) if p.is_file()]
        require(all(known.get(r['path'])==r for r in files),'Parent snapshot differs');originals+=files
        s=FrozenSystem(prefix);study=read(str(prefix)+'_native_balanced_study.json');n=len(s.full_b)
        Z=np.fromfile(study['target_Z_file'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(study['target_W_file'],dtype='<f8').reshape(n,8,order='F')
        L=np.fromfile(study['target_L_file'],dtype='<f8').reshape(8,8,order='F')
        require(rel(W,s.full_A@Z)<1e-9 and rel(L@L.T,.5*(Z.T@W+W.T@Z))<1e-9,'CSR/coarse mismatch')
        row=next(r for r in study['runs'] if r['pair']==1 and r['requested_balanced'] and not r['requested_fused'])
        x=np.fromfile(row['solution_file'],dtype='<f8');residual=s.full_b-s.full_A@x
        for name,r in [('rhs',s.full_b),('residual',residual),('zero',np.zeros(n))]:
            out=dest/(label+'_'+name);out.mkdir()
            base=s.full_apply(r);a=cho_solve((L,True),Z.T@r);u=s.full_apply(r-W@a)
            expected=u+Z@(a-cho_solve((L,True),W.T@u))
            arrays=dict(r=r,MAS_expected=base,balanced_expected=expected)
            require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite expected')
            generated={}
            for k,v in arrays.items():
                f=out/(k+'.bin')
                with f.open('xb') as h:np.asarray(v,dtype='<f8').tofile(h)
                generated[k]=str(f.resolve())
            cases.append(dict(name=label+'_'+name,scene=scene,probe=name,n=n,prefix=str(prefix.resolve()),
                files=generated,Z=study['target_Z_file'],W=study['target_W_file'],L=study['target_L_file'],
                historical_b=dict(MAS=study['MAS_b_action_file'],reference=study['balanced_b_action_file'],
                                  fused=study['fused_b_action_file']) if name=='rhs' else {}))
    write_new(RAW/'input_manifest.json',dict(cases=cases,original_inputs=originals,generated_inputs=tree(dest),
        source=record(Path(__file__)),protocol=record(REPORT/'PLAN.md'),calls=16,rounds=3,groups=6,
        methods=['MAS','reference','fused'],executions=['host16','graph16','graph1x16']))
    print(json.dumps(dict(prepared=True,cases=len(cases),frozen_files=len(frozen),buffers=len(names))),flush=True)

def build():
    parent=read(REPORT/'PARENT.json');verify(parent['original_source_files']+[r['seal'] for r in parent['parents']])
    require(not (BUILD/'identity.json').exists(),'Build already sealed')
    sources=tree(PROJECT)+tree(BUILD/'overlay')+parent['original_source_files']+[record(Path(__file__))]
    command=[str(CMAKE),'-S',str(PROJECT),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86','-DFROZEN_ROOT='+str(FROZEN),
        '-DOVERLAY_ROOT='+str(BUILD/'overlay'),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(command,BUILD/'configure.log')
    compile_command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','full_apply_graph','--','/m:2','/v:normal']
    checked(compile_command,BUILD/'build.log');verify(sources)
    log=(BUILD/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    require('driver.cu' in log and 'nvcc.exe' in log.lower(),'CUDA compile evidence missing')
    write_new(BUILD/'identity.json',dict(sources=sources,exe=record(BUILD/'Release/full_apply_graph.exe'),
        runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))],
        compilers=compiler_records(BUILD),configure=command,command=compile_command,
        build_evidence=[record(BUILD/name) for name in ['CMakeCache.txt','full_apply_graph.vcxproj','configure.log','build.log']],
        parents=parent['parents'],production_solver_modified=False))
    print(json.dumps(dict(compiled=True,binary=record(BUILD/'Release/full_apply_graph.exe'))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.action=='prepare' else build()
