"""Frozen-input stage-only benchmark; no edits to the native simulation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree,compiler_records
from build_windows import CMAKE,checked
from linux_runner import require,write_new

RAW=ROOT/'runs/innovation_p5_projection_probe_20261016'
REPORT=ROOT/'reports/innovation_p5_projection_probe_20261016'
BUILD=ROOT/'build/projection_probe_20261016'
PROJECT=ROOT/'tools/bench/projection_probe'
FROZEN=ROOT/'build/fused_apply_snapshot_20261016/StiffGIPC/linear_system/solver'
CUDA=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def inputs():
    from m3_numeric import FrozenSystem,rel
    import numpy as np
    from scipy.linalg import cho_solve
    require(not RAW.exists(),'Fresh inputs required');RAW.mkdir();folder=RAW/'inputs';folder.mkdir()
    parent=read(REPORT/'PARENT.json')['parent'];verify([parent]);known={r['path']:r for r in read(parent['path'])['files']}
    sources=[];cases=[]
    def save(path,array):
        with path.open('xb') as f:np.asarray(array,dtype='<f8').ravel(order='F').tofile(f)
        return str(path.resolve())
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        prefix=ROOT/'runs/innovation_p5_fused_apply_20261016'/label/'fixed'/f'f{frame}_n2'
        for p in prefix.parent.glob(prefix.name+'*'):
            if p.is_file():
                r=record(p);require(known.get(r['path'])==r,'Parent input differs');sources.append(r)
        system=FrozenSystem(prefix);study=read(str(prefix)+'_native_balanced_study.json');n=len(system.full_b)
        Z=np.fromfile(study['target_Z_file'],dtype='<f8').reshape(n,8,order='F')
        W=np.fromfile(study['target_W_file'],dtype='<f8').reshape(n,8,order='F')
        L=np.fromfile(study['target_L_file'],dtype='<f8').reshape(8,8,order='F')
        require(rel(W,system.full_A@Z)<1e-9 and rel(L@L.T,.5*(Z.T@W+W.T@Z))<1e-9,'Coarse original-CSR check')
        target=next(r for r in study['runs'] if r['pair']==1 and r['requested_balanced'] and not r['requested_fused'])
        x=np.fromfile(target['solution_file'],dtype='<f8');residual=system.full_b-system.full_A@x
        for name,rhs in (('rhs',system.full_b),('residual',residual)):
            r=rhs.copy();a=cho_solve((L,True),Z.T@r);t=r-W@a;u=system.full_apply(t);c=cho_solve((L,True),W.T@u)
            output=u+Z@(a-c);require(np.isfinite(np.r_[r,u,a,c,t,output]).all(),'Nonfinite stage input')
            case_name=label+'_'+name;dest=folder/case_name;dest.mkdir()
            arrays=dict(Z=Z,W=W,r=r,u=u,a=a,c=c,input_expected=t,output_expected=output)
            files={k:save(dest/(k+'.bin'),v) for k,v in arrays.items()}
            calibration=None
            if name=='rhs':
                native=np.fromfile(study['balanced_b_action_file'],dtype='<f8');calibration=rel(output,native)
                require(calibration<=1e-3,'Stage CPU reference not calibrated to saved native(b)')
            cases.append(dict(name=case_name,scene=scene,probe=name,n=n,files=files,
                construction='original Z/W/L; CPU fixed symmetrized stored MAS coefficients in FP64',
                rhs_norm=float(np.linalg.norm(r)),CPU_output_vs_native_b=calibration))
    verify(sources)
    write_new(RAW/'input_manifest.json',dict(cases=cases,parent=parent,source=record(Path(__file__)),
        original_inputs=sources,generated_inputs=tree(folder),protocol=record(REPORT/'PLAN.md'),
        calls_per_batch=32,rounds=3,repeats=6,production_solver_modified=False))
    print(json.dumps(dict(inputs_prepared=True,cases=len(cases))),flush=True)

def build():
    require(not BUILD.exists(),'Fresh build required');BUILD.mkdir()
    parent=read(REPORT/'PARENT.json')['parent'];verify([parent]);known={r['path']:r for r in read(parent['path'])['files']}
    headers=[record(FROZEN/name) for name in ('native_balanced_gpu.cuh','native_balanced_math.h')]
    require(all(known.get(r['path'])==r for r in headers),'Frozen headers differ from parent seal')
    sources=tree(PROJECT)+headers+[record(Path(__file__))]
    command=[str(CMAKE),'-S',str(PROJECT),'-B',str(BUILD),'-G','Visual Studio 17 2022','-A','x64',
        '-DCMAKE_GENERATOR_INSTANCE=D:/vs2022','-DCMAKE_CUDA_ARCHITECTURES=86',
        '-DFROZEN_NATIVE_DIR='+str(FROZEN),'-DCUDAToolkit_ROOT='+str(CUDA)]
    checked(command,BUILD/'configure.log')
    compile_command=[str(CMAKE),'--build',str(BUILD),'--config','Release','--target','projection_probe','--','/m:2','/v:normal']
    checked(compile_command,BUILD/'build.log');verify(sources)
    log=(BUILD/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    require('probe.cu' in log and 'nvcc.exe' in log.lower(),'Actual CUDA command missing')
    runtime=[record(p) for p in sorted((CUDA/'bin/x64').glob('cublas*.dll'))];require(len(runtime)>=2,'cuBLAS runtime missing')
    write_new(BUILD/'identity.json',dict(parent=parent,sources=sources,source=record(Path(__file__)),
        configure=command,command=compile_command,exe=record(BUILD/'Release/projection_probe.exe'),
        runtime=runtime,compilers=compiler_records(BUILD),build_evidence=[record(BUILD/name) for name in
            ('CMakeCache.txt','projection_probe.vcxproj','configure.log','build.log')],
        scope='Standalone stage-only probe. Includes verified frozen native kernels. External SDK/header closure not fully sealed.',
        production_solver_modified=False))
    print(json.dumps(dict(compiled=True,exe=record(BUILD/'Release/projection_probe.exe'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inputs','build'));args=p.parse_args()
    inputs() if args.action=='inputs' else build()
