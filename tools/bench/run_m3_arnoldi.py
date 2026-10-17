"""Independent, RHS-free low-eigenmode oracle; every returned mode is checked."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigs,ArpackNoConvergence
from m3_numeric import ROOT,FrozenSystem,Balanced,pcg,orthogonal,need,allow_transfer
from local_identity import record,verify
from linux_runner import write_new

PARENT=ROOT/'reports/innovation_m3_20261015/analysis_v2'
OUTPUT=ROOT/'reports/innovation_m3_20261015/arnoldi_control'


def spectral_space(A,B0,seconds=120,k=32):
    started=time.perf_counter();calls=[0];reason=None
    def action(v):
        calls[0]+=1
        if time.perf_counter()-started>seconds:raise TimeoutError('Arnoldi wall budget')
        return B0(A@v)
    operator=LinearOperator(A.shape,matvec=action,dtype=np.float64)
    try:
        values,vectors=eigs(operator,k=k,which='SM',ncv=min(len(A.indptr)-1,128),maxiter=1000,tol=1e-7,
            v0=np.random.default_rng(20261015).normal(size=A.shape[0]))
    except ArpackNoConvergence as error:
        values,vectors=error.eigenvalues,error.eigenvectors;reason='arpack_partial_convergence'
    except TimeoutError:
        return np.empty((A.shape[0],0)),dict(status='timeout',seconds=time.perf_counter()-started,calls=calls[0],qualified=0)
    modes=[];checks=[]
    for i in np.argsort(values.real):
        value=values[i];vector=vectors[:,i]
        real_value=float(value.real);imaginary_ratio=abs(value.imag)/max(abs(real_value),1e-300)
        vi=float(np.linalg.norm(vector.imag)/max(np.linalg.norm(vector.real),1e-300))
        v=vector.real;tv=B0(A@v)
        residual=float(np.linalg.norm(tv-real_value*v)/max(np.linalg.norm(tv)+abs(real_value)*np.linalg.norm(v),1e-300))
        qualified=real_value>0 and imaginary_ratio<1e-6 and vi<1e-6 and residual<1e-4
        checks.append(dict(value=real_value,imaginary_ratio=float(imaginary_ratio),vector_imaginary_ratio=vi,
            relative_eigen_residual=residual,qualified=bool(qualified)))
        if qualified:modes.append(v)
    raw=np.column_stack(modes) if modes else np.empty((A.shape[0],0));Z,_=orthogonal(raw,k)
    return Z,dict(status='complete' if Z.shape[1]==k else (reason or 'insufficient_qualified_modes'),
        seconds=time.perf_counter()-started,calls=calls[0],qualified=Z.shape[1],eigenpairs=checks,
        scope='General Arnoldi on B_ref A (not Euclidean-symmetric eigsh); independent of RHS; approximate eigenpairs')


def main():
    need(not OUTPUT.exists(),'Fresh Arnoldi control required');OUTPUT.mkdir()
    dependencies=[record(Path(__file__)),record(ROOT/'tools/bench/m3_numeric.py')]
    write_new(OUTPUT/'plan.json',dict(dependencies=dependencies,states=['Bunny14n1','Mixed15n1','Boxes17n1'],
        seconds_per_spectral_solve=120,requested_rank=32,residual_limit=1e-4,
        source_label='Expensive current-system spectral oracle; NOT an online basis generator',performance_certified=False))
    cases=[]
    for scene,frame in (('Bunny',14),('Mixed',15),('Boxes',17)):
        out=OUTPUT/scene;out.mkdir();folder=ROOT/f'runs/innovation_m3_20261015/adjacent_{scene.lower()}/fixed'
        source=FrozenSystem(folder/f'f{frame}_n1');z,diagnostic=spectral_space(source.A,source.apply)
        np.save(out/'source_basis.npy',z,allow_pickle=False);basis_file=record(out/'source_basis.npy');rows=[]
        for direction in (1,2):
            target=source if direction==1 else FrozenSystem(folder/f'f{frame}_n2')
            parent=json.loads((PARENT/scene/f'adjacent_f{frame}_n{direction}/result.json').read_text())
            if direction==2:allow_transfer(source.meta,target.meta);verify([basis_file])
            for rank in (8,16,32):
                if z.shape[1]<rank:continue
                for soft in (False,True):
                    vectors=z[:,:rank].copy()
                    if soft:vectors[~target.soft]=0
                    op=Balanced(target.A,target.apply,vectors);r,x,_=pcg(target.A,target.b,op,target.apply)
                    need(r['converged'],'Arnoldi control PCG failed');r.pop('trace')
                    name=f'n{direction}_'+('soft' if soft else 'global')+f'_{rank}'
                    np.save(out/(name+'_x.npy'),x,allow_pickle=False)
                    rows.append(dict(direction=direction,space='soft' if soft else 'global',requested_rank=rank,
                        actual_rank=op.Z.shape[1],baseline_iterations=parent['methods'][0]['iterations'],**r,
                        setup_seconds=op.setup_seconds,Z_W_bytes=op.Z.nbytes+op.W.nbytes,
                        solution=record(out/(name+'_x.npy')),source_basis=basis_file,
                        source_kind='current_spectral_oracle' if direction==1 else 'previous_spectral_oracle_persistence'))
        write_new(out/'result.json',dict(scene=scene,frame=frame,spectral=diagnostic,methods=rows,basis=basis_file))
        cases.append(record(out/'result.json'))
        print(json.dumps(dict(scene=scene,spectral_status=diagnostic['status'],qualified_rank=diagnostic['qualified'],
            seconds=diagnostic['seconds'],counts=[(r['direction'],r['space'],r['requested_rank'],r['iterations']) for r in rows])),flush=True)
    verify(dependencies);write_new(OUTPUT/'summary.json',dict(complete=True,cases=cases,performance_certified=False))


if __name__=='__main__':main()
