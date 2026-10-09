"""Screen two recycled Ritz vectors with a balanced coarse preconditioner.

Only earlier real systems provide the subspace. CPU float reductions differ
from CUDA; iteration counts are a hypothesis screen, never a speed certificate.
"""
from pathlib import Path
import argparse
import json
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
sys.path.insert(0,str(ROOT/'tools/bench'))
from audit_recycling_reference import preconditioner
from cpu_fixed_reference import load_snapshot
from config import read
from linux_runner import require,write_new
from local_identity import record


def solve(matrix,rhs,apply,reference=None,collect=False):
    x=np.zeros_like(rhs);residual=rhs.copy();z=apply(residual)
    rho=float(residual@z)
    require(rho>0 and np.isfinite(rho),'Nonpositive initial rho')
    ref=rho if reference is None else min(rho,reference)
    p=z.copy();vectors=[];alphas=[];betas=[]
    if collect:vectors.append(z/np.sqrt(rho))
    for iteration in range(1,501):
        ap=matrix@p;curvature=float(p@ap)
        require(curvature>0 and np.isfinite(curvature),'Nonpositive curvature')
        alpha=rho/curvature;x+=alpha*p;residual-=alpha*ap
        if collect and len(alphas)<64:alphas.append(alpha)
        if abs(rho)<=1e-4*ref:break
        z=apply(residual);new=float(residual@z)
        require(new>=0 and np.isfinite(new),'Nonpositive updated rho')
        if new==0:
            require(not np.any(residual),'Zero rho with nonzero residual')
            break
        beta=new/rho
        if collect and len(vectors)<64:
            betas.append(beta);vectors.append(z/np.sqrt(new))
        p=z+beta*p;rho=new
    else:raise ValueError('Surrogate iteration cap')
    result=dict(iterations=iteration,true_relative_residual=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs)),
        rho_reference=ref,rho_stop=rho)
    if not collect:return result,None
    size=len(alphas)
    require(size==len(vectors),'Incomplete Lanczos record')
    diagonal=1/np.asarray(alphas)
    if size>1:diagonal[1:]+=np.asarray(betas[:size-1])/np.asarray(alphas[:size-1])
    tri=np.diag(diagonal)
    if size>1:
        off=-np.sqrt(betas[:size-1])/np.asarray(alphas[:size-1])
        tri+=np.diag(off,1)+np.diag(off,-1)
    eigen,coefficients=np.linalg.eigh(tri)
    valid=np.flatnonzero(eigen>1e-12*max(eigen[-1],1e-300))[:2]
    require(len(valid)>0,'No positive Ritz values')
    basis=np.column_stack(vectors)@coefficients[:,valid]
    result['ritz_values']=eigen[valid].tolist();result['lanczos_dimension']=size
    return result,basis


def balanced(matrix,base,basis):
    u=basis/np.linalg.norm(basis,axis=0)
    au=matrix@u;gram=u.T@au;gram=.5*(gram+gram.T)
    eigen=np.linalg.eigvalsh(gram)
    require(eigen[0]>1e-12*eigen[-1] and eigen[-1]>0,'Degenerate coarse Gram matrix')
    inverse=np.linalg.inv(gram)
    def apply(r):
        coarse=inverse@(u.T@r)
        z=base(r-au@coarse)
        return z-u@(inverse@(au.T@z))+u@coarse
    return apply,len(eigen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for folder in sorted(args.capture.iterdir()):
        if not (folder/'fixed').is_dir():continue
        prior=None
        for metadata in sorted((folder/'fixed').glob('*_meta.json')):
            prefix=Path(str(metadata)[:-len('_meta.json')])
            matrix,rhs,_,_,evidence,details=load_snapshot(prefix)
            base,inputs=preconditioner(prefix,read(metadata))
            zero,next_basis=solve(matrix,rhs,base,collect=True)
            candidate=zero;rank=0;error=None
            if prior is not None:
                try:
                    action,rank=balanced(matrix,base,prior)
                    candidate,_=solve(matrix,rhs,action,zero['rho_reference'])
                except ValueError as failure:
                    error=str(failure)
            rows.append(dict(case=folder.name,prefix=str(prefix),zero=zero,balanced=candidate,rank=rank,
                error=error,inputs=evidence,preconditioner_inputs=inputs,
                added_spmv_for_setup=rank,added_coarse_dots_per_iteration=2*rank,
                true_residual_not_larger=candidate['true_relative_residual']<=zero['true_relative_residual']))
            prior=next_basis
    write_new(args.output,dict(scope=__doc__,cases=rows,controller=record(__file__),
        operators=record(ROOT/'tools/local/audit_recycling_reference.py'),
        performance_certified=False,quality_certified=False))
    print(json.dumps([{k:r[k] for k in ('case','prefix','zero','balanced','rank','error','true_residual_not_larger')}
        for r in rows]))


if __name__=='__main__':main()
