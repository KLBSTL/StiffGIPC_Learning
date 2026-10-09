"""Screen larger prior-system Ritz spaces with a positive additive correction.

CPU hypothesis only. M_new = M_old + U (U^T A U)^-1 U^T >= M_old; using
the ORIGINAL b^T M_old b reference keeps the old weighted stopping test at
least as strict in exact arithmetic. No current-system solution is reused.
"""
from pathlib import Path
import argparse
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
sys.path.insert(0,str(ROOT/'tools/bench'))
from audit_recycling_reference import preconditioner
from audit_ritz_reference import solve
from cpu_fixed_reference import load_snapshot
from config import read
from linux_runner import require,write_new
from local_identity import record


def prior_basis(a,b,apply):
    r=b.copy();z=apply(r);rho=float(r@z);p=z.copy();ref=rho
    vectors=[z/np.sqrt(rho)];alphas=[];betas=[]
    for _ in range(64):
        ap=a@p;curvature=float(p@ap)
        require(curvature>0 and rho>0,'Invalid prior Lanczos curvature/rho')
        alpha=rho/curvature;r-=alpha*ap;alphas.append(alpha)
        if rho<=1e-4*ref or len(alphas)==64:break
        z=apply(r);new=float(r@z)
        require(new>0,'Invalid prior Lanczos updated rho')
        beta=new/rho;betas.append(beta);vectors.append(z/np.sqrt(new))
        p=z+beta*p;rho=new
    diagonal=1/np.asarray(alphas)
    if len(alphas)>1:diagonal[1:]+=np.asarray(betas)/np.asarray(alphas[:-1])
    tri=np.diag(diagonal)
    if len(alphas)>1:
        off=-np.sqrt(betas)/np.asarray(alphas[:-1]);tri+=np.diag(off,1)+np.diag(off,-1)
    eigen,coefficients=np.linalg.eigh(tri)
    valid=eigen>1e-12*max(eigen[-1],1e-300)
    return np.column_stack(vectors)@coefficients[:,valid],eigen[valid]


def additive(a,base,basis):
    u=basis/np.linalg.norm(basis,axis=0);au=a@u
    gram=u.T@au;gram=.5*(gram+gram.T)
    spectrum=np.linalg.eigvalsh(gram)
    require(spectrum[0]>1e-12*spectrum[-1] and spectrum[-1]>0,'Degenerate coarse Gram')
    chol=np.linalg.cholesky(gram)
    def apply(r):
        weights=np.linalg.solve(chol.T,np.linalg.solve(chol,u.T@r))
        return base(r)+u@weights
    return apply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Independent dense fixture checks symmetry, positive increment and solve.
    from scipy.sparse import csr_matrix
    a=np.diag([1.,3.,17.,91.]);base=lambda r:r.copy()
    action=additive(a,base,np.eye(4)[:,:2])
    dense=np.column_stack([action(e) for e in np.eye(4)])
    require(np.allclose(dense,dense.T) and np.linalg.eigvalsh(dense-np.eye(4)).min()>=0,
            'Additive fixture symmetry/monotonicity failed')
    fixture,_=solve(csr_matrix(a),np.ones(4),action)
    require(fixture['true_relative_residual']<1e-10,'Additive solve fixture failed')
    rows=[]
    for folder in sorted(args.capture.iterdir()):
        if not (folder/'fixed').is_dir():continue
        prior=None
        for metadata in sorted((folder/'fixed').glob('*_meta.json')):
            prefix=Path(str(metadata)[:-len('_meta.json')])
            a,b,_,_,inputs,_=load_snapshot(prefix)
            base,pre_inputs=preconditioner(prefix,read(metadata))
            zero,_=solve(a,b,base)
            candidates=[]
            if prior is not None:
                for rank in (4,8,16):
                    basis,values=prior;rank=min(rank,basis.shape[1])
                    try:
                        action=additive(a,base,basis[:,:rank])
                        result,_=solve(a,b,action,zero['rho_reference'])
                        candidates.append(dict(rank=rank,result=result,
                            true_residual_not_larger=result['true_relative_residual']<=zero['true_relative_residual'],
                            original_rho_reference_unchanged=result['rho_reference']==zero['rho_reference'],
                            prior_ritz_values=values[:rank].tolist()))
                    except ValueError as failure:
                        candidates.append(dict(rank=rank,error=str(failure)))
            prior=prior_basis(a,b,base)
            row=dict(case=folder.name,prefix=str(prefix),zero=zero,candidates=candidates,
                inputs=inputs,preconditioner_inputs=pre_inputs)
            rows.append(row)
            print({k:row[k] for k in ('case','prefix','zero','candidates')},flush=True)
    write_new(args.output,dict(scope=__doc__,cases=rows,controller=record(__file__),
        dependencies=[record(ROOT/'tools/local'/name) for name in
            ('audit_recycling_reference.py','audit_ritz_reference.py','cpu_fixed_reference.py')],
        independent_fixture=fixture,performance_certified=False,quality_certified=False))


if __name__=='__main__':main()
