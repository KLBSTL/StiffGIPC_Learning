"""Stronger oracle: reconstruct the full recorded CG recurrence, not a sample."""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from m3_numeric import need,orthogonal


def full_cg_basis(A,b,B0,trace,expected_solution,max_modes=96):
    m=len(trace);need(m>0,'Empty CG recurrence')
    alpha=np.array([r['alpha'] for r in trace]);rho=np.array([r['rho'] for r in trace])
    need(np.all(alpha>0) and np.all(rho>0),'Nonpositive CG recurrence')
    beta=rho[1:]/rho[:-1]
    diagonal=1/alpha;diagonal[1:]+=beta/alpha[:-1];off=np.sqrt(beta)/alpha[:-1]
    eigen,coeff=eigh_tridiagonal(diagonal,off,select='i',select_range=(0,min(m,max_modes)-1))
    need(eigen[0]>0,'Invalid Lanczos Ritz spectrum')
    basis=np.zeros((len(b),len(eigen)));x=np.zeros_like(b);r=b.copy();z=B0(r);p=z.copy();rho_error=0.
    for i in range(m):
        actual=float(r@z);rho_error=max(rho_error,abs(actual-rho[i])/rho[i])
        q=z*((-1. if i%2 else 1.)/np.sqrt(rho[i]));basis+=np.outer(q,coeff[i])
        ap=A@p;x+=alpha[i]*p;r-=alpha[i]*ap
        if i+1<m:z=B0(r);p=z+beta[i]*p
    need(np.array_equal(x,expected_solution),'CG recurrence did not replay the saved solution bitwise')
    residuals=[];accepted=[];indices=[]
    for i,lam in enumerate(eigen):
        v=basis[:,i];tv=B0(A@v)
        error=float(np.linalg.norm(tv-lam*v)/max(np.linalg.norm(tv)+abs(lam)*np.linalg.norm(v),1e-300))
        residuals.append(error)
        if error<=.05:accepted.append(v);indices.append(i)
    raw=np.column_stack(accepted) if accepted else np.empty((len(b),0));Q,_=orthogonal(raw,32)
    return Q,dict(steps=m,raw_ritz_modes=len(eigen),accepted_spectral_modes=len(accepted),independent_rank=Q.shape[1],
        replay_solution_bitwise_equal=True,maximum_rho_relative_replay_error=rho_error,
        minimum_ritz=float(eigen[0]),selected_ritz_values=eigen.tolist(),raw_eigen_residuals=residuals,
        accepted_raw_indices=indices,spectral_residual_limit=.05,
        scope='Full finite-precision CG/Lanczos reconstruction with ghost/dependence filtering; not a certified global eigenspace')
