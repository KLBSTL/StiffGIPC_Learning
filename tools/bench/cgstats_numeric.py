"""Approximate history Gram from fixed-PCG statistics, not exact finite arithmetic."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import time

import numpy as np
from scipy.linalg import eigh

from m3_numeric import need


def statistical_gram(alpha,rho,indices,*,restarts=0,curvature=None):
    start=time.perf_counter()
    alpha=np.asarray(alpha,dtype=np.float64);rho=np.asarray(rho,dtype=np.float64)
    ids=np.asarray(indices)
    need(restarts==0,'Restarted recurrence is not supported by conjugacy model')
    need(alpha.ndim==rho.ndim==1 and len(alpha)==len(rho)>0,'Invalid scalar recurrence dimensions')
    need(ids.ndim==1 and len(ids)>0 and np.issubdtype(ids.dtype,np.integer),'Integer history indices required')
    need(np.all((ids>=0)&(ids<len(alpha))) and len(np.unique(ids))==len(ids),'Invalid or duplicate history indices')
    need(np.isfinite(alpha).all() and np.isfinite(rho).all() and np.all(alpha>0) and np.all(rho>0),
         'Nonpositive or nonfinite CG statistics')
    weights=alpha*rho
    need(np.isfinite(weights).all() and np.all(weights>0),'Invalid CG energy weights')
    if curvature is not None:
        pAp=np.asarray(curvature,dtype=np.float64)
        need(pAp.shape==alpha.shape and np.isfinite(pAp).all() and np.all(pAp>0),'Invalid curvature statistics')
        need(np.max(np.abs(alpha*pAp-rho)/rho)<1e-10,'alpha/rho/curvature context mismatch')
    suffix=np.cumsum(weights[::-1])[::-1]
    gram=suffix[np.maximum.outer(ids,ids)]
    return gram,dict(seconds=time.perf_counter()-start,source_steps=len(alpha),history_width=len(ids),
        approximate=True,scope='A-conjugate increment model; finite-precision Gram is independently audited')


def coefficients(gram,rank):
    start=time.perf_counter();gram=np.asarray(gram,dtype=np.float64)
    need(gram.ndim==2 and gram.shape[0]==gram.shape[1] and gram.shape[0]>0,'Invalid small Gram shape')
    need(isinstance(rank,int) and rank>0,'Positive integer rank required')
    need(np.isfinite(gram).all() and np.linalg.norm(gram-gram.T)<=1e-12*max(np.linalg.norm(gram),1e-300),'Nonsymmetric Gram')
    values,vectors=eigh(.5*(gram+gram.T));need(values[-1]>0,'No positive Gram energy')
    need(values[0]>=-1e-10*values[-1],'Gram is materially indefinite')
    eligible=np.flatnonzero(values>1e-10*values[-1])[::-1]
    selected=eligible[:rank];u=vectors[:,selected]
    return u,dict(requested_rank=rank,actual_rank=len(selected),seconds=time.perf_counter()-start,
        lowest_eigenvalue=float(values[0]),largest_eigenvalue=float(values[-1]),
        selected_eigenvalues=values[selected].tolist(),negative_eigenvalues_clipped=False)
