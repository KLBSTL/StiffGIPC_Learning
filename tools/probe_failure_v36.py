"""Bounded CPU spectral probe of the actual failed assembled system."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from analyze_perf_v36 import matrix_from_snapshot

p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
meta,A,b=matrix_from_snapshot(a.prefix)
d=A.diagonal();D=sparse.diags(1/np.sqrt(np.maximum(np.abs(d),1e-30)))
S=D@((A+A.T)*.5)@D
result={'dofs':len(b),'nnz':A.nnz,'finite':bool(np.isfinite(A.data).all() and np.isfinite(b).all()),
        'matrix_relative_asymmetry':float(sparse.linalg.norm(A-A.T)/max(sparse.linalg.norm(A),1e-300)),
        'negative_diagonal_entries':int(np.sum(d<0)), 'zero_diagonal_entries':int(np.sum(d==0)),
        'diag_min':float(d.min()),'diag_max':float(d.max()),
        'preconditioner_export_complete':meta['preconditioner_export_complete']}
start=time.monotonic()
try:
    values,vectors=eigsh(S,k=2,which='SA',tol=1e-6,maxiter=300,ncv=30,v0=np.random.default_rng(36).normal(size=len(b)))
    result['eigensolver_converged']=True
except ArpackNoConvergence as exc:
    values,vectors=exc.eigenvalues,exc.eigenvectors
    result['eigensolver_converged']=False
result['scaled_smallest_eigenvalues']=values.tolist()
result['spectral_seconds']=time.monotonic()-start
if len(values):
    v=vectors[:,0];x=D@v
    result['witness_xAx']=float(x@(A@x))
    result['scaled_eigen_residual']=float(np.linalg.norm(S@v-values[0]*v))
    np.save(str(a.output.with_suffix(''))+'_witness.npy',x)
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
