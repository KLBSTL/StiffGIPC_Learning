"""Bounded CPU same-matrix Jacobi-CG probe; never a performance comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from analyze_perf_v36 import matrix_from_snapshot

p=argparse.ArgumentParser();p.add_argument('prefix',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
meta,A,b=matrix_from_snapshot(a.prefix);d=A.diagonal();count=0;trace=[]
bn=np.linalg.norm(b)
def callback(x):
    global count
    count+=1
    if count%100==0:trace.append({'iterations':count,'true_relative_residual':float(np.linalg.norm(b-A@x)/bn)})
M=LinearOperator(A.shape,matvec=lambda x:x/d,dtype=np.float64)
x,info=cg(A,b,M=M,rtol=1e-6,atol=0,maxiter=3000,callback=callback)
result={'scope':'CPU same failed A/b, scalar Jacobi diagnostic; differs from native 3x3+ABD and MAS; no simulation or speedup claim',
        'iterations':count,'info':int(info),'true_relative_residual':float(np.linalg.norm(b-A@x)/bn),'trace':trace}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='trace'}))
