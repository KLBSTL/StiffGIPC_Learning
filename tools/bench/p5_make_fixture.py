"""Small independent GPU ABI fixture plus a deliberately deficient POD input."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from m3_numeric import ROOT,pod_basis,Balanced,need
from local_identity import record
from linux_runner import write_new


def create(folder,deficient=False):
    need(not folder.exists(),'Fresh fixture input');folder.mkdir(parents=True)
    rng=np.random.default_rng(577);n=60;width=20
    u=rng.normal(size=(n,n));source=u.T@u+np.eye(n)
    v=rng.normal(size=(n,n));target=v.T@v+2*np.eye(n)
    v=rng.normal(size=(48,48));D=v.T@v+np.eye(48)
    v=rng.normal(size=(12,12));abd=v.T@v+np.eye(12)
    inverse=np.zeros((n,n));inverse[:12,:12]=abd;inverse[12:,12:]=D
    final=rng.normal(size=n);X=rng.normal(size=(n,width))
    if deficient:X[:]=X[:,:1]
    S=final[:,None]-X
    history=[(i,None,None,X[:,i]) for i in range(width)]
    pod,_=pod_basis(csr_matrix(source),history,final)
    probes=rng.normal(size=(n,3));arrays={};records=[]
    def put(name,data,dtype='<f8',order='F'):
        a=np.asarray(data,dtype=dtype);p=folder/(name+'.bin');p.write_bytes(a.ravel(order=order).tobytes())
        arrays[name]=dict(file=p.name,shape=list(a.shape),dtype=dtype,order=order);records.append(record(p))
    for name,A in (('source',source),('target',target)):
        a=csr_matrix(A);put(name+'_rows',a.indptr,'<i4');put(name+'_cols',a.indices,'<i4')
        put(name+'_values',a.data);put(name+'_rhs',rng.normal(size=n))
    put('iterates',X);put('final_source_x',final);put('history_writes',np.column_stack([np.arange(width),np.arange(width)]),'<i4',order='C')
    put('expected_history',X);put('expected_S',S)
    for name in ('R','RT'):
        put(name+'_rows',np.arange(17),'<i4');put(name+'_cols',np.arange(16),'<i4');put(name+'_values',np.ones(16))
    put('mas_inverse',D);put('abd_inverse',abd);put('probes',probes);put('expected_MAS',inverse@probes)
    if not deficient:
        for rank in (8,16):
            op=Balanced(csr_matrix(target),lambda v:inverse@v,pod[:,:rank])
            put('expected_Z_'+str(rank),op.Z);put('expected_B_'+str(rank),np.column_stack([op(v) for v in probes.T]))
    write_new(folder/'manifest.json',dict(schema='p5_pod_gpu_input.v1',arrays=arrays,records=records,
        n=n,history_width=width,source_iterations=width,history_copy_calls=width,
        mas_nodes=16,mas_rows=16,mas_blocks=1,mas_offset=12,abd_offset=0,
        target_solution_used=False,fixture=True,expected_failure=deficient))


if __name__=='__main__':
    root=ROOT/'runs/innovation_p5_gpu_20261015'
    create(root/'fixture_positive_inputs');create(root/'fixture_deficient_inputs',True)
    print(json.dumps(dict(fixtures=2)))
