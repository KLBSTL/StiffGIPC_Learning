"""Prepare Mixed and algebraic GPU controls without copying historical payloads."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from cgstats_numeric import statistical_gram,coefficients
from m3_numeric import ROOT,FrozenSystem,Balanced,orthogonal,rel,need
from local_identity import record,verify
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def absolute_payloads(manifest,folder):
    result=json.loads(json.dumps(manifest))
    for a in result['arrays'].values():a['file']=str((folder/a['file']).resolve())
    return result


def writer(folder,manifest):
    need(not folder.exists(),'Fresh input directory');folder.mkdir(parents=True)
    def put(name,value,dtype='<f8',order='F'):
        a=np.asarray(value,dtype=dtype);path=folder/(name+'.bin');path.write_bytes(a.ravel(order=order).tobytes())
        manifest['arrays'][name]=dict(file=path.name,shape=list(a.shape),dtype=dtype,order=order)
        manifest['records'].append(record(path))
    return put


def prepare_real():
    folder=RAW/'inputs';old=ROOT/'runs/innovation_p5_gpu_20261015/inputs'
    original=read(old/'manifest.json');verify(original['records']);m=absolute_payloads(original,old);put=writer(folder,m)
    summary=read(RAW/'summary.json');need(summary['Mixed_primary_gate']['passed'],'Mixed primary CPU gate failed')
    case=RAW/'Mixed/adjacent_f15_n1'
    with np.load(case/'statistics.npz',allow_pickle=False) as data:
        alpha=data['alpha'];rho=data['rho'];sorted_indices=data['indices']
    n=m['n'];steps=len(alpha);writes=np.fromfile(old/'history_writes.bin',dtype='<i4').reshape(-1,2)
    indices=np.full(m['history_width'],-1,dtype=np.int32)
    for iteration,slot in writes:indices[slot]=iteration
    need(np.array_equal(np.sort(indices),sorted_indices),'Source bank/statistics context differs')
    S=np.fromfile(old/'expected_S.bin',dtype='<f8').reshape(n,-1,order='F')
    G,_=statistical_gram(alpha,rho,indices);u,_=coefficients(G,16)
    target=FrozenSystem(ROOT/'runs/innovation_m3_20261015/adjacent_mixed/fixed/f15_n2')
    probes=np.fromfile(old/'probes.bin',dtype='<f8').reshape(n,3,order='F')
    put('CG_scalars',np.column_stack([alpha,rho]));put('bank_indices',indices,'<i4')
    for rank in (8,16):
        z=S@u[:,:rank];op=Balanced(target.A,target.apply,z)
        prior=np.load(case/f'basis_{rank}.npy',allow_pickle=False);q,_=orthogonal(prior)
        need(rel(op.Z,q@(q.T@op.Z))<1e-8,'Permuted statistical space differs')
        put('expected_CG_Z_'+str(rank),op.Z)
        put('expected_CG_B_'+str(rank),np.column_stack([op(v) for v in probes.T]))
    m.update(schema='cgstats_pod_gpu_input.v1',source_restarts=0,
        approximate_Gram=True,CPU_gate=record(RAW/'summary.json'),statistics_source=record(case/'statistics.npz'),
        parent_GPU=record(ROOT/'reports/innovation_p5_gpu_20261015/DELIVERY_SEAL.json'))
    write_new(folder/'manifest.json',m)


def prepare_fixture():
    folder=RAW/'fixture_positive_inputs';m=dict(schema='cgstats_pod_gpu_input.v1',arrays={},records=[],fixture=True,
        n=60,history_width=20,source_iterations=20,history_copy_calls=20,mas_nodes=16,mas_rows=16,
        mas_blocks=1,mas_offset=12,abd_offset=0,source_restarts=0,target_solution_used=False)
    put=writer(folder,m);rng=np.random.default_rng(248);n=60;width=20
    q,_=np.linalg.qr(rng.normal(size=(n,n)));e=np.geomspace(.5,100.,n);A=(q*e)@q.T
    delta=q[:,:width]/np.sqrt(e[:width]);delta*=np.linspace(.2,3.,width)
    x=np.zeros(n);X=[]
    for i in range(width):X.append(x.copy());x+=delta[:,i]
    X=np.column_stack(X);S=x[:,None]-X
    weights=np.einsum('ij,ij->j',delta,A@delta);G,_=statistical_gram(np.ones(width),weights,np.arange(width))
    Gdirect=S.T@(A@S);need(rel(G,Gdirect)<1e-10,'Synthetic A-conjugate construction failed')
    h=rng.normal(size=(n,n));At=h.T@h+np.eye(n)
    h=rng.normal(size=(48,48));D=h.T@h+np.eye(48)
    h=rng.normal(size=(12,12));abd=h.T@h+np.eye(12)
    B=np.zeros((n,n));B[:12,:12]=abd;B[12:,12:]=D
    for name,matrix in (('source',A),('target',At)):
        mat=csr_matrix(matrix);put(name+'_rows',mat.indptr,'<i4');put(name+'_cols',mat.indices,'<i4')
        put(name+'_values',mat.data);put(name+'_rhs',rng.normal(size=n))
    for name in ('R','RT'):
        put(name+'_rows',np.arange(17),'<i4');put(name+'_cols',np.arange(16),'<i4');put(name+'_values',np.ones(16))
    put('mas_inverse',D);put('abd_inverse',abd);put('iterates',X);put('final_source_x',x)
    put('history_writes',np.column_stack([np.arange(width),np.arange(width)]),'<i4',order='C')
    put('expected_history',X);put('expected_S',S);put('CG_scalars',np.column_stack([np.ones(width),weights]))
    put('bank_indices',np.arange(width),'<i4');probes=rng.normal(size=(n,3));put('probes',probes);put('expected_MAS',B@probes)
    for gram,family in ((Gdirect,'expected'),(G,'expected_CG')):
        u,_=coefficients(.5*(gram+gram.T),16)
        for rank in (8,16):
            op=Balanced(csr_matrix(At),lambda v:B@v,S@u[:,:rank]);put(family+'_Z_'+str(rank),op.Z)
            put(family+'_B_'+str(rank),np.column_stack([op(v) for v in probes.T]))
    write_new(folder/'manifest.json',m)
    negative=RAW/'fixture_negative_inputs';bad=absolute_payloads(m,folder);negative_put=writer(negative,bad)
    bad_scalars=np.column_stack([np.ones(width),weights]);bad_scalars[0,1]*=-1
    negative_put('CG_scalars',bad_scalars);bad['expected_failure']='Invalid CG energy statistics'
    write_new(negative/'manifest.json',bad)


if __name__=='__main__':
    prepare_real();prepare_fixture();print(json.dumps(dict(prepared=True,fixtures=2)))
