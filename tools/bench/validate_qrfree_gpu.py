"""Independent Python QR is verification only, never a hidden setup dependency."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from m3_numeric import ROOT,Balanced,rel,need
from qrfree_numeric import CoordinateBalanced
from p5_validate import mixed_budget_source
from local_identity import record,verify
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_qrfree_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    folder=RAW/args.name;launch=read(folder/'launch.json');need(launch['passed'],'Launch failed')
    inputs=Path(launch['input']['path']).parent;m=read(inputs/'manifest.json');verify(m['records']);out=folder/'data'
    run=read(out/'result.json');need(run['passed'] and len(run['cases'])==2,'Incomplete paired probe')
    n=m['n'];rank=8;records=[]
    def original(name):
        a=m['arrays'][name];return np.fromfile(inputs/a['file'],dtype=a['dtype']).reshape(a['shape'],order=a['order'])
    def actual(name,shape):
        path=out/(name+'.bin');records.append(record(path));value=np.fromfile(path,dtype='<f8')
        need(value.size==int(np.prod(shape)),'Output shape mismatch');return value.reshape(shape,order='F')
    def sparse(name,r,c):return csr_matrix((original(name+'_values'),original(name+'_cols'),original(name+'_rows')),shape=(r,c))
    A=sparse('target',n,n);R=sparse('R',m['mas_rows'],m['mas_nodes']);Rt=sparse('RT',m['mas_nodes'],m['mas_rows'])
    need((R.T-Rt).nnz==0,'Restriction adjoint changed')
    inverse=np.fromfile(inputs/m['arrays']['mas_inverse']['file'],dtype='<f8').reshape(-1,48,48).transpose(0,2,1)
    abd=original('abd_inverse');offset=m['mas_offset'];ao=m['abd_offset'];nodes=m['mas_nodes']
    def B0(v):
        z=v.copy();z[ao:ao+12]=abd@v[ao:ao+12];r=(R@v[offset:offset+3*nodes].reshape(nodes,3)).reshape(-1,48)
        local=np.einsum('bij,bj->bi',inverse,r).reshape(-1,3);z[offset:offset+3*nodes]=(Rt@local).ravel();return z
    need(np.array_equal(actual('history_bank',(n,m['history_width'])),original('expected_history')),'History bank changed')
    reference=original('expected_CG_Z_8');qref,_=np.linalg.qr(reference,mode='reduced');probes=original('probes');checks=[]
    actions={};setup_pairs=[];conditional=[];source=mixed_budget_source(m);native=read(source) if source else None
    for case in run['cases']:
        tag=case['method'];z=actual('Z_'+tag,(n,rank));w=actual('W_'+tag,(n,rank));L=np.tril(actual('factor_'+tag,(rank,rank)))
        q,_=np.linalg.qr(z,mode='reduced');span=rel(q,qref@(qref.T@q));we=rel(w,A@z);raw=z.T@(A@z);E=.5*(raw+raw.T)
        ev=np.linalg.eigvalsh(E);le=rel(L@L.T,E)
        need(span<1e-8 and we<1e-10 and le<1e-10 and ev[0]>1e-12*ev[-1] and np.all(np.diag(L)>0),
             'Space/AZ/coarse factor audit failed')
        op=CoordinateBalanced(A,B0,z);errors=[];actions[tag]=[]
        for i in range(3):
            value=actual(f'{tag}_action_{i}',(n,));errors.append(rel(value,op(probes[:,i])));actions[tag].append(value)
        need(max(errors)<1e-8 and case['graph_action_error']<1e-12,'Independent action audit failed')
        need(sorted(t['pair'] for t in case['pipeline_samples'])==list(range(5)),'Missing setup pair')
        checks.append(dict(method=tag,span_error=span,AZ_error=we,factor_error=le,coarse_condition=float(ev[-1]/ev[0]),
            action_errors=errors,orthogonality=float(np.linalg.norm(z.T@z-np.eye(rank))),Graph_error=case['graph_action_error']))
        if native:
            N=read(RAW/'Mixed/result.json')['methods'][0]['iterations'];setup=float(np.median([t['wall_ms'] for t in case['pipeline_samples']]))
            for mode in ('host','graph'):
                differences=[]
                for pair in range(3):
                    rows=[t for t in case['apply_samples'] if t['mode']==mode and t['pair']==pair]
                    need(len(rows)==2 and {t['order'] for t in rows}=={0,1},'Apply pair incomplete')
                    base=next(t for t in rows if t['arm']=='MAS_reference');aug=next(t for t in rows if t['arm']=='balanced')
                    differences.append(aug['gpu_us']-base['gpu_us'])
                delta=float(np.median(differences));budget=(native['methods'][0]['iterations']-N)*native['native_default_iteration_ms']
                conditional.append(dict(method=tag,mode=mode,setup_wall_ms=setup,added_us=differences,
                    added_us_median=delta,conditional_iterations=N,saved_before_extra_ms=budget,
                    conditional_net_ms=budget-setup-N*delta/1000,performance_certified=False))
    cross=[rel(a,b) for a,b in zip(actions['QR_free'],actions['QR'])];need(max(cross)<1e-8,'Changing coordinates changed GPU action')
    qr,free=run['cases'];need(qr['method']=='QR' and free['method']=='QR_free','Case schedule changed')
    for pair in range(5):
        a=next(t for t in qr['pipeline_samples'] if t['pair']==pair);b=next(t for t in free['pipeline_samples'] if t['pair']==pair)
        need(a['order']==pair%2 and a['order']!=b['order'],'AB/BA schedule changed')
        setup_pairs.append(dict(pair=pair,QR_ms=a['wall_ms'],QR_free_ms=b['wall_ms'],saved_ms=a['wall_ms']-b['wall_ms'],ratio=a['wall_ms']/b['wall_ms']))
    write_new(folder/'validation.json',dict(passed=True,checks=checks,cross_action_errors=cross,setup_pairs=setup_pairs,
        conditional=conditional,inputs=m['records']+[record(inputs/'manifest.json')],results=records+[record(out/'result.json')],
        validator=record(Path(__file__)),native_PCG=False,performance_certified=False))
    print(json.dumps(dict(passed=True,cross_action_max=max(cross),setup_pairs=setup_pairs,conditional=conditional)),flush=True)


if __name__=='__main__':main()
