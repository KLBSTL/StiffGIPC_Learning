"""Independent numerical and matched-cost checks, keeping approximate spaces explicit."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix

from cgstats_numeric import statistical_gram
from m3_numeric import ROOT,Balanced,rel,need
from p5_validate import mixed_budget_source
from local_identity import record,verify
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9_-]+',args.name) is not None,'Simple run name')
    folder=RAW/args.name;launch=read(folder/'launch.json');need(launch['passed'],'Launch failed')
    input_dir=Path(launch['input']['path']).parent;m=read(input_dir/'manifest.json');verify(m['records'])
    output=folder/'data';run=read(output/'result.json');need(run['passed'],'Native validation failed')
    n=m['n'];width=m['history_width'];records=[]
    def original(name):
        a=m['arrays'][name];return np.fromfile(input_dir/a['file'],dtype=a['dtype']).reshape(a['shape'],order=a['order'])
    def actual(name,shape):
        path=output/(name+'.bin');records.append(record(path));value=np.fromfile(path,dtype='<f8')
        need(value.size==int(np.prod(shape)),'Output shape');return value.reshape(shape,order='F')
    def sparse(name,r,c):return csr_matrix((original(name+'_values'),original(name+'_cols'),original(name+'_rows')),shape=(r,c))
    A_source=sparse('source',n,n);A_target=sparse('target',n,n)
    R=sparse('R',m['mas_rows'],m['mas_nodes']);Rt=sparse('RT',m['mas_nodes'],m['mas_rows']);need((R.T-Rt).nnz==0,'Adjoint restriction mismatch')
    inverse=np.fromfile(input_dir/m['arrays']['mas_inverse']['file'],dtype='<f8').reshape(-1,48,48).transpose(0,2,1)
    abd=original('abd_inverse');offset=m['mas_offset'];abd_offset=m['abd_offset'];nodes=m['mas_nodes']
    def B0(v):
        z=v.copy();z[abd_offset:abd_offset+12]=abd@v[abd_offset:abd_offset+12]
        r=(R@v[offset:offset+3*nodes].reshape(nodes,3)).reshape(-1,48)
        local=np.einsum('bij,bj->bi',inverse,r).reshape(-1,3);z[offset:offset+3*nodes]=(Rt@local).ravel();return z
    bank=actual('history_bank',(n,width));need(np.array_equal(bank,original('expected_history')),'GPU source history changed')
    S=original('final_source_x')[:,None]-bank;need(np.array_equal(S,original('expected_S')),'Source correction history changed')
    G=S.T@(A_source@S);G=.5*(G+G.T)
    scalar=original('CG_scalars');H,_=statistical_gram(scalar[:,0],scalar[:,1],original('bank_indices'),restarts=m['source_restarts'])
    checks=[];budget=[];pairs=[];source=mixed_budget_source(m);native=read(source) if source else None
    for case in run['ranks']:
        rank=case['rank'];family=case['family'];tag=case['tag'];stats=family=='statistics'
        z=actual('Z_'+tag,(n,rank));w=actual('W_'+tag,(n,rank));L=np.tril(actual('factor_'+tag,(rank,rank)))
        expected=original(('expected_CG_Z_' if stats else 'expected_Z_')+str(rank))
        span=rel(z,expected@(expected.T@z));orth=float(np.linalg.norm(z.T@z-np.eye(rank)))
        we=rel(w,A_target@z);E=z.T@(A_target@z);E=.5*(E+E.T);le=rel(L@L.T,E)
        small=actual('gram_'+tag,(width,width));ref=H if stats else G;ge=rel(small,ref)
        ev=actual('eigen_'+tag,(width,));ee=rel(ev,np.linalg.eigvalsh(ref))
        need(span<1e-6 and orth<1e-10 and we<1e-11 and le<1e-11 and ge<1e-10 and ee<1e-10 and
             np.all(np.diag(L)>0),'GPU matrix/space/factor validation failed')
        op=Balanced(A_target,B0,z);probes=original('probes');actions=[]
        for i in range(3):
            value=actual(f'balanced_{tag}_p{i}',(n,));error=rel(value,op(probes[:,i]));need(error<1e-8,'GPU wrapper validation failed');actions.append(error)
        checks.append(dict(rank=rank,family=family,projection_relative=span,orthogonality_frobenius=orth,
            target_AZ_relative=we,factor_relative=le,Gram_implementation_relative=ge,eigenvalues_relative=ee,
            approximate_Gram_vs_direct=rel(H,G) if stats else 0.,independent_action_relative=actions))
        need(sorted(x['pair'] for x in case['pipeline_samples'])==[0,1,2],'Missing matched setup sample')
        if native is not None:
            CPU_case=read(RAW/'Mixed/adjacent_f15_n1/result.json')
            method=('CG_statistics_' if stats else 'direct_POD_')+str(rank)
            N=next(r['iterations'] for r in CPU_case['methods'] if r['method']==method)
            setup=float(np.median([x['wall_ms'] for x in case['pipeline_samples']]))
            for execution in ('graph','host'):
                deltas=[]
                for pair in range(3):
                    group=[x for x in case['apply_samples'] if x['mode']==execution and x['pair']==pair]
                    base=next(x for x in group if x['arm']=='MAS_reference');candidate=next(x for x in group if x['arm']=='balanced')
                    deltas.append(candidate['GPU_us_per_apply']-base['GPU_us_per_apply'])
                delta=float(np.median(deltas));saved=(native['methods'][0]['iterations']-N)*native['native_default_iteration_ms']
                budget.append(dict(rank=rank,family=family,execution=execution,conditional_iterations=N,
                    setup_wall_ms=setup,paired_added_us=deltas,median_added_us=delta,
                    conditional_saved_before_extra_ms=saved,conditional_net_ms=saved-setup-N*delta/1000,
                    performance_certified=False,scope='Reference MAS delta and transferred CPU counts are assumptions, not native PCG timing'))
    for rank in (8,16):
        direct=next(x for x in run['ranks'] if x['rank']==rank and x['family']=='direct')
        stats=next(x for x in run['ranks'] if x['rank']==rank and x['family']=='statistics')
        for pair in range(3):
            d=next(x for x in direct['pipeline_samples'] if x['pair']==pair)
            s=next(x for x in stats['pipeline_samples'] if x['pair']==pair)
            need(d['order']!=s['order'] and d['order']==pair%2,'Setup AB/BA schedule changed')
            pairs.append(dict(rank=rank,pair=pair,direct_wall_ms=d['wall_ms'],statistics_wall_ms=s['wall_ms'],
                saved_setup_ms=d['wall_ms']-s['wall_ms'],construction_ratio=d['wall_ms']/s['wall_ms']))
    write_new(folder/'validation.json',dict(passed=True,history_bitwise_equal=True,checks=checks,setup_pairs=pairs,
        conditional_budgets=budget,budget_source=record(source) if source else None,
        inputs=m['records'],results=records+[record(output/'result.json')],source=record(Path(__file__)),
        approximate_statistics_Gram=True,native_PCG_implemented=False,performance_certified=False))
    print(json.dumps(dict(passed=True,checks=checks,setup_pairs=pairs,conditional_budgets=budget)),flush=True)


if __name__=='__main__':main()
