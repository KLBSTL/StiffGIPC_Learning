"""Independent CPU checks for the frozen-input CUDA cost probe, then budgets."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix

from m3_numeric import ROOT,Balanced,rel,need
from local_identity import record,verify
from linux_runner import write_new

SESSION=ROOT/'runs/innovation_p5_gpu_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def mixed_budget_source(manifest):
    if manifest.get('fixture'):
        return None
    need(manifest['n']==60858 and manifest['source_context']==dict(frame=15,inner=0,linear_system_id=1)
         and manifest['target_context']==dict(frame=15,inner=1,linear_system_id=2),'Wrong cost-budget context')
    verify([manifest['parent']])
    return ROOT/'reports/innovation_m3_20261015/analysis_v2/Mixed/adjacent_f15_n2/result.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--output-name',default='validation.json');args=parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9_-]+\.json',args.output_name) is not None,'Simple result filename required')
    folder=SESSION/args.name;launch=read(folder/'launch.json');need(launch['passed'],'Launch failed')
    source=Path(launch['input']['path']).parent;manifest=read(source/'manifest.json');verify(manifest['records'])
    raw=folder/'data';run=read(raw/'result.json');need(run['passed'],'Native checks failed')
    n=manifest['n'];m=manifest['history_width'];records=[]
    budget_source=mixed_budget_source(manifest);budget=read(budget_source) if budget_source else None
    def original(name):
        a=manifest['arrays'][name];return np.fromfile(source/a['file'],dtype=a['dtype']).reshape(a['shape'],order=a['order'])
    def actual(name,shape):
        path=raw/(name+'.bin');records.append(record(path));a=np.fromfile(path,dtype='<f8');need(a.size==int(np.prod(shape)),'Output shape')
        return a.reshape(shape,order='F')
    def matrix(name,rows,cols):
        return csr_matrix((original(name+'_values'),original(name+'_cols'),original(name+'_rows')),shape=(rows,cols))
    source_A=matrix('source',n,n);target_A=matrix('target',n,n)
    R=matrix('R',manifest['mas_rows'],manifest['mas_nodes']);Rt=matrix('RT',manifest['mas_nodes'],manifest['mas_rows'])
    need((R.T-Rt).nnz==0,'R and RT differ')
    inverse=np.fromfile(source/'mas_inverse.bin',dtype='<f8').reshape(-1,48,48).transpose(0,2,1)
    abd=original('abd_inverse');offset=manifest['mas_offset'];abd_offset=manifest['abd_offset'];nodes=manifest['mas_nodes']
    def B0(v):
        answer=v.copy();answer[abd_offset:abd_offset+12]=abd@v[abd_offset:abd_offset+12]
        r=(R@v[offset:offset+3*nodes].reshape(nodes,3)).reshape(-1,48)
        z=np.einsum('bij,bj->bi',inverse,r).reshape(-1,3)
        answer[offset:offset+3*nodes]=(Rt@z).ravel();return answer
    bank=actual('history_bank',(n,m));need(np.array_equal(bank,original('expected_history')),'GPU history copy changed source samples')
    S=original('final_source_x')[:,None]-bank;need(np.array_equal(S,original('expected_S')),'History subtraction changed samples')
    gram=S.T@(source_A@S);eigen=np.linalg.eigvalsh(.5*(gram+gram.T));checks=[];costs=[]
    for entry in run['ranks']:
        rank=entry['rank'];z=actual('Z_r'+str(rank),(n,rank));w=actual('W_r'+str(rank),(n,rank));factor=actual('factor_r'+str(rank),(rank,rank))
        expected_z=original('expected_Z_'+str(rank));span=rel(z,expected_z@(expected_z.T@z))
        orth=float(np.linalg.norm(z.T@z-np.eye(rank)));w_error=rel(w,target_A@z)
        L=np.tril(factor);E=z.T@(target_A@z);E=.5*(E+E.T);factor_error=rel(L@L.T,E)
        raw_gram=actual('gram_r'+str(rank),(m,m));gram_error=rel(raw_gram,.5*(gram+gram.T))
        ev=actual('eigen_r'+str(rank),(m,));eigen_error=rel(ev,eigen)
        need(span<1e-6 and orth<1e-10 and w_error<1e-11 and factor_error<1e-11 and gram_error<1e-10
             and eigen_error<1e-10 and np.all(np.diag(L)>0),'GPU numerical contract failed')
        op=Balanced(target_A,B0,z);probes=original('probes');action=[]
        for probe in range(3):
            v=actual(f'balanced_r{rank}_p{probe}',(n,));error=rel(v,op(probes[:,probe]));need(error<1e-8,'Independent wrapper calibration failed');action.append(error)
        checks.append(dict(rank=rank,source_POD_span_relative=span,orthogonality_frobenius=orth,
            target_AZ_relative=w_error,Cholesky_relative=factor_error,source_Gram_relative=gram_error,
            source_eigenvalues_relative=eigen_error,independent_action_relative=action))
        setup_gpu=float(np.median([x['gpu_ms'] for x in entry['pipeline_samples']]))
        setup_wall=float(np.median([x['wall_ms'] for x in entry['pipeline_samples']]))
        for mode in ('graph','host'):
            pairs=[]
            for i in range(3):
                group=[x for x in entry['apply_samples'] if x['mode']==mode and x['pair']==i]
                base=next(x for x in group if x['arm']=='MAS_reference');candidate=next(x for x in group if x['arm']=='balanced')
                pairs.append(dict(base_us=base['GPU_us_per_apply'],balanced_us=candidate['GPU_us_per_apply'],
                    added_us=candidate['GPU_us_per_apply']-base['GPU_us_per_apply']))
            if budget is None:continue # Tiny QA has no Mixed scene acceleration budget.
            delta=float(np.median([p['added_us'] for p in pairs]))
            method=next(v for v in budget['methods'] if v['method']=='transfer_pod_'+str(rank))
            N=method['iterations'];baseline=budget['methods'][0]['iterations']
            unit=budget['native_default_iteration_ms'];saved=(baseline-N)*unit
            costs.append(dict(rank=rank,execution=mode,paired_samples=pairs,median_added_GPU_us=delta,
                pipeline_GPU_ms=setup_gpu,pipeline_wall_ms=setup_wall,assumed_target_iterations=N,
                assumed_original_GPU_ms_per_iteration=unit,conditional_saved_ms_before_extra=saved,
                conditional_added_ms=setup_wall+N*delta/1000,
                conditional_net_ms=saved-setup_wall-N*delta/1000,
                interpretation='Assumes M3 CPU iterations and old native mean cost transfer; not native GPU PCG evidence'))
    write_new(folder/args.output_name,dict(passed=True,history_bitwise_equal=True,checks=checks,costs=costs,
        budget_source=record(budget_source) if budget_source else None,
        source=record(Path(__file__)),inputs=manifest['records'],results=records+[record(raw/'result.json')],
        performance_certified=False,quality_certified=False,native_solver_modified=False))
    print(json.dumps(dict(passed=True,checks=checks,costs=costs)),flush=True)


if __name__=='__main__':main()
