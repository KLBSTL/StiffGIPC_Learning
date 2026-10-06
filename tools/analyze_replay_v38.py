"""CPU hierarchical MAS action and symmetric A residual cross-checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import unpack,difference

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
task=Path(__file__).resolve().parents[1]
previous=task/'downloads/autodl_perf_v37_20261003/runs/autodl'
prefixes={'smoke':previous/'autodl_perf_v37_mas_smoke/mas_audit_initial',
          'initial':previous/'autodl_perf_v37_bunny_mas/mas_audit_initial',
          'failure':previous/'autodl_perf_v37_bunny_mas/mas_audit_failure'}
records=[]
for case,prefix in prefixes.items():
    dirs=sorted((a.root/'runs').glob(case+'_m*'))
    if not dirs:continue
    meta,matrix,rhs=matrix_from_snapshot(prefix)
    local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];offset=3*local['offset']
    count=local['buffers']['d_inverseMatMas']['count'];bank=clusters//count
    assert bank==16 and offset+3*nodes==len(rhs)
    partial=lambda field,dtype:np.fromfile(str(prefix)+'_mas_'+field+'.bin',dtype)
    part=partial('d_partId_map_real','<i4')[:mapped]
    real=partial('d_real_map_partId','<i4')[:nodes]
    coarse=partial('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    assert np.array_equal(part[real],np.arange(nodes))
    assert coarse.min()>=mapped and coarse.max()<clusters
    h=unpack(str(prefix)+'_mas_d_inverseMatMas.bin','<f8',count,bank)
    d=np.arange(3*bank);h[:,d,d]=np.where(h[:,d,d]==0,1,h[:,d,d])
    inv=np.linalg.inv((h+h.transpose(0,2,1))/2)
    stored=unpack(str(prefix)+'_mas_d_precondMatMas.bin','<f4',count,bank)
    abd=[x for x in meta['local_preconditioners'] if x['kind']=='abd']
    def apply(v,blocks):
        lr=np.zeros((clusters,3));fem=v[offset:].reshape(nodes,3)
        valid=part>=0;lr[np.flatnonzero(valid)]=fem[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],fem)
        lz=np.einsum('nij,nj->ni',blocks,lr.reshape(count,bank*3)).reshape(clusters,3)
        out=np.zeros_like(v);f=lz[real].copy()
        for level in range(levels-1):f+=lz[coarse[:,level]]
        out[offset:]=f.ravel()
        for local_abd in abd:
            mats=np.fromfile(str(prefix)+'_'+local_abd['buffer']+'.bin','<f8').reshape(-1,12,12).transpose(0,2,1)
            start=3*local_abd['offset'];end=start+len(mats)*12
            out[start:end]=np.einsum('nij,nj->ni',mats,v[start:end].reshape(-1,12)).ravel()
        return out
    reference={kind:[apply(np.fromfile(str(prefix)+f'_v{k}.bin','<f8'),blocks) for k in range(10)] for kind,blocks in [('stored',stored),('fp64',inv)]}
    gpu_inverse=a.root/'reports'/f'{case}_inverse64.bin'
    if gpu_inverse.exists():
        exported=unpack(gpu_inverse,'<f8',count,bank)
        reference['fp64_gpu']=[apply(np.fromfile(str(prefix)+f'_v{k}.bin','<f8'),exported) for k in range(10)]
        inverse_check={'relative_vs_cpu_reference':float(np.linalg.norm(exported-inv)/np.linalg.norm(inv)),
            'max_inverse_product_error':float(np.max(np.linalg.norm(h@exported-np.eye(3*bank),axis=(1,2))/np.sqrt(3*bank)))}
    for run in dirs:
        raw=json.loads((run/'result.json').read_text());mode=int(run.name[-1]);kind='fp64' if mode in [3,5] else 'stored'
        row={'case':case,'mode':raw['mode'],'inverse32_identical':raw['native_inverse_recomputed_bitwise'],
             'max_repeat':max(d['relative'] for v in raw['operators'] for d in v['repeats']),
             'max_vs_v37':max(v['versus_v37']['relative'] for v in raw['operators']),
             'max_scaled_asymmetry':max(raw['bilinear']),
             'cpu_hierarchy_operator_checks':[],'solves':[]}
        for k in range(10):
            gpu=np.fromfile(run/f'Mv{k}.bin','<f8')
            row['cpu_hierarchy_operator_checks'].append({'index':k,**difference(reference[kind][k],gpu)})
        if mode in [3,5] and 'fp64_gpu' in reference:
            row['gpu_inverse_check']=inverse_check
            row['same_inverse_cpu_hierarchy_checks']=[{'index':k,**difference(reference['fp64_gpu'][k],np.fromfile(run/f'Mv{k}.bin','<f8'))} for k in range(10)]
        first={}
        for solve in raw['solves']:
            x=np.fromfile(run/solve['file'],'<f8');res=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs))
            key=solve['tolerance'];first.setdefault(key,x)
            row['solves'].append({**solve,'cpu_true_residual':res,
                'cpu_gpu_residual_disagreement':abs(res-solve['true_relative_residual']),
                'solution_repeat':difference(first[key],x)})
        records.append(row)
assert not a.output.exists()
a.output.write_text(json.dumps(records,indent=2,allow_nan=False))
for row in records:
    print(json.dumps({'case':row['case'],'mode':row['mode'],'max_repeat':row['max_repeat'],
        'max_cpu_operator_difference':max(x['relative'] for x in row['cpu_hierarchy_operator_checks']),
        'max_residual_disagreement':max(x['cpu_gpu_residual_disagreement'] for x in row['solves']),
        'solves':[(x['tolerance'],x['reason'],x['iterations'],x['cpu_true_residual']) for x in row['solves']]}))
