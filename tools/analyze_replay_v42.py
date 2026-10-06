"""Independent CPU sparse residual, actual-factor action, and fixed gates."""
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import unpack,difference

ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/replay_v42_20261003';rows=[]
cases=[('v39_default',39,'wide_bunny'),('v39_strict',39,'wide_bunny_strict'),
       ('v41_graph',41,'double_bunny'),('v41_host',41,'double_bunny_host'),('v41_wide',41,'wide_guard')]
for case,version,name in cases:
    prefix=ROOT/f'downloads/autodl_perf_v{version}_20261003/runs/autodl'/f'autodl_perf_v{version}_{name}'/'mas_audit_failure'
    meta,A,b=matrix_from_snapshot(prefix)
    m=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    nodes,mapped,levels,_,clusters,*_=m['dimensions'];offset=3*m['offset'];count=m['buffers']['d_inverseMatMas']['count'];bank=clusters//count;size=bank*3
    read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];real=read('d_real_map_partId','<i4')[:nodes];coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    h=unpack(str(prefix)+'_mas_d_inverseMatMas.bin','<f8',count,bank);h=(h+h.transpose(0,2,1))/2
    diagonal=np.arange(size);h[:,diagonal,diagonal]=np.where(h[:,diagonal,diagonal]==0,1,h[:,diagonal,diagonal])
    for mode in [5,6,7]:
        run=data/'runs'/f'{case}_m{mode}'
        if not (run/'result.json').exists():rows.append({'case':case,'mode':mode,'missing':True});continue
        raw=json.loads((run/'result.json').read_text());row={'case':case,'mode':mode,'operators':[],'solves':[]}
        row['operator_repeat_max']=max(d['relative'] for p in raw['operators'] for key in ['repeats','graph_repeats'] for d in p[key])
        row['operator_bitwise']=all(d['unequal']==0 for p in raw['operators'] for key in ['repeats','graph_repeats'] for d in p[key])
        row['operator_repeat_passed']=raw['operator_repeat_passed'];row['min_probe_quadratic']=min(p['v_Mv'] for p in raw['operators'])
        if mode>=6:
            factor=np.fromfile(run/'factors.bin','<f8').reshape(count,size,size)
            reconstructed=factor@factor.transpose(0,2,1)
            errors=np.linalg.norm(reconstructed-h,axis=(1,2))/np.linalg.norm(h,axis=(1,2))
            row.update(factor_max_reconstruction_relative=float(errors.max()),factor_min_diagonal=float(np.diagonal(factor,axis1=1,axis2=2).min()))
            for k in range(10):
                v=np.fromfile(str(prefix)+f'_v{k}.bin','<f8');fem=v[offset:].reshape(nodes,3)
                lr=np.zeros((clusters,3));valid=part>=0;lr[np.flatnonzero(valid)]=fem[part[valid]]
                for level in range(levels-1):np.add.at(lr,coarse[:,level],fem)
                w=lr.reshape(count,size).copy()
                for j in range(size):
                    w[:,j]/=factor[:,j,j];w[:,j+1:]-=factor[:,j+1:,j]*w[:,j,None]
                for j in range(size-1,-1,-1):
                    w[:,j]/=factor[:,j,j];w[:,:j]-=factor[:,j,:j]*w[:,j,None]
                lz=w.reshape(clusters,3);fem_z=lz[real].copy()
                for level in range(levels-1):fem_z+=lz[coarse[:,level]]
                out=np.zeros_like(v);out[offset:]=fem_z.ravel()
                for local in meta['local_preconditioners']:
                    if local['kind']=='abd':
                        inverse=np.fromfile(str(prefix)+'_'+local['buffer']+'.bin','<f8').reshape(-1,12,12).transpose(0,2,1)
                        start=3*local['offset'];stop=start+12*len(inverse)
                        out[start:stop]=np.einsum('nij,nj->ni',inverse,v[start:stop].reshape(-1,12)).ravel()
                row['operators'].append({'probe':k,**difference(out,np.fromfile(run/f'Mv{k}.bin','<f8'))})
            row['cpu_operator_max_relative']=max(c['relative'] for c in row['operators'])
            row['factor_passed']=row['factor_min_diagonal']>0 and row['factor_max_reconstruction_relative']<=1e-12
            row['cpu_operator_passed']=row['cpu_operator_max_relative']<=1e-10
        first={}
        for s in raw['solves']:
            x=np.fromfile(run/s['file'],'<f8');res=float(np.linalg.norm(b-A@x)/np.linalg.norm(b));first.setdefault(s['tolerance'],x)
            item={k:v for k,v in s.items() if k!='history'}
            item.update(cpu_true_residual=res,cpu_gpu_residual_difference=abs(res-s['true_relative_residual']),solution_repeat=difference(first[s['tolerance']],x))
            row['solves'].append(item)
        strict=[s for s in row['solves'] if s['tolerance']==1e-14]
        row['all_rho_stop']=len(row['solves'])==4 and all(s['reason']=='rho_stop' for s in row['solves'])
        row['strict_residual_max']=max(s['cpu_true_residual'] for s in strict)
        row['strict_repeat_max']=max(s['solution_repeat']['relative'] for s in strict)
        row['strict_residual_passed']=row['strict_residual_max']<=1e-8
        row['strict_repeat_passed']=row['strict_repeat_max']<=1e-6
        row['all_fixed_gates_passed']=mode>=6 and all(row[k] for k in ['factor_passed','cpu_operator_passed','operator_repeat_passed','all_rho_stop','strict_residual_passed','strict_repeat_passed'])
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('solves','operators')}),flush=True)
target=ROOT/'reports/REPLAY_V42_CPU.json';assert not target.exists();target.write_text(json.dumps(rows,indent=2,allow_nan=False))
