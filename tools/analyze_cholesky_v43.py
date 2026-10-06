"""Independent CPU action using actual MAS-class factors, both fixtures and simulations."""
import json
import argparse
from pathlib import Path
import numpy as np
from analyze_perf_v37 import unpack,difference
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/autodl_perf_v43_20261003';rows=[]
parser=argparse.ArgumentParser();parser.add_argument('--fixtures-only',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
cases=[('smoke',37,'mas_smoke','initial'),('default',39,'wide_bunny','failure'),('strict',39,'wide_bunny_strict','failure'),('v41_graph',41,'double_bunny','failure'),('v41_host',41,'double_bunny_host','failure'),('v41_wide',41,'wide_guard','failure')]
targets=[]
for case,version,name,kind in cases:
    prefix=ROOT/f'downloads/autodl_perf_v{version}_20261003/runs/autodl'/f'autodl_perf_v{version}_{name}'/f'mas_audit_{kind}'
    folder=data/'runs/fixtures_corrected'/case
    if not folder.exists():folder=data/'runs/fixtures'/case
    targets.append((case,prefix,folder,True))
for audit in ([] if args.fixtures_only else sorted((data/'runs/autodl').glob('*/mas_audit_*_audit.json'))):
    targets.append((audit.parent.name+'_'+audit.name,Path(str(audit)[:-len('_audit.json')]),audit.parent,False))
for name,prefix,folder,fixture in targets:
    if fixture and not (folder/'fixture.json').exists():continue
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text());m=next(v for v in meta['local_preconditioners'] if v['kind']=='MAS_full_owned_buffers')
    if not fixture and not m.get('cholesky'):continue
    nodes,mapped,levels,_,clusters,*_=m['dimensions'];offset=3*m['offset'];count=m['buffers']['d_inverseMatMas']['count'];bank=clusters//count;size=bank*3
    factor_path=folder/'factors.bin' if fixture else Path(str(prefix)+'_mas_d_cholesky.bin')
    factor=np.fromfile(factor_path,'<f8').reshape(count,size,size)
    h=unpack(str(prefix)+'_mas_d_inverseMatMas.bin','<f8',count,bank);h=(h+h.transpose(0,2,1))/2
    d=np.arange(size);h[:,d,d]=np.where(h[:,d,d]==0,1,h[:,d,d]);error=np.linalg.norm(factor@factor.transpose(0,2,1)-h,axis=(1,2))/np.linalg.norm(h,axis=(1,2))
    read=lambda field,dtype:np.fromfile(str(prefix)+'_mas_'+field+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];real=read('d_real_map_partId','<i4')[:nodes];coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    probes=[]
    for k in range(10):
        v=np.fromfile(str(prefix)+f'_v{k}.bin','<f8');lr=np.zeros((clusters,3));fem=v[offset:].reshape(nodes,3);valid=part>=0;lr[np.flatnonzero(valid)]=fem[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],fem)
        w=lr.reshape(count,size).copy()
        for j in range(size):w[:,j]/=factor[:,j,j];w[:,j+1:]-=factor[:,j+1:,j]*w[:,j,None]
        for j in range(size-1,-1,-1):w[:,j]/=factor[:,j,j];w[:,:j]-=factor[:,j,:j]*w[:,j,None]
        lz=w.reshape(clusters,3);z=lz[real].copy()
        for level in range(levels-1):z+=lz[coarse[:,level]]
        if fixture:expected=z.ravel();actual=np.fromfile(folder/f'p1_v{k}.bin','<f8')
        else:
            expected=np.zeros_like(v);expected[offset:]=z.ravel();actual=np.fromfile(str(prefix)+f'_Mv{k}.bin','<f8')
            for abd in meta['local_preconditioners']:
                if abd['kind']=='abd':
                    inv=np.fromfile(str(prefix)+'_'+abd['buffer']+'.bin','<f8').reshape(-1,12,12).transpose(0,2,1);start=3*abd['offset'];stop=start+12*len(inv)
                    expected[start:stop]=np.einsum('nij,nj->ni',inv,v[start:stop].reshape(-1,12)).ravel()
        check={'probe':k,**difference(expected,actual)}
        oldname={'default':'v39_default','strict':'v39_strict'}.get(name,name)
        old=ROOT/'downloads/replay_v42_20261003/runs'/f'{oldname}_m7'
        if fixture and (old/'result.json').exists():check['v42_action']=difference(np.fromfile(old/f'Mv{k}.bin','<f8')[offset:],actual)
        probes.append(check)
    row={'name':name,'fixture':fixture,'factor_min_diagonal':float(np.diagonal(factor,axis1=1,axis2=2).min()),
         'factor_max_reconstruction_relative':float(error.max()),'cpu_operator_max_relative':max(p['relative'] for p in probes),'probes':probes}
    if fixture:row['lifecycle']=json.loads((folder/'fixture.json').read_text())
    else:
        audit=json.loads(Path(str(prefix)+'_audit.json').read_text());_,A,b=matrix_from_snapshot(prefix);x=np.fromfile(str(prefix)+'_x.bin','<f8')
        row.update(solution_unchanged=audit['solution_unchanged_bitwise'],cpu_true_residual=float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),max_gpu_repeat=max(max(p['repeat_relative']) for p in audit['probes']))
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('probes','lifecycle')}),flush=True)
target=args.output or ROOT/'reports/CHOLESKY_V43_CPU.json';assert not target.exists();target.write_text(json.dumps(rows,indent=2,allow_nan=False))
