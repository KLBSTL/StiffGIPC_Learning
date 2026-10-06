"""Independent CPU checks of actual v41 snapshots, selecting the active inverse precision."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import unpack,difference

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();rows=[]
for audit in sorted((a.root/'runs/autodl').glob('*/mas_audit_*_audit.json')):
    prefix=Path(str(audit)[:-len('_audit.json')]);record=json.loads(audit.read_text());meta,A,b=matrix_from_snapshot(prefix)
    active=next(m for m in meta['local_preconditioners'] if m['kind']=='MAS_full_owned_buffers')
    double=active.get('inverse64',False);field='d_precondMatMas64' if double else 'd_precondMatMas'
    nodes,mapped,levels,_,clusters,*_=active['dimensions'];offset=3*active['offset'];count=active['buffers'][field]['count'];bank=clusters//count
    blocks=unpack(str(prefix)+'_mas_'+field+'.bin','<f8' if double else '<f4',count,bank)
    eigen=np.linalg.eigvalsh((blocks+blocks.transpose(0,2,1))/2)
    read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];real=read('d_real_map_partId','<i4')[:nodes]
    coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    def apply(v):
        out=np.zeros_like(v);fem=v[offset:].reshape(nodes,3);lr=np.zeros((clusters,3));valid=part>=0
        lr[np.flatnonzero(valid)]=fem[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],fem)
        lz=np.einsum('nij,nj->ni',blocks,lr.reshape(count,bank*3)).reshape(clusters,3)
        z=lz[real].copy()
        for level in range(levels-1):z+=lz[coarse[:,level]]
        out[offset:]=z.ravel()
        for m in meta['local_preconditioners']:
            if m['kind']=='abd':
                inv=np.fromfile(str(prefix)+'_'+m['buffer']+'.bin','<f8').reshape(-1,12,12).transpose(0,2,1)
                begin=3*m['offset'];end=begin+12*len(inv)
                out[begin:end]=np.einsum('nij,nj->ni',inv,v[begin:end].reshape(-1,12)).ravel()
        return out
    checks=[]
    for i in range(10):
        v=np.fromfile(str(prefix)+f'_v{i}.bin','<f8');gpu=np.fromfile(str(prefix)+f'_Mv{i}.bin','<f8')
        checks.append(difference(apply(v),gpu))
    x=np.fromfile(str(prefix)+'_x.bin','<f8')
    row={'path':str(audit),'frame':record['frame'],'inverse64':double,'solution_unchanged':record['solution_unchanged_bitwise'],
         'cpu_true_residual':float(np.linalg.norm(b-A@x)/np.linalg.norm(b)),
         'initial_rho_cpu':float(b@apply(b)),'inverse_min_eigenvalue':float(eigen.min()),
         'inverse_negative_blocks':np.flatnonzero(eigen[:,0]<0).tolist(),
         'max_cpu_operator_relative':max(c['relative'] for c in checks),'cpu_operator_checks':checks,
         'max_gpu_repeat':max(max(x['repeat_relative']) for x in record['probes'])}
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cpu_operator_checks'}),flush=True)
a.output.write_text(json.dumps(rows,indent=2,allow_nan=False))
