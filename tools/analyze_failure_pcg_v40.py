"""Independent split of b^T P b across FEM MAS and ABD inverse blocks."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
from analyze_perf_v37 import unpack

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();results=[]
for path in sorted(a.root.glob('*/mas_audit_failure_meta.json')):
    prefix=Path(str(path)[:-len('_meta.json')]);meta,A,b=matrix_from_snapshot(prefix)
    row={'snapshot':str(prefix),'dofs':len(b),'parts':[]}
    full=np.zeros_like(b)
    for local in meta['local_preconditioners']:
        offset=3*local['offset']
        if local['kind']=='abd':
            mats=np.fromfile(str(prefix)+'_'+local['buffer']+'.bin','<f8').reshape(-1,12,12).transpose(0,2,1)
            values=b[offset:offset+12*len(mats)].reshape(-1,12)
            action=np.einsum('nij,nj->ni',mats,values).ravel()
            full[offset:offset+len(action)]=action
            vals=np.linalg.eigvalsh((mats+mats.transpose(0,2,1))/2)
            h=A[offset:offset+12*len(mats),offset:offset+12*len(mats)].toarray()
            row['parts'].append({'kind':'ABD','b_P_b':float(values.ravel()@action),
                 'inverse_eigenvalues':vals.tolist(),'A_principal_eigenvalues':np.linalg.eigvalsh((h+h.T)/2).tolist(),
                 'inverse_asymmetry':float(np.linalg.norm(mats-mats.transpose(0,2,1))/np.linalg.norm(mats))})
        elif local['kind']=='MAS_full_owned_buffers':
            nodes,mapped,levels,_,clusters,*_=local['dimensions'];count=local['buffers']['d_precondMatMas']['count'];bank=clusters//count
            read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
            part=read('d_partId_map_real','<i4')[:mapped];real=read('d_real_map_partId','<i4')[:nodes]
            coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
            blocks=unpack(str(prefix)+'_mas_d_precondMatMas.bin','<f4',count,bank)
            v=b[offset:offset+3*nodes].reshape(nodes,3);lr=np.zeros((clusters,3));valid=part>=0
            lr[np.flatnonzero(valid)]=v[part[valid]]
            for level in range(levels-1):np.add.at(lr,coarse[:,level],v)
            lz=np.einsum('nij,nj->ni',blocks,lr.reshape(count,bank*3)).reshape(clusters,3)
            action=lz[real].copy()
            for level in range(levels-1):action+=lz[coarse[:,level]]
            full[offset:offset+3*nodes]=action.ravel()
            row['parts'].append({'kind':'MAS','b_P_b':float(v.ravel()@action.ravel())})
    row['rho0_cpu']=float(b@full)
    row['negative_rho_makes_native_stop_impossible']=row['rho0_cpu']<0
    row['b_A_b']=float(b@(A@b))
    result=json.loads((path.parent/'output/stats.json').read_text())['frames'][-1]['newton'][-1]['pcg']
    row['simulator_pcg']=result
    if 'rho_initial' in result:row['rho_initial_absolute_disagreement']=abs(row['rho0_cpu']-result['rho_initial'])
    results.append(row);print(json.dumps(row),flush=True)
a.output.write_text(json.dumps(results,indent=2,allow_nan=False))
