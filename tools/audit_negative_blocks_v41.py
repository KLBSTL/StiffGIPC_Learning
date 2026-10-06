"""Inspect active MAS inverses and corresponding input blocks at new failures."""
import argparse
import json
from pathlib import Path
import numpy as np
def unpack(path,dtype,count,bank):
    packed=np.fromfile(path,dtype).reshape(count,bank*(bank+1)//2,3,3).transpose(0,1,3,2)
    out=np.zeros((count,3*bank,3*bank))
    for r in range(bank):
        for c in range(r,bank):
            block=packed[:,bank*r-r*(r+1)//2+c]
            out[:,3*r:3*r+3,3*c:3*c+3]=block
            if r!=c:out[:,3*c:3*c+3,3*r:3*r+3]=block.transpose(0,2,1)
    return out

def gauss_jordan(matrix):
    gj=matrix.copy()
    for j in range(len(gj)):
        pivot=gj[j,j];column=gj[:,j].copy();gj[:,j]=0;gj[j,j]=1;gj[j]/=pivot
        for k in range(len(gj)):
            if k!=j:gj[k]-=column[k]*gj[j]
    return np.triu(gj)+np.triu(gj,1).T

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();records=[]
for metadata in sorted((a.root/'runs/autodl').glob('*/mas_audit_failure_meta.json')):
    prefix=str(metadata)[:-len('_meta.json')];meta=json.loads(metadata.read_text())
    local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    double=local.get('inverse64',False);field='d_precondMatMas64' if double else 'd_precondMatMas'
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];count=local['buffers'][field]['count'];bank=clusters//count
    blocks=unpack(prefix+'_mas_'+field+'.bin','<f8' if double else '<f4',count,bank)
    blocks=(blocks+blocks.transpose(0,2,1))/2
    h=unpack(prefix+'_mas_d_inverseMatMas.bin','<f8',count,bank)
    d=np.arange(3*bank);h[:,d,d]=np.where(h[:,d,d]==0,1,h[:,d,d]);raw_h=h.copy();h=(h+h.transpose(0,2,1))/2
    vals=np.linalg.eigvalsh(blocks);hv=np.linalg.eigvalsh(h)
    negative=np.flatnonzero(vals[:,0]<0)
    read=lambda name,dtype:np.fromfile(prefix+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    probes=[];selected=set(negative.tolist()+np.flatnonzero(hv[:,0]<=0).tolist())
    for k in [0,1]:
        b=np.fromfile(prefix+f'_v{k}.bin','<f8')[3*local['offset']:].reshape(nodes,3)
        lr=np.zeros((clusters,3));valid=part>=0;lr[np.flatnonzero(valid)]=b[part[valid]]
        for level in range(levels-1):np.add.at(lr,coarse[:,level],b)
        lr=lr.reshape(count,bank*3);q=np.einsum('ni,nij,nj->n',lr,blocks,lr)
        selected.update(np.argsort(q)[:3].tolist())
        probes.append({'probe':k,'kind':'rhs' if k==0 else 'true_residual','total_mas_quadratic':float(q.sum()),
                       'negative_terms_sum':float(q[q<0].sum()),'negative_quadratic_blocks':np.flatnonzero(q<0).tolist()})
    entries=[]
    for i in sorted(selected):
        cpu=np.linalg.inv(h[i]);sym=(cpu+cpu.T)/2
        gj=gauss_jordan(raw_h[i]);gj_sym=gauss_jordan(h[i])
        raw_inverse=np.linalg.inv(raw_h[i]);raw_inverse=np.triu(raw_inverse)+np.triu(raw_inverse,1).T
        try:np.linalg.cholesky(h[i]);chol=True
        except np.linalg.LinAlgError:chol=False
        entries.append({'block':i,'input_min_eigenvalue':float(hv[i,0]),'input_condition_number':float(np.linalg.cond(h[i])),
                        'cpu_cholesky_passed':chol,'stored_min_eigenvalue':float(vals[i,0]),
                        'input_relative_asymmetry':float(np.linalg.norm(raw_h[i]-raw_h[i].T)/np.linalg.norm(raw_h[i])),
                        'cpu_gauss_jordan_min_eigenvalue':float(np.linalg.eigvalsh(gj)[0]),
                        'cpu_gauss_jordan_product_error':float(np.linalg.norm(h[i]@gj-np.eye(3*bank))/np.sqrt(3*bank)),
                        'cpu_gauss_jordan_vs_gpu':float(np.linalg.norm(gj-blocks[i])/np.linalg.norm(gj)),
                        'cpu_symmetric_gauss_jordan_min_eigenvalue':float(np.linalg.eigvalsh(gj_sym)[0]),
                        'cpu_symmetric_gauss_jordan_product_error':float(np.linalg.norm(h[i]@gj_sym-np.eye(3*bank))/np.sqrt(3*bank)),
                        'cpu_raw_lapack_upper_min_eigenvalue':float(np.linalg.eigvalsh(raw_inverse)[0]),
                        'cpu_raw_lapack_upper_product_error':float(np.linalg.norm(h[i]@raw_inverse-np.eye(3*bank))/np.sqrt(3*bank)),
                        'cpu_inverse_min_eigenvalue':float(np.linalg.eigvalsh(sym)[0]),
                        'stored_inverse_product_error':float(np.linalg.norm(h[i]@blocks[i]-np.eye(3*bank))/np.sqrt(3*bank)),
                        'cpu_inverse_product_error':float(np.linalg.norm(h[i]@cpu-np.eye(3*bank))/np.sqrt(3*bank))})
    row={'run':metadata.parent.name,'inverse64':double,'input_nonpositive_blocks':np.flatnonzero(hv[:,0]<=0).tolist(),
         'negative_inverse_blocks':negative.tolist(),'probes':probes,'block_details':entries}
    records.append(row);print(json.dumps(row),flush=True)
a.output.write_text(json.dumps(records,indent=2,allow_nan=False))
