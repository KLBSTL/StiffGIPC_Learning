import json
from pathlib import Path
import numpy as np
from analyze_perf_v37 import unpack

ROOT=Path(__file__).resolve().parents[1]
runs=ROOT/'downloads/autodl_perf_v39_20261003/runs/autodl'
records=[]
for name in ['autodl_perf_v39_wide_bunny','autodl_perf_v39_wide_bunny_strict']:
    prefix=runs/name/'mas_audit_failure';meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    local=next(x for x in meta['local_preconditioners'] if x['kind']=='MAS_full_owned_buffers')
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];count=local['buffers']['d_precondMatMas']['count'];bank=clusters//count
    b=np.fromfile(str(prefix)+'_rhs.bin','<f8')[3*local['offset']:].reshape(nodes,3)
    read=lambda name,dtype:np.fromfile(str(prefix)+'_mas_'+name+'.bin',dtype)
    part=read('d_partId_map_real','<i4')[:mapped];coarse=read('d_coarseTable','<i4').reshape(-1,6)[:nodes,:levels-1]
    lr=np.zeros((clusters,3));valid=part>=0;lr[np.flatnonzero(valid)]=b[part[valid]]
    for level in range(levels-1):np.add.at(lr,coarse[:,level],b)
    lr=lr.reshape(count,bank*3)
    blocks=unpack(str(prefix)+'_mas_d_precondMatMas.bin','<f4',count,bank)
    h=unpack(str(prefix)+'_mas_d_inverseMatMas.bin','<f8',count,bank)
    d=np.arange(3*bank);h[:,d,d]=np.where(h[:,d,d]==0,1,h[:,d,d])
    q=np.einsum('ni,nij,nj->n',lr,blocks,lr)
    vals=np.linalg.eigvalsh(blocks);negative=np.flatnonzero(vals[:,0]<0)
    entries=[]
    for i in sorted(set(negative.tolist()+np.argsort(q)[:3].tolist())):
        symmetric=(h[i]+h[i].T)/2
        cpu=np.linalg.inv(symmetric);rounded=cpu.astype(np.float32).astype(np.float64)
        entries.append({'block':i,'rhs_norm':float(np.linalg.norm(lr[i])),
            'stored_min_eigenvalue':float(vals[i,0]),'stored_quadratic':float(q[i]),
            'input_min_eigenvalue':float(np.linalg.eigvalsh(symmetric)[0]),'input_condition_number':float(np.linalg.cond(symmetric)),
            'cpu_inverse_quadratic':float(lr[i]@cpu@lr[i]),
            'rounded_cpu_inverse_quadratic':float(lr[i]@rounded@lr[i]),
            'rounded_cpu_inverse_min_eigenvalue':float(np.linalg.eigvalsh((rounded+rounded.T)/2)[0]),
            'stored_inverse_product_error':float(np.linalg.norm(symmetric@blocks[i]-np.eye(48))/np.sqrt(48)),
            'cpu_inverse_product_error':float(np.linalg.norm(symmetric@cpu-np.eye(48))/np.sqrt(48))})
    row={'run':name,'negative_stored_inverse_blocks':negative.tolist(),'negative_quadratic_blocks':np.flatnonzero(q<0).tolist(),
         'total_mas_quadratic':float(q.sum()),'negative_terms_sum':float(q[q<0].sum()),'block_details':entries}
    records.append(row);print(json.dumps(row),flush=True)
out=ROOT/'reports/PCG_V40_NEGATIVE_BLOCKS.json';assert not out.exists();out.write_text(json.dumps(records,indent=2,allow_nan=False))
