"""NumPy-only CPU eigenanalysis of live MAS blocks. No GPU work."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
assert not a.output.exists()
records=[]
for path in sorted(a.root.glob('runs/autodl/*/mas_audit_*_audit.json')):
    audit=json.loads(path.read_text())
    prefix=str(path)[:-len('_audit.json')]
    for local in audit['system']['local_preconditioners']:
        if local['kind']!='MAS_full_owned_buffers':continue
        count=local['buffers']['d_inverseMatMas']['count']
        packed=local['buffers']['d_inverseMatMas']['item_bytes']//(9*8)
        bank=int((np.sqrt(1+8*packed)-1)/2)
        width=3*bank
        assert bank*(bank+1)//2==packed and count*bank==local['dimensions'][4]
        matrices=[]
        record={'file':str(path),'frame':audit['frame'],'blocks':count,'bank_size':bank,'spectra':[]}
        for name,dtype in [('d_inverseMatMas','<f8'),('d_precondMatMas','<f4')]:
            data=np.fromfile(prefix+'_mas_'+name+'.bin',dtype).reshape(count,packed,3,3).transpose(0,1,3,2)
            blocks=np.zeros((count,width,width))
            for row in range(bank):
                for col in range(row,bank):
                    m=data[:,bank*row-row*(row+1)//2+col]
                    blocks[:,3*row:3*row+3,3*col:3*col+3]=m
                    if row!=col:blocks[:,3*col:3*col+3,3*row:3*row+3]=m.transpose(0,2,1)
            if name=='d_inverseMatMas':
                # The actual inversion kernel substitutes 1 for zero diagonals.
                d=np.arange(width); diagonal=blocks[:,d,d]
                blocks[:,d,d]=np.where(diagonal==0,1,diagonal)
            eigenvalues=np.linalg.eigvalsh((blocks+blocks.transpose(0,2,1))/2)
            scale=np.maximum(np.max(np.abs(eigenvalues),axis=1),1e-300)
            rel=eigenvalues[:,0]/scale
            worst=int(np.argmin(rel))
            record['spectra'].append({'buffer':name,'finite':bool(np.isfinite(blocks).all()),
                'min_eigenvalue':float(eigenvalues.min()),'min_relative_eigenvalue':float(rel.min()),
                'nonpositive_blocks':int(np.count_nonzero(eigenvalues[:,0]<=0)),
                'negative_blocks_relative_1e7':int(np.count_nonzero(rel < -1e-7)),
                'worst_relative_block':worst,'worst_eigen_range':[float(eigenvalues[worst,0]),float(eigenvalues[worst,-1])],
                'max_relative_asymmetry':float(np.max(np.linalg.norm(blocks-blocks.transpose(0,2,1),axis=(1,2))/np.maximum(np.linalg.norm(blocks,axis=(1,2)),1e-300)))})
            matrices.append(blocks)
        error=np.linalg.norm(matrices[0]@matrices[1]-np.eye(width),axis=(1,2))/np.sqrt(width)
        record['inverse_product_error']={'max':float(error.max()),'median':float(np.median(error)),
                                          'max_block':int(np.argmax(error))}
        records.append(record)
a.output.write_text(json.dumps(records,indent=2,allow_nan=False))
print(json.dumps(records,indent=2))
