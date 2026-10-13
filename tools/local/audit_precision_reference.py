"""Separate legacy MAS coefficient storage and apply precision on fixed A,b.

Use the identical saved hierarchy and local numeric matrices. Replacing exact
zero diagonals by one reproduces the native padding rule. CPU FP64 inversion
and reductions are diagnostics, not a native speed or equivalence claim.
"""
from pathlib import Path
import argparse
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):sys.path.insert(0,str(ROOT/'tools'/folder))
from audit_recycling_reference import preconditioner
from audit_contact_reference import cg
from cpu_fixed_reference import load_snapshot,fnv1a64
from local_identity import record
from config import read
from linux_runner import require,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--session',required=True)
    args=p.parse_args();require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session;require(not out.exists(),'Fresh output required');out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    matrix,rhs,active,zero,evidence,detail=load_snapshot(args.prefix)
    require(active.all(),'Need explicit exact-zero-row treatment')
    meta=read(Path(str(args.prefix)+'_meta.json'))
    native,operator_evidence,operators=preconditioner(args.prefix,meta,with_data=True)
    mas=[(offset,data) for kind,offset,data in operators if kind=='mas'];require(len(mas)==1,'One MAS owner required')
    offset,data=mas[0];nodes,restriction,stored,fine,coarse=data
    owner=next(m for m in meta['local_preconditioners'] if m['kind']=='MAS_full_owned_buffers')
    path=Path(str(args.prefix)+'_mas_d_inverseMatMas.bin');raw=path.read_bytes()
    expected=owner['buffers']['d_inverseMatMas'];require(len(raw)==expected['count']*expected['item_bytes']
        and fnv1a64(raw)==expected['fnv1a64'],'Local input matrix fingerprint differs')
    packed=np.frombuffer(raw,dtype='<f8').reshape(-1,136,3,3).transpose(0,1,3,2)
    h=np.zeros(stored.shape,dtype=np.float64)
    for i in range(16):
        for j in range(i,16):
            tile=packed[:,16*i-i*(i+1)//2+j];h[:,3*i:3*i+3,3*j:3*j+3]=tile
            if i!=j:h[:,3*j:3*j+3,3*i:3*i+3]=tile.transpose(0,2,1)
    indices=np.arange(48);diagonal=h[:,indices,indices];padding=diagonal==0
    h[:,indices,indices]=np.where(padding,1,diagonal)
    eig=np.linalg.eigvalsh(h);require(eig.min()>0,'Nonpositive local input matrix')
    exact=np.linalg.inv(h)
    old_error=np.linalg.norm(h@stored-np.eye(48),axis=(1,2))/np.sqrt(48)
    new_error=np.linalg.norm(h@exact-np.eye(48),axis=(1,2))/np.sqrt(48)
    require(new_error.max()<1e-6,'Inaccurate FP64 local inverse')
    def action(inverses,wide):
        dtype=np.float64 if wide else np.float32
        rmap=restriction.astype(dtype);coeff=inverses.astype(dtype)
        def apply(vector):
            output=vector.copy()
            for kind,local_offset,local_data in operators:
                if kind=='abd':
                    count=len(local_data)*12
                    output[local_offset:local_offset+count]=np.einsum('bij,bj->bi',local_data,
                        vector[local_offset:local_offset+count].reshape(-1,12)).ravel()
            r=rmap@vector[offset:offset+3*nodes].reshape(nodes,3).astype(dtype)
            z=np.einsum('bij,bj->bi',coeff,r.reshape(-1,48)).reshape(-1,3)
            value=z[fine].copy()
            for level in range(coarse.shape[1]):value+=z[coarse[:,level]]
            output[offset:offset+3*nodes]=value.ravel()
            return output
        return apply
    cases=[]
    for label,apply in (('native_CPU_MAS',native),('stored_FP32_coefficients_FP64_apply',action(stored,True)),
        ('FP64_inverse_FP32_storage_apply',action(exact,False)),('FP64_inverse_storage_apply',action(exact,True))):
        result=cg(matrix,rhs,apply,stop_apply=native);result['name']=label;cases.append(result)
        print({k:v for k,v in result.items() if k!='history'},flush=True)
    report=dict(controller=record(__file__),helper=record(ROOT/'tools/local/audit_recycling_reference.py'),
        evidence=evidence,operator_evidence=operator_evidence,input_matrix=record(path),scope=__doc__,
        local_matrices=len(h),padding_diagonals=int(padding.sum()),
        local_condition_quantiles=np.quantile(eig[:,-1]/eig[:,0],[0,.5,.9,.99,1]).tolist(),
        stored_inverse_error_quantiles=np.quantile(old_error,[0,.5,.9,.99,1]).tolist(),
        FP64_inverse_error_max=float(new_error.max()),cases=cases,
        stopping_norm='Common original MAS norm and original-b reference; previous-rho timing',
        physical_quality_certified=False,performance_certified=False)
    write_new(out/'analysis.json',report)


if __name__=='__main__':main()
