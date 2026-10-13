"""Independent contact/MAS diagnostics on immutable native snapshots.

Matrix-only counterfactuals hold b fixed. CPU arithmetic is not the production
GPU reduction order; iteration changes do not certify speed or physical quality.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):
    sys.path.insert(0,str(ROOT/'tools'/folder))
from cpu_fixed_reference import load_snapshot
from audit_recycling_reference import preconditioner
from linux_runner import require,write_new
from local_identity import record
from config import read


CONTACT=np.dtype(dict(names=['kind','ids','grad','anchor','offset','lambda','gamma','slack',
    'penalty_mu','friction_lambda','friction_gamma','age','coordinates'],
    formats=['<i4',('<i4',4),('<f8',(4,3)),('<f8',(4,3)),'<f8','<f8','<f8','<f8',
             '<f8','<f8','<f8','<i4',('<f8',2)],
    offsets=[0,4,24,120,216,224,232,240,248,256,264,272,288],itemsize=304))


def cg(matrix,rhs,apply,cap=500,stop_apply=None):
    reference=float(rhs@apply(rhs));require(reference>0,'Invalid reference rho')
    norm_apply=stop_apply or apply
    common_reference=float(rhs@norm_apply(rhs))
    x=np.zeros_like(rhs);r=rhs.copy();z=apply(r);p=z.copy();rho=float(r@z)
    history=[]
    for iteration in range(1,cap+1):
        metric=float(r@norm_apply(r))/common_reference
        ap=matrix@p;curvature=float(p@ap)
        if not curvature>0 or not np.isfinite(curvature):
            return dict(iterations=iteration,error='nonpositive_curvature',history=history)
        alpha=rho/curvature;x+=alpha*p;r-=alpha*ap
        new_z=apply(r);new=float(r@new_z)
        history.append(dict(iteration=iteration,rho_before=rho/reference,rho_after=new/reference,
            common_norm_before=metric,
            recursive_residual=float(np.linalg.norm(r)/np.linalg.norm(rhs)),alpha=alpha,beta=new/rho))
        if abs(metric)<=1e-4:
            return dict(iterations=iteration,true_residual=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs)),
                rho_ratio=new/reference,history=history,cap=False)
        if not new>0 or not np.isfinite(new):
            return dict(iterations=iteration,error='nonpositive_rho',history=history)
        p=new_z+(new/rho)*p;rho=new
    return dict(iterations=cap,cap=True,true_residual=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs)),history=history)


def block_jacobi(matrix):
    # One affine body (12 DOFs), then physical FEM vertices (3 DOFs each).
    require(matrix.shape[0]>=12 and (matrix.shape[0]-12)%3==0,'Unexpected system partition')
    abd=matrix[:12,:12].toarray();abd_inv=np.linalg.inv(abd)
    count=(matrix.shape[0]-12)//3;indices=np.arange(count)*3+12
    blocks=np.empty((count,3,3))
    for i in range(3):
        for j in range(3):blocks[:,i,j]=np.asarray(matrix[indices+i,indices+j]).ravel()
    eigen=np.linalg.eigvalsh(blocks)
    require(np.min(eigen)>0,'Nonpositive Jacobi block; no diagonal repair allowed')
    inverse=np.linalg.inv(blocks)
    def apply(vector):
        output=np.empty_like(vector);output[:12]=abd_inv@vector[:12]
        output[12:]=np.einsum('bij,bj->bi',inverse,vector[12:].reshape(-1,3)).ravel()
        return output
    return apply


def contact_matrix(prefix,meta,matrix,rhs,folder):
    event=read(Path(str(prefix)+'_event.json'))
    require(event['contact_bytes']==304 and event['capture_state_unchanged'],'Contact ABI/state protection differs')
    contacts=np.fromfile(str(prefix)+'_contacts.bin',dtype=CONTACT)
    x=np.fromfile(str(prefix)+'_vertices.bin',dtype='<f8').reshape(-1,3)
    direction=np.fromfile(str(prefix)+'_direction.bin',dtype='<f8').reshape(-1,3)
    q=np.fromfile(str(prefix)+'_abd_q.bin',dtype='<f8').reshape(-1,12)
    dq=np.fromfile(str(prefix)+'_abd_dq.bin',dtype='<f8').reshape(-1,12)
    trace=read(folder/'trace/metadata.json');abd_points=trace['abd_point_num']
    require(len(contacts)==event['contacts'] and len(x)==event['vertices'] and len(q)==1,'Unexpected contact or body count')
    affine=q[0,3:].reshape(3,3)
    rest=np.linalg.solve(affine,(x[:abd_points]-q[0,:3]).T).T
    reconstructed=rest@dq[0,3:].reshape(3,3).T+dq[0,:3]
    j_error=float(np.linalg.norm(reconstructed-direction[:abd_points])/max(np.linalg.norm(direction[:abd_points]),1e-30))
    require(j_error<1e-9,'Recovered affine Jacobian disagrees with native direction')
    rows=[];cols=[];values=[];gap=[]
    for i,c in enumerate(contacts):
        require(c['kind'] in (0,1,2) and 0<c['gamma']<=1 and 0<=c['age']<=26,'Contact ABI scalar check failed')
        count=1 if c['kind']==2 else 4;value=float(c['offset'])
        for j in range(count):
            node=int(c['ids'][j]);g=c['grad'][j];require(0<=node<len(x),'Contact vertex outside domain')
            value+=float(g@(x[node]-c['anchor'][j]))
            if node<abd_points:
                indices=np.arange(12);v=np.r_[g,(g[:,None]*rest[node]).ravel()]
            else:
                indices=12+3*(node-abd_points)+np.arange(3);v=g
            rows.extend([i]*len(indices));cols.extend(indices);values.extend(v)
        gap.append(value)
    g=coo_matrix((values,(rows,cols)),shape=(len(contacts),matrix.shape[0])).tocsr();g.sum_duplicates()
    mu=np.where(contacts['penalty_mu']>0,contacts['penalty_mu'],event['mu'])
    # There were no in-frame mu doublings; current initial/current values agree.
    require(np.max(np.abs(mu-event['mu']))<1e-12,'Per-contact penalty requires recorded initial/current scaling')
    gap=np.asarray(gap);slack=np.maximum(0,gap-contacts['lambda']/mu)
    slack_error=float(np.max(np.abs(slack-contacts['slack'])))
    require(slack_error<1e-10,'Contact layout/slack does not match captured assembly state')
    weight=mu*contacts['gamma'];c_all=g.T@g.multiply(weight[:,None])
    inactive=(contacts['slack']>0)&(contacts['lambda']==0)
    c_inactive=g.T@g.multiply((weight*inactive)[:,None])
    free=matrix-c_all
    diagonal=matrix.diagonal();free_d=free.diagonal()
    require(np.min(free_d)>0,'Subtracted contact matrix has nonpositive diagonal; check reconstruction')
    numerator=g.T@(contacts['gamma']*(mu*(gap-contacts['slack'])-contacts['lambda']))
    positive=diagonal>0
    info=dict(contact_count=len(contacts),kinds={str(k):int(np.sum(contacts['kind']==k)) for k in (0,1,2)},
        inactive_count=int(inactive.sum()),inactive_fraction=float(inactive.mean()),
        inactive_contact_trace_fraction=float(c_inactive.diagonal().sum()/c_all.diagonal().sum()),
        contact_trace_fraction=float(c_all.diagonal().sum()/diagonal.sum()),
        contact_diagonal_fraction_quantiles=np.quantile(c_all.diagonal()[positive]/diagonal[positive],[0,.5,.9,.99,1]).tolist(),
        contact_to_free_max=float(np.max(c_all.diagonal()/free_d)),
        min_free_diagonal=float(free_d.min()),slack_error=slack_error,jacobian_direction_relative_error=j_error,
        contact_gradient_to_total_rhs=float(np.linalg.norm(numerator)/np.linalg.norm(rhs)),
        gamma_quantiles=np.quantile(contacts['gamma'],[0,.5,.9,1]).tolist(),
        release_age_histogram={str(k):int(np.sum(contacts['age']==k)) for k in np.unique(contacts['age'])})
    return c_all.tocsr(),c_inactive.tocsr(),info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--common-norm',action='store_true')
    args=p.parse_args();require(args.session.replace('_','').isalnum(),'Invalid session')
    out=ROOT/'runs'/args.session;require(not out.exists(),'Fresh output required');out.mkdir(parents=True)
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    reports=[]
    folder=args.capture/'linear113'
    for label in ('e1','e2'):
        start=time.monotonic();prefix=folder/'events'/label
        matrix,rhs,active,zero,evidence,detail=load_snapshot(prefix)
        meta=read(Path(str(prefix)+'_meta.json'))
        require(active.all(),'Exact-zero DOFs need explicit elimination')
        apply,operator_evidence=preconditioner(prefix,meta)
        total,inactive,contact=contact_matrix(prefix,meta,matrix,rhs,folder)
        cases=[]
        definitions=[
            ('native_coefficients_CPU_MAS',matrix,apply),
            ('inactive_curvature_removed_frozen_MAS',matrix-inactive,apply),
            ('all_contact_0p1_frozen_MAS',matrix-.9*total,apply),
            ('original_block_Jacobi',matrix,block_jacobi(matrix)),
            ('inactive_removed_block_Jacobi',matrix-inactive,block_jacobi(matrix-inactive)),
            ('all_contact_0p1_block_Jacobi',matrix-.9*total,block_jacobi(matrix-.9*total))]
        full_abd_action=block_jacobi(matrix)
        abd3=np.stack([np.linalg.inv(matrix[i:i+3,i:i+3].toarray()) for i in range(0,12,3)])
        def global3(vector):
            output=full_abd_action(vector)
            output[:12]=np.einsum('bij,bj->bi',abd3,vector[:12].reshape(4,3)).ravel()
            return output
        definitions.append(('global_3x3_Jacobi',matrix,global3))
        for name,mat,action in definitions:
            result=cg(mat,rhs,action,stop_apply=apply if args.common_norm else None)
            result['name']=name;cases.append(result)
            print(json.dumps(dict(event=label,name=name,**{k:v for k,v in result.items() if k not in ('history','name')})),flush=True)
        report=dict(event=label,contact=contact,cases=cases,system=detail,evidence=evidence,
            operator_evidence=operator_evidence,elapsed_seconds=time.monotonic()-start,
            scope=__doc__,common_original_M_norm=args.common_norm,
            quality_certified=False,performance_certified=False)
        write_new(out/(label+'.json'),report);reports.append(report)
    write_new(out/'summary.json',dict(controller=record(__file__),events=[dict(event=r['event'],contact=r['contact'],
        cases=[{k:v for k,v in c.items() if k!='history'} for c in r['cases']]) for r in reports]))


if __name__=='__main__':main()
