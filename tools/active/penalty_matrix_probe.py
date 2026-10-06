"""Finite A-only normal-contact curvature probe; RHS and saved MAS stay fixed.

This is not a physical candidate or a total-linear speed benchmark. A live run
must rebuild AL derivatives and MAS consistently for any acceptance claim.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from config import ROOT, read, sha
from cross_rhs import operator, pcg
sys.path.insert(0, str(ROOT/'tools'))
from event_model_v48 import CONTACT


def probe(prefix, new_mu):
    state=read(str(prefix)+'_state.json')
    run=prefix.parent.parent
    scene=read(run/'output/scene.json')
    if scene['effective_scalar_fields']['friction_compiled'] is not False:
        raise ValueError('Cannot isolate normal AL when friction is present')
    frame=read(run/'output/stats.json')['frames'][state['frame']-1]
    if frame.get('toi_reduced_slack') is not False:
        raise ValueError('Need the fixed-slack normal AL operator')
    A,b,M=operator(prefix)
    meta=read(str(prefix)+'_meta.json')
    contacts=np.fromfile(str(prefix)+'_state_contacts.bin',CONTACT)
    mu,initial_mu,*_=np.fromfile(str(prefix)+'_state_scalars.bin','<f8')
    trace=read(run/'trace/metadata.json')
    abd=trace['abd_point_num']
    abd_dofs=sum(12*p['count'] for p in meta['local_preconditioners'] if p['kind']=='abd')
    boundaries=np.fromfile(run/'trace/boundary_types.bin','<i4')
    rr,cc,vv=[],[],[]
    # Only frozen FEM-only contacts are representable without production J.
    # Refuse a partial subtraction instead of treating ABD contacts as FEM.
    for c in contacts:
        count=1 if c['kind']==2 else 4
        ids=c['ids'][:count].astype(int)
        if np.any(ids<abd):raise ValueError('ABD contact needs saved production J')
        weights=(c['penalty_mu']*mu/initial_mu if c['penalty_mu']>0 else mu)*c['gamma']
        columns=[];values=[]
        for slot,vertex in enumerate(ids):
            if boundaries[vertex]!=0:continue
            for axis in range(3):
                columns.append(abd_dofs+3*(vertex-abd)+axis)
                values.append(float(c['grad'][slot][axis]))
        for i,ci in enumerate(columns):
            for j,cj in enumerate(columns):
                value=weights*values[i]*values[j]
                if value:rr.append(ci);cc.append(cj);vv.append(value)
    H=sparse.coo_matrix((vv,(rr,cc)),shape=A.shape).tocsr()
    mu_source='explicit CLI'
    if new_mu is None:
        # Source-verified affine mapping x=t+A*xbar. Recover J from the exact
        # saved x/q; this is an independently reconstructed J, not a production
        # J dump. Only the all-free mixed ABD input is supported here.
        abd_objects=[o for o in scene['objects'] if o['body_type']=='ABD']
        if any(o.get('fixed_mode')=='all' for o in abd_objects):
            raise ValueError('Frozen world estimator requires all-free ABD bodies')
        q=np.fromfile(str(prefix)+'_state_q.bin','<f8').reshape(-1,12)
        vertices=np.fromfile(str(prefix)+'_state_vertices.bin','<f8').reshape(-1,3)
        body_ids=np.fromfile(run/'trace/body_ids.bin','<i4')[:abd]
        A0=(A-H).tocsr()
        d0=A0.diagonal()[abd_dofs:].reshape(-1,3)
        free=boundaries[abd:]==0
        maximum=float(np.abs(d0[free]).max(initial=0))
        for body in range(len(q)):
            hb=A0[12*body:12*(body+1),12*body:12*(body+1)].toarray()
            if np.linalg.eigvalsh(hb).min()<=0:raise ValueError('Nonpositive ABD block')
            inv=np.linalg.inv(hb)
            points=np.flatnonzero(body_ids==body)
            xbar=np.linalg.solve(q[body,3:].reshape(3,3),(vertices[points]-q[body,:3]).T).T
            J=np.zeros((len(points),3,12));J[:,:,:3]=np.eye(3)
            for axis in range(3):J[:,axis,3+3*axis:6+3*axis]=xbar
            c=J@inv@J.transpose(0,2,1)
            if np.linalg.eigvalsh(c).min()<=0:raise ValueError('Nonpositive vertex compliance')
            k=np.linalg.inv(c)
            maximum=max(maximum,float(np.diagonal(k,axis1=1,axis2=2).max(initial=0)))
        new_mu=.1*maximum
        mu_source='reconstructed world-block estimator at saved inner state; live estimator is evaluated at frame start'
    changed=(A+(new_mu/mu-1)*H).tocsr()
    asym=sparse.linalg.norm(changed-changed.T)/max(sparse.linalg.norm(changed),1e-300)
    if asym>1e-12:raise ValueError('Changed matrix asymmetric')
    report={'prefix':str(prefix),'contacts':len(contacts),'old_mu':float(mu),'new_mu':new_mu,
            'new_mu_source':mu_source,'scale':new_mu/mu,'asymmetry':float(asym),'runs':[],
            'input_sha256':{p.name:sha(p) for p in sorted(prefix.parent.glob(prefix.name+'*')) if p.is_file()}}
    for name,matrix in [('historical_A',A),('scaled_normal_AL_A',changed)]:
        r=pcg(matrix,b,M)
        report['runs'].append({'arm':name,**r})
        print(json.dumps({'system':prefix.name,'arm':name,**{k:v for k,v in r.items() if k!='ledger'}}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefixes',nargs='+')
    p.add_argument('--new-mu',type=float);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.new_mu is not None and (not np.isfinite(a.new_mu) or a.new_mu<=0):raise ValueError('Positive mu required')
    target=ROOT/a.output
    if target.exists():raise FileExistsError(target)
    result={'scope':'A-only causal diagnostic; same b and saved M. Not consistent physical solve, rebuilt MAS performance or quality acceptance',
            'max_iterations':1000,'seconds_per_arm':90,'rho_tol':1e-4,
            'new_mu':a.new_mu,'script_sha256':sha(__file__),'systems':[]}
    for s in a.prefixes:
        result['systems'].append(probe(ROOT/s,a.new_mu))
        target.write_text(json.dumps(result,indent=2,allow_nan=False))
