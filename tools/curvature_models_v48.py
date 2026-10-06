"""Independent 9x9 eigendecomposition of the exact Stable NH1 Hessian."""
import hashlib,json
from pathlib import Path
import numpy as np
from event_model_v48 import EventModel
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/CURVATURE_V48_BUNNY.json';assert not out.exists()
baseline=json.loads((ROOT/'reports/MODELS_V48_BUNNY_R2.json').read_text());assert baseline['passed']
epsilon=np.zeros((3,3,3))
for i,j,k in [(0,1,2),(1,2,0),(2,0,1)]:epsilon[i,j,k]=1;epsilon[i,k,j]=-1
rows=[]
for entry in baseline['events']:
    prefix=ROOT/'runs/local/v47_bunny_graph/events'/entry['event'];m=EventModel(prefix)
    for name,digest in entry['sha256'].items():assert hashlib.sha256((prefix.parent/name).read_bytes()).hexdigest()==digest
    F=m.edges(m.x)@m.inv;D=m.edges(m.d)@m.inv;vol=m.vol*m.dt**2
    raw=[];projected=[];negative=[];pg=np.zeros_like(m.x)
    for begin in range(0,len(F),2048):
        f=F[begin:begin+2048];d=D[begin:begin+2048];v=vol[begin:begin+2048];t=m.tets[begin:begin+2048]
        co=np.stack([np.cross(f[:,:,1],f[:,:,2]),np.cross(f[:,:,2],f[:,:,0]),np.cross(f[:,:,0],f[:,:,1])],axis=-1).reshape(-1,9)
        det2=np.einsum('ikm,jln,tmn->tijkl',epsilon,epsilon,f).reshape(-1,9,9)
        h=m.u*np.eye(9)[None]+m.v*co[:,:,None]*co[:,None,:]+(m.v*(np.linalg.det(f)-1)-m.u)[:,None,None]*det2
        ev,vec=np.linalg.eigh(h);dz=d.reshape(-1,9)
        raw.extend(np.einsum('ti,tij,tj->t',dz,h,dz)*v)
        coeff=np.einsum('tij,ti->tj',vec,dz)
        projected.extend(np.sum(np.maximum(ev,0)*coeff**2,axis=1)*v)
        negative.extend(np.sum(np.minimum(ev,0)*coeff**2,axis=1)*v)
        action=np.einsum('tij,tj->ti',vec,np.maximum(ev,0)*coeff).reshape(-1,3,3)
        local=(action@m.inv[begin:begin+len(f)].transpose(0,2,1))*v[:,None,None]
        np.add.at(pg,t[:,0],-local.sum(axis=2))
        for k in range(3):np.add.at(pg,t[:,k+1],local[:,:,k])
    raw=np.array(raw);projected=np.array(projected);negative=np.array(negative)
    check=entry['terms']['fem']['finite_differences'][1]['gradient_derivative']
    rel=float(abs(raw.sum()-check)/max(abs(check),1e-12))
    assert rel<=1e-6,('exact_hessian_vs_gradient_fd',entry['event'],rel)
    meta,A,b=matrix_from_snapshot(prefix);z=m.load('solution')
    constraint_d=np.einsum('nki,nki->n',m.cg,m.d[m.cids]);ag=np.zeros_like(m.x)
    np.add.at(ag,m.cids.ravel(),(m.weight[:,None,None]*constraint_d[:,None,None]*m.cg).reshape(-1,3))
    remainder=A@z-m.lift(pg)-m.lift(ag)
    _,g0,per0=m.fem(m.x,True);after=m.load('after_vertices').reshape(-1,3);_,_,per1=m.fem(after,True)
    # Element directional derivative directly in F space, independent of vertex accumulation.
    cof=np.stack([np.cross(F[:,:,1],F[:,:,2]),np.cross(F[:,:,2],F[:,:,0]),np.cross(F[:,:,0],F[:,:,1])],axis=-1)
    P=m.u*F+(m.v*(np.linalg.det(F)-1)-m.u)[:,None,None]*cof
    derivative=-np.einsum('tij,tij->t',P,D)*vol;r=m.e['line_search_r']
    predicted=r*derivative+.5*r*r*projected;error=per1-per0-predicted
    top=np.argsort(error)[-5:][::-1];J=np.linalg.det(F)
    row={'event':entry['event'],'exact_fem_curvature':float(raw.sum()),'projected_fem_curvature':float(projected.sum()),
        'negative_eigenspace_curvature_removed':float(-negative.sum()),'gradient_fd_relative_error':rel,
        'assembled_curvature':entry['assembled_xAx'],'al_curvature':entry['terms']['al']['analytic_curvature'],
        'unseparated_other_curvature':float(z@remainder),
        'unseparated_other_action_norm':float(np.linalg.norm(remainder)),
        'assembled_action_norm':float(np.linalg.norm(A@z)),
        'fem_predicted_delta':float(predicted.sum()),'fem_actual_delta':float(np.sum(per1-per0)),
        'fem_nonlinear_model_error':float(error.sum()),
        'top5_model_error_tets':[{'fem_tet_index':int(i),'vertex_ids':m.tets[i].tolist(),'initial_trial_J':float(J[i]),
            'actual_delta':float(per1[i]-per0[i]),'predicted_delta':float(predicted[i]),'error':float(error[i])} for i in top]}
    rows.append(row);out.write_text(json.dumps({'events':rows},indent=2));print(json.dumps({k:v for k,v in row.items() if k!='top5_model_error_tets'}),flush=True)
final={'events':rows,'passed':True,'scope':'Exact FEM Hessian and its PSD projection; remaining assembled terms are not separated.',
    'helper_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tools/event_model_v48.py']}}
out.write_text(json.dumps(final,indent=2))
